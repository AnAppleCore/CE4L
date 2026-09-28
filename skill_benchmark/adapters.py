from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


class ChannelMLPAdapter(nn.Module):
    """Channel-mixing adapter applied per time step.

    x: [B,T,C] -> y: [B,T,C]
    Implemented as residual bottleneck MLP with a learnable residual scale (init 0).
    """

    def __init__(self, *, dim: int, bottleneck: int = 64, use_layernorm: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.use_layernorm = bool(use_layernorm)

        self.ln = nn.LayerNorm(self.dim) if self.use_layernorm else nn.Identity()
        self.fc1 = nn.Linear(self.dim, self.bottleneck)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.bottleneck, self.dim)

        # Residual scale: start from 0 so adapter is identity at init.
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))

        # Initialize close to identity behavior
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ChannelMLPAdapter expects x [B,T,C], got shape={tuple(x.shape)}")
        if int(x.shape[-1]) != self.dim:
            raise ValueError(f"ChannelMLPAdapter dim mismatch: expected C={self.dim}, got C={int(x.shape[-1])}")
        h = self.ln(x)
        d = self.fc2(self.act(self.fc1(h)))
        return x + self.alpha * d


@dataclass
class MixtureSpec:
    task_ids: torch.Tensor  # [B,L] int64 task ids
    weights: torch.Tensor  # [B,L] float weights


class AdapterBank(nn.Module):
    """Holds per-task adapters.

    Training uses ONLY current task adapter.
    Inference can mix adapters using MixtureSpec (task-id unknown, computed by router).
    """

    def __init__(self, *, input_dim: int, bottleneck: int = 64, use_layernorm: bool = True) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.bottleneck = int(bottleneck)
        self.use_layernorm = bool(use_layernorm)

        # Anchor buffer so `.to(device)` on the bank persists even before adapters are added.
        # Newly created adapters will be moved to this device.
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

        self._adapters: Dict[int, ChannelMLPAdapter] = {}
        self._current_task: Optional[int] = None

    def task_ids(self) -> List[int]:
        return list(sorted(int(t) for t in self._adapters.keys()))

    def num_tasks(self) -> int:
        return int(len(self._adapters))

    def has_task(self, task_id: int) -> bool:
        return int(task_id) in self._adapters

    def get(self, task_id: int) -> ChannelMLPAdapter:
        tid = int(task_id)
        if tid not in self._adapters:
            raise KeyError(f"AdapterBank: task_id={tid} not found.")
        return self._adapters[tid]

    def add_task(self, task_id: int, *, init_from_task: Optional[int] = None) -> None:
        tid = int(task_id)
        if tid in self._adapters:
            return
        ad = ChannelMLPAdapter(dim=self.input_dim, bottleneck=self.bottleneck, use_layernorm=self.use_layernorm)
        if init_from_task is not None and int(init_from_task) in self._adapters:
            src = self._adapters[int(init_from_task)]
            ad.load_state_dict(src.state_dict(), strict=True)
        # Ensure adapter is on same device as the bank (important for task-end dynamic creation).
        ad = ad.to(device=self._device_anchor.device)
        self._adapters[tid] = ad
        self.add_module(f"adapter_task_{tid:02d}", ad)

    def set_current_task(self, task_id: int) -> None:
        tid = int(task_id)
        if tid not in self._adapters:
            raise KeyError(f"set_current_task: task_id={tid} not in bank. Call add_task first.")
        self._current_task = tid

    def current_task(self) -> Optional[int]:
        return self._current_task

    def freeze_all(self) -> None:
        for ad in self._adapters.values():
            for p in ad.parameters():
                p.requires_grad = False

    def freeze_all_except(self, task_id: int) -> None:
        tid = int(task_id)
        for t, ad in self._adapters.items():
            req = (int(t) == tid)
            for p in ad.parameters():
                p.requires_grad = bool(req)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ONLY current adapter."""
        if self._current_task is None:
            return x
        return self.get(self._current_task)(x)

    @torch.no_grad()
    def forward_mixture(self, x: torch.Tensor, mix: MixtureSpec) -> torch.Tensor:
        """Mix adapters according to per-sample weights. Does not require gradients (inference)."""
        if self.num_tasks() <= 0:
            return x
        if x.dim() != 3:
            raise ValueError(f"forward_mixture expects x [B,T,C], got shape={tuple(x.shape)}")
        B = int(x.shape[0])
        task_ids = mix.task_ids
        weights = mix.weights
        if task_ids.shape[0] != B or weights.shape[0] != B:
            raise ValueError("MixtureSpec batch mismatch.")
        if task_ids.shape != weights.shape:
            raise ValueError("MixtureSpec shapes must match.")

        device = x.device
        out = torch.zeros_like(x)
        # Iterate selected slots (L small, default 2)
        L = int(task_ids.shape[1])
        for j in range(L):
            tids_j = task_ids[:, j].to(device=device)
            w_j = weights[:, j].to(device=device).view(B, 1, 1)
            # For each unique task id in this slot, apply its adapter to the corresponding subset
            for tid in torch.unique(tids_j).tolist():
                tid_int = int(tid)
                if tid_int not in self._adapters:
                    continue
                mask = (tids_j == tid_int)
                if not torch.any(mask):
                    continue
                x_sub = x[mask]
                y_sub = self._adapters[tid_int](x_sub)
                out[mask] = out[mask] + (w_j[mask] * y_sub)
        return out

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        payload = {"tasks": self.task_ids(), "input_dim": self.input_dim, "bottleneck": self.bottleneck, "use_layernorm": self.use_layernorm}
        torch.save(payload, os.path.join(output_dir, "adapter_bank_meta.pt"))
        for tid, ad in self._adapters.items():
            torch.save(ad.state_dict(), os.path.join(output_dir, f"adapter_task_{tid:02d}.pt"))

