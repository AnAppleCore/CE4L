from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from skill_benchmark.adapters import ChannelMLPAdapter


@dataclass
class L2PSelection:
    indices: torch.Tensor  # [B,K] adapter indices
    match: torch.Tensor    # [B,K] match scores (lower is better)


class L2PPool(nn.Module):
    """L2P prompt pool implemented with adapters + learnable keys."""

    def __init__(
        self,
        *,
        pool_size: int,
        topk: int,
        adapter_dim: int,
        key_dim: int,
        adapter_bottleneck: int = 64,
        diversed_selection: bool = True,
        batchwise_selection: bool = False,
    ) -> None:
        super().__init__()
        self.pool_size = int(pool_size)
        self.topk = int(topk)
        self.adapter_dim = int(adapter_dim)
        self.key_dim = int(key_dim)
        self.adapter_bottleneck = int(adapter_bottleneck)
        self.diversed_selection = bool(diversed_selection)
        self.batchwise_selection = bool(batchwise_selection)

        if self.pool_size <= 0:
            raise ValueError(f"pool_size must be > 0, got {self.pool_size}")
        if self.topk <= 0 or self.topk > self.pool_size:
            raise ValueError(f"topk must be in [1, pool_size], got topk={self.topk} pool_size={self.pool_size}")

        self.keys = nn.Parameter(torch.randn(self.pool_size, self.key_dim, requires_grad=True))
        self.adapters = nn.ModuleList(
            [
                ChannelMLPAdapter(dim=self.adapter_dim, bottleneck=self.adapter_bottleneck, use_layernorm=True)
                for _ in range(self.pool_size)
            ]
        )

        nn.init.uniform_(self.keys, -1.0, 1.0)

        # Frequency tracking for diversified selection
        self.register_buffer("frequency", torch.ones(self.pool_size))
        self.register_buffer("counter", torch.zeros(self.pool_size))

    def cosine_match(self, query: torch.Tensor) -> torch.Tensor:
        """Return 1 - cosine(query, keys): [B, pool_size], lower is better."""
        if query.dim() != 2:
            raise ValueError(f"query must be [B,D], got {tuple(query.shape)}")
        if int(query.shape[1]) != self.key_dim:
            raise ValueError(f"query dim mismatch: expected {self.key_dim}, got {int(query.shape[1])}")
        return 1.0 - F.cosine_similarity(query.unsqueeze(1), self.keys, dim=-1)

    def select_topk(self, match: torch.Tensor, *, training: bool) -> L2PSelection:
        """Select top-k adapters based on match scores (lower is better)."""
        if match.dim() != 2 or int(match.shape[1]) != self.pool_size:
            raise ValueError(f"match must be [B,{self.pool_size}], got {tuple(match.shape)}")

        scores = match
        if training and self.diversed_selection:
            freq = F.normalize(self.frequency, p=1, dim=0)
            scores = scores * freq.view(1, -1)

        _, idx = scores.topk(self.topk, dim=-1, largest=False, sorted=True)
        if self.batchwise_selection:
            uniq, counts = idx.unique(sorted=True, return_counts=True)
            _, mosts = counts.topk(self.topk, largest=True, sorted=True)
            idx = uniq[mosts].clone().expand(idx.shape[0], -1)

        # update frequency counters (training only)
        if training:
            self.counter += torch.bincount(idx.reshape(-1), minlength=self.pool_size).to(self.counter.device)
        selected_match = match.gather(1, idx)
        return L2PSelection(indices=idx, match=selected_match)

    def apply_adapters(self, x: torch.Tensor, sel: L2PSelection) -> torch.Tensor:
        """Apply selected adapters and average their outputs."""
        if x.dim() != 3:
            raise ValueError(f"x must be [B,T,C], got {tuple(x.shape)}")
        if int(x.shape[-1]) != self.adapter_dim:
            raise ValueError(f"adapter dim mismatch: expected {self.adapter_dim}, got {int(x.shape[-1])}")
        idx = sel.indices
        B = int(x.shape[0])
        K = int(idx.shape[1])
        if K <= 0:
            return x
        out = torch.zeros_like(x)
        for j in range(K):
            tids_j = idx[:, j].to(device=x.device)
            for tid in torch.unique(tids_j).tolist():
                tid_int = int(tid)
                if tid_int < 0 or tid_int >= self.pool_size:
                    continue
                mask = (tids_j == tid_int)
                if not torch.any(mask):
                    continue
                x_sub = x[mask]
                y_sub = self.adapters[tid_int](x_sub)
                out[mask] = out[mask] + y_sub
        return out / float(K)

    def update_frequency(self) -> torch.Tensor:
        """Update frequency from counter and reset counter."""
        if self.training:
            self.frequency += self.counter
        counter = self.counter.clone()
        self.counter *= 0
        if self.training:
            return self.frequency - 1
        return counter

    def save(self, output_dir: str, *, filename: str = "l2p_pool.pt") -> str:
        """Save L2P pool state + metadata to a single file."""
        import os

        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "meta": {
                "pool_size": self.pool_size,
                "topk": self.topk,
                "adapter_dim": self.adapter_dim,
                "key_dim": self.key_dim,
                "adapter_bottleneck": self.adapter_bottleneck,
                "diversed_selection": self.diversed_selection,
                "batchwise_selection": self.batchwise_selection,
            },
            "state_dict": self.state_dict(),
        }
        path = os.path.join(output_dir, filename)
        torch.save(payload, path)
        return path

    @staticmethod
    def load(path: str, *, device: Optional[torch.device] = None) -> "L2PPool":
        """Load L2P pool from a file created by save()."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})
        pool = L2PPool(
            pool_size=int(meta.get("pool_size", 1)),
            topk=int(meta.get("topk", 1)),
            adapter_dim=int(meta.get("adapter_dim", 1)),
            key_dim=int(meta.get("key_dim", 1)),
            adapter_bottleneck=int(meta.get("adapter_bottleneck", 64)),
            diversed_selection=bool(meta.get("diversed_selection", True)),
            batchwise_selection=bool(meta.get("batchwise_selection", False)),
        )
        state = payload.get("state_dict", {})
        pool.load_state_dict(state, strict=True)
        if device is not None:
            pool = pool.to(device=device)
        return pool
