from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch


def _as_3d(x: torch.Tensor) -> torch.Tensor:
    """Ensure x is [B,T,C]."""
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected tensor with dim 2 or 3, got shape={tuple(x.shape)}")


def extract_r(x: torch.Tensor, *, M: int = 1) -> torch.Tensor:
    """Extract stable router representation r(x) from input features.

    - If x is [B,T,C]: time-chunk mean pooling with M segments -> [B, M*C]
    - If x is [B,C]: treated as M=1 -> [B, C]
    """
    if not torch.is_tensor(x):
        raise TypeError(f"extract_r expects torch.Tensor, got {type(x)}")
    if x.dim() == 2:
        # [B,C]
        if int(M) != 1:
            raise ValueError(f"extract_r: M must be 1 for vector inputs, got M={M}")
        return x

    x3 = _as_3d(x)
    B, T, C = x3.shape
    M = int(M)
    if M <= 0:
        raise ValueError(f"extract_r: M must be >= 1, got {M}")
    if M == 1:
        return x3.mean(dim=1)

    # Chunk into M approximately equal segments.
    # For simplicity (and determinism), use floor division boundaries.
    seg_means: List[torch.Tensor] = []
    for m in range(M):
        s = (m * T) // M
        e = ((m + 1) * T) // M
        if e <= s:
            # Degenerate chunk; fall back to global mean
            seg_means.append(x3.mean(dim=1))
        else:
            seg_means.append(x3[:, s:e, :].mean(dim=1))
    return torch.cat(seg_means, dim=1)  # [B, M*C]


@dataclass
class WhitenedSubspaceStats:
    """Per-task stats for the whitened-subspace router."""

    mu: np.ndarray  # [d]
    var: np.ndarray  # [d] (diagonal variance)
    Bw: np.ndarray  # [d, k+1] orthonormal basis in whitened space

    def to_torch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = torch.from_numpy(self.mu).to(device=device, dtype=torch.float32)
        var = torch.from_numpy(self.var).to(device=device, dtype=torch.float32)
        Bw = torch.from_numpy(self.Bw).to(device=device, dtype=torch.float32)
        return mu, var, Bw


class TaskWhitenedSubspaceRouter:
    """Whitened-subspace router: diagonal whitening + augmented whitened subspace residual.

    For each task t:
      1) Estimate mean mu_t and diagonal variance var_t from r(x) on the task train split.
      2) Define whitening weights w_t = 1/sqrt(var_t + eps).
      3) In whitened space, build an augmented subspace basis:
         - mean direction: m_w = normalize(mu_t ⊙ w_t)
         - variation directions: U_w = top-k PCA directions of (r - mu_t) ⊙ w_t
         - basis: B_w = orth([m_w, U_w])  (QR)
      4) Score sample r by residual ratio to this task basis (lower is better):
         e_t(r) = 1 - ||B_w^T (r ⊙ w_t)||^2 / (||r ⊙ w_t||^2 + eps)
    """

    def __init__(self, *, M: int = 1, k: int = 32, eps: float = 1e-6) -> None:
        self.M = int(M)
        self.k = int(k)
        self.eps = float(eps)
        self._stats: Dict[int, WhitenedSubspaceStats] = {}

    def task_ids(self) -> List[int]:
        return list(sorted(int(t) for t in self._stats.keys()))

    def num_tasks(self) -> int:
        return int(len(self._stats))

    def add_task_stats(self, task_id: int, *, mu: np.ndarray, var: np.ndarray, Bw: np.ndarray) -> None:
        tid = int(task_id)
        mu = np.asarray(mu, dtype=np.float32).reshape(-1)
        var = np.asarray(var, dtype=np.float32).reshape(-1)
        Bw = np.asarray(Bw, dtype=np.float32)
        if Bw.ndim != 2:
            raise ValueError(f"Bw must be 2D, got shape={Bw.shape}")
        if int(Bw.shape[0]) != int(mu.shape[0]):
            raise ValueError(f"Bw rows must match mu dim, got Bw={Bw.shape} mu={mu.shape}")
        if mu.shape != var.shape:
            raise ValueError(f"mu/var shape mismatch: mu={mu.shape} var={var.shape}")
        self._stats[tid] = WhitenedSubspaceStats(mu=mu, var=var, Bw=Bw)

    def fit_from_loader(
        self,
        *,
        task_id: int,
        loader: Iterable,
        device: Union[torch.device, str] = "cpu",
        verbose: bool = False,
    ) -> WhitenedSubspaceStats:
        """Fit (mu, var, Bw) from a task-specific loader. Uses only ego vid1/vid2 tensors."""
        dev = torch.device(device)
        tid = int(task_id)
        M = int(self.M)
        k = int(self.k)

        # Pass 1: mean + diag variance
        n = 0
        sum_r: Optional[torch.Tensor] = None
        sum_r2: Optional[torch.Tensor] = None
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                input1 = batch[0]
                input2 = batch[1]
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            x1 = input1.to(dev, dtype=torch.float32, non_blocking=True)
            x2 = input2.to(dev, dtype=torch.float32, non_blocking=True)
            r1 = extract_r(x1, M=M)  # [B,d]
            r2 = extract_r(x2, M=M)
            r = torch.cat([r1, r2], dim=0)  # [2B,d]
            if sum_r is None:
                sum_r = r.sum(dim=0)
                sum_r2 = (r * r).sum(dim=0)
            else:
                sum_r = sum_r + r.sum(dim=0)
                sum_r2 = sum_r2 + (r * r).sum(dim=0)  # type: ignore[operator]
            n += int(r.shape[0])
        if n <= 1 or sum_r is None or sum_r2 is None:
            raise RuntimeError(f"fit_from_loader: empty/too-small loader for task_id={tid} (n={n})")

        mu = (sum_r / float(n)).detach()
        ex2 = (sum_r2 / float(n)).detach()
        var = (ex2 - mu * mu).clamp(min=0.0)
        d = int(mu.numel())
        if verbose:
            print(f"[router fit whitened-subspace] task={tid} n={n} d={d} M={M} k={k}")

        # Pass 2: covariance in whitened space (centered) for U_w
        w = 1.0 / torch.sqrt(var + float(self.eps))  # [d]
        S = torch.zeros((d, d), dtype=torch.float64, device=dev)
        for batch in loader:
            input1 = batch[0]
            input2 = batch[1]
            x1 = input1.to(dev, dtype=torch.float32, non_blocking=True)
            x2 = input2.to(dev, dtype=torch.float32, non_blocking=True)
            r1 = extract_r(x1, M=M).to(torch.float64)
            r2 = extract_r(x2, M=M).to(torch.float64)
            r = torch.cat([r1, r2], dim=0)  # [2B,d]
            z = (r - mu.to(torch.float64)) * w.to(torch.float64)  # whitened centered
            S = S + (z.transpose(0, 1) @ z)
        cov = (S / float(max(n - 1, 1))).to(dtype=torch.float32)
        cov_np = cov.detach().cpu().numpy()
        eig, V = np.linalg.eigh(cov_np)  # ascending, V columns orthonormal
        order = np.argsort(eig)[::-1]
        order = order[: int(min(k, V.shape[1]))]
        Uw = V[:, order].astype(np.float32, copy=False)  # [d,k]
        mw = (mu * w).to(torch.float32)
        mw = mw / (mw.norm(p=2).clamp(min=float(self.eps)))
        A = np.concatenate([mw.detach().cpu().numpy().astype(np.float32, copy=False)[:, None], Uw], axis=1)  # [d,1+k]
        Bw, _ = np.linalg.qr(A)  # [d,1+k]

        st = WhitenedSubspaceStats(
            mu=mu.detach().cpu().numpy().astype(np.float32, copy=False),
            var=var.detach().cpu().numpy().astype(np.float32, copy=False),
            Bw=Bw.astype(np.float32, copy=False),
        )
        self._stats[tid] = st
        return st

    def augmented_residual_scores(self, r: torch.Tensor, *, device: torch.device) -> Tuple[torch.Tensor, List[int]]:
        """Return residual ratio to augmented whitened subspace (mw + Uw): [B, Ttasks]. Lower is better."""
        if self.num_tasks() <= 0:
            raise RuntimeError("augmented_residual_scores called with empty router (no task stats).")
        r = r.to(device=device, dtype=torch.float32)
        tids = self.task_ids()
        B, d = int(r.shape[0]), int(r.shape[1])
        out = torch.empty((B, len(tids)), device=device, dtype=torch.float32)
        eps = float(self.eps)
        for j, tid in enumerate(tids):
            sp = self._stats[int(tid)]
            mu, var, Bw = sp.to_torch(device)
            if int(mu.numel()) != d or int(var.numel()) != d or int(Bw.shape[0]) != d:
                raise RuntimeError(f"Router dim mismatch for task_id={tid}: r_dim={d}, mu_dim={mu.numel()}, var_dim={var.numel()}, Bw={tuple(Bw.shape)}")
            w = 1.0 / torch.sqrt(var.clamp(min=0.0) + eps)
            x = r * w[None, :]  # uncentered whitened
            x2 = (x * x).sum(dim=1)
            proj = x @ Bw  # [B,1+k]
            proj2 = (proj * proj).sum(dim=1)
            out[:, j] = 1.0 - (proj2 / (x2 + eps))
        return out, tids

    def save_task(self, *, output_dir: str, task_id: int) -> None:
        os.makedirs(output_dir, exist_ok=True)
        tid = int(task_id)
        if tid not in self._stats:
            raise KeyError(f"save_task: task_id={tid} not found in router.")
        st = self._stats[tid]
        np.savez(
            os.path.join(output_dir, f"router_task_{tid:02d}.npz"),
            mu=st.mu,
            var=st.var,
            Bw=st.Bw,
            M=self.M,
            k=self.k,
            eps=self.eps,
            type="whitened_subspace",
        )

    def save_index(self, *, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        payload = {"tasks": self.task_ids(), "M": int(self.M), "k": int(self.k), "eps": float(self.eps), "type": "whitened_subspace"}
        with open(os.path.join(output_dir, "router_index.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

