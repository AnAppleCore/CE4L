from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


@dataclass
class EWCConfig:
    """EWC hyperparameters (diagonal Fisher)."""

    lambda_: float = 1e-2
    gamma: float = 1.0
    fisher_batches: int = 50


class EWC:
    """Elastic Weight Consolidation (online, diagonal Fisher)."""

    name = "ewc"

    def __init__(self, *, cfg: Optional[EWCConfig] = None) -> None:
        self.cfg = cfg or EWCConfig()
        if float(self.cfg.lambda_) < 0.0:
            raise ValueError(f"EWC lambda must be >= 0, got {self.cfg.lambda_}")
        if float(self.cfg.gamma) < 0.0:
            raise ValueError(f"EWC gamma must be >= 0, got {self.cfg.gamma}")
        if int(self.cfg.fisher_batches) < 0:
            raise ValueError(f"EWC fisher_batches must be >= 0, got {self.cfg.fisher_batches}")

        self._models: Optional[Dict[str, torch.nn.Module]] = None
        self._model_uniform: Optional[torch.nn.Module] = None
        self._theta_star: Dict[str, torch.Tensor] = {}
        self._fisher_diag: Dict[str, torch.Tensor] = {}

    def bind_models(self, *, models: Dict[str, torch.nn.Module], model_uniform: Optional[torch.nn.Module] = None) -> None:
        self._models = models
        self._model_uniform = model_uniform

    def _iter_named_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        if self._models is None:
            raise RuntimeError("EWC.bind_models must be called before use.")
        params: List[Tuple[str, torch.nn.Parameter]] = []
        for mk, m in self._models.items():
            for name, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                params.append((f"models.{mk}.{name}", p))
        if self._model_uniform is not None:
            for name, p in self._model_uniform.named_parameters():
                if not p.requires_grad:
                    continue
                params.append((f"model_uniform.{name}", p))
        return params

    def _zero_grads(self) -> None:
        if self._models is None:
            return
        for m in self._models.values():
            try:
                m.zero_grad(set_to_none=True)
            except TypeError:
                m.zero_grad()
        if self._model_uniform is not None:
            try:
                self._model_uniform.zero_grad(set_to_none=True)
            except TypeError:
                self._model_uniform.zero_grad()

    def update_fisher_from_loader(
        self,
        *,
        loader: Iterable[Any],
        loss_fn,
        max_batches: Optional[int] = None,
    ) -> None:
        """Estimate diagonal Fisher via grad^2 averaged over batches."""
        params = self._iter_named_params()
        if len(params) == 0:
            raise RuntimeError("EWC: no trainable parameters found for Fisher estimation.")

        fisher: Dict[str, torch.Tensor] = {n: torch.zeros_like(p, device="cpu") for n, p in params}

        # Switch to eval for deterministic stats, but restore original state.
        train_states: Dict[str, bool] = {}
        if self._models is not None:
            for k, m in self._models.items():
                train_states[f"models.{k}"] = bool(m.training)
                m.eval()
        if self._model_uniform is not None:
            train_states["model_uniform"] = bool(self._model_uniform.training)
            self._model_uniform.eval()

        n_batches = 0
        for batch in loader:
            if max_batches is not None and int(max_batches) > 0 and n_batches >= int(max_batches):
                break
            self._zero_grads()
            loss = loss_fn(batch)
            if loss is None:
                raise RuntimeError("EWC loss_fn returned None.")
            loss.backward()
            for name, p in params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                # Defensive: AMP / unstable batches can produce NaN/Inf grads; keep Fisher finite.
                if not torch.isfinite(g).all():
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                fisher[name] += (g.cpu() ** 2)
            n_batches += 1

        # Restore original train/eval state
        if self._models is not None:
            for k, m in self._models.items():
                if train_states.get(f"models.{k}", False):
                    m.train()
        if self._model_uniform is not None and train_states.get("model_uniform", False):
            self._model_uniform.train()

        if n_batches <= 0:
            raise RuntimeError("EWC Fisher estimation saw 0 batches.")

        for name in fisher:
            fisher[name] = torch.nan_to_num(
                fisher[name] / float(n_batches),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        # Online merge
        if self._fisher_diag:
            gamma = float(self.cfg.gamma)
            merged: Dict[str, torch.Tensor] = {}
            for name, f_new in fisher.items():
                f_old = self._fisher_diag.get(name, None)
                if f_old is None:
                    merged[name] = f_new
                else:
                    merged[name] = (gamma * f_old) + f_new
            self._fisher_diag = merged
        else:
            self._fisher_diag = fisher

        # Snapshot current params
        self._theta_star = {name: p.detach().cpu().clone() for name, p in params}

    def regularization_loss(self) -> torch.Tensor:
        """EWC penalty term computed on current parameters."""
        if not self._fisher_diag or not self._theta_star:
            return torch.zeros((), device=self._infer_device())
        lam = float(self.cfg.lambda_)
        if lam <= 0.0:
            return torch.zeros((), device=self._infer_device())

        loss = torch.zeros((), device=self._infer_device())
        for name, p in self._iter_named_params():
            if name not in self._fisher_diag or name not in self._theta_star:
                continue
            f = self._fisher_diag[name].to(device=p.device, dtype=p.dtype)
            theta = self._theta_star[name].to(device=p.device, dtype=p.dtype)
            loss = loss + (f * (p - theta).pow(2)).sum()
        return loss * (lam * 0.5)

    def _infer_device(self) -> torch.device:
        if self._models:
            for m in self._models.values():
                for p in m.parameters():
                    return p.device
        if self._model_uniform is not None:
            for p in self._model_uniform.parameters():
                return p.device
        return torch.device("cpu")
