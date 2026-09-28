from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import copy
import torch
import torch.nn.functional as F


@dataclass
class LwFConfig:
    alpha: float = 0.5


class LwF:
    """Learning without Forgetting (teacher distillation)."""

    name = "lwf"

    def __init__(self, *, cfg: Optional[LwFConfig] = None) -> None:
        self.cfg = cfg or LwFConfig()
        if float(self.cfg.alpha) < 0.0:
            raise ValueError(f"LwF alpha must be >= 0, got {self.cfg.alpha}")
        self._teacher_models: Optional[Dict[str, torch.nn.Module]] = None
        self._teacher_uniform: Optional[torch.nn.Module] = None

    def has_teacher(self) -> bool:
        return self._teacher_models is not None and len(self._teacher_models) > 0

    def update_teacher_from_models(
        self,
        *,
        models: Dict[str, torch.nn.Module],
        model_uniform: Optional[torch.nn.Module] = None,
    ) -> None:
        """Snapshot current student weights as teacher (frozen models)."""
        teacher_models: Dict[str, torch.nn.Module] = {}
        for mk, m in models.items():
            tm = copy.deepcopy(m)
            for p in tm.parameters():
                p.requires_grad = False
            tm.eval()
            teacher_models[mk] = tm
        self._teacher_models = teacher_models
        if model_uniform is not None:
            tu = copy.deepcopy(model_uniform)
            for p in tu.parameters():
                p.requires_grad = False
            tu.eval()
            self._teacher_uniform = tu
        else:
            self._teacher_uniform = None

    def lwf_loss(
        self,
        *,
        inputs: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        student_z: Tuple[torch.Tensor, torch.Tensor],
        models: Dict[str, torch.nn.Module],
        model_uniform: Optional[torch.nn.Module],
        use_exo: bool,
        use_RN: bool,
    ) -> torch.Tensor:
        """Compute distillation loss on ranking scores."""
        if not self.has_teacher():
            return torch.zeros((), device=student_z[0].device)
        alpha = float(self.cfg.alpha)
        if alpha <= 0.0:
            return torch.zeros((), device=student_z[0].device)

        teacher_models = self._teacher_models or {}
        device = student_z[0].device
        for tm in teacher_models.values():
            tm.to(device=device)
            tm.eval()
        if self._teacher_uniform is not None:
            self._teacher_uniform.to(device=device)
            self._teacher_uniform.eval()

        with torch.no_grad():
            input_var1, input_var2, input_exo_var = inputs
            if use_RN and use_exo and input_exo_var is not None:
                input_var1 = torch.cat((input_var1, input_exo_var), dim=1)
                input_var2 = torch.cat((input_var2, input_exo_var), dim=1)

            out1_all = None
            out2_all = None
            for k in teacher_models.keys():
                all1, _, _ = teacher_models[k](input_var1)
                all2, _, _ = teacher_models[k](input_var2)
                o1 = all1.mean(dim=1)
                o2 = all2.mean(dim=1)
                out1_all = o1 if out1_all is None else (out1_all + o1)
                out2_all = o2 if out2_all is None else (out2_all + o2)
            teacher_z = (out1_all, out2_all)

        # NOTE:
        # In continual settings, the student output dimension can grow across tasks
        # (e.g., more action/skill candidates). The teacher snapshot from the previous
        # task may therefore have a smaller last-dimension than the current student.
        # We distill only on the overlapping part to avoid shape mismatch crashes.
        def _align_last_dim(
            s: torch.Tensor, t: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if s.shape == t.shape:
                return s, t
            # Prefer aligning only the last dimension when prefixes match.
            if s.ndim == t.ndim and s.shape[:-1] == t.shape[:-1]:
                d = min(int(s.shape[-1]), int(t.shape[-1]))
                if d <= 0:
                    # Degenerate case: no overlap, return zero loss later.
                    return s[..., :0], t[..., :0]
                return s[..., :d], t[..., :d]
            raise RuntimeError(
                f"[LwF] teacher/student shape mismatch: student={tuple(s.shape)} teacher={tuple(t.shape)}"
            )

        s0, t0 = _align_last_dim(student_z[0], teacher_z[0])
        s1, t1 = _align_last_dim(student_z[1], teacher_z[1])
        if s0.numel() == 0 or s1.numel() == 0:
            return torch.zeros((), device=student_z[0].device)  # no overlap
        loss = F.mse_loss(s0, t0, reduction="mean")
        loss = loss + F.mse_loss(s1, t1, reduction="mean")
        return loss * alpha
