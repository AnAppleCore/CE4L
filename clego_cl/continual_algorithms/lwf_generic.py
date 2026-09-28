from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import copy
import torch
import torch.nn.functional as F


@dataclass
class LwFGenericConfig:
    alpha: float = 0.5


class LwFGeneric:
    """Generic Learning without Forgetting (teacher distillation via MSE).

    This implementation is model-agnostic and only requires the caller to provide
    student/teacher tensors (logits or embeddings) of matching shapes.
    """

    name = "lwf"

    def __init__(self, *, cfg: Optional[LwFGenericConfig] = None) -> None:
        self.cfg = cfg or LwFGenericConfig()
        if float(self.cfg.alpha) < 0.0:
            raise ValueError(f"LwF alpha must be >= 0, got {self.cfg.alpha}")
        self._teacher: Optional[torch.nn.Module] = None

    def has_teacher(self) -> bool:
        return self._teacher is not None

    def update_teacher(self, model: torch.nn.Module) -> None:
        """Snapshot current student weights as teacher (frozen model)."""
        # Keep teacher on the same device as the student model. This is important for
        # wrappers like DataParallel, which require module params/buffers on device_ids[0].
        device = None
        try:
            device = next(model.parameters()).device
        except Exception:
            device = None

        tm = copy.deepcopy(model)
        if device is not None:
            tm = tm.to(device=device)
        for p in tm.parameters():
            p.requires_grad = False
        tm.eval()
        self._teacher = tm

    def teacher(self) -> Optional[torch.nn.Module]:
        return self._teacher

    def lwf_loss(self, student_z: torch.Tensor, teacher_z: torch.Tensor) -> torch.Tensor:
        """Compute MSE distillation loss (alpha-weighted)."""
        if not self.has_teacher():
            return torch.zeros((), device=student_z.device)
        alpha = float(self.cfg.alpha)
        if alpha <= 0.0:
            return torch.zeros((), device=student_z.device)
        loss = F.mse_loss(student_z, teacher_z, reduction="mean")
        return loss * alpha
