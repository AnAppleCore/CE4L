from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .utils import (
    collate_samples,
    merge_batches,
    split_batch_to_samples,
    take_batch_indices,
    to_cpu_detached,
    to_device,
)


@dataclass
class DERPPConfig:
    """DER++ hyperparameters.

    - buffer_ratio: memory size relative to total train samples.
    - replay_batch_ratio: fraction of each batch replaced by replay samples (effective batch size stays constant).
    - distill_alpha: weight for distillation loss on replay samples (MSE).
    """

    buffer_ratio: float = 0.2
    replay_batch_ratio: float = 0.2
    distill_alpha: float = 0.5


def _infer_device_from_nested(x: Any) -> Optional[torch.device]:
    if torch.is_tensor(x):
        return x.device
    if isinstance(x, dict):
        for v in x.values():
            d = _infer_device_from_nested(v)
            if d is not None:
                return d
        return None
    if isinstance(x, (list, tuple)):
        for v in x:
            d = _infer_device_from_nested(v)
            if d is not None:
                return d
        return None
    return None


def _mse_nested(a: Any, b: Any) -> torch.Tensor:
    """Compute mean MSE over nested tensors."""
    if torch.is_tensor(a) and torch.is_tensor(b):
        return F.mse_loss(a, b, reduction="mean")
    if isinstance(a, dict) and isinstance(b, dict):
        keys = [k for k in a.keys() if k in b]
        if not keys:
            raise ValueError("DERPP mse_nested: empty dict intersection")
        return sum(_mse_nested(a[k], b[k]) for k in keys)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("DERPP mse_nested: length mismatch")
        return sum(_mse_nested(a[i], b[i]) for i in range(len(a)))
    raise TypeError(f"DERPP mse_nested: unsupported types a={type(a)} b={type(b)}")


class DERPP:
    """DER++ (Dark Experience Replay++).

    Stores a balanced replay buffer (like ER) but also caches a per-sample distillation target `z`
    computed from the model at the time the sample is added to memory.

    Training-time behavior:
    - Mix replay into the current batch (replacement, constant effective batch size).
    - Add distillation loss on replay items only: alpha * MSE(z_cur, z_mem).

    Notes for this repo:
    - `z` is benchmark-specific:
      - AAP: logits used for classification (source stream).
      - Skill: aggregate ranking scores (output1_all/output2_all).
      - Association: (image_embed, text_embed) embeddings (per-sample).
    """

    name = "derpp"

    def __init__(self, *, cfg: Optional[DERPPConfig] = None, seed: int = 0) -> None:
        self.cfg = cfg or DERPPConfig()
        if not (0.0 <= float(self.cfg.buffer_ratio) <= 1.0):
            raise ValueError(f"DERPP buffer_ratio must be in [0,1], got {self.cfg.buffer_ratio}")
        if not (0.0 <= float(self.cfg.replay_batch_ratio) <= 1.0):
            raise ValueError(f"DERPP replay_batch_ratio must be in [0,1], got {self.cfg.replay_batch_ratio}")
        if float(self.cfg.distill_alpha) < 0.0:
            raise ValueError(f"DERPP distill_alpha must be >= 0, got {self.cfg.distill_alpha}")

        self._rng = np.random.RandomState(int(seed))
        self._capacity: Optional[int] = None
        # per-task list of items: (sample_in_cpu, sample_z_cpu)
        self._per_task: Dict[int, List[Tuple[Any, Any]]] = {}
        self._tasks_seen: List[int] = []

        # iteration cache (set by mix_in_replay)
        self._last_replay_k: int = 0
        self._last_replay_z_cpu: Optional[Any] = None
        self._last_total_bs: int = 0

    def configure_total_capacity(self, *, total_train_samples: int) -> None:
        total = int(total_train_samples)
        if total <= 0:
            raise ValueError(f"DERPP total_train_samples must be > 0, got {total_train_samples}")
        cap = int(round(float(self.cfg.buffer_ratio) * float(total)))
        if float(self.cfg.buffer_ratio) > 0 and cap <= 0:
            cap = 1
        if cap < 0:
            raise ValueError("DERPP capacity computed negative (bug).")
        self._capacity = int(cap)

    @property
    def capacity(self) -> int:
        if self._capacity is None:
            raise RuntimeError("DERPP not configured: call configure_total_capacity(total_train_samples=...) first.")
        return int(self._capacity)

    def is_enabled(self) -> bool:
        return self.capacity > 0 and float(self.cfg.replay_batch_ratio) > 0.0

    def num_items(self) -> int:
        return int(sum(len(v) for v in self._per_task.values()))

    def _target_quota_per_task(self) -> int:
        if len(self._tasks_seen) == 0:
            return 0
        return int(self.capacity // len(self._tasks_seen))

    def update_memory_from_loader(
        self,
        *,
        task_id: int,
        loader: Any,
        model: torch.nn.Module,
        distill_target_fn: Callable[[Any, torch.nn.Module], Any],
        max_items_from_task: Optional[int] = None,
        distill_batch_size: int = 64,
    ) -> None:
        """At task end, sample a balanced subset into memory and store distillation targets.

        `distill_target_fn(batch_in, model) -> z_batch` must return a batch-aligned tensor/nested structure.
        """
        tid = int(task_id)
        if tid not in self._tasks_seen:
            self._tasks_seen.append(tid)

        quota = self._target_quota_per_task()
        if max_items_from_task is not None:
            quota = min(int(quota), int(max_items_from_task))
        if quota <= 0:
            self._per_task.setdefault(tid, [])
            return

        # 1) Reservoir sample inputs
        reservoir_in: List[Any] = []
        seen = 0
        for batch in loader:
            samples = split_batch_to_samples(batch)
            for s in samples:
                seen += 1
                s_cpu = to_cpu_detached(s)
                if len(reservoir_in) < quota:
                    reservoir_in.append(s_cpu)
                else:
                    j = int(self._rng.randint(0, seen))
                    if j < quota:
                        reservoir_in[j] = s_cpu

        if seen <= 0:
            raise RuntimeError(f"DERPP: task_id={tid} loader produced 0 samples; cannot build replay memory.")

        # 2) Compute distillation targets for selected samples
        model_was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        items: List[Tuple[Any, Any]] = []
        with torch.no_grad():
            bs = max(int(distill_batch_size), 1)
            for i in range(0, len(reservoir_in), bs):
                chunk_in_cpu = reservoir_in[i : i + bs]
                batch_in_cpu = collate_samples(chunk_in_cpu)
                batch_in_dev = to_device(batch_in_cpu, device)
                z_dev = distill_target_fn(batch_in_dev, model)
                z_cpu = to_cpu_detached(z_dev)
                z_samples = split_batch_to_samples(z_cpu)
                if len(z_samples) != len(chunk_in_cpu):
                    raise RuntimeError(
                        f"DERPP distill_target_fn returned {len(z_samples)} samples, expected {len(chunk_in_cpu)}"
                    )
                for s_in_cpu, s_z_cpu in zip(chunk_in_cpu, z_samples):
                    items.append((s_in_cpu, s_z_cpu))
        model.train(model_was_training)

        self._per_task[tid] = items
        self._rebalance_strict()

    def _rebalance_strict(self) -> None:
        cap = self.capacity
        if cap <= 0:
            self._per_task = {t: [] for t in self._tasks_seen}
            return
        if len(self._tasks_seen) == 0:
            return

        base = cap // len(self._tasks_seen)
        rem = cap - base * len(self._tasks_seen)
        quotas: Dict[int, int] = {}
        for i, t in enumerate(self._tasks_seen):
            quotas[int(t)] = int(base + (1 if i < rem else 0))

        for t in list(self._per_task.keys()):
            if t not in quotas:
                del self._per_task[t]

        for t, q in quotas.items():
            cur = self._per_task.get(t, [])
            if len(cur) <= q:
                self._per_task[t] = cur
                continue
            idx = self._rng.choice(len(cur), size=int(q), replace=False)
            self._per_task[t] = [cur[int(i)] for i in idx]

        total = self.num_items()
        if total > cap:
            raise RuntimeError(f"DERPP rebalance bug: total items {total} > capacity {cap}")

    def _sample_replay(self, *, cur_batch_size: int) -> Tuple[Optional[Any], int]:
        """Return (replay_batch_in, k). Also caches replay z targets for distillation."""
        self._last_replay_k = 0
        self._last_replay_z_cpu = None
        self._last_total_bs = int(cur_batch_size)

        if not self.is_enabled():
            return None, 0
        total = self.num_items()
        if total <= 0:
            return None, 0

        bs = int(cur_batch_size)
        k = int(round(float(self.cfg.replay_batch_ratio) * float(bs)))
        if k <= 0:
            return None, 0

        all_items: List[Tuple[Any, Any]] = []
        for t in self._tasks_seen:
            all_items.extend(self._per_task.get(int(t), []))
        if len(all_items) <= 0:
            return None, 0

        idx = self._rng.choice(len(all_items), size=int(k), replace=(len(all_items) < k))
        picked = [all_items[int(i)] for i in idx]
        picked_in = [p[0] for p in picked]
        picked_z = [p[1] for p in picked]

        rb_in = collate_samples(picked_in)
        rb_z = collate_samples(picked_z)

        self._last_replay_k = int(k)
        self._last_replay_z_cpu = to_cpu_detached(rb_z)
        return rb_in, int(k)

    def mix_in_replay(self, *, cur_batch: Any, cur_batch_size: int) -> Any:
        """Mix replay into current batch (replacement, constant effective batch size)."""
        bs = int(cur_batch_size)
        if bs <= 0:
            raise ValueError(f"DERPP mix_in_replay: cur_batch_size must be > 0, got {cur_batch_size}")

        rb_in, k = self._sample_replay(cur_batch_size=bs)
        if rb_in is None or k <= 0:
            return cur_batch

        keep = bs - int(k)
        if keep <= 0:
            # full replay
            return rb_in

        idx_keep = self._rng.choice(bs, size=int(keep), replace=False)
        idx_keep = [int(i) for i in idx_keep]
        cur_sub = take_batch_indices(cur_batch, idx_keep)
        merged = merge_batches(cur_sub, rb_in)
        # replay items are at the end by construction
        self._last_total_bs = int(bs)
        return merged

    def distill_loss(self, current_z: Any) -> torch.Tensor:
        """Compute DER++ distillation loss on replay items only.

        `current_z` must be batch-aligned, and *must correspond to the same samples* as the
        mixed batch produced by the latest `mix_in_replay` call.
        """
        k = int(self._last_replay_k)
        if k <= 0 or self._last_replay_z_cpu is None or float(self.cfg.distill_alpha) <= 0.0:
            # return a zero tensor on the right device if possible
            dev = _infer_device_from_nested(current_z)
            return torch.zeros((), device=dev) if dev is not None else torch.zeros(())

        bs = int(self._last_total_bs)
        if bs <= 0:
            dev = _infer_device_from_nested(current_z)
            return torch.zeros((), device=dev) if dev is not None else torch.zeros(())

        # select replay slice (last k)
        idx = list(range(bs - k, bs))
        z_cur_rep = take_batch_indices(current_z, idx)

        dev = _infer_device_from_nested(current_z)
        if dev is None:
            # fall back to model-independent cpu compute
            z_mem_rep = self._last_replay_z_cpu
        else:
            z_mem_rep = to_device(self._last_replay_z_cpu, dev)

        loss = _mse_nested(z_cur_rep, z_mem_rep)
        return loss * float(self.cfg.distill_alpha)

