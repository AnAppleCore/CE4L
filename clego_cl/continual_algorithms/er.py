from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .utils import collate_samples, merge_batches, split_batch_to_samples, take_batch_indices, to_cpu_detached
from .derpp import DERPP, DERPPConfig
from .ewc import EWC, EWCConfig
from .lwf import LwF, LwFConfig


def _as_name(x: Any) -> str:
    return str(x or "").strip().lower()


@dataclass
class ERConfig:
    buffer_ratio: float = 0.2
    replay_batch_ratio: float = 0.2


class ExperienceReplay:
    """Experience Replay (ER) for task-incremental continual learning.

    Design goals (matching your requirements):
    - Minimal hyperparams: only buffer_ratio and replay_batch_ratio.
    - Strict: no "skip on error" fallbacks; raise if misconfigured.
    - Balanced memory: roughly equal quota per seen task (after each task ends).
    - Buffer stores batch-ready tensors (CPU, detached) so it can be merged into future batches.
    """

    name = "er"

    def __init__(self, *, cfg: Optional[ERConfig] = None, seed: int = 0) -> None:
        self.cfg = cfg or ERConfig()
        if not (0.0 <= float(self.cfg.buffer_ratio) <= 1.0):
            raise ValueError(f"ER buffer_ratio must be in [0,1], got {self.cfg.buffer_ratio}")
        if not (0.0 <= float(self.cfg.replay_batch_ratio) <= 1.0):
            raise ValueError(f"ER replay_batch_ratio must be in [0,1], got {self.cfg.replay_batch_ratio}")
        self._rng = np.random.RandomState(int(seed))

        self._capacity: Optional[int] = None
        self._per_task: Dict[int, List[Any]] = {}
        self._tasks_seen: List[int] = []

    # -------------------------
    # Lifecycle / configuration
    # -------------------------
    def configure_total_capacity(self, *, total_train_samples: int) -> None:
        total = int(total_train_samples)
        if total <= 0:
            raise ValueError(f"ER total_train_samples must be > 0, got {total_train_samples}")
        cap = int(round(float(self.cfg.buffer_ratio) * float(total)))
        if float(self.cfg.buffer_ratio) > 0 and cap <= 0:
            cap = 1
        if cap < 0:
            raise ValueError("ER capacity computed negative (bug).")
        self._capacity = int(cap)

    @property
    def capacity(self) -> int:
        if self._capacity is None:
            raise RuntimeError("ER not configured: call configure_total_capacity(total_train_samples=...) first.")
        return int(self._capacity)

    def is_enabled(self) -> bool:
        return self.capacity > 0 and float(self.cfg.replay_batch_ratio) > 0.0

    def num_items(self) -> int:
        return int(sum(len(v) for v in self._per_task.values()))

    # -------------
    # Memory update
    # -------------
    def _target_quota_per_task(self) -> int:
        if len(self._tasks_seen) == 0:
            return 0
        return int(self.capacity // len(self._tasks_seen))

    def update_memory_from_loader(
        self,
        *,
        task_id: int,
        loader: Any,
        max_items_from_task: Optional[int] = None,
    ) -> None:
        """At task end, sample a balanced subset of this task's training data into memory.

        Sampling is uniform over loader samples using reservoir sampling.
        """
        tid = int(task_id)
        if tid not in self._tasks_seen:
            self._tasks_seen.append(tid)

        quota = self._target_quota_per_task()
        if max_items_from_task is not None:
            quota = min(int(quota), int(max_items_from_task))
        if quota <= 0:
            # capacity can be 0 if buffer_ratio==0; strict but intentional.
            self._per_task.setdefault(tid, [])
            return

        # Reservoir sample 'quota' examples from the loader (per-sample, not per-batch).
        reservoir: List[Any] = []
        seen = 0

        for batch in loader:
            samples = split_batch_to_samples(batch)
            for s in samples:
                seen += 1
                s_cpu = to_cpu_detached(s)
                if len(reservoir) < quota:
                    reservoir.append(s_cpu)
                else:
                    j = int(self._rng.randint(0, seen))
                    if j < quota:
                        reservoir[j] = s_cpu

        if seen <= 0:
            raise RuntimeError(f"ER: task_id={tid} loader produced 0 samples; cannot build replay memory.")

        self._per_task[tid] = reservoir
        self._rebalance_strict()

    def _rebalance_strict(self) -> None:
        """Rebalance memory to maintain equal per-task quotas (strictly bounded by capacity)."""
        cap = self.capacity
        if cap <= 0:
            self._per_task = {t: [] for t in self._tasks_seen}
            return

        if len(self._tasks_seen) == 0:
            return

        # Equal quota; distribute remainder to earliest tasks deterministically
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
            # Downsample without replacement
            idx = self._rng.choice(len(cur), size=int(q), replace=False)
            self._per_task[t] = [cur[int(i)] for i in idx]

        # Final strict check
        total = self.num_items()
        if total > cap:
            raise RuntimeError(f"ER rebalance bug: total items {total} > capacity {cap}")

    # -----------------------
    # Replay sampling / merge
    # -----------------------
    def sample_replay_batch(self, *, cur_batch_size: int) -> Optional[Any]:
        """Sample and collate replay items. Returns a batch structure compatible with merge_batches()."""
        if not self.is_enabled():
            return None
        total = self.num_items()
        if total <= 0:
            return None

        # NOTE: replay_batch_ratio is interpreted as the *fraction of the batch* that should come
        # from replay (not "additional samples"). This keeps the effective batch size constant,
        # preventing OOM at task>1 when replay becomes active.
        k = int(round(float(self.cfg.replay_batch_ratio) * float(int(cur_batch_size))))
        if k <= 0:
            return None
        # Strict: if buffer exists but is smaller than requested, sample with replacement to keep ratio stable.
        # This avoids silently reducing replay strength early, which can hide bugs in academic comparisons.
        all_items: List[Any] = []
        for t in self._tasks_seen:
            all_items.extend(self._per_task.get(int(t), []))
        if len(all_items) <= 0:
            return None
        idx = self._rng.choice(len(all_items), size=int(k), replace=(len(all_items) < k))
        picked = [all_items[int(i)] for i in idx]
        return collate_samples(picked)

    def mix_in_replay(self, *, cur_batch: Any, cur_batch_size: int) -> Any:
        """Return merged batch with constant effective batch size.

        If replay is enabled, we replace a fraction of the current batch with replay samples:
        - total batch size stays == cur_batch_size
        - replay samples count ~= replay_batch_ratio * cur_batch_size
        """
        if not self.is_enabled():
            return cur_batch

        bs = int(cur_batch_size)
        if bs <= 0:
            raise ValueError(f"ER mix_in_replay: cur_batch_size must be > 0, got {cur_batch_size}")

        k = int(round(float(self.cfg.replay_batch_ratio) * float(bs)))
        if k <= 0:
            return cur_batch

        rb = self.sample_replay_batch(cur_batch_size=bs)
        if rb is None:
            return cur_batch

        keep = bs - int(k)
        if keep <= 0:
            # Corner case: full replay batch.
            return rb

        # Subsample current batch to keep total batch size constant.
        idx = self._rng.choice(bs, size=int(keep), replace=False)
        idx = [int(i) for i in idx]
        cur_sub = take_batch_indices(cur_batch, idx)
        return merge_batches(cur_sub, rb)


def build_continual_algorithm(
    *,
    algo_name: Any,
    buffer_ratio: Optional[float] = None,
    replay_batch_ratio: Optional[float] = None,
    seed: int = 0,
    distill_alpha: Optional[float] = None,
    ewc_lambda: Optional[float] = None,
    ewc_gamma: Optional[float] = None,
    ewc_fisher_batches: Optional[int] = None,
    lwf_alpha: Optional[float] = None,
) -> Optional[Union[ExperienceReplay, DERPP, EWC, LwF]]:
    name = _as_name(algo_name)
    if name in ("", "none", "null", "no"):
        return None
    if name in ("er", "experience_replay"):
        cfg = ERConfig(
            buffer_ratio=float(0.2 if buffer_ratio is None else buffer_ratio),
            replay_batch_ratio=float(0.2 if replay_batch_ratio is None else replay_batch_ratio),
        )
        return ExperienceReplay(cfg=cfg, seed=int(seed))
    if name in ("derpp", "der++", "derpp++", "dark_experience_replay++"):
        cfg = DERPPConfig(
            buffer_ratio=float(0.2 if buffer_ratio is None else buffer_ratio),
            replay_batch_ratio=float(0.2 if replay_batch_ratio is None else replay_batch_ratio),
            distill_alpha=float(0.5 if distill_alpha is None else distill_alpha),
        )
        return DERPP(cfg=cfg, seed=int(seed))
    if name in ("ewc", "elastic_weight_consolidation"):
        cfg = EWCConfig(
            lambda_=float(1e-2 if ewc_lambda is None else ewc_lambda),
            gamma=float(1.0 if ewc_gamma is None else ewc_gamma),
            fisher_batches=int(50 if ewc_fisher_batches is None else ewc_fisher_batches),
        )
        return EWC(cfg=cfg)
    if name in ("lwf", "learning_without_forgetting"):
        cfg = LwFConfig(alpha=float(0.5 if lwf_alpha is None else lwf_alpha))
        return LwF(cfg=cfg)
    raise ValueError(f"Unknown continual_algorithm: {algo_name!r}. Supported: none | er | derpp | ewc | lwf")

