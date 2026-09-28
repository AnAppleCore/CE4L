import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _as_value_array(v: Any) -> Tuple[np.ndarray, bool]:
    """Return (array, is_scalar). Scalars become shape (1,)."""
    if isinstance(v, (int, float, np.floating, np.integer)):
        return np.asarray([float(v)], dtype=np.float32), True
    arr = np.asarray(v)
    if arr.ndim != 1:
        raise ValueError(f"Vector metric must be 1D array, got shape={arr.shape}")
    return arr.astype(np.float32), False


@dataclass
class ContinualRecorder:
    task_order: List[int]
    metrics_spec: Optional[Dict[str, Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        self.T = len(self.task_order)
        self._task_to_j = {int(tid): j for j, tid in enumerate(self.task_order)}
        self._A: Dict[str, np.ndarray] = {}
        self._A0: Dict[str, np.ndarray] = {}
        self._is_scalar: Dict[str, bool] = {}
        # 0-based index of the last task for which update_after_task() has been called.
        # This is critical to avoid computing CL statistics for future (unfilled) rows.
        self._last_completed_t: int = -1

    def set_A0(self, metric_key: str, per_task_values: Dict[int, Any], weights: Dict[int, float]) -> None:
        _ = weights  # weights not used for A0 itself; kept for symmetry
        # infer K + scalar/vector from any present value
        present = None
        for tid in self.task_order:
            if int(tid) in per_task_values:
                present = per_task_values[int(tid)]
                break
        if present is None:
            raise ValueError(f"set_A0: no values provided for metric={metric_key}")
        arr0, is_scalar = _as_value_array(present)
        K = int(arr0.shape[0])
        self._is_scalar[metric_key] = is_scalar
        if is_scalar:
            A0 = np.full((self.T,), np.nan, dtype=np.float32)
        else:
            A0 = np.full((self.T, K), np.nan, dtype=np.float32)
        for j, tid in enumerate(self.task_order):
            v = per_task_values.get(int(tid), None)
            if v is None:
                continue
            vv, _ = _as_value_array(v)
            if is_scalar:
                A0[j] = float(vv[0])
            else:
                if vv.shape[0] != K:
                    raise ValueError(f"A0 metric={metric_key} expects K={K}, got {vv.shape[0]}")
                A0[j, :] = vv
        self._A0[metric_key] = A0

    def update_after_task(
        self,
        t_idx: int,
        metrics: Dict[str, Dict[int, Any]],
        weights: Dict[int, float],
    ) -> None:
        """Update A matrix row for a completed task index (1-based t_idx)."""
        t = int(t_idx) - 1
        if t < 0 or t >= self.T:
            raise ValueError(f"t_idx out of range: {t_idx}, T={self.T}")
        self._last_completed_t = max(self._last_completed_t, t)
        for metric_key, per_task_values in metrics.items():
            # infer shapes if first time
            if metric_key not in self._A:
                present = None
                for tid in self.task_order:
                    if int(tid) in per_task_values:
                        present = per_task_values[int(tid)]
                        break
                if present is None:
                    raise ValueError(f"update_after_task: no values for metric={metric_key}")
                arr, is_scalar = _as_value_array(present)
                K = int(arr.shape[0])
                self._is_scalar[metric_key] = is_scalar
                if is_scalar:
                    self._A[metric_key] = np.full((self.T, self.T), np.nan, dtype=np.float32)
                else:
                    self._A[metric_key] = np.full((self.T, self.T, K), np.nan, dtype=np.float32)

            is_scalar = self._is_scalar[metric_key]
            for tid, v in per_task_values.items():
                tid_int = int(tid)
                if tid_int not in self._task_to_j:
                    continue
                j = self._task_to_j[tid_int]
                if j > t:
                    continue  # only fill seen tasks
                vv, _ = _as_value_array(v)
                if is_scalar:
                    self._A[metric_key][t, j] = float(vv[0])
                else:
                    self._A[metric_key][t, j, :] = vv.astype(np.float32)
            # keep weights for micro averages
        # store weights per task id (micro)
        self._weights = {int(k): float(v) for k, v in weights.items()}

    def compute_barA_macro_micro(self, metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
        A = self._A[metric_key]
        is_scalar = self._is_scalar[metric_key]
        t_end = min(max(self._last_completed_t, -1), self.T - 1)
        if is_scalar:
            macro = np.full((self.T,), np.nan, dtype=np.float32)
            micro = np.full((self.T,), np.nan, dtype=np.float32)
            for t in range(t_end + 1):
                vals = A[t, : t + 1]
                macro[t] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan
                ws = np.array([self._weights.get(int(self.task_order[j]), 0.0) for j in range(t + 1)], dtype=np.float32)
                if float(ws.sum()) > 0 and np.any(~np.isnan(vals)):
                    micro[t] = float(np.nansum(vals * ws) / ws.sum())
                else:
                    micro[t] = np.nan
            return macro, micro
        else:
            K = A.shape[-1]
            macro = np.full((self.T, K), np.nan, dtype=np.float32)
            micro = np.full((self.T, K), np.nan, dtype=np.float32)
            for t in range(t_end + 1):
                vals = A[t, : t + 1, :]  # (t+1, K)
                macro[t, :] = np.nanmean(vals, axis=0)
                ws = np.array([self._weights.get(int(self.task_order[j]), 0.0) for j in range(t + 1)], dtype=np.float32)
                denom = float(ws.sum())
                if denom > 0:
                    micro[t, :] = np.nansum(vals * ws[:, None], axis=0) / denom
            return macro, micro

    def compute_cl_stats(self, metric_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (forgetting, bwt, fwt) each shaped [T] or [T,K]."""
        A = self._A[metric_key]
        A0 = self._A0.get(metric_key, None)
        is_scalar = self._is_scalar[metric_key]
        t_end = min(max(self._last_completed_t, -1), self.T - 1)
        if is_scalar:
            forgetting = np.full((self.T,), np.nan, dtype=np.float32)
            bwt = np.full((self.T,), np.nan, dtype=np.float32)
            fwt = np.full((self.T,), np.nan, dtype=np.float32)
            for t in range(t_end + 1):
                if t >= 1:
                    # forgetting and bwt over j < t
                    diffs_f: List[float] = []
                    diffs_b: List[float] = []
                    for j in range(t):
                        cur = A[t, j]
                        # CL stats are only defined when the needed entry exists.
                        if np.isnan(cur):
                            continue
                        prev_slice = A[j:t, j]  # k in [j, t-1] in 0-based
                        if np.all(np.isnan(prev_slice)):
                            continue
                        best_prev = float(np.nanmax(prev_slice))
                        diffs_f.append(best_prev - float(cur))
                        diffs_b.append(float(cur) - float(A[j, j]))
                    if len(diffs_f) > 0:
                        forgetting[t] = float(np.mean(diffs_f))
                    if len(diffs_b) > 0:
                        bwt[t] = float(np.mean(diffs_b))
                    if A0 is not None:
                        # FWT is only defined when A[t-1, t] exists (i.e. we evaluated on unseen task t at time t-1).
                        if not np.isnan(A[t - 1, t]) and not np.isnan(A0[t]):
                            fwt[t] = float(A[t - 1, t] - A0[t])
            return forgetting, bwt, fwt
        else:
            K = A.shape[-1]
            forgetting = np.full((self.T, K), np.nan, dtype=np.float32)
            bwt = np.full((self.T, K), np.nan, dtype=np.float32)
            fwt = np.full((self.T, K), np.nan, dtype=np.float32)
            for t in range(t_end + 1):
                if t >= 1:
                    diffs_f = []
                    diffs_b = []
                    for j in range(t):
                        cur = A[t, j, :]
                        if np.all(np.isnan(cur)):
                            continue
                        prev_slice = A[j:t, j, :]  # (t-j, K)
                        if np.all(np.isnan(prev_slice)):
                            continue
                        best_prev = np.nanmax(prev_slice, axis=0)  # (K,)
                        diffs_f.append(best_prev - cur)
                        diffs_b.append(cur - A[j, j, :])
                    if len(diffs_f) > 0:
                        forgetting[t, :] = np.nanmean(np.stack(diffs_f, axis=0), axis=0)
                    if len(diffs_b) > 0:
                        bwt[t, :] = np.nanmean(np.stack(diffs_b, axis=0), axis=0)
                    if A0 is not None:
                        if not np.all(np.isnan(A[t - 1, t, :])) and not np.all(np.isnan(A0[t, :])):
                            fwt[t, :] = A[t - 1, t, :] - A0[t, :]
            return forgetting, bwt, fwt

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for metric_key, A in self._A.items():
            np.save(os.path.join(output_dir, f"continual_A_{metric_key}.npy"), A)
            if metric_key in self._A0:
                np.save(os.path.join(output_dir, f"continual_A0_{metric_key}.npy"), self._A0[metric_key])
            macro, micro = self.compute_barA_macro_micro(metric_key)
            np.save(os.path.join(output_dir, f"continual_bar_A_macro_{metric_key}.npy"), macro)
            np.save(os.path.join(output_dir, f"continual_bar_A_micro_{metric_key}.npy"), micro)
            forgetting, bwt, fwt = self.compute_cl_stats(metric_key)
            np.save(os.path.join(output_dir, f"continual_forgetting_{metric_key}.npy"), forgetting)
            np.save(os.path.join(output_dir, f"continual_bwt_{metric_key}.npy"), bwt)
            np.save(os.path.join(output_dir, f"continual_fwt_{metric_key}.npy"), fwt)


