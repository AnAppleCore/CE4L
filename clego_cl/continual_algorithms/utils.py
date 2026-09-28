from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import torch


Nested = Union[torch.Tensor, Dict[str, Any], List[Any], Tuple[Any, ...]]


def to_cpu_detached(x: Any) -> Any:
    """Move tensors to CPU and detach (recursively)."""
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: to_cpu_detached(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        out = [to_cpu_detached(v) for v in x]
        return tuple(out) if isinstance(x, tuple) else out
    return x


def to_device(x: Any, device: torch.device) -> Any:
    """Move tensors to device (recursively)."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        out = [to_device(v, device) for v in x]
        return tuple(out) if isinstance(x, tuple) else out
    return x


def split_batch_to_samples(batch: Any) -> List[Any]:
    """Split a collated batch into a list of per-sample structures.

    Supports:
    - Tensor: split on dim0
    - dict: split each value then zip by keys
    - (list/tuple): split each element then zip
    """
    if torch.is_tensor(batch):
        if batch.dim() == 0:
            return [batch]
        return [batch[i] for i in range(batch.shape[0])]

    if isinstance(batch, dict):
        # For experience replay we only keep tensor fields (batch-aligned on dim0).
        # Many benchmarks include non-tensor metadata (strings/ints) that should not
        # participate in replay (and cannot be merged safely).
        tensor_items = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        if not tensor_items:
            return []

        # Infer batch size from tensor fields (prefer dim0 tensors).
        bs = None
        for v in tensor_items.values():
            if v.dim() > 0:
                bs = int(v.shape[0]) if bs is None else max(bs, int(v.shape[0]))
        if bs is None:
            bs = 1

        # Enforce consistency across tensor fields.
        for k, v in tensor_items.items():
            if v.dim() > 0 and int(v.shape[0]) != bs:
                raise ValueError(
                    f"split_batch_to_samples: tensor field '{k}' has batch={int(v.shape[0])}, expected {bs}"
                )

        out: List[Dict[str, Any]] = []
        for i in range(bs):
            sample: Dict[str, Any] = {}
            for k, v in tensor_items.items():
                sample[k] = v if v.dim() == 0 else v[i]
            out.append(sample)
        return out

    if isinstance(batch, (list, tuple)):
        parts = [split_batch_to_samples(v) for v in batch]
        if not parts:
            return []
        n = max(len(p) for p in parts)
        out = []
        for i in range(n):
            row = [p[i] for p in parts]
            out.append(tuple(row) if isinstance(batch, tuple) else row)
        return out

    # Non-splittable; treat as a single sample
    return [batch]


def collate_samples(samples: Sequence[Any]) -> Any:
    """Collate per-sample structures into a batch (best-effort, minimal).

    - Tensor: stack on dim0
    - dict: collate each key
    - tuple/list: collate each position
    - otherwise: return list
    """
    if len(samples) == 0:
        raise ValueError("collate_samples: empty samples")

    s0 = samples[0]
    if torch.is_tensor(s0):
        return torch.stack([s for s in samples], dim=0)
    if isinstance(s0, dict):
        keys = list(s0.keys())
        return {k: collate_samples([s[k] for s in samples]) for k in keys}
    if isinstance(s0, (list, tuple)):
        m = len(s0)
        cols = []
        for j in range(m):
            cols.append(collate_samples([s[j] for s in samples]))
        return tuple(cols) if isinstance(s0, tuple) else cols
    return list(samples)


def merge_batches(cur: Any, replay: Any) -> Any:
    """Merge replay batch into current batch by concatenating on dim0 (recursively)."""
    if replay is None:
        return cur
    if torch.is_tensor(cur) and torch.is_tensor(replay):
        if cur.dim() == 0 or replay.dim() == 0:
            raise ValueError("merge_batches: cannot merge scalar tensors")
        return torch.cat([cur, replay], dim=0)
    if isinstance(cur, dict) and isinstance(replay, dict):
        out = dict(cur)
        # Only merge keys that exist in both; keep other current-only keys as-is.
        for k in cur.keys():
            if k in replay:
                out[k] = merge_batches(cur[k], replay[k])
        return out
    if isinstance(cur, (list, tuple)) and isinstance(replay, (list, tuple)):
        if len(cur) != len(replay):
            raise ValueError("merge_batches: length mismatch")
        merged = [merge_batches(cur[i], replay[i]) for i in range(len(cur))]
        return tuple(merged) if isinstance(cur, tuple) else merged
    # If types don't match, fall back to keeping current (strict)
    raise TypeError(f"merge_batches: unsupported types cur={type(cur)} replay={type(replay)}")


def take_batch_indices(batch: Any, indices: Sequence[int]) -> Any:
    """Select a subset of samples from a collated batch using indices along dim0.

    This is used to keep the *effective* batch size constant when mixing replay.
    Best-effort behavior:
    - Tensor with dim>0: index along dim0.
    - dict: index tensor fields with dim0==bs; index list/tuple fields with len==bs.
      Other fields are kept as-is.
    - tuple/list: treated as a structure (e.g. (x, y)), index each element recursively.
    """
    idx = list(int(i) for i in indices)
    if len(idx) == 0:
        # Represent an empty batch by taking the first element and slicing to zero.
        # This keeps shapes consistent (e.g., [0, ...]) for tensors.
        if torch.is_tensor(batch) and batch.dim() > 0:
            return batch[:0]
        if isinstance(batch, dict):
            return {k: take_batch_indices(v, idx) for k, v in batch.items() if torch.is_tensor(v) or isinstance(v, (dict, list, tuple))}
        if isinstance(batch, (list, tuple)):
            out = [take_batch_indices(v, idx) for v in batch]
            return tuple(out) if isinstance(batch, tuple) else out
        return batch

    if torch.is_tensor(batch):
        if batch.dim() == 0:
            return batch
        t_idx = torch.as_tensor(idx, device=batch.device, dtype=torch.long)
        return batch.index_select(0, t_idx)

    if isinstance(batch, dict):
        # Infer batch size from tensor fields (dim0), if any.
        bs = None
        for v in batch.values():
            if torch.is_tensor(v) and v.dim() > 0:
                bs = int(v.shape[0]) if bs is None else max(bs, int(v.shape[0]))
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.dim() > 0 and (bs is None or int(v.shape[0]) == bs):
                out[k] = take_batch_indices(v, idx)
            elif isinstance(v, list) and bs is not None and len(v) == bs:
                out[k] = [v[i] for i in idx]
            elif isinstance(v, tuple) and bs is not None and len(v) == bs:
                out[k] = tuple(v[i] for i in idx)
            elif isinstance(v, (dict, list, tuple)) or torch.is_tensor(v):
                out[k] = take_batch_indices(v, idx)
            else:
                out[k] = v
        return out

    if isinstance(batch, (list, tuple)):
        out = [take_batch_indices(v, idx) for v in batch]
        return tuple(out) if isinstance(batch, tuple) else out

    return batch

