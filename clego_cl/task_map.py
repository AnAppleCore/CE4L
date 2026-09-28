import os
from typing import Optional, Dict

import numpy as np


_STRIP_SUFFIXES = (".txt", ".mp4", ".avi", ".mkv")


def load_video_to_task(path: str) -> Dict[str, int]:
    """Load `video_to_task.npy` as a python dict: {video_uid(str): task_id(int)}.

    The repo convention is:
    - key: pure video_uid (no extension, no directory)
    - value: task_id in 1..8
    """
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, dict):
        mp = arr
    else:
        try:
            mp = arr.item()
        except Exception as e:
            raise ValueError(
                f"Unsupported video_to_task format at {path}: type={type(arr)}, shape={getattr(arr, 'shape', None)}"
            ) from e
    if not isinstance(mp, dict):
        raise ValueError(f"video_to_task must be a dict, got {type(mp)} from {path}")
    def _parse_task_id(v) -> int:
        if isinstance(v, (int, np.integer)):
            return int(v)
        s = str(v).strip().lower()
        # accept formats like "task5", "Task5", "5"
        digits = "".join([ch for ch in s if ch.isdigit()])
        if digits:
            return int(digits)
        raise ValueError(f"Cannot parse task id from value: {v!r}")

    out: Dict[str, int] = {}
    for k, v in mp.items():
        kk = normalize_video_id(str(k))
        out[kk] = _parse_task_id(v)
    return out


def normalize_video_id(raw: str) -> str:
    raw = (raw or "").strip()
    raw = os.path.basename(raw)
    for suf in _STRIP_SUFFIXES:
        if raw.endswith(suf):
            raw = raw[: -len(suf)]
            break
    return raw


def get_task_id(video_to_task: Dict[str, int], raw_id: str) -> Optional[int]:
    vid = normalize_video_id(raw_id)
    return video_to_task.get(vid, None)


