from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .task_map import normalize_video_id


@dataclass(frozen=True)
class FilterStats:
    total: int
    kept: int
    dropped_not_in_tasks: int
    dropped_unknown_task: int


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def filter_text_lines_by_tasks(
    *,
    in_path: str,
    out_path: str,
    video_to_task: Dict[str, int],
    allowed_tasks: Set[int],
    line_to_video_id: Callable[[str], str],
) -> FilterStats:
    """
    Read a text file, keep only lines whose parsed video_id maps to a task in `allowed_tasks`.

    `line_to_video_id` should return a raw identifier; we always normalize with `normalize_video_id`.
    """
    total = 0
    kept = 0
    dropped_unknown = 0
    dropped_not_allowed = 0
    out_lines: List[str] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            total += 1
            vid = normalize_video_id(line_to_video_id(line))
            tid = video_to_task.get(vid)
            if tid is None:
                dropped_unknown += 1
                continue
            if int(tid) not in allowed_tasks:
                dropped_not_allowed += 1
                continue
            out_lines.append(line)
            kept += 1
    _ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
    return FilterStats(
        total=total,
        kept=kept,
        dropped_not_in_tasks=dropped_not_allowed,
        dropped_unknown_task=dropped_unknown,
    )


def filter_bundle_by_tasks(
    *,
    in_bundle: str,
    out_bundle: str,
    video_to_task: Dict[str, int],
    allowed_tasks: Set[int],
) -> FilterStats:
    # bundle format: one video filename per line (may include extension like .txt)
    return filter_text_lines_by_tasks(
        in_path=in_bundle,
        out_path=out_bundle,
        video_to_task=video_to_task,
        allowed_tasks=allowed_tasks,
        line_to_video_id=lambda ln: ln.split()[0],
    )


def filter_aap_list_by_tasks(
    *,
    in_list: str,
    out_list: str,
    video_to_task: Dict[str, int],
    allowed_tasks: Set[int],
) -> FilterStats:
    # AAP list format: video_uid|timestamp|[...]
    return filter_text_lines_by_tasks(
        in_path=in_list,
        out_path=out_list,
        video_to_task=video_to_task,
        allowed_tasks=allowed_tasks,
        line_to_video_id=lambda ln: ln.split("|", 1)[0],
    )


def filter_mcq_json_by_tasks(
    *,
    in_json: str,
    out_json: str,
    video_to_task: Dict[str, int],
    allowed_tasks: Set[int],
    save_file: bool = False,
) -> Tuple[int, int, int]:
    """
    Filter association MCQ json file, keeping only entries whose query video is in allowed tasks.

    Args:
        save_file: If False, only return stats without saving the filtered JSON (for backward compatibility).
    
    Returns: (total, kept, dropped)
    """
    with open(in_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = 0
    kept = 0
    out = {}
    for k, item in data.items():
        total += 1
        q = item.get("query", {}) if isinstance(item, dict) else {}
        vid = q.get("video_uid") or q.get("video_id")
        vid = normalize_video_id(str(vid))
        tid = video_to_task.get(vid)
        if tid is None or int(tid) not in allowed_tasks:
            continue
        out[str(k)] = item
        kept += 1
    if save_file:
        _ensure_dir(out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return total, kept, total - kept


def filter_mcq_json_by_tasks_to_dict(
    *,
    in_json: str,
    video_to_task: Dict[str, int],
    allowed_tasks: Set[int],
) -> Tuple[Dict, int, int, int]:
    """
    Filter association MCQ json file in memory, returning the filtered dictionary.
    
    Returns: (filtered_dict, total, kept, dropped)
    """
    with open(in_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = 0
    kept = 0
    out = {}
    for k, item in data.items():
        total += 1
        q = item.get("query", {}) if isinstance(item, dict) else {}
        vid = q.get("video_uid") or q.get("video_id")
        vid = normalize_video_id(str(vid))
        tid = video_to_task.get(vid)
        if tid is None or int(tid) not in allowed_tasks:
            continue
        out[str(k)] = item
        kept += 1
    return out, total, kept, total - kept




