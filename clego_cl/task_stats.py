import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UnknownTracker:
    max_examples: int = 200
    count: int = 0
    examples: List[str] = field(default_factory=list)
    _seen: set = field(default_factory=set, repr=False)

    def add(self, raw_id: str) -> None:
        self.count += 1
        if raw_id in self._seen:
            return
        if len(self.examples) < self.max_examples:
            self.examples.append(raw_id)
        self._seen.add(raw_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unknown_count": int(self.count),
            "unknown_examples": list(self.examples),
        }


def ensure_unknown_tracker(obj: Optional[Any]) -> UnknownTracker:
    if obj is None:
        return UnknownTracker()
    if isinstance(obj, UnknownTracker):
        return obj
    raise TypeError(f"unknown_tracker must be UnknownTracker or None, got {type(obj)}")


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_task_stats(output_dir: str, stats: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    save_json(os.path.join(output_dir, "task_stats.json"), stats)


def save_unknown_ids(output_dir: str, unknown: UnknownTracker) -> None:
    os.makedirs(output_dir, exist_ok=True)
    save_json(os.path.join(output_dir, "unknown_ids.json"), unknown.to_dict())


