import json
from typing import Optional, List


def make_task_order(
    all_task_ids: List[int],
    num_tasks: Optional[int],
    randomize: bool,
    seed: int,
) -> List[int]:
    task_ids = list(sorted(set(int(x) for x in all_task_ids)))
    if num_tasks is not None:
        task_ids = task_ids[: int(num_tasks)]
    if randomize:
        import random

        rng = random.Random(int(seed))
        rng.shuffle(task_ids)
    return task_ids


def save_task_order(path: str, order: List[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([int(x) for x in order], f, indent=2)


def load_task_order(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return [int(x) for x in arr]


