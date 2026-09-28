import os
import sys
from typing import Dict, Set, Tuple, Union, List

import numpy as np

# Ensure repo root is on sys.path so `clego_cl` is importable even if cwd is this folder.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.task_map import normalize_video_id
from egolearner_eval import edit_score, f_score, read_file


def eval_grouped(
    results_dir: str,
    gt_dir: str,
    split_bundle: Union[str, List[str]],
    video_to_task: Dict[str, int],
    seen_tasks: Set[int],
) -> Tuple[Dict[str, Dict[int, float]], Dict[int, float]]:
    """Evaluate per-video prediction files in `results_dir`, grouped by cooking task.

    Returns:
    - per_task_metrics[metric_key][task_id] = value
    - weights[task_id] = num_videos_in_task (micro weight)
    """
    bundles = [split_bundle] if isinstance(split_bundle, str) else list(split_bundle)
    vids_raw = []
    for b in bundles:
        with open(b, "r") as f:
            vids_raw.extend([x.strip() for x in f.read().strip().split("\n") if x.strip()])

    # per task accumulators
    per_task_videos = {}
    for v in vids_raw:
        uid = normalize_video_id(v)
        tid = video_to_task.get(uid, None)
        if tid is None or tid not in seen_tasks:
            continue
        per_task_videos.setdefault(int(tid), []).append(v)

    overlap_list = [0.1, 0.25, 0.5]
    per_task_metrics: Dict[str, Dict[int, float]] = {
        "acc": {},
        "edit": {},
        "f1_010": {},
        "f1_025": {},
        "f1_050": {},
    }
    weights: Dict[int, float] = {}

    for tid, vids in per_task_videos.items():
        tp = np.zeros(3)
        fp = np.zeros(3)
        fn = np.zeros(3)
        correct = 0
        total = 0
        edit_sum = 0.0

        for vid in vids:
            gt_file = os.path.join(gt_dir, vid)
            gt_content = read_file(gt_file).split("\n")[0:-1]
            recog_file = os.path.join(results_dir, vid.split(".")[0])
            recog_content = read_file(recog_file).split("\n")[1].split()

            for i in range(len(gt_content)):
                total += 1
                if gt_content[i] == recog_content[i]:
                    correct += 1
            edit_sum += edit_score(recog_content, gt_content)
            for s, ov in enumerate(overlap_list):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, ov)
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1

        acc = 100.0 * float(correct) / float(total) if total > 0 else float("nan")
        edit_val = float(edit_sum) / float(len(vids)) if len(vids) > 0 else float("nan")

        f1s = []
        for s in range(len(overlap_list)):
            precision = 0.0 if (tp[s] + fp[s]) == 0 else tp[s] / float(tp[s] + fp[s])
            recall = 0.0 if (tp[s] + fn[s]) == 0 else tp[s] / float(tp[s] + fn[s])
            f1 = 0.0 if (precision + recall) == 0.0 else 2.0 * (precision * recall) / (precision + recall)
            f1s.append(float(np.nan_to_num(f1) * 100.0))

        per_task_metrics["acc"][tid] = float(acc)
        per_task_metrics["edit"][tid] = float(edit_val)
        per_task_metrics["f1_010"][tid] = float(f1s[0])
        per_task_metrics["f1_025"][tid] = float(f1s[1])
        per_task_metrics["f1_050"][tid] = float(f1s[2])
        weights[tid] = float(len(vids))

    return per_task_metrics, weights


