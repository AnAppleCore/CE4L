from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Set, Tuple, Union

# Ensure repo root on path so `clego_cl` is importable even if cwd is this folder.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.task_map import load_video_to_task
from egolearner_eval_grouped import eval_grouped


def _wavg(per_task_metrics: Dict[str, Dict[int, float]], weights: Dict[int, float], key: str) -> float:
    num = 0.0
    den = 0.0
    for tid, w in weights.items():
        v = per_task_metrics.get(key, {}).get(int(tid), float("nan"))
        if isinstance(v, float) and math.isnan(v):
            continue
        num += float(v) * float(w)
        den += float(w)
    return float("nan") if den <= 0 else num / den


def _load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_eval_bundles(manifest: dict, split_dir: str) -> List[str]:
    bundles = []
    for item in manifest.get("eval_target", []) or []:
        p = item.get("out", None)
        if isinstance(p, str) and os.path.exists(p):
            bundles.append(p)
    if bundles:
        return bundles

    # Fallback: scan split_dir for eval_target_*.bundle
    if os.path.isdir(split_dir):
        for name in sorted(os.listdir(split_dir)):
            if name.startswith("eval_target_") and name.endswith(".bundle"):
                p = os.path.join(split_dir, name)
                if os.path.exists(p):
                    bundles.append(p)
    return bundles


def _write_eval_log(path: str, line: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _maybe_update_best(best_path: str, result_line: str, avg_score: float) -> bool:
    """Update best_result file if avg_score improves. Returns True if updated."""
    best_dir = os.path.dirname(best_path)
    if best_dir:
        os.makedirs(best_dir, exist_ok=True)

    update = False
    if os.path.exists(best_path):
        try:
            with open(best_path, "r", encoding="utf-8") as f:
                best_line = f.readline().strip()
            if best_line.startswith("Best "):
                best_line = best_line[5:]
            # Use rsplit to get the last "Avg: " (not "F1@Avg:")
            best_avg = float(best_line.rsplit("Avg: ", 1)[1].strip())
            if avg_score > best_avg:
                update = True
        except Exception:
            update = True
    else:
        update = True

    if update:
        with open(best_path, "w", encoding="utf-8") as f:
            f.write("Best " + result_line.rstrip("\n") + "\n")
    return update


def main() -> None:
    p = argparse.ArgumentParser("CLEGO TAS joint_fair eval (fair_split_manifest + grouped metrics)")
    p.add_argument("--output_root", type=str, required=True, help="Experiment output directory (contains fair_splits/ and results/).")
    p.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--epoch", type=int, default=-1, help="Epoch number to log (optional).")
    p.add_argument(
        "--video_to_task_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "video_to_task.npy"),
        help="Path to video_to_task.npy",
    )
    p.add_argument(
        "--path_data",
        type=str,
        default=os.path.join(_REPO_ROOT, "temporal_action_segmentation_benchmark", "tas_annotation"),
        help="Path to TAS annotation root (contains gts_fps25/).",
    )
    p.add_argument(
        "--results_subdir",
        type=str,
        default="results",
        help="Subdir under output_root where predictions are stored (default: results).",
    )
    p.add_argument("--no_write", action="store_true", help="If set, do not write eval_results/best_result files; only print.")
    args = p.parse_args()

    output_root = os.path.abspath(args.output_root)
    split = str(args.eval_split)

    split_dir = os.path.join(output_root, "fair_splits", split)
    manifest_path = os.path.join(split_dir, "fair_split_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing fair_split_manifest.json: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    allowed_tasks: Set[int] = set(int(x) for x in (manifest.get("allowed_tasks", []) or []))
    if not allowed_tasks:
        raise RuntimeError(f"Manifest has empty allowed_tasks: {manifest_path}")

    bundles = _resolve_eval_bundles(manifest, split_dir=split_dir)
    if not bundles:
        raise RuntimeError(f"No eval_target bundles found in manifest or {split_dir}")

    results_dir = os.path.join(output_root, args.results_subdir, split)
    if not os.path.isdir(results_dir) or len(os.listdir(results_dir)) == 0:
        raise FileNotFoundError(
            f"Missing prediction files in {results_dir}. "
            f"Run predict first (egolearner_joint_fair_main.py --action predict)."
        )

    gt_dir = os.path.join(os.path.abspath(args.path_data), "gts_fps25")
    video_to_task = load_video_to_task(args.video_to_task_path)

    per_task_metrics, weights = eval_grouped(
        results_dir=results_dir,
        gt_dir=gt_dir,
        split_bundle=bundles,
        video_to_task=video_to_task,
        seen_tasks=allowed_tasks,
    )

    # Micro-weighted across tasks (weights are #videos per task)
    acc = _wavg(per_task_metrics, weights, "acc")
    edit = _wavg(per_task_metrics, weights, "edit")
    f1_010 = _wavg(per_task_metrics, weights, "f1_010")
    f1_025 = _wavg(per_task_metrics, weights, "f1_025")
    f1_050 = _wavg(per_task_metrics, weights, "f1_050")

    f1_avg = (f1_010 + f1_025 + f1_050) / 3.0
    avg_score = (acc + edit + f1_010 + f1_025 + f1_050) / 5.0

    epoch_str = f"Epoch {int(args.epoch)}" if int(args.epoch) >= 0 else "Epoch Unknown"
    result_line = (
        f"{epoch_str}: "
        f"Acc: {acc:.4f}, Edit: {edit:.4f}, "
        f"F1@0.10: {f1_010:.4f}, F1@0.25: {f1_025:.4f}, F1@0.50: {f1_050:.4f}, "
        f"F1@Avg: {f1_avg:.4f}, Avg: {avg_score:.4f}"
    )
    print(result_line)

    if not args.no_write:
        path_result = os.path.join(output_root, args.results_subdir)
        eval_log_path = os.path.join(path_result, f"eval_results_{split}.log")
        best_path = os.path.join(path_result, f"best_result_{split}.txt")
        _write_eval_log(eval_log_path, result_line)
        updated = _maybe_update_best(best_path, result_line, avg_score)
        if updated:
            print("*** New best result saved! ***")


if __name__ == "__main__":
    main()



