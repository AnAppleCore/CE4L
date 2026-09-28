import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure repo root on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.config_utils import apply_config_as_defaults, load_yaml_config, split_argv_config
from clego_cl.task_order import make_task_order, save_task_order
from clego_cl.task_stats import UnknownTracker, save_task_stats, save_unknown_ids

from dataset import SkillDataSet
from model import RAAN
from opts import parser as base_parser, update_paths_from_args

import train as base


TASK_TO_ACTIONS: Dict[int, str] = {1: "18", 2: "06", 3: "20", 4: "13,14,15"}
ALL_TASK_IDS: List[int] = [1, 2, 3, 4]


@dataclass(frozen=True)
class JointFairEpochMetrics:
    epoch: int
    per_task_ranking_acc: Dict[int, float]
    weights: Dict[int, float]
    micro_avg: float
    macro_avg: float
    train_time_sec: float


def _union_action_select(task_order: List[int]) -> str:
    ids: List[int] = []
    for tid in task_order:
        s = TASK_TO_ACTIONS[int(tid)]
        for x in s.split(","):
            x = x.strip()
            if x:
                ids.append(int(x))
    uniq = sorted(set(ids))
    return ",".join([f"{x:02d}" for x in uniq])


def _dump_args_json(output_root: str, args: argparse.Namespace, extra: Dict[str, object]) -> None:
    try:
        payload = {"args": {}, "extra": dict(extra)}
        for k, v in vars(args).items():
            try:
                json.dumps(v)
                payload["args"][k] = v
            except Exception:
                payload["args"][k] = str(v)
        os.makedirs(output_root, exist_ok=True)
        with open(os.path.join(output_root, "args_effective.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _macro_avg(per_task: Dict[int, float], task_order: List[int]) -> float:
    vals = []
    for tid in task_order:
        v = per_task.get(int(tid), float("nan"))
        if np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if len(vals) else float("nan")


def _micro_avg(per_task: Dict[int, float], weights: Dict[int, float], task_order: List[int]) -> float:
    num = 0.0
    den = 0.0
    for tid in task_order:
        t = int(tid)
        w = float(weights.get(t, 0.0))
        v = per_task.get(t, float("nan"))
        if w <= 0 or not np.isfinite(float(v)):
            continue
        num += float(v) * w
        den += w
    return float("nan") if den <= 0 else float(num / den)


def _build_loader(args: argparse.Namespace, list_path: str, action_select: str, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = SkillDataSet(
        args.root_path,
        list_path,
        ftr_tmpl="{}_{}.npz",
        action_select=action_select,
        use_exo=bool(getattr(args, "use_exo", False)),
        exo_root_path=getattr(args, "exo_root_path", None),
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=int(args.workers),
        pin_memory=True,
    )


def _eval_per_task_val(
    *,
    args: argparse.Namespace,
    task_order: List[int],
    criterion,
    models: Dict[str, torch.nn.Module],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    per_task: Dict[int, float] = {}
    weights: Dict[int, float] = {}
    for tid in task_order:
        action_select = TASK_TO_ACTIONS[int(tid)]
        val_loader = _build_loader(args, args.val_list, action_select=action_select, shuffle=False)
        weights[int(tid)] = float(len(val_loader.dataset))
        if len(val_loader.dataset) == 0:
            per_task[int(tid)] = float("nan")
        else:
            per_task[int(tid)] = float(
                base.validate(val_loader, models, criterion, epoch=0, use_exo=args.use_exo, use_RN=args.relation_network)
            )
    return per_task, weights


def _save_ckpt(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _finalize_jsonl_to_json(jsonl_path: str, json_path: str) -> None:
    if not os.path.isfile(jsonl_path):
        return
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"schema": "skill_joint_fair_epoch_metrics_v1", "epochs": rows},
            f,
            indent=2,
            ensure_ascii=False,
        )
    try:
        os.remove(jsonl_path)
    except Exception:
        pass


def main() -> None:
    cfg_path, remaining = split_argv_config(sys.argv[1:])
    parser = argparse.ArgumentParser(parents=[base_parser], add_help=False)
    parser.add_argument("--config", type=str, default=None, help="YAML config file (optional). CLI args override config.")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Only write task_order/task_stats then exit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomize_order", action="store_true", default=True)
    parser.add_argument("--no_randomize_order", action="store_true")
    # Override: in joint_fair we interpret opts.py's --epochs as the *total* training budget.
    # Config must set epochs == continual epochs_per_task.
    if cfg_path is not None:
        cfg = load_yaml_config(cfg_path)
        apply_config_as_defaults(parser, cfg)
    args = parser.parse_args(remaining)
    if getattr(args, "no_randomize_order", False):
        args.randomize_order = False
    if not args.output_root:
        raise ValueError("--output_root must be provided via CLI or --config")

    os.makedirs(args.output_root, exist_ok=True)

    # Make train.py helper functions use our args and disable tensorboard
    base.args = args
    base.writer = None

    # Task order (randomized file is required by protocol, even though joint training doesn't depend on order)
    task_order: List[int] = make_task_order(
        all_task_ids=ALL_TASK_IDS,
        num_tasks=None,
        randomize=bool(getattr(args, "randomize_order", True)),
        seed=int(args.seed),
    )
    save_task_order(os.path.join(args.output_root, "task_order.json"), task_order)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    # Keep baseline path updates for internal flags, but DO NOT rely on them for output management.
    update_paths_from_args(args)

    # Build models (match continual)
    if args.rank_aware_loss:
        models = {"pos": None, "neg": None}
    else:
        models = {"att": None}
    for k in list(models.keys()):
        models[k] = RAAN(args.num_samples, args.attention, args.num_filters, args.input_size).cuda()

    model_uniform = None
    if args.disparity_loss or args.rank_aware_loss:
        model_uniform = RAAN(args.num_samples, attention=False, num_filters=1, input_size=args.input_size).cuda()

    criterion = torch.nn.MarginRankingLoss(margin=args.m1).cuda()

    if args.disparity_loss or args.rank_aware_loss:
        attention_params = []
        model_params = []
        for m in models.values():
            for name, param in m.named_parameters():
                if not param.requires_grad:
                    continue
                if "att" in name:
                    attention_params.append(param)
                else:
                    model_params.append(param)
        optimizer = torch.optim.Adam(list(model_uniform.parameters()) + model_params, args.lr)
        optimizer_attention = torch.optim.Adam(attention_params, args.lr * 0.1)
    else:
        only_model = models[list(models.keys())[0]]
        optimizer = torch.optim.Adam(only_model.parameters(), args.lr)
        optimizer_attention = None

    # Stats
    unknown = UnknownTracker(max_examples=200)
    stats = {
        "benchmark": "skill_benchmark_joint_fair",
        "task_order": [int(x) for x in task_order],
        "task_to_actions": {str(k): v for k, v in TASK_TO_ACTIONS.items()},
        "micro_weight_unit": "num_pairs_in_eval_split",
        "splits": {"train": {}, "val": {}, "train_union_action_select": _union_action_select(task_order)},
    }
    # Cheap counts by building datasets (no NPZ load happens during __len__? it parses file only)
    for tid in task_order:
        act_sel = TASK_TO_ACTIONS[int(tid)]
        ds_tr = SkillDataSet(args.root_path, args.train_list, action_select=act_sel, use_exo=bool(args.use_exo), exo_root_path=getattr(args, "exo_root_path", None))
        ds_va = SkillDataSet(args.root_path, args.val_list, action_select=act_sel, use_exo=bool(args.use_exo), exo_root_path=getattr(args, "exo_root_path", None))
        stats["splits"]["train"][str(int(tid))] = int(len(ds_tr))
        stats["splits"]["val"][str(int(tid))] = int(len(ds_va))

    # Save args snapshot
    _dump_args_json(
        args.output_root,
        args,
        extra={
            "entrypoint": "skill_benchmark/joint_fair_train.py",
            "task_order": [int(x) for x in task_order],
            "train_union_action_select": _union_action_select(task_order),
        },
    )

    if args.dry_run:
        stats.update(unknown.to_dict())
        stats["unknown_ratio"] = 0.0
        save_task_stats(args.output_root, stats)
        save_unknown_ids(args.output_root, unknown)
        return

    # Loaders: JOINT training on union actions
    union_action_select = _union_action_select(task_order)
    args.action_select = union_action_select  # for transparency only
    train_loader = _build_loader(args, args.train_list, action_select=union_action_select, shuffle=True)

    # Training loop with per-epoch per-task val
    out_ckpt_dir = os.path.join(args.output_root, "checkpoints")
    os.makedirs(out_ckpt_dir, exist_ok=True)
    train_log_path = os.path.join(args.output_root, "train.log")
    # Persist per-epoch metrics into a single aggregated JSON file (avoid thousands of small files).
    # We append per-epoch records to JSONL for low overhead, then finalize to one JSON at the end.
    epoch_metrics_jsonl = os.path.join(args.output_root, "metrics_epochs.jsonl")
    epoch_metrics_json = os.path.join(args.output_root, "metrics_epochs.json")
    # If rerun into an existing directory, remove old aggregated files to avoid mixing runs.
    for p in [epoch_metrics_jsonl, epoch_metrics_json]:
        if os.path.isfile(p):
            try:
                os.remove(p)
            except Exception:
                pass

    best_epoch: Optional[int] = None
    best_micro = float("-inf")
    best_macro = float("-inf")
    best_per_task: Dict[int, float] = {}
    best_weights: Dict[int, float] = {}

    phase = 0
    start_all = time.time()
    with open(train_log_path, "a", encoding="utf-8") as log_f:
        for ep in range(int(args.epochs)):
            t0 = time.time()
            if model_uniform is None:
                raise ValueError("joint_fair_train expects disparity_loss/rank_aware_loss enabled (default).")
            phase = base.train_with_uniform(
                train_loader,
                models,
                model_uniform,
                criterion,
                optimizer,
                optimizer_attention,
                epoch=int(ep),
                phase=phase,
                use_exo=args.use_exo,
                use_triplet_loss=args.triplet_loss,
                use_RN=args.relation_network,
            )

            # Per-task val at the end of every epoch (requested)
            per_task, weights = _eval_per_task_val(args=args, task_order=task_order, criterion=criterion, models=models)
            micro = _micro_avg(per_task, weights, task_order)
            macro = _macro_avg(per_task, task_order)
            dt = time.time() - t0

            # Persist epoch metrics (fine-grained)
            m = JointFairEpochMetrics(
                epoch=int(ep + 1),
                per_task_ranking_acc={int(k): _safe_float(v) for k, v in per_task.items()},
                weights={int(k): float(v) for k, v in weights.items()},
                micro_avg=float(micro),
                macro_avg=float(macro),
                train_time_sec=float(dt),
            )
            _append_jsonl(
                epoch_metrics_jsonl,
                {
                    "epoch": m.epoch,
                    "per_task_ranking_acc": {str(k): float(v) for k, v in m.per_task_ranking_acc.items()},
                    "weights": {str(k): float(v) for k, v in m.weights.items()},
                    "micro_avg": float(m.micro_avg) if np.isfinite(m.micro_avg) else None,
                    "macro_avg": float(m.macro_avg) if np.isfinite(m.macro_avg) else None,
                    "train_time_sec": float(m.train_time_sec),
                },
            )

            log_f.write(
                f"Epoch {ep+1:04d}/{int(args.epochs):04d} | micro_avg={micro:.6f} macro_avg={macro:.6f} | "
                + " ".join([f"T{int(tid)}={per_task.get(int(tid), float('nan')):.6f}" for tid in task_order])
                + "\n"
            )
            log_f.flush()

            # Save "task_end.pth" snapshot every epoch? (overwrite) to reflect latest state
            _save_ckpt(
                os.path.join(out_ckpt_dir, "task_end.pth"),
                {
                    "epoch": int(ep + 1),
                    "task_order": [int(x) for x in task_order],
                    "models": {k: m_.state_dict() for k, m_ in models.items()},
                    "model_uniform": None if model_uniform is None else model_uniform.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimizer_attention": None if optimizer_attention is None else optimizer_attention.state_dict(),
                    "phase": int(phase),
                    "best_epoch": int(best_epoch) if best_epoch is not None else None,
                    "best_micro_avg": float(best_micro) if np.isfinite(best_micro) else None,
                },
            )

            # Best selection by micro_avg (requested)
            if np.isfinite(micro) and float(micro) > float(best_micro):
                best_micro = float(micro)
                best_macro = float(macro) if np.isfinite(macro) else float("nan")
                best_epoch = int(ep + 1)
                best_per_task = {int(k): float(v) for k, v in per_task.items()}
                best_weights = {int(k): float(v) for k, v in weights.items()}
                _save_ckpt(
                    os.path.join(out_ckpt_dir, "task_best.pth"),
                    {
                        "epoch": int(ep + 1),
                        "task_order": [int(x) for x in task_order],
                        "models": {k: m_.state_dict() for k, m_ in models.items()},
                        "model_uniform": None if model_uniform is None else model_uniform.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "optimizer_attention": None if optimizer_attention is None else optimizer_attention.state_dict(),
                        "phase": int(phase),
                        "best_epoch": int(best_epoch),
                        "best_micro_avg": float(best_micro),
                        "best_macro_avg": float(best_macro) if np.isfinite(best_macro) else None,
                    },
                )

    total_time = float(time.time() - start_all)

    # Finalize metrics into a single JSON file (and remove JSONL).
    _finalize_jsonl_to_json(epoch_metrics_jsonl, epoch_metrics_json)

    # Save task stats / unknown ids (for parity)
    stats.update(unknown.to_dict())
    stats["unknown_ratio"] = 0.0
    save_task_stats(args.output_root, stats)
    save_unknown_ids(args.output_root, unknown)

    # Final joint_fair summary
    with open(os.path.join(args.output_root, "joint_fair_eval.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": "val",
                "task_order": [int(x) for x in task_order],
                "train_union_action_select": union_action_select,
                "epochs": int(args.epochs),
                "seed": int(args.seed),
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
                "best_micro_avg": float(best_micro) if np.isfinite(best_micro) else None,
                "best_macro_avg": float(best_macro) if np.isfinite(best_macro) else None,
                "best_per_task_ranking_acc": {str(k): float(v) for k, v in best_per_task.items()},
                "best_weights": {str(k): float(v) for k, v in best_weights.items()},
                "total_train_time_sec": total_time,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Write best.log as single scalar (best_micro_avg) for quick glance
    with open(os.path.join(args.output_root, "best.log"), "w", encoding="utf-8") as f:
        f.write(f"{best_micro}\n")


if __name__ == "__main__":
    main()

