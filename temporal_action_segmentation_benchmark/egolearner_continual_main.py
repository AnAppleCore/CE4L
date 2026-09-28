import argparse
import json
import os
import re
import sys
import time
import shutil
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import torch

# Ensure repo root on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.config_utils import load_yaml_config, split_argv_config, apply_config_as_defaults
from clego_cl.task_map import load_video_to_task, normalize_video_id
from clego_cl.task_order import make_task_order, save_task_order
from clego_cl.continual_recorder import ContinualRecorder
from clego_cl.task_stats import UnknownTracker, save_task_stats, save_unknown_ids
from clego_cl.continual_algorithms import build_continual_algorithm
from clego_cl.continual_algorithms.lwf_generic import LwFGeneric, LwFGenericConfig
from clego_cl.vista import VISTAState, build_vista_router, vista_eval_router_grouped
from skill_benchmark.adapters import AdapterBank
from clego_cl.l2p import L2PPool

from model import MultiStageModel
import egolearner_train as base_train
import egolearner_predict as base_predict
from egolearner_train import Trainer
from egolearner_predict import predict
from egolearner_batch_gen import BatchGenerator
from egolearner_eval_grouped import eval_grouped
from egobridge_settings import get_annotations_from_settings


def _dump_args_json(output_root: str, args: argparse.Namespace, extra: Dict[str, object]) -> None:
    """Write a JSON snapshot of parsed args for reproducibility / hyperparam matching."""
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


def _read_bundle_vids(bundle_path: str) -> List[str]:
    with open(bundle_path, "r") as f:
        return [x.strip() for x in f.read().strip().split("\n") if x.strip()]


def _collect_task_counts(vids: List[str], video_to_task: Dict[str, int], unknown: UnknownTracker) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for v in vids:
        uid = normalize_video_id(v)
        tid = video_to_task.get(uid, None)
        if tid is None:
            unknown.add(uid)
            continue
        counts[int(tid)] = counts.get(int(tid), 0) + 1
    return counts


def _allowed_set_for_task(vids: List[str], video_to_task: Dict[str, int], task_id: int, unknown: UnknownTracker) -> Set[str]:
    allowed = set()
    for v in vids:
        uid = normalize_video_id(v)
        tid = video_to_task.get(uid, None)
        if tid is None:
            unknown.add(uid)
            continue
        if int(tid) == int(task_id):
            allowed.add(uid)
    return allowed


def _save_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


TAS_ALLOWED_TASK_IDS = [1, 3, 4, 5]

def _load_state_dict_flexible(model: torch.nn.Module, state_dict: dict) -> None:
    """Load checkpoints saved with or without DataParallel 'module.' prefix."""
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    # Try stripping 'module.' prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        stripped = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(stripped)
        return
    # Try adding 'module.' prefix
    prefixed = {("module." + k): v for k, v in state_dict.items()}
    model.load_state_dict(prefixed)


def _avg_score_joint_fair_style_single_task(per_task_metrics: Dict[str, Dict[int, float]], task_id: int) -> float:
    """Match joint_fair Avg definition, but for a single task only."""
    tid = int(task_id)
    try:
        acc = float(per_task_metrics["acc"][tid])
        edit = float(per_task_metrics["edit"][tid])
        f1_010 = float(per_task_metrics["f1_010"][tid])
        f1_025 = float(per_task_metrics["f1_025"][tid])
        f1_050 = float(per_task_metrics["f1_050"][tid])
    except Exception:
        return float("nan")
    return (acc + edit + f1_010 + f1_025 + f1_050) / 5.0


def _format_task_eval_line(prefix: str, epoch: int, task_id: int, per_task_metrics: Dict[str, Dict[int, float]]) -> str:
    tid = int(task_id)
    acc = float(per_task_metrics.get("acc", {}).get(tid, float("nan")))
    edit = float(per_task_metrics.get("edit", {}).get(tid, float("nan")))
    f1_010 = float(per_task_metrics.get("f1_010", {}).get(tid, float("nan")))
    f1_025 = float(per_task_metrics.get("f1_025", {}).get(tid, float("nan")))
    f1_050 = float(per_task_metrics.get("f1_050", {}).get(tid, float("nan")))
    f1_avg = (f1_010 + f1_025 + f1_050) / 3.0
    avg = (acc + edit + f1_010 + f1_025 + f1_050) / 5.0
    return (
        f"{prefix} Epoch {int(epoch)} task_id={tid}: "
        f"Acc: {acc:.4f}, Edit: {edit:.4f}, "
        f"F1@0.10: {f1_010:.4f}, F1@0.25: {f1_025:.4f}, F1@0.50: {f1_050:.4f}, "
        f"F1@Avg: {f1_avg:.4f}, Avg: {avg:.4f}"
    )


def _require_no_overrides_for_fixed_tasks(args: argparse.Namespace) -> None:
    # Strict protocol: fixed *task set* (and count), but order may be randomized if enabled.
    if getattr(args, "num_tasks", None) not in (None, len(TAS_ALLOWED_TASK_IDS)):
        raise RuntimeError(f"TAS continual uses fixed num_tasks={len(TAS_ALLOWED_TASK_IDS)}; got --num_tasks={args.num_tasks}")


def _require_task_coverage_strict(
    *,
    mode: str,
    task_order: List[int],
    video_to_task: Dict[str, int],
    train_source_bundles,
    train_target_bundles,
    val_source_bundles,
    val_target_bundles,
    test_source_bundles,
    test_target_bundles,
) -> None:
    """Strict academic protocol: every task must appear in all splits/domains."""

    def _iter_bundle_paths(b):
        if isinstance(b, str):
            return [b]
        return list(b)

    def _tasks_in_bundle(bundle) -> Set[int]:
        vids = []
        for p in _iter_bundle_paths(bundle):
            vids.extend(_read_bundle_vids(p))
        out: Set[int] = set()
        for v in vids:
            uid = normalize_video_id(v)
            tid = video_to_task.get(uid, None)
            if tid is not None:
                out.add(int(tid))
        return out

    coverage = {
        "train_source": _tasks_in_bundle(train_source_bundles),
        "train_target": _tasks_in_bundle(train_target_bundles),
        "val_source": _tasks_in_bundle(val_source_bundles),
        "val_target": _tasks_in_bundle(val_target_bundles),
        "test_source": _tasks_in_bundle(test_source_bundles),
        "test_target": _tasks_in_bundle(test_target_bundles),
    }
    missing = {k: [t for t in task_order if int(t) not in v] for k, v in coverage.items()}
    bad = {k: v for k, v in missing.items() if len(v) > 0}
    if bad:
        msg = [f"[TAS strict task coverage failed] mode={mode} task_order={task_order}"]
        for split_name, miss in bad.items():
            msg.append(f"- {split_name} missing tasks: {miss}")
        msg.append("This indicates your split bundles + video_to_task mapping do not satisfy the strict protocol.")
        raise RuntimeError("\n".join(msg))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("CLEGO TAS continual runner (task-incremental sequential finetuning)")
    p.add_argument("--config", type=str, default=None, help="YAML config file (optional). CLI args override config.")
    # CL settings
    p.add_argument("--video_to_task_path", type=str, default=os.path.join(_REPO_ROOT, "video_to_task.npy"))
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--dry_run", action="store_true", help="Only write task_order/task_stats/unknown_ids then exit.")
    p.add_argument("--num_tasks", type=int, default=None)
    # Default: randomize task order (but keep the allowed task set fixed).
    # Use --no_randomize_order to reproduce the legacy fixed order behavior.
    p.add_argument("--randomize_order", action="store_true", default=True)
    p.add_argument("--no_randomize_order", action="store_true", help="Disable task order randomization (legacy fixed order).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs_per_task", type=int, default=1)
    p.add_argument(
        "--val_every",
        type=int,
        default=0,
        help=(
            "How often to run the *current-task* VAL sweep during training. "
            "Set to 0 to disable per-epoch sweeps and evaluate ONLY once at task end (default: 0). "
            "Set to N>=1 to keep legacy online VAL-best tracking every N epochs."
        ),
    )
    p.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--max_train_videos_per_task", type=int, default=None, help="Optional cap on #train videos per task (sanity/fast runs).")
    p.add_argument("--max_eval_videos", type=int, default=None, help="Optional cap on #eval videos total (sanity/fast runs).")
    # Continual algorithm (pluggable)
    p.add_argument(
        "--continual_algorithm",
        type=str,
        default="none",
        help="Continual algorithm to add on top of baseline. Supported: none | er | derpp | ewc | lwf",
    )
    p.add_argument("--continual_algorithm_buffer_ratio", type=float, default=0.2, help="ER buffer size as ratio of total train videos (default: 0.2)")
    p.add_argument("--continual_algorithm_replay_batch_ratio", type=float, default=0.2, help="ER replay batch size as ratio of current batch (default: 0.2)")
    p.add_argument(
        "--continual_algorithm_distill_alpha",
        type=float,
        default=0.5,
        help="DERPP distillation loss weight (default: 0.5). Only used when continual_algorithm=derpp.",
    )
    # EWC/LwF hyperparams
    p.add_argument("--ewc_lambda", type=float, default=1e-2, help="EWC regularization weight (default: 1e-2)")
    p.add_argument("--ewc_gamma", type=float, default=1.0, help="EWC online fisher decay (default: 1.0)")
    p.add_argument("--ewc_fisher_batches", type=int, default=50, help="EWC fisher batches per task (default: 50)")
    p.add_argument("--lwf_alpha", type=float, default=0.5, help="LwF distillation weight (default: 0.5)")

    # ------------------
    # VISTA configs
    # ------------------
    p.add_argument("--vista_enabled", action="store_true", help="Enable VISTA (whitened-subspace router + adapters).")
    p.add_argument("--vista_router_M", type=int, default=1)
    p.add_argument("--vista_subspace_k", type=int, default=32)
    p.add_argument("--vista_topL", type=int, default=2, help="Inference-time top-L tasks to mix adapters (default: 2).")
    p.add_argument("--vista_gamma", type=float, default=10.0)
    p.add_argument("--vista_eps", type=float, default=1e-6)
    p.add_argument("--vista_adapter_bottleneck", type=int, default=64)
    p.add_argument("--vista_apply_to_target", type=int, default=1, help="Apply mixture adapters to target stream (1/0).")
    p.add_argument("--vista_train_backbone_after_task1", action="store_true")
    p.add_argument("--vista_router_fit_max_samples", type=int, default=0)

    # ------------------
    # L2P configs
    # ------------------
    p.add_argument("--l2p_enabled", action="store_true", help="Enable L2P (fixed adapter pool + key-query selection).")
    p.add_argument("--l2p_pool_size", type=int, default=4, help="L2P: prompt pool size (default: 4).")
    p.add_argument("--l2p_topK", type=int, default=2, help="L2P: top-K adapters selected per sample (default: 2).")
    p.add_argument("--l2p_router_M", type=int, default=1, help="L2P: time-chunk pooling segments M for query representation (default: 1).")
    p.add_argument("--l2p_adapter_bottleneck", type=int, default=64, help="L2P: adapter bottleneck dim r (default: 64).")
    p.add_argument("--l2p_sim_lambda", type=float, default=0.5, help="L2P: similarity loss weight (default fixed: 0.5).")
    p.add_argument("--l2p_diversed_selection", action="store_true", default=True, help="L2P: enable frequency-based diversified selection during training.")
    p.add_argument("--l2p_batchwise_selection", action="store_true", default=False, help="L2P: batch-wise top-K selection (optional).")

    # Key baseline args (a subset, but includes what model/trainer expects)
    p.add_argument("--feature_path", type=str, default=None)
    p.add_argument("--exp_type", type=str, default="ego-only")
    # By default, point to the repo-provided split bundles. For full training/eval, set this to your dataset root.
    p.add_argument("--path_data", type=str, default="temporal_action_segmentation_benchmark/tas_annotation")
    p.add_argument("--num_stages", default=4, type=int)
    p.add_argument("--num_layers", default=10, type=int)
    p.add_argument("--num_f_maps", default=64, type=int)
    p.add_argument("--features_dim", default=2048, type=int)
    p.add_argument("--lr", default=0.0005, type=float)
    p.add_argument("--bS", default=1, type=int)
    p.add_argument("--alpha", default=0.15, type=float)
    p.add_argument("--tau", default=4, type=float)
    p.add_argument("--use_target", default="none", choices=["none", "uSv"])
    p.add_argument("--split_target", default="0")
    p.add_argument("--ratio_source", default=1.0, type=float)
    p.add_argument("--ratio_label_source", default=1.0, type=float)
    p.add_argument("--resume_epoch", default=0, type=int)
    p.add_argument("--use_best_model", type=str, default="none", choices=["none", "source", "target"])
    p.add_argument("--multi_gpu", default=False, action="store_true")
    p.add_argument("--verbose", default=False, action="store_true")
    p.add_argument("--use_tensorboard", default=False, action="store_true")
    p.add_argument("--epoch_embedding", default=50, type=int)
    p.add_argument("--stage_embedding", default=-1, type=int)
    p.add_argument("--num_frame_video_embedding", default=50, type=int)

    # Sampling
    p.add_argument("--feat_sample_rate", default=1, type=int)
    p.add_argument("--label_sample_rate", default=2, type=int)
    p.add_argument("--all_sample_rate", default=1, type=int)

    # DA knobs used in Trainer.train
    p.add_argument("--DA_adv", default="none", type=str)
    p.add_argument("--DA_adv_video", default="none", type=str)
    p.add_argument("--pair_ssl", default="all", type=str)
    p.add_argument("--num_seg", default=10, type=int)
    p.add_argument("--place_adv", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--multi_adv", default=["N", "N"], type=str, nargs="+")
    p.add_argument("--weighted_domain_loss", default="Y", type=str)
    p.add_argument("--ps_lb", default="soft", type=str)
    p.add_argument("--source_lb_weight", default="pseudo", type=str)
    p.add_argument("--method_centroid", default="none", type=str)
    p.add_argument("--DA_sem", default="mse", type=str)
    p.add_argument("--place_sem", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--ratio_ma", default=0.7, type=float)
    p.add_argument("--iter_max_beta", default=[1000, 1000], type=float, nargs="+")
    p.add_argument("--beta", default=[-2, -2], type=float, nargs="+")
    p.add_argument("--gamma", default=-2, type=float)
    p.add_argument("--iter_max_gamma", default=1000, type=float)
    p.add_argument("--DA_ent", default="none", type=str)
    p.add_argument("--place_ent", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--mu", default=1, type=float)
    p.add_argument("--use_attn", default="none", choices=["none", "domain_attn"])
    p.add_argument("--DA_dis", default="none", choices=["none", "JAN"])
    p.add_argument("--place_dis", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--nu", default=-2, type=float)
    p.add_argument("--iter_max_nu", default=1000, type=float)
    p.add_argument("--DA_ens", default="none", choices=["none", "MCD", "SWD"])
    p.add_argument("--place_ens", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--dim_proj", default=128, type=int)
    p.add_argument("--SS_video", default="none", choices=["none", "VCOP"])
    p.add_argument("--place_ss", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--eta", default=1, type=float)

    return p


def main() -> None:
    cfg_path, remaining = split_argv_config(sys.argv[1:])
    parser = build_parser()
    if cfg_path is not None:
        cfg = load_yaml_config(cfg_path)
        apply_config_as_defaults(parser, cfg)
    args = parser.parse_args(remaining)
    if getattr(args, "no_randomize_order", False):
        args.randomize_order = False
    if not args.output_root:
        raise ValueError("--output_root must be provided via CLI or --config")
    if not args.feature_path and not args.dry_run:
        raise ValueError("--feature_path must be provided via CLI or --config (unless --dry_run)")

    os.makedirs(args.output_root, exist_ok=True)
    _dump_args_json(args.output_root, args, extra={"entrypoint": "egolearner_continual_main.py"})

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unknown = UnknownTracker(max_examples=200)

    # Resolve split bundles etc from settings
    # We emulate egolearner_main behavior with get_annotations_from_settings.
    args.dataset = "egobridge"
    _require_no_overrides_for_fixed_tasks(args)
    video_to_task = load_video_to_task(args.video_to_task_path)

    # Always resolve BOTH val and test bundles. We will:
    # - select best checkpoints using VAL
    # - report TEST results using the val-best epoch
    args.test = False
    train_source_vid_list_file, train_source_feat_suffix, val_source_vid_list_file, val_source_feat_suffix, \
        train_target_vid_list_file, train_target_feat_suffix, val_target_vid_list_file, val_target_feat_suffix = get_annotations_from_settings(args)
    args.test = True
    _, _, test_source_vid_list_file, test_source_feat_suffix, \
        _, _, test_target_vid_list_file, test_target_feat_suffix = get_annotations_from_settings(args)

    # Strict protocol: fixed task set for ALL modes.
    task_order = [int(x) for x in TAS_ALLOWED_TASK_IDS]
    if getattr(args, "randomize_order", False):
        # Deterministic shuffle controlled by seed; used only for task order.
        rng = np.random.RandomState(int(args.seed))
        rng.shuffle(task_order)
    task_ids_present = sorted(set(int(v) for v in video_to_task.values()))
    save_task_order(os.path.join(args.output_root, "task_order.json"), task_order)

    # Collect basic stats (video counts per task per split)
    stats = {
        "benchmark": "temporal_action_segmentation_benchmark",
        "task_order": task_order,
        "effective_num_tasks": int(len(task_order)),
        "tasks_present_in_train": [int(x) for x in task_ids_present],
        "micro_weight_unit": "num_videos_in_eval_split",
        "splits": {},
    }
    for name, bundle in [
        ("train_source", train_source_vid_list_file),
        ("train_target", train_target_vid_list_file),
        ("val_target", val_target_vid_list_file),
        ("test_target", test_target_vid_list_file),
    ]:
        vids = []
        if isinstance(bundle, str):
            vids = _read_bundle_vids(bundle)
        else:
            for b in bundle:
                vids.extend(_read_bundle_vids(b))
        counts = _collect_task_counts(vids, video_to_task, unknown)
        stats["splits"][name] = {str(k): int(v) for k, v in counts.items()}
        stats["splits"][name]["total"] = int(len(vids))

    # unknown ratio based on union of splits processed above
    total_seen = sum(int(stats["splits"][k]["total"]) for k in stats["splits"])
    stats.update(unknown.to_dict())
    stats["unknown_ratio"] = float(unknown.count) / float(total_seen) if total_seen > 0 else 0.0
    save_task_stats(args.output_root, stats)
    save_unknown_ids(args.output_root, unknown)

    if args.dry_run:
        return

    # Strictly require that the chosen tasks appear in ALL splits/domains (train_source/train_target/val/test).
    # This avoids silent empty-task training and mismatched continual protocols.
    # Note: get_annotations_from_settings() switches test bundles when args.test=True.
    # We already resolved bundles above; reuse them here in a consistent naming scheme.
    ts_val, ts_suf_val, vs_val, vs_suf_val, tt_val, tt_suf_val, vt_val, vt_suf_val = (
        train_source_vid_list_file,
        train_source_feat_suffix,
        val_source_vid_list_file,
        val_source_feat_suffix,
        train_target_vid_list_file,
        train_target_feat_suffix,
        val_target_vid_list_file,
        val_target_feat_suffix,
    )
    ts_test, ts_suf_test, vs_test, vs_suf_test, tt_test, tt_suf_test, vt_test, vt_suf_test = (
        train_source_vid_list_file,
        train_source_feat_suffix,
        test_source_vid_list_file,
        test_source_feat_suffix,
        train_target_vid_list_file,
        train_target_feat_suffix,
        test_target_vid_list_file,
        test_target_feat_suffix,
    )
    _require_task_coverage_strict(
        mode=str(args.exp_type),
        task_order=task_order,
        video_to_task=video_to_task,
        train_source_bundles=ts_val,
        train_target_bundles=tt_val,
        val_source_bundles=vs_val,
        val_target_bundles=vt_val,
        test_source_bundles=vs_test,
        test_target_bundles=vt_test,
    )

    # Build model + trainer
    actions_dict = {str(i): i for i in range(28)}  # 1 + 27, consistent with baseline
    num_classes = len(actions_dict)
    model = MultiStageModel(args, num_classes)
    trainer = Trainer(num_classes)

    # ------------------------------
    # VISTA setup (feature space)
    # ------------------------------
    vista_enabled = bool(getattr(args, "vista_enabled", False))
    vista_state = None
    base_train.vista_adapter_optimizer = None
    if vista_enabled:
        embed_dim = int(getattr(args, "features_dim", 2048))
        adapter_bank = AdapterBank(
            input_dim=embed_dim,
            bottleneck=int(getattr(args, "vista_adapter_bottleneck", 64)),
            use_layernorm=True,
        ).to(device)
        router = build_vista_router(
            router_M=int(getattr(args, "vista_router_M", 1)),
            subspace_k=int(getattr(args, "vista_subspace_k", 32)),
            eps=float(getattr(args, "vista_eps", 1e-6)),
        )
        vista_state = VISTAState(
            enabled=True,
            adapter_bank=adapter_bank,
            router=router,
            router_M=int(getattr(args, "vista_router_M", 1)),
            topL=int(getattr(args, "vista_topL", 2)),
            gamma=float(getattr(args, "vista_gamma", 10.0)),
            eps=float(getattr(args, "vista_eps", 1e-6)),
            apply_to_target=bool(getattr(args, "vista_apply_to_target", 1)),
            train_backbone_after_task1=bool(getattr(args, "vista_train_backbone_after_task1", False)),
        )
        base_train.vista_enabled = True
        base_train.vista_state = vista_state
        base_train.vista_mode = "none"
        base_predict.vista_enabled = True
        base_predict.vista_state = vista_state
        base_predict.vista_mode = "none"
    else:
        base_train.vista_enabled = False
        base_train.vista_state = None
        base_train.vista_mode = "none"
        base_predict.vista_enabled = False
        base_predict.vista_state = None
        base_predict.vista_mode = "none"

    # ------------------------------
    # L2P setup (feature space)
    # ------------------------------
    l2p_pool = None
    if bool(getattr(args, "l2p_enabled", False)):
        embed_dim = int(getattr(args, "features_dim", 2048))
        key_dim = int(getattr(args, "l2p_router_M", 1)) * int(embed_dim)
        pool_size = int(len(task_order))
        l2p_pool = L2PPool(
            pool_size=pool_size,
            topk=int(getattr(args, "l2p_topK", 2)),
            adapter_dim=int(embed_dim),
            key_dim=int(key_dim),
            adapter_bottleneck=int(getattr(args, "l2p_adapter_bottleneck", 64)),
            diversed_selection=bool(getattr(args, "l2p_diversed_selection", True)),
            batchwise_selection=bool(getattr(args, "l2p_batchwise_selection", False)),
        ).to(device)
        base_train.l2p_enabled = True
        base_train.l2p_pool = l2p_pool
        base_train.l2p_topk = int(getattr(args, "l2p_topK", 2))
        base_train.l2p_router_M = int(getattr(args, "l2p_router_M", 1))
        base_train.l2p_sim_lambda = float(getattr(args, "l2p_sim_lambda", 0.5))
        base_train.l2p_diversed_selection = bool(getattr(args, "l2p_diversed_selection", True))
        base_train.l2p_batchwise_selection = bool(getattr(args, "l2p_batchwise_selection", False))
        base_train.l2p_mode = "train"
        base_train.l2p_optimizer = torch.optim.Adam(l2p_pool.parameters(), lr=float(getattr(args, "lr", 5e-4)))
        base_predict.l2p_enabled = True
        base_predict.l2p_pool = l2p_pool
        base_predict.l2p_topk = int(getattr(args, "l2p_topK", 2))
        base_predict.l2p_router_M = int(getattr(args, "l2p_router_M", 1))
        base_predict.l2p_sim_lambda = float(getattr(args, "l2p_sim_lambda", 0.5))
        base_predict.l2p_diversed_selection = bool(getattr(args, "l2p_diversed_selection", True))
        base_predict.l2p_batchwise_selection = bool(getattr(args, "l2p_batchwise_selection", False))
        base_predict.l2p_mode = "none"
    else:
        base_train.l2p_enabled = False
        base_train.l2p_pool = None
        base_train.l2p_mode = "none"
        base_train.l2p_optimizer = None
        base_predict.l2p_enabled = False
        base_predict.l2p_pool = None
        base_predict.l2p_mode = "none"

    def _vista_fit_router_for_task(*, task_id: int, batch_gen) -> None:
        if vista_state is None or vista_state.router is None:
            return
        max_samples = int(getattr(args, "vista_router_fit_max_samples", 0))

        class _PairLoader:
            def __iter__(self_inner):
                n_samples = 0
                batch_gen.reset()
                while batch_gen.has_next():
                    x, _, _ = batch_gen.next_batch(args.bS, 'source')
                    if max_samples > 0:
                        remaining = int(max_samples - n_samples)
                        if remaining <= 0:
                            break
                        x = x[:remaining]
                    if x.numel() == 0:
                        continue
                    # BatchGenerator returns (B, C, T); VISTA router expects (B, T, C).
                    x = x.transpose(1, 2).contiguous()
                    yield x, x
                    n_samples += int(x.shape[0])
                    if max_samples > 0 and n_samples >= max_samples:
                        break

        vista_state.router.fit_from_loader(task_id=int(task_id), loader=_PairLoader(), device="cpu", verbose=False)

    # shared checkpoint folder
    ckpt_dir = os.path.join(args.output_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # No need to save epoch-0 checkpoints: we can run A0 prediction directly from in-memory weights.

    # Recorder
    rec_val = ContinualRecorder(task_order=task_order)
    rec_test = ContinualRecorder(task_order=task_order)

    # ------------------------------
    # Continual algorithm (ER, etc.)
    # ------------------------------
    algo_name = str(getattr(args, "continual_algorithm", "none")).strip().lower()
    if algo_name == "lwf":
        algo = LwFGeneric(cfg=LwFGenericConfig(alpha=float(getattr(args, "lwf_alpha", 0.5))))
    else:
        algo = build_continual_algorithm(
            algo_name=getattr(args, "continual_algorithm", "none"),
            buffer_ratio=getattr(args, "continual_algorithm_buffer_ratio", 0.2),
            replay_batch_ratio=getattr(args, "continual_algorithm_replay_batch_ratio", 0.2),
            seed=int(getattr(args, "seed", 42)),
            distill_alpha=getattr(args, "continual_algorithm_distill_alpha", 0.5),
            ewc_lambda=getattr(args, "ewc_lambda", 1e-2),
            ewc_gamma=getattr(args, "ewc_gamma", 1.0),
            ewc_fisher_batches=getattr(args, "ewc_fisher_batches", 50),
            lwf_alpha=getattr(args, "lwf_alpha", 0.5),
        )
    if algo is not None:
        # Disallow VISTA/L2P combinations with EWC/LwF (per protocol).
        algo_name_str = str(getattr(algo, "name", "")).strip().lower()
        if algo_name_str in ("ewc", "lwf") and (
            bool(getattr(args, "vista_enabled", False)) or bool(getattr(args, "l2p_enabled", False))
        ):
            raise ValueError("EWC/LwF are not supported with VISTA or L2P in temporal_action_segmentation_benchmark (disable vista_enabled/l2p_enabled).")
        # Configure capacity based on total *train videos* across both domains, restricted to this protocol's task set.
        # (We keep it video-level for TAS to avoid mixing video-length/frame-level hyperparams.)
        vids_source: List[str] = []
        if isinstance(train_source_vid_list_file, str):
            vids_source = _read_bundle_vids(train_source_vid_list_file)
        else:
            for b in train_source_vid_list_file:
                vids_source.extend(_read_bundle_vids(b))
        vids_target: List[str] = []
        if isinstance(train_target_vid_list_file, str):
            vids_target = _read_bundle_vids(train_target_vid_list_file)
        else:
            for b in train_target_vid_list_file:
                vids_target.extend(_read_bundle_vids(b))
        allowed_tasks = set(int(x) for x in task_order)
        total_train = 0
        for v in vids_source:
            tid = video_to_task.get(normalize_video_id(v), None)
            if tid is not None and int(tid) in allowed_tasks:
                total_train += 1
        for v in vids_target:
            tid = video_to_task.get(normalize_video_id(v), None)
            if tid is not None and int(tid) in allowed_tasks:
                total_train += 1
        if hasattr(algo, "configure_total_capacity"):
            algo.configure_total_capacity(total_train_samples=int(total_train))

        # Inject into baseline trainer module; Trainer.train() reads a module-global.
        import egolearner_train as train_mod  # noqa

        train_mod.continual_algo = algo
        if hasattr(algo, "bind_models"):
            algo.bind_models(models={"main": model})

    class _BatchGenIterable:
        """Make BatchGenerator compatible with ER's update_memory_from_loader() contract."""

        def __init__(self, bg: BatchGenerator, bs: int):
            self.bg = bg
            self.bs = int(bs)

        def __iter__(self):
            self.bg.reset()
            while self.bg.has_next():
                yield self.bg.next_batch(self.bs, "source")

    def _tas_derpp_distill_target_fn(batch_in, model_obj: torch.nn.Module):
        """Return per-sample distillation target z for DERPP.

        We use the last-stage *source logits* (per frame), masked to valid frames using mask_source.
        Shapes:
          - input_source: (B, dim, T)
          - mask_source : (B, C, T)
          - logits      : (B, C, T)
        """
        input_source, _label_source, mask_source = batch_in
        # Only compute source branch logits; DA-related outputs are irrelevant here.
        pred, _prob, _feat, _feat_video, _pred_d, _pred_d_video, _lb_d, _lb_d_video, _pred2, _prob2 = model_obj.forward_domain(
            input_source,
            mask_source,
            0,  # domain_GT=source
            beta=[0.0, 0.0],
            reverse=False,
        )
        # last stage logits: (B, C, T)
        z = pred[:, -1, :, :]
        # Mask valid frames (padding -> 0). Use a frame mask broadcast across classes.
        frame_mask = (mask_source[:, :1, :] > 0).to(z.dtype)
        return z * frame_mask

    # A0 grouped eval (no training), on BOTH val and test
    for split_name, bundle, feat_suffix in [
        ("val", vt_val, vt_suf_val),
        ("test", vt_test, vt_suf_test),
    ]:
        a0_results_dir = os.path.join(args.output_root, "A0", split_name)
        os.makedirs(a0_results_dir, exist_ok=True)
        # Router has not been fitted yet at A0; do not apply VISTA mixture.
        base_predict.vista_mode = "none"
        base_predict.l2p_mode = "none"
        predict(
            model=model,
            model_dir=ckpt_dir,
            results_dir=a0_results_dir,
            features_path=args.feature_path,
            vid_list_file=bundle,
            feat_suffix=feat_suffix,
            feat_sample_rate=args.feat_sample_rate,
            all_sample_rate=args.all_sample_rate,
            epoch=0,
            actions_dict=actions_dict,
            device=device,
            args=args,
            load_model=False,
        )
        per_task_metrics_all, weights_all = eval_grouped(
            results_dir=a0_results_dir,
            gt_dir=os.path.join(args.path_data, "gts_fps25/"),
            split_bundle=bundle,
            video_to_task=video_to_task,
            seen_tasks=set(task_order),
        )
        for t in task_order:
            if float(weights_all.get(int(t), 0.0)) <= 0:
                raise RuntimeError(
                    f"[TAS strict] A0 eval has zero weight for task_id={t} on split={split_name}. "
                    f"Do not cap eval to miss tasks (max_eval_videos), and ensure bundles cover all tasks."
                )
        for metric_key, mp in per_task_metrics_all.items():
            if split_name == "val":
                rec_val.set_A0(metric_key, mp, weights_all)
            else:
                rec_test.set_A0(metric_key, mp, weights_all)

    # Continual training over tasks (per-task epoch semantics)
    epochs_per_task = int(args.epochs_per_task)
    val_every = int(getattr(args, "val_every", 0) or 0)
    if val_every < 0:
        raise ValueError(f"--val_every must be >= 0, got {val_every}")
    for t_idx, task_id in enumerate(task_order, start=1):
        task_dir = os.path.join(args.output_root, f"task_{t_idx:02d}")
        os.makedirs(task_dir, exist_ok=True)
        task_ckpt_dir = os.path.join(task_dir, "checkpoints")
        os.makedirs(task_ckpt_dir, exist_ok=True)
        # Do not save per-task epoch-0 checkpoints; this runner keeps only val_best.* per task.

        # allowed ids for this task (per-domain; for mixed ego/exo settings, source/target video IDs never overlap)
        train_source_vids: List[str] = []
        if isinstance(train_source_vid_list_file, str):
            train_source_vids = _read_bundle_vids(train_source_vid_list_file)
        else:
            for b in train_source_vid_list_file:
                train_source_vids.extend(_read_bundle_vids(b))

        train_target_vids: List[str] = []
        if isinstance(train_target_vid_list_file, str):
            train_target_vids = _read_bundle_vids(train_target_vid_list_file)
        else:
            for b in train_target_vid_list_file:
                train_target_vids.extend(_read_bundle_vids(b))

        allowed_source = _allowed_set_for_task(train_source_vids, video_to_task, task_id, unknown)
        allowed_target = _allowed_set_for_task(train_target_vids, video_to_task, task_id, unknown)

        if args.max_train_videos_per_task is not None:
            cap = int(args.max_train_videos_per_task)
            if cap > 0:
                if len(allowed_source) > cap:
                    allowed_source = set(sorted(list(allowed_source))[:cap])
                if len(allowed_target) > cap:
                    allowed_target = set(sorted(list(allowed_target))[:cap])

        # Strict: do NOT fallback. Missing data is a protocol/configuration error.
        if len(allowed_source) == 0 or len(allowed_target) == 0:
            raise RuntimeError(
                f"[TAS strict] Empty training set after task filtering. "
                f"task_index={t_idx} task_id={task_id} allowed_source={len(allowed_source)} allowed_target={len(allowed_target)} "
                f"(exp_type={args.exp_type}, eval_split={args.eval_split}). "
                f"This indicates the dataset split does not support the selected tasks for this mode."
            )

        # (No skip path in strict mode.)

        # Build batch generators (train only on allowed subset)
        #
        gt_path = os.path.join(args.path_data, "gts_fps25/")
        batch_gen_source_train = BatchGenerator(
            num_classes, actions_dict, gt_path, args.feature_path,
            feat_sample_rate=args.feat_sample_rate, label_sample_rate=args.label_sample_rate,
            all_sample_rate=args.all_sample_rate, feat_suffix=train_source_feat_suffix,
            allowed_video_ids=allowed_source, video_to_task=video_to_task, unknown_tracker=unknown,
        )
        batch_gen_target_train = BatchGenerator(
            num_classes, actions_dict, gt_path, args.feature_path,
            feat_sample_rate=args.feat_sample_rate, label_sample_rate=args.label_sample_rate,
            all_sample_rate=args.all_sample_rate, feat_suffix=train_target_feat_suffix,
            allowed_video_ids=allowed_target, video_to_task=video_to_task, unknown_tracker=unknown,
        )
        batch_gen_source_val = BatchGenerator(
            num_classes, actions_dict, gt_path, args.feature_path,
            feat_sample_rate=args.feat_sample_rate, label_sample_rate=args.label_sample_rate,
            all_sample_rate=args.all_sample_rate, feat_suffix=vs_suf_val,
        )
        batch_gen_target_val = BatchGenerator(
            num_classes, actions_dict, gt_path, args.feature_path,
            feat_sample_rate=args.feat_sample_rate, label_sample_rate=args.label_sample_rate,
            all_sample_rate=args.all_sample_rate, feat_suffix=vt_suf_val,
        )
        batch_gen_source_train.read_data(train_source_vid_list_file)
        batch_gen_target_train.read_data(train_target_vid_list_file)
        batch_gen_source_val.read_data(vs_val)
        batch_gen_target_val.read_data(vt_val)

        # ---- VISTA: create current task adapter (hot-start from previous) ----
        if vista_state is not None and vista_state.adapter_bank is not None:
            init_from = None
            if t_idx >= 2:
                init_from = int(task_order[t_idx - 2])
            vista_state.adapter_bank.add_task(int(task_id), init_from_task=init_from)
            vista_state.adapter_bank.set_current_task(int(task_id))
            vista_state.adapter_bank.freeze_all_except(int(task_id))
            base_train.vista_adapter_optimizer = torch.optim.Adam(
                [p for p in vista_state.adapter_bank.get(int(task_id)).parameters() if p.requires_grad],
                lr=float(getattr(args, "lr", 5e-4)),
            )
            base_train.vista_mode = "train"
        if l2p_pool is not None:
            base_train.l2p_mode = "train"

        # ---- VISTA: optionally freeze backbone after task 1 ----
        if vista_state is not None and int(t_idx) >= 2:
            freeze = not bool(getattr(args, "vista_train_backbone_after_task1", False))
            if freeze:
                for p in model.parameters():
                    p.requires_grad = False

        # Per-task epoch semantics: match standard epochs within each task.
        args.resume_epoch = 0
        args.num_epochs = epochs_per_task
        task_results_dir = os.path.join(task_dir, "results")

        start = time.time()
        _save_json(
            os.path.join(task_dir, "train_task_filter.json"),
            {
                "task_index": t_idx,
                "task_id": int(task_id),
                "allowed_source_videos": int(len(allowed_source)),
                "allowed_target_videos": int(len(allowed_target)),
            },
        )
        # ------------------------------
        # VAL selection/evaluation strategy for this task:
        # - Legacy (val_every >= 1): online VAL sweep every N epochs; keep only the best weights on disk.
        # - New default (val_every == 0): NO per-epoch sweeps. Evaluate ONLY once at task end and save
        #   exactly one checkpoint (still named val_best.* for backward compatibility).
        #
        # Note: Regardless of mode, this runner stores only a single checkpoint per task on disk.
        # ------------------------------

        # Helper: optionally cap a bundle for fast sanity (kept consistent with legacy behavior)
        def _maybe_cap_bundle(bundle_in, tag: str) -> Union[str, List[str]]:
            if args.max_eval_videos is None:
                return bundle_in
            capped_path = os.path.join(task_dir, f"{tag}_capped.bundle")
            vids_eval: List[str] = []
            if isinstance(bundle_in, str):
                vids_eval = _read_bundle_vids(bundle_in)
            else:
                for b in bundle_in:
                    vids_eval.extend(_read_bundle_vids(b))
            vids_eval = vids_eval[: int(args.max_eval_videos)]
            with open(capped_path, "w", encoding="utf-8") as f:
                f.write("\n".join(vids_eval) + ("\n" if len(vids_eval) > 0 else ""))
            return capped_path

        val_bundle_for_sweep = _maybe_cap_bundle(vt_val, tag="val_sweep")
        sweep_dir = os.path.join(task_dir, "_val_sweep_tmp")
        os.makedirs(sweep_dir, exist_ok=True)
        sweep_log = os.path.join(task_dir, "val_sweep_current_task.log")
        # Overwrite sweep log for deterministic runs
        with open(sweep_log, "w", encoding="utf-8") as _:
            pass

        best_epoch: Union[int, None] = None
        best_score = float("-inf")

        def _save_val_best(epoch_num: int, score: float, model_obj: torch.nn.Module, opt_obj) -> None:
            nonlocal best_epoch, best_score
            best_epoch = int(epoch_num)
            best_score = float(score)
            # Handle DataParallel when saving
            state_dict = model_obj.module.state_dict() if hasattr(model_obj, "module") else model_obj.state_dict()
            torch.save(state_dict, os.path.join(task_ckpt_dir, "val_best.model"))
            try:
                torch.save(opt_obj.state_dict(), os.path.join(task_ckpt_dir, "val_best.opt"))
            except Exception:
                pass
            _save_json(
                os.path.join(task_dir, "val_best.json"),
                {
                    "task_index": int(t_idx),
                    "task_id": int(task_id),
                    "best_epoch": int(best_epoch),
                    "best_avg_score": float(best_score),
                    "metric": "Avg=(Acc+Edit+F1@0.10+F1@0.25+F1@0.50)/5, current-task only",
                    "sweep_log": os.path.abspath(sweep_log),
                    "val_every": int(val_every),
                },
            )

        def _eval_current_task_on_val_once(*, epoch_num: int, model_obj: torch.nn.Module, opt_obj) -> float:
            """Run a single current-task VAL eval using in-memory weights. Returns the scalar score used by legacy sweep."""
            # Predict using in-memory weights (no disk checkpoints)
            if (
                vista_state is not None
                and vista_state.router is not None
                and vista_state.router.num_tasks() > 0
                and vista_state.adapter_bank is not None
                and vista_state.adapter_bank.num_tasks() > 0
            ):
                base_predict.vista_mode = "infer"
            else:
                base_predict.vista_mode = "none"
            predict(
                model=model_obj,
                model_dir=task_ckpt_dir,
                results_dir=sweep_dir,
                features_path=args.feature_path,
                vid_list_file=val_bundle_for_sweep,
                feat_suffix=vt_suf_val,
                feat_sample_rate=args.feat_sample_rate,
                all_sample_rate=args.all_sample_rate,
                epoch=int(epoch_num),
                actions_dict=actions_dict,
                device=device,
                args=args,
                load_model=False,
            )
            cur_metrics, cur_weights = eval_grouped(
                results_dir=sweep_dir,
                gt_dir=os.path.join(args.path_data, "gts_fps25/"),
                split_bundle=val_bundle_for_sweep,
                video_to_task=video_to_task,
                seen_tasks={int(task_id)},
            )
            if float(cur_weights.get(int(task_id), 0.0)) <= 0:
                raise RuntimeError(
                    f"[TAS strict] val sweep has zero weight for current task_id={int(task_id)} at epoch={int(epoch_num)}."
                )
            line = _format_task_eval_line(prefix="[val_sweep]", epoch=int(epoch_num), task_id=int(task_id), per_task_metrics=cur_metrics)
            with open(sweep_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            score = _avg_score_joint_fair_style_single_task(cur_metrics, int(task_id))
            return float(score) if np.isfinite(score) else float("-inf")

        def _epoch_end_cb(epoch_num: int, model_obj: torch.nn.Module, opt_obj) -> None:
            # New default (val_every==0): evaluate exactly once at task end.
            if int(val_every) == 0:
                if int(epoch_num) != int(epochs_per_task):
                    return
                score = _eval_current_task_on_val_once(epoch_num=int(epoch_num), model_obj=model_obj, opt_obj=opt_obj)
                _save_val_best(epoch_num=int(epoch_num), score=float(score), model_obj=model_obj, opt_obj=opt_obj)
                return

            # Legacy online mode (val_every>=1): evaluate every N epochs.
            if int(epoch_num) % int(val_every) != 0:
                return
            score = _eval_current_task_on_val_once(epoch_num=int(epoch_num), model_obj=model_obj, opt_obj=opt_obj)
            if float(score) > float(best_score):
                _save_val_best(epoch_num=int(epoch_num), score=float(score), model_obj=model_obj, opt_obj=opt_obj)

        # Train; do NOT save per-epoch checkpoints.
        base_train.vista_mode = "train"
        trainer.train(
            model=model,
            model_dir=task_ckpt_dir,
            results_dir=task_results_dir,
            batch_gen_source_train=batch_gen_source_train,
            batch_gen_target_train=batch_gen_target_train,
            batch_gen_source_test=batch_gen_source_val,
            batch_gen_target_test=batch_gen_target_val,
            device=device,
            args=args,
            epoch_end_callback=_epoch_end_cb,
            save_epoch_checkpoints=False,
        )

        # ---- VISTA: fit router and save adapters/router at task end ----
        if vista_state is not None:
            _vista_fit_router_for_task(task_id=int(task_id), batch_gen=batch_gen_source_train)
            vista_state.adapter_bank.save(os.path.join(task_dir, "adapters"))
            vista_state.router.save_task(output_dir=os.path.join(task_dir, "router"), task_id=int(task_id))
            vista_state.router.save_task(output_dir=os.path.join(args.output_root, "router"), task_id=int(task_id))

        if best_epoch is None or not os.path.isfile(os.path.join(task_ckpt_dir, "val_best.model")):
            raise RuntimeError(f"[continual] failed to select/save val-best for task_id={int(task_id)} (best_epoch={best_epoch})")

        # Carry over: load best model weights into the in-memory model before next task begins.
        # (Also used for DERPP memory distillation targets, by your protocol.)
        state = torch.load(os.path.join(task_ckpt_dir, "val_best.model"), map_location="cpu")
        _load_state_dict_flexible(model, state)

        # ---- LwF: update teacher snapshot at task end (after loading val-best) ----
        if algo is not None and str(getattr(algo, "name", "")).strip().lower() == "lwf":
            if hasattr(algo, "update_teacher"):
                algo.update_teacher(model)

        # ---- EWC: estimate Fisher at task end (using val-best weights) ----
        if algo is not None and str(getattr(algo, "name", "")).strip().lower() == "ewc":
            ewc_loader = _BatchGenIterable(batch_gen_source_train, bs=int(getattr(args, "bS", 1)))
            ce = torch.nn.CrossEntropyLoss(ignore_index=-100).to(device=device)

            def _ewc_loss_from_batch(batch_in):
                input_source, label_source, mask_source = batch_in
                input_source = input_source.to(device)
                label_source = label_source.to(device)
                mask_source = mask_source.to(device)
                pred, _prob, _feat, _feat_video, _pred_d, _pred_d_video, _lb_d, _lb_d_video, _pred2, _prob2 = model.forward_domain(
                    input_source,
                    mask_source,
                    0,  # domain_GT=source
                    beta=[0.0, 0.0],
                    reverse=False,
                )
                # last stage logits: (B, C, T)
                z = pred[:, -1, :, :]
                # mask invalid frames
                label = label_source.clone()
                frame_mask = (mask_source[:, :1, :] > 0)
                label[~frame_mask.squeeze(1)] = -100
                z_flat = z.transpose(1, 2).contiguous().view(-1, z.shape[1])
                label_flat = label.view(-1)
                return ce(z_flat, label_flat)

            max_batches = int(getattr(args, "ewc_fisher_batches", 0))
            max_batches = max_batches if max_batches > 0 else None
            algo.update_fisher_from_loader(loader=ewc_loader, loss_fn=_ewc_loss_from_batch, max_batches=max_batches)

        # ---- ER/DERPP memory update at task end (strict, task-balanced) ----
        if algo is not None and hasattr(algo, "capacity") and int(getattr(algo, "capacity", 0)) > 0:
            # Sample from the *source* training stream only (domain losses are out of scope by your protocol).
            # We reuse the already-filtered batch generator to keep the sample semantics identical to training.
            if getattr(algo, "name", "") == "derpp":
                algo.update_memory_from_loader(
                    task_id=int(task_id),
                    loader=_BatchGenIterable(batch_gen_source_train, bs=int(getattr(args, "bS", 1))),
                    model=model,
                    distill_target_fn=_tas_derpp_distill_target_fn,
                    # TAS batches are variable-length (padded per-batch). To keep DERPP robust
                    # without introducing new padding semantics, compute distill targets one sample at a time.
                    distill_batch_size=1,
                )
            else:
                algo.update_memory_from_loader(
                    task_id=int(task_id),
                    loader=_BatchGenIterable(batch_gen_source_train, bs=int(getattr(args, "bS", 1))),
                )
        model.to(device)

        # Evaluate ALL seen tasks on VAL using val-best epoch (store predictions)
        seen_tasks = set(task_order[:t_idx])
        val_results_dir = os.path.join(task_dir, "eval", "val")
        os.makedirs(val_results_dir, exist_ok=True)
        val_bundle_eval = _maybe_cap_bundle(vt_val, tag="val_eval")
        base_predict.vista_mode = "infer"
        if l2p_pool is not None:
            base_predict.l2p_mode = "infer"
        predict(
            model=model,
            model_dir=task_ckpt_dir,
            results_dir=val_results_dir,
            features_path=args.feature_path,
            vid_list_file=val_bundle_eval,
            feat_suffix=vt_suf_val,
            feat_sample_rate=args.feat_sample_rate,
            all_sample_rate=args.all_sample_rate,
            epoch=int(best_epoch or epochs_per_task),
            actions_dict=actions_dict,
            device=device,
            args=args,
            load_model=False,
        )
        per_task_metrics_val, weights_val = eval_grouped(
            results_dir=val_results_dir,
            gt_dir=os.path.join(args.path_data, "gts_fps25/"),
            split_bundle=val_bundle_eval,
            video_to_task=video_to_task,
            seen_tasks=seen_tasks,
        )
        for t in task_order[:t_idx]:
            if float(weights_val.get(int(t), 0.0)) <= 0:
                raise RuntimeError(
                    f"[TAS strict] Grouped VAL eval has zero weight for task_id={t} at t_idx={t_idx} (best_epoch={best_epoch})."
                )
        rec_val.update_after_task(t_idx=t_idx, metrics=per_task_metrics_val, weights=weights_val)
        rec_val.save(args.output_root)

        # Evaluate ALL seen tasks on TEST using the same val-best epoch (store predictions)
        test_results_dir = os.path.join(task_dir, "eval", "test")
        os.makedirs(test_results_dir, exist_ok=True)
        test_bundle_eval = _maybe_cap_bundle(vt_test, tag="test_eval")
        base_predict.vista_mode = "infer"
        if l2p_pool is not None:
            base_predict.l2p_mode = "infer"
        predict(
            model=model,
            model_dir=task_ckpt_dir,
            results_dir=test_results_dir,
            features_path=args.feature_path,
            vid_list_file=test_bundle_eval,
            feat_suffix=vt_suf_test,
            feat_sample_rate=args.feat_sample_rate,
            all_sample_rate=args.all_sample_rate,
            epoch=int(best_epoch or epochs_per_task),
            actions_dict=actions_dict,
            device=device,
            args=args,
            load_model=False,
        )
        per_task_metrics_test, weights_test = eval_grouped(
            results_dir=test_results_dir,
            gt_dir=os.path.join(args.path_data, "gts_fps25/"),
            split_bundle=test_bundle_eval,
            video_to_task=video_to_task,
            seen_tasks=seen_tasks,
        )
        for t in task_order[:t_idx]:
            if float(weights_test.get(int(t), 0.0)) <= 0:
                raise RuntimeError(
                    f"[TAS strict] Grouped TEST eval has zero weight for task_id={t} at t_idx={t_idx} (best_epoch={best_epoch})."
                )
        test_out_dir = os.path.join(args.output_root, "test_metrics")
        rec_test.update_after_task(t_idx=t_idx, metrics=per_task_metrics_test, weights=weights_test)
        rec_test.save(test_out_dir)

        # Write concise per-task summaries (val-best on val, and test-at-val-best)
        with open(os.path.join(task_dir, "eval_results_val.log"), "w", encoding="utf-8") as f:
            f.write(_format_task_eval_line(prefix="[val_best]", epoch=int(best_epoch), task_id=int(task_id), per_task_metrics=per_task_metrics_val) + "\n")
        with open(os.path.join(task_dir, "eval_results_test.log"), "w", encoding="utf-8") as f:
            f.write(_format_task_eval_line(prefix="[test_at_val_best]", epoch=int(best_epoch), task_id=int(task_id), per_task_metrics=per_task_metrics_test) + "\n")

        # ---- VISTA: evaluate router hit rates on VAL/TEST bundles (seen tasks) ----
        router_stats_val: Dict[int, Dict[str, float]] = {}
        router_hit_val: Dict[int, Dict[str, float]] = {}
        router_stats_test: Dict[int, Dict[str, float]] = {}
        router_hit_test: Dict[int, Dict[str, float]] = {}
        if vista_state is not None and vista_state.router is not None and bool(getattr(vista_state, "enabled", False)):
            def _router_eval_on_bundle(bundle_in, feat_suffixes, *, split_name: str):
                # bundle_in can be str or list[str]
                bundles = [bundle_in] if isinstance(bundle_in, str) else list(bundle_in)
                suf_list = feat_suffixes if isinstance(feat_suffixes, list) else [feat_suffixes]
                if len(bundles) != len(suf_list):
                    raise ValueError(f"[TAS vista] feat_suffix mismatch for split={split_name}: bundles={len(bundles)} suffixes={len(suf_list)}")

                # Collect (x, gt_task_id) pairs in small batches to amortize overhead.
                X_buf: List[torch.Tensor] = []
                gt_buf: List[int] = []
                out_stats: Dict[int, Dict[str, float]] = {}
                out_hits: Dict[int, Dict[str, float]] = {}
                sum_stats: Dict[int, Dict[str, float]] = {}
                sum_hits: Dict[int, Dict[str, float]] = {}
                sum_n: Dict[int, int] = {}

                def _flush():
                    nonlocal X_buf, gt_buf, out_stats, out_hits, sum_stats, sum_hits, sum_n
                    if len(X_buf) == 0:
                        return
                    # TAS features are variable-length (T differs per video). Do NOT stack along T.
                    # Evaluate router per-sample to avoid padding artifacts and shape errors.
                    with torch.no_grad():
                        for x_tc, gt_tid in zip(X_buf, gt_buf):
                            x = x_tc.unsqueeze(0).to(device=device)  # (1, T, C)
                            gt = torch.tensor([int(gt_tid)], device=device, dtype=torch.long)
                            b_stats, b_hits = vista_eval_router_grouped(
                                router=vista_state.router,
                                x=x,
                                gt_task_ids=gt,
                                M=int(getattr(args, "vista_router_M", 1)),
                                topL=int(getattr(args, "vista_topL", 2)),
                                gamma=float(getattr(args, "vista_gamma", 10.0)),
                            )
                            for tid, hh in b_hits.items():
                                n = int(hh.get("n_samples", 0))
                                if n <= 0:
                                    continue
                                sum_n[tid] = sum_n.get(tid, 0) + n
                                acc = sum_hits.get(
                                    tid,
                                    {"top1": 0.0, "topL": 0.0, "prob": 0.0, "topL_cfg": int(hh.get("topL", 1))},
                                )
                                acc["top1"] += float(hh.get("top1_hit_rate", 0.0)) * float(n)
                                acc["topL"] += float(hh.get("topL_hit_rate", 0.0)) * float(n)
                                acc["prob"] += float(hh.get("true_task_prob_mean", 0.0)) * float(n)
                                acc["topL_cfg"] = int(hh.get("topL", acc["topL_cfg"]))
                                sum_hits[tid] = acc

                                st = b_stats.get(tid, {}) or {}
                                ss = sum_stats.get(tid, {"res_best_mean": 0.0, "res_gap_mean": 0.0, "entropy_mean": 0.0})
                                ss["res_best_mean"] += float(st.get("res_best_mean", 0.0)) * float(n)
                                ss["res_gap_mean"] += float(st.get("res_gap_mean", 0.0)) * float(n)
                                ss["entropy_mean"] += float(st.get("entropy_mean", 0.0)) * float(n)
                                sum_stats[tid] = ss
                    X_buf = []
                    gt_buf = []

                for bundle_path, suf in zip(bundles, suf_list):
                    for vid in _read_bundle_vids(bundle_path):
                        uid = normalize_video_id(vid)
                        tid = video_to_task.get(uid, None)
                        if tid is None or int(tid) not in seen_tasks:
                            continue
                        feat_file = f"{args.feature_path}{vid.split('.')[0]}{suf}.pt"
                        features = torch.load(feat_file)
                        features = features.transpose(1, 0)
                        features = features[:, :: int(args.feat_sample_rate)]
                        features = features[:, :: int(args.all_sample_rate)]
                        x_ct = torch.as_tensor(features, dtype=torch.float32)
                        x_tc = x_ct.transpose(0, 1).contiguous()  # (T, C)
                        X_buf.append(x_tc)
                        gt_buf.append(int(tid))
                        if len(X_buf) >= 16:
                            _flush()
                _flush()

                for tid in sorted(sum_n.keys()):
                    n = int(sum_n[tid])
                    if n <= 0:
                        continue
                    hs = sum_hits.get(tid, {})
                    ss = sum_stats.get(tid, {})
                    out_hits[int(tid)] = {
                        "top1_hit_rate": float(hs.get("top1", 0.0)) / float(n),
                        "topL_hit_rate": float(hs.get("topL", 0.0)) / float(n),
                        "topL": int(hs.get("topL_cfg", int(getattr(args, "vista_topL", 2)))),
                        "n_samples": int(n),
                        "true_task_prob_mean": float(hs.get("prob", 0.0)) / float(n),
                    }
                    out_stats[int(tid)] = {
                        "res_best_mean": float(ss.get("res_best_mean", 0.0)) / float(n),
                        "res_gap_mean": float(ss.get("res_gap_mean", 0.0)) / float(n),
                        "entropy_mean": float(ss.get("entropy_mean", 0.0)) / float(n),
                    }
                return out_stats, out_hits

            try:
                router_stats_val, router_hit_val = _router_eval_on_bundle(val_bundle_eval, vt_suf_val, split_name="val")
                router_stats_test, router_hit_test = _router_eval_on_bundle(test_bundle_eval, vt_suf_test, split_name="test")
            except Exception as e:
                raise RuntimeError(f"[TAS vista] router eval on val/test failed at task_index={t_idx} task_id={task_id}") from e

        _save_json(
            os.path.join(task_dir, "metrics_task_end.json"),
            {
                "task_index": int(t_idx),
                "task_id": int(task_id),
                "seen_tasks": [int(x) for x in task_order[:t_idx]],
                "train_time_sec": float(time.time() - start),
                "val_best": {
                    "best_epoch": int(best_epoch),
                    "best_avg_score_current_task": float(best_score),
                    "weights": {str(k): float(v) for k, v in weights_val.items()},
                    "metrics": {mk: {str(k): float(v) for k, v in mp.items()} for mk, mp in per_task_metrics_val.items()},
                },
                "test_at_val_best": {
                    "epoch": int(best_epoch),
                    "weights": {str(k): float(v) for k, v in weights_test.items()},
                    "metrics": {mk: {str(k): float(v) for k, v in mp.items()} for mk, mp in per_task_metrics_test.items()},
                },
                "vista": {
                    "enabled": bool(vista_state is not None),
                    "router_M": int(getattr(args, "vista_router_M", 1)),
                    "subspace_k": int(getattr(args, "vista_subspace_k", 32)),
                    "topL": int(getattr(args, "vista_topL", 2)),
                    "gamma": float(getattr(args, "vista_gamma", 10.0)),
                    "eps": float(getattr(args, "vista_eps", 1e-6)),
                    "apply_to_target": bool(getattr(args, "vista_apply_to_target", 1)),
                    "router_fit_max_samples": int(getattr(args, "vista_router_fit_max_samples", 0)),
                    "router_stats_val": {str(k): v for k, v in router_stats_val.items()},
                    "router_hit_val": {str(k): v for k, v in router_hit_val.items()},
                    "router_stats_test": {str(k): v for k, v in router_stats_test.items()},
                    "router_hit_test": {str(k): v for k, v in router_hit_test.items()},
                },
            },
        )

        # Save router eval separately for convenience
        if vista_state is not None and vista_state.router is not None:
            _save_json(
                os.path.join(task_dir, "router_eval_val.json"),
                {
                    "task_index": int(t_idx),
                    "task_id": int(task_id),
                    "seen_tasks": [int(x) for x in task_order[:t_idx]],
                    "router_stats_val": {str(k): v for k, v in router_stats_val.items()},
                    "router_hit_val": {str(k): v for k, v in router_hit_val.items()},
                },
            )
            _save_json(
                os.path.join(task_dir, "router_eval_test.json"),
                {
                    "task_index": int(t_idx),
                    "task_id": int(task_id),
                    "seen_tasks": [int(x) for x in task_order[:t_idx]],
                    "router_stats_test": {str(k): v for k, v in router_stats_test.items()},
                    "router_hit_test": {str(k): v for k, v in router_hit_test.items()},
                },
            )

        # Save router index (skill parity)
        if vista_state is not None and vista_state.router is not None:
            try:
                vista_state.router.save_index(output_dir=os.path.join(args.output_root, "router"))
            except Exception:
                pass

    # Final save
    rec_val.save(args.output_root)
    rec_test.save(os.path.join(args.output_root, "test_metrics"))
    save_unknown_ids(args.output_root, unknown)


if __name__ == "__main__":
    main()


