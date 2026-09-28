import argparse
import copy
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Ensure repo root on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.config_utils import load_yaml_config, split_argv_config, apply_config_as_defaults, inject_required_positionals_if_missing
from clego_cl.task_map import load_video_to_task, normalize_video_id
from clego_cl.task_order import save_task_order
from clego_cl.continual_recorder import ContinualRecorder
from clego_cl.task_stats import UnknownTracker, save_task_stats, save_unknown_ids
from clego_cl.continual_algorithms import build_continual_algorithm
from clego_cl.continual_algorithms.lwf_generic import LwFGeneric, LwFGenericConfig
from clego_cl.vista import VISTAState, build_vista_router, vista_eval_router_grouped
from skill_benchmark.adapters import AdapterBank
from clego_cl.l2p import L2PPool

from anticipation_opts import parser as base_parser
from anticipation_dataset import TSNDataSet
from anticipation_models import VideoModel
from loss import *
from anticipation_eval_grouped import eval_grouped
import anticipation_main as base_main


ANTICIPATION_ALLOWED_TASK_IDS = [1, 2, 3, 4, 5, 6, 7, 8]


def _require_no_overrides_for_fixed_tasks(args: argparse.Namespace) -> None:
    # Strict protocol: fixed *task set* (and count), but order may be randomized if enabled.
    if getattr(args, "num_tasks", None) not in (None, len(ANTICIPATION_ALLOWED_TASK_IDS)):
        raise RuntimeError(f"AAP anticipation uses fixed num_tasks={len(ANTICIPATION_ALLOWED_TASK_IDS)}; got --num_tasks={args.num_tasks}")


def _require_task_coverage_strict(
    *,
    exp_name: str,
    task_order: List[int],
    video_to_task: Dict[str, int],
    train_source_list: str,
    train_target_list: str,
    val_list: str,
) -> None:
    """Strict academic protocol: every task must appear in train_source/train_target/val."""

    def _tasks_in_list(list_path: str) -> set[int]:
        out: set[int] = set()
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split("|")[0].strip()
                uid = normalize_video_id(vid)
                tid = video_to_task.get(uid, None)
                if tid is not None:
                    out.add(int(tid))
        return out

    cov = {
        "train_source": _tasks_in_list(train_source_list),
        "train_target": _tasks_in_list(train_target_list),
        "val": _tasks_in_list(val_list),
    }
    missing = {k: [t for t in task_order if int(t) not in v] for k, v in cov.items()}
    bad = {k: v for k, v in missing.items() if len(v) > 0}
    if bad:
        msg = [f"[AAP anticipation strict task coverage failed] exp={exp_name} task_order={task_order}"]
        for split_name, miss in bad.items():
            msg.append(f"- {split_name} missing tasks: {miss}")
        raise RuntimeError("\n".join(msg))


def _parse_list_video_ids(list_path: str) -> List[str]:
    vids = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vid = line.split("|")[0].strip()
            vids.append(normalize_video_id(vid))
    return vids


def _allowed_for_task(list_path: str, video_to_task: Dict[str, int], task_id: int, unknown: UnknownTracker) -> Set[str]:
    allowed = set()
    for vid in _parse_list_video_ids(list_path):
        tid = video_to_task.get(vid, None)
        if tid is None:
            unknown.add(vid)
            continue
        if int(tid) == int(task_id):
            allowed.add(vid)
    return allowed


def _counts_by_task(list_path: str, video_to_task: Dict[str, int], unknown: UnknownTracker) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for vid in _parse_list_video_ids(list_path):
        tid = video_to_task.get(vid, None)
        if tid is None:
            unknown.add(vid)
            continue
        counts[int(tid)] = counts.get(int(tid), 0) + 1
    return counts


def _merge_counts(*counts: Dict[int, int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for mp in counts:
        for k, v in (mp or {}).items():
            out[int(k)] = out.get(int(k), 0) + int(v)
    return out


def _ensure_nonempty_metrics(
    per_task_metrics: Dict[str, Dict[int, float]],
    weights: Dict[int, float],
) -> Tuple[Dict[str, Dict[int, float]], Dict[int, float]]:
    # Strict protocol: do not inject NaNs. If eval produces empty metrics, treat it as a configuration/data bug.
    if not per_task_metrics:
        raise RuntimeError("[AAP anticipation strict] eval_grouped returned empty per_task_metrics.")
    for mk, mp in per_task_metrics.items():
        if mp is None or len(mp) == 0:
            raise RuntimeError(f"[AAP anticipation strict] eval_grouped returned empty metric dict for metric={mk}.")
    if weights is None or len(weights) == 0:
        raise RuntimeError("[AAP anticipation strict] eval_grouped returned empty weights dict.")
    return (
        {mk: {int(k): float(v) for k, v in mp.items()} for mk, mp in per_task_metrics.items()},
        {int(k): float(v) for k, v in weights.items()},
    )


def _dump_args_json(output_root: str, args: argparse.Namespace, extra: Dict[str, object]) -> None:
    """Write a JSON snapshot of parsed args for reproducibility / hyperparam matching."""
    try:
        import json

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
        # best-effort only
        pass


def main() -> None:
    # extend base parser
    cfg_path, remaining = split_argv_config(sys.argv[1:])
    cfg = load_yaml_config(cfg_path) if cfg_path is not None else {}
    remaining = inject_required_positionals_if_missing(
        remaining,
        ["class_file", "modality", "train_source_list", "train_target_list", "val_list"],
        cfg,
    )

    p = argparse.ArgumentParser(parents=[base_parser], add_help=False)
    p.add_argument("--config", type=str, default=None, help="YAML config file (optional). CLI args override config.")
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
    p.add_argument("--max_train_samples_per_task", type=int, default=None, help="Optional cap on train samples per task (sanity/fast runs).")
    p.add_argument("--max_val_samples", type=int, default=None, help="Optional cap on val samples (sanity/fast runs).")
    # Continual algorithm (pluggable)
    p.add_argument(
        "--continual_algorithm",
        type=str,
        default="none",
        help="Continual algorithm to add on top of baseline. Supported: none | er | derpp | ewc | lwf",
    )
    p.add_argument("--continual_algorithm_buffer_ratio", type=float, default=0.2, help="ER buffer size as ratio of total train samples (default: 0.2)")
    p.add_argument("--continual_algorithm_replay_batch_ratio", type=float, default=0.2, help="ER replay batch size as ratio of current batch (default: 0.2)")
    p.add_argument("--continual_algorithm_distill_alpha", type=float, default=0.5, help="DER++ distillation weight (default: 0.5)")
    # EWC/LwF hyperparams
    p.add_argument("--ewc_lambda", type=float, default=1e-2, help="EWC regularization weight (default: 1e-2)")
    p.add_argument("--ewc_gamma", type=float, default=1.0, help="EWC online fisher decay (default: 1.0)")
    p.add_argument("--ewc_fisher_batches", type=int, default=50, help="EWC fisher batches per task (default: 50)")
    p.add_argument("--lwf_alpha", type=float, default=0.5, help="LwF distillation weight (default: 0.5)")
    # ------------------------------
    # L2P: fixed adapter pool (adapters + learnable keys)
    # ------------------------------
    p.add_argument("--l2p_enabled", action="store_true", default=False, help="Enable L2P (fixed adapter pool + key-query selection).")
    p.add_argument("--l2p_pool_size", type=int, default=4, help="L2P: prompt pool size (default: 4).")
    p.add_argument("--l2p_topK", type=int, default=2, help="L2P: top-K adapters selected per sample (default: 2).")
    p.add_argument("--l2p_router_M", type=int, default=1, help="L2P: time-chunk pooling segments M for query representation (default: 1).")
    p.add_argument("--l2p_adapter_bottleneck", type=int, default=64, help="L2P: adapter bottleneck dim r (default: 64).")
    p.add_argument("--l2p_sim_lambda", type=float, default=0.5, help="L2P: similarity loss weight (default fixed: 0.5).")
    p.add_argument("--l2p_diversed_selection", action="store_true", default=True, help="L2P: enable frequency-based diversified selection during training.")
    p.add_argument("--l2p_batchwise_selection", action="store_true", default=False, help="L2P: batch-wise top-K selection (optional).")
    if cfg:
        apply_config_as_defaults(p, cfg)
    args = p.parse_args(remaining)
    if getattr(args, "no_randomize_order", False):
        args.randomize_order = False
    if not args.output_root:
        raise ValueError("--output_root must be provided via CLI or --config")
    _require_no_overrides_for_fixed_tasks(args)

    # The original `anticipation_main.py` uses a module-level global `args` in train()/validate().
    # We call those functions directly here, so we must set it.
    import anticipation_main as base_main
    base_main.args = args

    os.makedirs(args.output_root, exist_ok=True)
    _dump_args_json(args.output_root, args, extra={"entrypoint": "anticipation_continual_main.py"})
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    unknown = UnknownTracker(max_examples=200)
    video_to_task = load_video_to_task(args.video_to_task_path)
    # Fixed task set for anticipation (8 tasks)
    train_counts_source = _counts_by_task(args.train_source_list, video_to_task, unknown)
    train_counts_target = _counts_by_task(args.train_target_list, video_to_task, unknown)
    train_counts_all = _merge_counts(train_counts_source, train_counts_target)
    tasks_present = sorted([int(t) for t, n in train_counts_all.items() if int(n) > 0])
    task_order = [int(x) for x in ANTICIPATION_ALLOWED_TASK_IDS]
    if getattr(args, "randomize_order", False):
        rng = np.random.RandomState(int(args.seed))
        rng.shuffle(task_order)
    save_task_order(os.path.join(args.output_root, "task_order.json"), task_order)
    _require_task_coverage_strict(
        exp_name=os.path.basename(os.path.normpath(args.output_root)),
        task_order=task_order,
        video_to_task=video_to_task,
        train_source_list=args.train_source_list,
        train_target_list=args.train_target_list,
        val_list=args.val_list,
    )

    stats = {
        "benchmark": "action_anticipation_planning_benchmark/anticipation",
        "task_order": task_order,
        "effective_num_tasks": int(len(task_order)),
        "tasks_present_in_train": [int(x) for x in tasks_present],
        "micro_weight_unit": "num_samples_in_eval_split",
        "splits": {
            "train_source": {str(k): int(v) for k, v in train_counts_source.items()},
            "train_target": {str(k): int(v) for k, v in train_counts_target.items()},
            "val": {str(k): int(v) for k, v in _counts_by_task(args.val_list, video_to_task, unknown).items()},
        },
    }
    total_seen = sum(len(_parse_list_video_ids(x)) for x in [args.train_source_list, args.train_target_list, args.val_list])
    stats.update(unknown.to_dict())
    stats["unknown_ratio"] = float(unknown.count) / float(total_seen) if total_seen > 0 else 0.0
    save_task_stats(args.output_root, stats)
    save_unknown_ids(args.output_root, unknown)

    if args.dry_run:
        return

    num_class = args.num_classes
    model = VideoModel(
        num_class,
        args.baseline_type,
        args.frame_aggregation,
        args.modality,
        train_segments=args.num_segments,
        val_segments=args.val_segments,
        base_model=args.arch,
        path_pretrained=args.pretrained,
        add_fc=args.add_fc,
        fc_dim=args.fc_dim,
        dropout_i=args.dropout_i,
        dropout_v=args.dropout_v,
        partial_bn=not args.no_partialbn,
        use_bn=args.use_bn if args.use_target != "none" else "none",
        ens_DA=args.ens_DA if args.use_target != "none" else "none",
        n_rnn=args.n_rnn,
        rnn_cell=args.rnn_cell,
        n_directions=args.n_directions,
        n_ts=args.n_ts,
        use_attn=args.use_attn,
        n_attn=args.n_attn,
        use_attn_frame=args.use_attn_frame,
        verbose=args.verbose,
        share_params=args.share_params,
    )
    model = torch.nn.DataParallel(model, args.gpus).cuda()

    # ------------------------------
    # EMA copy (main model only)
    # ------------------------------
    ema_enabled = bool(getattr(args, "vista_ema_enabled", False))
    ema_decay = float(getattr(args, "vista_ema_decay", 0.999))
    ema_model = None
    if ema_enabled:
        if not (0.0 < float(ema_decay) < 1.0):
            raise ValueError(f"--vista_ema_decay must be in (0,1), got {ema_decay}")
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    # Inject into baseline module (train()/eval read module globals).
    base_main.ema_enabled = bool(ema_enabled)
    base_main.ema_decay = float(ema_decay)
    base_main.ema_model = ema_model

    # ------------------------------
    # VISTA setup (feature space)
    # ------------------------------
    vista_enabled = bool(getattr(args, "vista_enabled", False))
    vista_state = None
    base_main.vista_adapter_optimizer = None
    if vista_enabled:
        embed_dim = int(getattr(getattr(model, "module", model), "feature_dim", 768))
        adapter_bank = AdapterBank(
            input_dim=embed_dim,
            bottleneck=int(getattr(args, "vista_adapter_bottleneck", 64)),
            use_layernorm=True,
        ).cuda()
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
            apply_to_target=bool(getattr(args, "vista_apply_to_target", True)),
            train_backbone_after_task1=bool(getattr(args, "vista_train_backbone_after_task1", False)),
        )
        base_main.vista_enabled = True
        base_main.vista_state = vista_state
        base_main.vista_mode = "none"
    else:
        base_main.vista_enabled = False
        base_main.vista_state = None
        base_main.vista_mode = "none"

    # ------------------------------
    # L2P setup (feature space)
    # ------------------------------
    l2p_pool = None
    if bool(getattr(args, "l2p_enabled", False)):
        embed_dim = int(getattr(getattr(model, "module", model), "feature_dim", 768))
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
        ).cuda()
        base_main.l2p_enabled = True
        base_main.l2p_pool = l2p_pool
        base_main.l2p_topk = int(getattr(args, "l2p_topK", 2))
        base_main.l2p_router_M = int(getattr(args, "l2p_router_M", 1))
        base_main.l2p_sim_lambda = float(getattr(args, "l2p_sim_lambda", 0.5))
        base_main.l2p_diversed_selection = bool(getattr(args, "l2p_diversed_selection", True))
        base_main.l2p_batchwise_selection = bool(getattr(args, "l2p_batchwise_selection", False))
        base_main.l2p_mode = "train"
        base_main.l2p_optimizer = torch.optim.Adam(l2p_pool.parameters(), lr=float(getattr(args, "lr", 1e-4)))
    else:
        base_main.l2p_enabled = False
        base_main.l2p_pool = None
        base_main.l2p_mode = "none"
        base_main.l2p_optimizer = None

    def _vista_fit_router_for_task(*, task_id: int, loader) -> None:
        if vista_state is None or vista_state.router is None:
            return
        max_samples = int(getattr(args, "vista_router_fit_max_samples", 0))

        class _PairLoader:
            def __iter__(self_inner):
                n_samples = 0
                for data_batch in loader:
                    x = data_batch[0] if isinstance(data_batch, (tuple, list)) else data_batch
                    if max_samples > 0:
                        remaining = int(max_samples - n_samples)
                        if remaining <= 0:
                            break
                        x = x[:remaining]
                    if x.numel() == 0:
                        continue
                    yield x, x
                    n_samples += int(x.shape[0])
                    if max_samples > 0 and n_samples >= max_samples:
                        break

        vista_state.router.fit_from_loader(task_id=int(task_id), loader=_PairLoader(), device="cpu", verbose=False)

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("optimizer not supported")

    # losses
    class_ids_list = [line.strip().split("|")[3] for line in open(args.train_source_list)]
    class_id_list = []
    for class_ids in class_ids_list:
        class_id_list.extend(eval(class_ids))
    class_id, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
    class_freq = (class_data_counts / class_data_counts.sum()).tolist()
    weight_source_class = torch.ones(num_class).cuda()
    if args.weighted_class_loss == "Y":
        weight_source_class = (1 / torch.Tensor(class_freq) ** args.class_weight_scale).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weight_source_class).cuda()
    weight_domain_loss = torch.Tensor([1, 1]).cuda()
    criterion_domain = torch.nn.CrossEntropyLoss(weight=weight_domain_loss).cuda()

    # Build a single val loader (not filtered; we group by task in eval)
    num_val = sum(1 for _ in open(args.val_list))
    if args.max_val_samples is not None:
        num_val = min(int(num_val), int(args.max_val_samples))
    val_segments = args.val_segments if args.val_segments > 0 else args.num_segments
    val_set = TSNDataSet(
        "",
        args.val_list,
        args.feat_path,
        num_dataload=num_val,
        num_segments=val_segments,
        num_classes=num_class,
        new_length=1 if args.modality == "RGB" else 5,
        modality=args.modality,
        image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
        random_shift=False,
        test_mode=True,
        video_to_task=video_to_task,
        unknown_tracker=unknown,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size[2],
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )

    rec = ContinualRecorder(task_order=task_order)

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
            distill_alpha=getattr(args, "continual_algorithm_distill_alpha", 0.5),
            ewc_lambda=getattr(args, "ewc_lambda", 1e-2),
            ewc_gamma=getattr(args, "ewc_gamma", 1.0),
            ewc_fisher_batches=getattr(args, "ewc_fisher_batches", 50),
            lwf_alpha=getattr(args, "lwf_alpha", 0.5),
            seed=int(getattr(args, "seed", 42)),
        )
    if algo is not None:
        # Disallow VISTA/L2P combinations with EWC/LwF (per protocol).
        algo_name = str(getattr(algo, "name", "")).strip().lower()
        if algo_name in ("ewc", "lwf") and (
            bool(getattr(args, "vista_enabled", False)) or bool(getattr(args, "l2p_enabled", False))
        ):
            raise ValueError("EWC/LwF are not supported with VISTA or L2P in action_anticipation_planning_benchmark (disable vista_enabled/l2p_enabled).")
        if hasattr(algo, "configure_total_capacity"):
            total_train = 0
            for tid in task_order:
                total_train += int(train_counts_source.get(int(tid), 0))
                total_train += int(train_counts_target.get(int(tid), 0))
            algo.configure_total_capacity(total_train_samples=int(total_train))
        if hasattr(algo, "bind_models"):
            algo.bind_models(models={"main": model})
        # Inject into the underlying baseline module (train() reads module globals).
        base_main.continual_algo = algo

    # A0: evaluate before training
    base_main.vista_mode = "infer"
    if l2p_pool is not None:
        base_main.l2p_mode = "none"
    per_task_metrics_all, weights_all = eval_grouped(
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        num_class=num_class,
        args=args,
        video_to_task=video_to_task,
        seen_tasks=set(task_order),
        ema_model=ema_model,
    )
    per_task_metrics_all, weights_all = _ensure_nonempty_metrics(per_task_metrics_all, weights_all)
    for metric_key, mp in per_task_metrics_all.items():
        rec.set_A0(metric_key, mp, weights_all)

    epoch_global = 0  # for bookkeeping only
    # For lr_adaptive == 'loss' we need to track classification loss across epochs (best-effort).
    loss_c_current = 999.0
    loss_c_previous = 999.0

    for t_idx, task_id in enumerate(task_order, start=1):
        task_dir = os.path.join(args.output_root, f"task_{t_idx:02d}")
        os.makedirs(os.path.join(task_dir, "checkpoints"), exist_ok=True)
        # Make standard code save artifacts inside this task directory.
        args.exp_path = task_dir + "/"
        args.save_best_log = os.path.join(task_dir, "best.log")

        # Build task-specific loaders
        # IMPORTANT: source/target lists can be different sets of videos. Filter them independently.
        allowed_source = _allowed_for_task(args.train_source_list, video_to_task, task_id, unknown)
        allowed_target = _allowed_for_task(args.train_target_list, video_to_task, task_id, unknown)

        # Use per-task record counts (not full-list counts), otherwise tiny tasks get massively over-repeated.
        num_source = int(train_counts_source.get(int(task_id), 0))
        num_target = int(train_counts_target.get(int(task_id), 0))
        if num_source <= 0 or num_target <= 0:
            # Can't train because RandomSampler requires positive num_samples and train() zips loaders.
            # Still run evaluation and record metrics.
            seen = set(task_order[:t_idx])
            base_main.vista_mode = "infer"
            per_task_metrics, weights = eval_grouped(
                val_loader=val_loader,
                model=model,
                criterion=criterion,
                num_class=num_class,
                args=args,
                video_to_task=video_to_task,
                seen_tasks=seen,
                ema_model=ema_model,
            )
            per_task_metrics, weights = _ensure_nonempty_metrics(per_task_metrics, weights)
            rec.update_after_task(t_idx=t_idx, metrics=per_task_metrics, weights=weights)
            rec.save(args.output_root)
            continue

        num_iter_source = num_source / args.batch_size[0]
        num_iter_target = num_target / args.batch_size[1]
        num_max_iter = max(num_iter_source, num_iter_target)
        num_source_train = round(num_max_iter * args.batch_size[0]) if args.copy_list[0] == "Y" else num_source
        num_target_train = round(num_max_iter * args.batch_size[1]) if args.copy_list[1] == "Y" else num_target
        if args.max_train_samples_per_task is not None:
            num_source_train = min(int(num_source_train), int(args.max_train_samples_per_task))
            num_target_train = min(int(num_target_train), int(args.max_train_samples_per_task))

        source_set = TSNDataSet(
            "",
            args.train_source_list,
            args.feat_path,
            num_dataload=num_source_train,
            num_segments=args.num_segments,
            num_classes=num_class,
            new_length=1 if args.modality == "RGB" else 5,
            modality=args.modality,
            image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
            random_shift=False,
            test_mode=True,
            allowed_video_ids=allowed_source,
            video_to_task=video_to_task,
            unknown_tracker=unknown,
        )
        target_set = TSNDataSet(
            "",
            args.train_target_list,
            args.feat_path,
            num_dataload=num_target_train,
            num_segments=args.num_segments,
            num_classes=num_class,
            new_length=1 if args.modality == "RGB" else 5,
            modality=args.modality,
            image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
            random_shift=False,
            test_mode=True,
            allowed_video_ids=allowed_target,
            video_to_task=video_to_task,
            unknown_tracker=unknown,
        )

        source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
        target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
        source_loader = torch.utils.data.DataLoader(
            source_set,
            batch_size=args.batch_size[0],
            shuffle=False,
            sampler=source_sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True,
        )
        target_loader = torch.utils.data.DataLoader(
            target_set,
            batch_size=args.batch_size[1],
            shuffle=False,
            sampler=target_sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True,
        )

        # ---- VISTA: create current task adapter (hot-start from previous) ----
        if vista_state is not None and vista_state.adapter_bank is not None:
            init_from = None
            if t_idx >= 2:
                init_from = int(task_order[t_idx - 2])
            vista_state.adapter_bank.add_task(int(task_id), init_from_task=init_from)
            vista_state.adapter_bank.set_current_task(int(task_id))
            vista_state.adapter_bank.freeze_all_except(int(task_id))
            base_main.vista_adapter_optimizer = torch.optim.Adam(
                [p for p in vista_state.adapter_bank.get(int(task_id)).parameters() if p.requires_grad],
                lr=float(getattr(args, "lr", 1e-4)),
            )
            base_main.vista_mode = "train"

        if l2p_pool is not None:
            base_main.l2p_mode = "train"

        # ---- VISTA: optionally freeze backbone after task 1 ----
        if vista_state is not None and int(t_idx) >= 2:
            freeze = not bool(getattr(args, "vista_train_backbone_after_task1", False))
            if freeze:
                for p in model.parameters():
                    p.requires_grad = False

        # Train epochs for this task (no reset optimizer/model)
        from anticipation_main import train as train_one_epoch
        from anticipation_main import adjust_learning_rate, adjust_learning_rate_loss

        start = time.time()
        train_log_path = os.path.join(task_dir, "train.log")
        train_short_log_path = os.path.join(task_dir, "train_short.log")
        with open(train_log_path, "a", encoding="utf-8") as log_f, open(train_short_log_path, "a", encoding="utf-8") as log_short_f:
            # Make per-task epoch semantics match original code's `for epoch in range(..., args.epochs+1)`.
            args.epochs = int(args.epochs_per_task)
            for ep_local in range(int(args.epochs_per_task)):
                epoch_global += 1

                # ---- Match original `anticipation_main.main()` scheduling as closely as possible ----
                epoch_1based = int(ep_local) + 1
                # alpha schedule (if args.alpha < 0 in config)
                if float(args.alpha) < 0:
                    # same as original: alpha = 2/(1+exp(-epoch/args.epochs)) - 1
                    import math

                    alpha = 2.0 / (1.0 + math.exp(-1.0 * (epoch_1based) / float(max(args.epochs, 1)))) - 1.0
                else:
                    alpha = float(args.alpha)

                # LR schedule (epoch-level)
                if getattr(args, "lr_adaptive", "none") == "loss":
                    adjust_learning_rate_loss(optimizer, args.lr_decay, loss_c_current, loss_c_previous, ">")
                elif getattr(args, "lr_adaptive", "none") == "none" and hasattr(args, "lr_steps") and (epoch_1based in list(args.lr_steps)):
                    adjust_learning_rate(optimizer, args.lr_decay)

                base_main.vista_mode = "train"
                loss_ret = train_one_epoch(
                    num_class,
                    source_loader,
                    target_loader,
                    model,
                    criterion,
                    criterion_domain,
                    optimizer,
                    epoch_1based,
                    log=log_f,
                    log_short=log_short_f,
                    alpha=alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                    mu=args.mu,
                )
                # `train()` returns (loss_c.avg, attn_epoch_source.mean(0), attn_epoch_target.mean(0))
                if isinstance(loss_ret, (tuple, list)) and len(loss_ret) >= 1:
                    try:
                        loss_c_previous = float(loss_c_current)
                        loss_c_current = float(loss_ret[0])
                    except Exception:
                        pass
                # NOTE: We intentionally do not reset optimizer/model between tasks.

        # Save checkpoint
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch_global}, os.path.join(task_dir, "checkpoints", "task_end.pth"))

        # ---- EWC: estimate Fisher at task end ----
        if algo is not None and str(getattr(algo, "name", "")).strip().lower() == "ewc":
            # Build a non-duplicated loader for Fisher estimation (source stream only).
            mem_source_set = TSNDataSet(
                "",
                args.train_source_list,
                args.feat_path,
                num_dataload=int(num_source),
                num_segments=args.num_segments,
                num_classes=num_class,
                new_length=1 if args.modality == "RGB" else 5,
                modality=args.modality,
                image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                random_shift=False,
                test_mode=True,
                allowed_video_ids=allowed_source,
                video_to_task=video_to_task,
                unknown_tracker=unknown,
            )
            if len(mem_source_set) <= 0:
                raise RuntimeError(f"[EWC strict] Empty mem_source_set for task_id={task_id} (num_source={num_source})")
            fisher_loader = torch.utils.data.DataLoader(
                mem_source_set,
                batch_size=args.batch_size[0],
                shuffle=False,
                sampler=torch.utils.data.sampler.RandomSampler(mem_source_set),
                num_workers=args.workers,
                pin_memory=True,
                persistent_workers=True,
            )

            def _ewc_loss_from_batch(batch_in):
                x, y = batch_in
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                dummy_tgt = torch.zeros_like(x)
                beta = getattr(args, "beta", [0.0])
                mu = getattr(args, "mu", 0.0)
                _, out_s, _out_s2, _pd_s, _f_s, _att_t, _out_t, _out_t2, _pd_t, _f_t = model(
                    x, dummy_tgt, beta, mu, is_train=False, reverse=False
                )
                if args.baseline_type == "frame":
                    y = y.unsqueeze(1).repeat(1, args.num_segments).view(-1)
                return criterion(out_s, y)

            max_batches = int(getattr(args, "ewc_fisher_batches", 0))
            max_batches = max_batches if max_batches > 0 else None
            algo.update_fisher_from_loader(loader=fisher_loader, loss_fn=_ewc_loss_from_batch, max_batches=max_batches)

        # ---- LwF: update teacher snapshot at task end ----
        if algo is not None and str(getattr(algo, "name", "")).strip().lower() == "lwf":
            if hasattr(algo, "update_teacher"):
                algo.update_teacher(model)

        # ---- VISTA: fit router and save adapters/router at task end ----
        if vista_state is not None:
            _vista_fit_router_for_task(task_id=int(task_id), loader=source_loader)
            vista_state.adapter_bank.save(os.path.join(task_dir, "adapters"))
            vista_state.router.save_task(output_dir=os.path.join(task_dir, "router"), task_id=int(task_id))
            vista_state.router.save_task(output_dir=os.path.join(args.output_root, "router"), task_id=int(task_id))

        # Eval grouped on seen tasks
        seen = set(task_order[:t_idx])
        base_main.vista_mode = "infer"
        if l2p_pool is not None:
            base_main.l2p_mode = "infer"
        per_task_metrics, weights = eval_grouped(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            num_class=num_class,
            args=args,
            video_to_task=video_to_task,
            seen_tasks=seen,
            ema_model=ema_model,
        )
        per_task_metrics, weights = _ensure_nonempty_metrics(per_task_metrics, weights)
        rec.update_after_task(t_idx=t_idx, metrics=per_task_metrics, weights=weights)
        rec.save(args.output_root)

        # ---- VISTA: evaluate router hit rates on VAL (seen tasks) ----
        router_stats_val: Dict[int, Dict[str, float]] = {}
        router_hit_val: Dict[int, Dict[str, float]] = {}
        if vista_state is not None and vista_state.router is not None and bool(getattr(vista_state, "enabled", False)):
            try:
                # Grouped by GT task from val_loader.dataset.video_list (same semantics as eval_grouped).
                ds = val_loader.dataset
                global_index = 0
                # Accumulate sums to produce task-weighted means (per sample).
                sum_stats: Dict[int, Dict[str, float]] = {}
                sum_hits: Dict[int, Dict[str, float]] = {}
                sum_n: Dict[int, int] = {}

                for val_data, _val_label in val_loader:
                    batch_val_ori = int(val_data.size(0))
                    # task ids for *original* samples only (ignore any dummy padding logic used in eval_grouped)
                    gt_ids: List[int] = []
                    keep_idx: List[int] = []
                    for j in range(batch_val_ori):
                        try:
                            rec_item = ds.video_list[global_index + j]
                            uid = normalize_video_id(rec_item.path)
                            tid = video_to_task.get(uid, None)
                        except Exception:
                            tid = None
                        if tid is None:
                            continue
                        if int(tid) not in seen:
                            continue
                        gt_ids.append(int(tid))
                        keep_idx.append(int(j))
                    global_index += batch_val_ori
                    if len(keep_idx) == 0:
                        continue
                    x = val_data[keep_idx].cuda(non_blocking=True)
                    gt = torch.tensor(gt_ids, device=x.device, dtype=torch.long)
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
                        # hit counts as sums
                        acc = sum_hits.get(tid, {"top1": 0.0, "topL": 0.0, "prob": 0.0, "topL_cfg": int(hh.get("topL", 1))})
                        acc["top1"] += float(hh.get("top1_hit_rate", 0.0)) * float(n)
                        acc["topL"] += float(hh.get("topL_hit_rate", 0.0)) * float(n)
                        acc["prob"] += float(hh.get("true_task_prob_mean", 0.0)) * float(n)
                        acc["topL_cfg"] = int(hh.get("topL", acc["topL_cfg"]))
                        sum_hits[tid] = acc
                        # stats sums
                        st = b_stats.get(tid, {}) or {}
                        ss = sum_stats.get(tid, {"res_best_mean": 0.0, "res_gap_mean": 0.0, "entropy_mean": 0.0})
                        ss["res_best_mean"] += float(st.get("res_best_mean", 0.0)) * float(n)
                        ss["res_gap_mean"] += float(st.get("res_gap_mean", 0.0)) * float(n)
                        ss["entropy_mean"] += float(st.get("entropy_mean", 0.0)) * float(n)
                        sum_stats[tid] = ss

                for tid in sorted(sum_n.keys()):
                    n = int(sum_n[tid])
                    if n <= 0:
                        continue
                    hs = sum_hits.get(tid, {})
                    ss = sum_stats.get(tid, {})
                    router_hit_val[int(tid)] = {
                        "top1_hit_rate": float(hs.get("top1", 0.0)) / float(n),
                        "topL_hit_rate": float(hs.get("topL", 0.0)) / float(n),
                        "topL": int(hs.get("topL_cfg", int(getattr(args, "vista_topL", 2)))),
                        "n_samples": int(n),
                        "true_task_prob_mean": float(hs.get("prob", 0.0)) / float(n),
                    }
                    router_stats_val[int(tid)] = {
                        "res_best_mean": float(ss.get("res_best_mean", 0.0)) / float(n),
                        "res_gap_mean": float(ss.get("res_gap_mean", 0.0)) / float(n),
                        "entropy_mean": float(ss.get("entropy_mean", 0.0)) / float(n),
                    }
            except Exception as e:
                raise RuntimeError(f"[AAP anticipation vista] router eval on val failed at task_index={t_idx} task_id={task_id}") from e

        with open(os.path.join(task_dir, "metrics_task_end.json"), "w", encoding="utf-8") as f:
            import json

            json.dump(
                {
                    "task_index": t_idx,
                    "task_id": int(task_id),
                    "seen_tasks": [int(x) for x in task_order[:t_idx]],
                    "train_time_sec": float(time.time() - start),
                    "weights": {str(k): float(v) for k, v in weights.items()},
                    "metrics": {mk: {str(k): float(v) for k, v in mp.items()} for mk, mp in per_task_metrics.items()},
                    "vista": {
                        "enabled": bool(vista_state is not None),
                        "router_M": int(getattr(args, "vista_router_M", 1)),
                        "subspace_k": int(getattr(args, "vista_subspace_k", 32)),
                        "topL": int(getattr(args, "vista_topL", 2)),
                        "gamma": float(getattr(args, "vista_gamma", 10.0)),
                        "eps": float(getattr(args, "vista_eps", 1e-6)),
                        "apply_to_target": bool(getattr(args, "vista_apply_to_target", True)),
                        "router_fit_max_samples": int(getattr(args, "vista_router_fit_max_samples", 0)),
                        "router_stats_val": {str(k): v for k, v in router_stats_val.items()},
                        "router_hit_val": {str(k): v for k, v in router_hit_val.items()},
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save router eval separately for convenience (val only in AAP continual).
        if vista_state is not None and vista_state.router is not None:
            try:
                import json

                with open(os.path.join(task_dir, "router_eval_val.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task_index": int(t_idx),
                            "task_id": int(task_id),
                            "seen_tasks": [int(x) for x in task_order[:t_idx]],
                            "router_stats_val": {str(k): v for k, v in router_stats_val.items()},
                            "router_hit_val": {str(k): v for k, v in router_hit_val.items()},
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            except Exception as e:
                raise RuntimeError(f"[AAP anticipation vista] failed to save router_eval_val.json at task_index={t_idx}") from e

        # Save router index (skill parity)
        if vista_state is not None and vista_state.router is not None:
            try:
                vista_state.router.save_index(output_dir=os.path.join(args.output_root, "router"))
            except Exception:
                # router_index is a convenience artifact; do not fail training if unavailable
                pass

        # ---- ER/DER++ memory update at task end (strict, task-balanced) ----
        if algo is not None and hasattr(algo, "capacity") and int(getattr(algo, "capacity", 0)) > 0:
            # Build a non-duplicated loader for memory sampling (avoid num_dataload expansion bias).
            mem_source_set = TSNDataSet(
                "",
                args.train_source_list,
                args.feat_path,
                num_dataload=int(num_source),
                num_segments=args.num_segments,
                num_classes=num_class,
                new_length=1 if args.modality == "RGB" else 5,
                modality=args.modality,
                image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                random_shift=False,
                test_mode=True,
                allowed_video_ids=allowed_source,
                video_to_task=video_to_task,
                unknown_tracker=unknown,
            )
            if len(mem_source_set) <= 0:
                raise RuntimeError(f"[ER strict] Empty mem_source_set for task_id={task_id} (num_source={num_source})")
            mem_loader = torch.utils.data.DataLoader(
                mem_source_set,
                batch_size=args.batch_size[0],
                shuffle=False,
                sampler=torch.utils.data.sampler.RandomSampler(mem_source_set),
                num_workers=args.workers,
                pin_memory=True,
                persistent_workers=True,
            )
            # DER++ requires storing distillation targets (logits) in memory.
            if getattr(algo, "name", "") == "derpp":
                # distill target: source logits (same semantics as classification on source stream)
                def _distill_target_fn(batch_in, model_obj):
                    # batch_in is (data, label)
                    x = batch_in[0]
                    # AAP model forward requires both source and target streams; target stream is unused when DA is disabled.
                    dummy_tgt = torch.zeros_like(x)
                    beta = getattr(args, "beta", [0.0])
                    mu = getattr(args, "mu", 0.0)
                    _, out_s, _, _, _, _, _, _, _, _ = model_obj(x, dummy_tgt, beta, mu, is_train=False, reverse=False)
                    return out_s

                algo.update_memory_from_loader(task_id=int(task_id), loader=mem_loader, model=model, distill_target_fn=_distill_target_fn)
            else:
                algo.update_memory_from_loader(task_id=int(task_id), loader=mem_loader)

    rec.save(args.output_root)
    save_unknown_ids(args.output_root, unknown)


if __name__ == "__main__":
    main()


