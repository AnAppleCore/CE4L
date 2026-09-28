from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Set

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Ensure repo root on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.config_utils import apply_config_as_defaults, inject_required_positionals_if_missing, load_yaml_config, split_argv_config
from clego_cl.task_map import load_video_to_task
from clego_cl.fair_data import filter_aap_list_by_tasks

import planning_main as base
from planning_dataset import TSNDataSet
from planning_models import VideoModel
from planning_eval_grouped import eval_grouped


# Match continual protocol (fixed task set)
PLANNING_ALLOWED_TASK_IDS = [1, 3, 4, 5]


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


def _micro_average(per_task: Dict[int, float], weights: Dict[int, float], allowed: Set[int]) -> float:
    num = 0.0
    den = 0.0
    for t in allowed:
        w = float(weights.get(int(t), 0.0))
        v = per_task.get(int(t), float("nan"))
        if w <= 0 or not math.isfinite(float(v)):
            continue
        num += float(v) * w
        den += w
    return float("nan") if den <= 0 else float(num / den)


def main() -> None:
    cfg_path, remaining = split_argv_config(sys.argv[1:])
    cfg = load_yaml_config(cfg_path) if cfg_path is not None else {}

    # planning_main uses a module-level `args` global; we set it after parsing.
    from plannnig_opts import parser as base_parser

    remaining = inject_required_positionals_if_missing(
        remaining,
        ["class_file", "modality", "train_source_list", "train_target_list", "val_list"],
        cfg,
    )

    p = argparse.ArgumentParser(parents=[base_parser], add_help=False)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--video_to_task_path", type=str, default=os.path.join(_REPO_ROOT, "video_to_task.npy"))
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    if cfg:
        apply_config_as_defaults(p, cfg)
    args = p.parse_args(remaining)

    if not args.output_root:
        raise ValueError("--output_root must be provided via CLI or --config")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    os.makedirs(args.output_root, exist_ok=True)
    _dump_args_json(args.output_root, args, extra={"entrypoint": "planning_joint_fair_main.py", "allowed_tasks": PLANNING_ALLOWED_TASK_IDS})

    # Fair lists (filter to allowed tasks)
    video_to_task = load_video_to_task(args.video_to_task_path)
    allowed_tasks = set(int(x) for x in PLANNING_ALLOWED_TASK_IDS)
    fair_dir = os.path.join(args.output_root, "fair_lists")
    os.makedirs(fair_dir, exist_ok=True)

    ts_fair = os.path.join(fair_dir, "train_source_list_fair.txt")
    tt_fair = os.path.join(fair_dir, "train_target_list_fair.txt")
    vl_fair = os.path.join(fair_dir, "val_list_fair.txt")
    st_ts = filter_aap_list_by_tasks(in_list=args.train_source_list, out_list=ts_fair, video_to_task=video_to_task, allowed_tasks=allowed_tasks)
    st_tt = filter_aap_list_by_tasks(in_list=args.train_target_list, out_list=tt_fair, video_to_task=video_to_task, allowed_tasks=allowed_tasks)
    st_vl = filter_aap_list_by_tasks(in_list=args.val_list, out_list=vl_fair, video_to_task=video_to_task, allowed_tasks=allowed_tasks)

    with open(os.path.join(fair_dir, "fair_list_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "allowed_tasks": sorted(list(allowed_tasks)),
                "train_source": {"in": os.path.abspath(args.train_source_list), "out": os.path.abspath(ts_fair), "stats": st_ts.__dict__},
                "train_target": {"in": os.path.abspath(args.train_target_list), "out": os.path.abspath(tt_fair), "stats": st_tt.__dict__},
                "val": {"in": os.path.abspath(args.val_list), "out": os.path.abspath(vl_fair), "stats": st_vl.__dict__},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Overwrite args lists to fair versions
    args.train_source_list = ts_fair
    args.train_target_list = tt_fair
    args.val_list = vl_fair

    # Make base module use our args
    base.args = args

    # Set exp paths to output_root (planning_main expects exp_path ending with '/')
    args.exp_path = args.output_root.rstrip("/") + "/"
    args.save_best_log = os.path.join(args.output_root, "best.log")

    if args.dry_run:
        return

    FUTURE_LENGTH = base.FUTURE_LENGTH
    ACTION_NUM_CLASSES = base.ACTION_NUM_CLASSES
    num_class = ACTION_NUM_CLASSES * FUTURE_LENGTH

    # Model
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

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("optimizer not supported")

    # class-balanced weights (same as planning_main/planning_continual_main)
    class_ids_list = [line.strip().split("|")[2] for line in open(args.train_source_list)]
    class_id_list = []
    for class_ids in class_ids_list:
        class_id_list.extend(eval(class_ids))
    class_id, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
    class_freq = (class_data_counts / class_data_counts.sum()).tolist()
    weight_source_class = torch.ones(ACTION_NUM_CLASSES).cuda()
    if args.weighted_class_loss == "Y":
        weight_source_class = 1 / torch.Tensor(class_freq).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weight_source_class, ignore_index=-1).cuda()
    weight_domain_loss = torch.Tensor([1, 1]).cuda()
    criterion_domain = torch.nn.CrossEntropyLoss(weight=weight_domain_loss).cuda()

    # loaders
    num_source = sum(1 for _ in open(args.train_source_list))
    num_target = sum(1 for _ in open(args.train_target_list))
    num_iter_source = num_source / args.batch_size[0]
    num_iter_target = num_target / args.batch_size[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter * args.batch_size[0]) if args.copy_list[0] == "Y" else num_source
    num_target_train = round(num_max_iter * args.batch_size[1]) if args.copy_list[1] == "Y" else num_target

    source_set = TSNDataSet(
        "",
        args.train_source_list,
        args.feat_path,
        num_dataload=num_source_train,
        num_segments=args.num_segments,
        new_length=1 if args.modality == "RGB" else 5,
        modality=args.modality,
        image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
        random_shift=False,
        test_mode=True,
        video_to_task=video_to_task,
    )
    target_set = TSNDataSet(
        "",
        args.train_target_list,
        args.feat_path,
        num_dataload=num_target_train,
        num_segments=args.num_segments,
        new_length=1 if args.modality == "RGB" else 5,
        modality=args.modality,
        image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
        random_shift=False,
        test_mode=True,
        video_to_task=video_to_task,
    )
    if len(source_set) == 0 or len(target_set) == 0:
        raise RuntimeError(f"[planning joint_fair] Empty training set after filtering: source={len(source_set)} target={len(target_set)}")

    source_loader = torch.utils.data.DataLoader(
        source_set,
        batch_size=args.batch_size[0],
        shuffle=False,
        sampler=torch.utils.data.sampler.RandomSampler(source_set),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )
    target_loader = torch.utils.data.DataLoader(
        target_set,
        batch_size=args.batch_size[1],
        shuffle=False,
        sampler=torch.utils.data.sampler.RandomSampler(target_set),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # val loader (filtered)
    num_val = sum(1 for _ in open(args.val_list))
    val_segments = args.val_segments if args.val_segments > 0 else args.num_segments
    val_set = TSNDataSet(
        "",
        args.val_list,
        args.feat_path,
        num_dataload=num_val,
        num_segments=val_segments,
        new_length=1 if args.modality == "RGB" else 5,
        modality=args.modality,
        image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
        random_shift=False,
        test_mode=True,
        video_to_task=video_to_task,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size[2],
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # train logs
    train_log_path = os.path.join(args.output_root, "train.log")
    train_short_log_path = os.path.join(args.output_root, "train_short.log")
    start = time.time()
    with open(train_log_path, "a", encoding="utf-8") as log_f, open(train_short_log_path, "a", encoding="utf-8") as log_short_f:
        for ep in range(1, int(args.epochs) + 1):
            base.train(
                num_class,
                source_loader,
                target_loader,
                model,
                criterion,
                criterion_domain,
                optimizer,
                int(ep),
                log=log_f,
                log_short=log_short_f,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                mu=args.mu,
            )

    # final eval on allowed tasks (same as CL final seen task set)
    per_task_metrics, weights = eval_grouped(val_loader=val_loader, model=model, args=args, video_to_task=video_to_task, seen_tasks=set(allowed_tasks))
    ed_map = per_task_metrics.get("ed_final", {})
    weights = {int(k): float(v) for k, v in weights.items()}
    ed_map = {int(k): float(v) for k, v in ed_map.items()}
    micro = _micro_average(ed_map, weights, allowed_tasks)

    with open(os.path.join(args.output_root, "joint_fair_eval.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "allowed_tasks": sorted(list(allowed_tasks)),
                "metric": "ed_final",
                "per_task": {str(k): float(v) for k, v in ed_map.items()},
                "weights": {str(k): float(v) for k, v in weights.items()},
                "micro_avg": float(micro) if math.isfinite(micro) else None,
                "train_time_sec": float(time.time() - start),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # For plot_aap_results baseline reader (single float)
    with open(args.save_best_log, "w", encoding="utf-8") as f:
        f.write(f"{micro}\n")


if __name__ == "__main__":
    main()


