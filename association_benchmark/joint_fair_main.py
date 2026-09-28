from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from typing import Dict, List, Set

import numpy as np
import torch
import torch.amp as amp
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from omegaconf import OmegaConf

# Ensure repo root on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from function.config import get_config
from function.logger import get_logger
from function import distributed as dist_utils
from function.utils import build_optimizer, build_scheduler, resume_checkpoint
from function.utils import build_val_transform  # noqa: F401 (kept for style parity)
from models.builder import build_model
from models import model_utils
from models.tokenizer import generate_tokenizer
from clego_cl.fair_data import filter_mcq_json_by_tasks

# Reuse training loop from original main
from main import train_one_epoch, get_args_parser as get_base_args_parser  # noqa

# Reuse eval aggregation and loaders from continual_main (keeps CL metric definition)
from continual_main import _load_video_to_task_map, _build_train_loader_for_task, _build_val_loader, evaluate_mcq_grouped_by_task  # noqa

from function.random import random_seed as set_random_seed


def get_args_parser():
    return get_base_args_parser()


def _derive_task_subset(video_to_task: Dict[str, int], num_tasks: int) -> List[int]:
    all_task_ids_sorted = sorted(set(int(v) for v in video_to_task.values()))
    return list(all_task_ids_sorted[: int(num_tasks)])


def main_worker(args):
    cfg = get_config(args)
    os.makedirs(cfg.output, exist_ok=True)

    dist_utils.init_distributed_mode(args)
    logger = get_logger(cfg)

    if dist_utils.get_rank() == 0:
        path = os.path.join(cfg.output, "config.yml")
        OmegaConf.save(cfg, path)
        logger.info(f"Full config save to {path}")
    logger.info(OmegaConf.to_yaml(cfg))

    set_random_seed(getattr(cfg.train, "seed", 42), dist_utils.get_rank())
    cudnn.deterministic = True
    cudnn.benchmark = False

    if args.distributed:
        dist.barrier()

    # Task subset definition (match continual)
    continual_cfg = getattr(cfg, "continual", None)
    assert continual_cfg is not None and getattr(continual_cfg, "enabled", False), (
        "joint_fair_main requires cfg.continual.enabled: true (we reuse it only for task subset definition)."
    )
    video_to_task_path = getattr(continual_cfg, "video_to_task_path", None)
    assert video_to_task_path is not None, "cfg.continual.video_to_task_path must be provided"
    video_to_task = _load_video_to_task_map(video_to_task_path)
    num_tasks = int(getattr(continual_cfg, "num_tasks", 5))
    task_subset = _derive_task_subset(video_to_task, num_tasks=num_tasks)
    task_subset_set: Set[int] = set(int(x) for x in task_subset)

    if dist_utils.is_main_process():
        logger.info(f"[joint_fair] Using task subset (first {num_tasks} tasks): {task_subset}")

    # Optional: filter MCQ json to only allowed tasks (saves compute and makes metrics explicit)
    # We patch cfg.test.ourdata.metadata path to the filtered version.
    try:
        meta_in = str(cfg.test.ourdata.metadata)
        meta_out = os.path.join(cfg.output, "fair_eval", os.path.basename(meta_in).replace(".json", "_joint_fair.json"))
        total, kept, dropped = filter_mcq_json_by_tasks(
            in_json=meta_in,
            out_json=meta_out,
            video_to_task={str(k): int(v) for k, v in video_to_task.items()},
            allowed_tasks=task_subset_set,
        )
        if dist_utils.is_main_process():
            logger.info(f"[joint_fair] Filtered MCQ metadata: {meta_in} -> {meta_out} (total={total}, kept={kept}, dropped={dropped})")
        try:
            OmegaConf.set_struct(cfg, False)
        except Exception:
            pass
        cfg.test.ourdata.metadata = meta_out
    except Exception as e:
        if dist_utils.is_main_process():
            logger.info(f"[joint_fair] WARNING: failed to filter MCQ metadata, will rely on seen_task_ids filtering: {e}")

    # Build model and components (same as main.py)
    logger.info(f"Creating model: {cfg.model.name}")
    model = build_model(cfg.model)
    if cfg.model.freeze_temperature and hasattr(model, "logit_scale"):
        logger.info("Freeze logit temperature")
        model.logit_scale.requires_grad = False
    model.cuda(args.gpu)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=cfg.train.find_unused_parameters
        )
    tokenizer = generate_tokenizer(cfg.model.name)
    criterion = model_utils.get_loss(cfg.model.name, args, cfg, tokenizer=tokenizer).cuda(args.gpu)
    optimizer = build_optimizer(cfg.train, model, criterion)
    scaler = amp.GradScaler("cuda", enabled=not cfg.train.disable_amp)

    loaded_resume = resume_checkpoint(cfg, model, optimizer, scaler, criterion)
    start_epoch = int(loaded_resume["start_epoch"])
    best_metric = float(loaded_resume["best_acc1"])

    # Validation loader (shared)
    val_loader = _build_val_loader(args, cfg, deepcopy(tokenizer))

    # Test-only mode: evaluate on task subset and exit
    if args.testonly or getattr(cfg.test, "testonly", False):
        seen_tasks = set(task_subset)
        per_task_acc, overall_acc, direction_acc = evaluate_mcq_grouped_by_task(val_loader, model, cfg, args, logger, video_to_task, seen_tasks)
        if dist_utils.is_main_process():
            out = {
                "task_subset": [int(x) for x in task_subset],
                "overall_acc": float(overall_acc),
                "per_task_acc": {str(k): float(v) for k, v in per_task_acc.items()},
                "direction_acc": {k: float(v) for k, v in direction_acc.items()},
            }
            with open(os.path.join(cfg.output, "joint_fair_testonly.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            logger.info(out)
        return

    # Build a single train loader filtered to the task subset (joint training on the CL-matched data version)
    allowed_video_uids = {k for k, v in video_to_task.items() if int(v) in task_subset_set}
    train_loader, train_sampler = _build_train_loader_for_task(args, cfg, tokenizer, allowed_video_uids)
    lr_schedule = build_scheduler(cfg, train_loader)

    logger.info("=> beginning joint_fair training")
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg, logger)
        for k, v in train_stats.items():
            logger.info(f"Epoch {epoch}: Train_{k}: {round(v, 3)}")

        # Evaluate on task subset (align to CL final eval protocol)
        seen_tasks = set(task_subset)
        per_task_acc, overall_acc, direction_acc = evaluate_mcq_grouped_by_task(val_loader, model, cfg, args, logger, video_to_task, seen_tasks)

        # Select best by overall_acc (bar_A at final seen set)
        if dist_utils.is_main_process():
            logger.info(f"[joint_fair] Eval overall_acc={overall_acc:.3f} | Ego->Exo={direction_acc['Ego->Exo']:.3f} | Exo->Ego={direction_acc['Exo->Ego']:.3f}")

        if dist_utils.is_main_process() and float(overall_acc) > float(best_metric):
            is_best = True
            best_metric = float(overall_acc)
        else:
            is_best = False

        # save checkpoint (same style as main.py)
        is_epoch = ((epoch + 1) % int(cfg.train.save_freq)) == 0
        if args.distributed and cfg.train.use_zero:
            logger.info("=> consolidating state_dict before saving (due to ZeRO)")
            optimizer.consolidate_state_dict()

        dist_utils.save_on_master(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
                "scaler": scaler.state_dict(),
                "best_acc1": best_metric,
                "cfg": cfg,
                "joint_fair": True,
                "task_subset": [int(x) for x in task_subset],
            },
            is_best,
            cfg.output,
            is_epoch=is_epoch,
        )

        # Save a compact eval snapshot per epoch (rank0 only)
        if dist_utils.is_main_process():
            with open(os.path.join(cfg.output, "joint_fair_eval_latest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": int(epoch + 1),
                        "task_subset": [int(x) for x in task_subset],
                        "overall_acc": float(overall_acc),
                        "per_task_acc": {str(k): float(v) for k, v in per_task_acc.items()},
                        "direction_acc": {k: float(v) for k, v in direction_acc.items()},
                        "best_overall_acc": float(best_metric),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )


def main():
    parser = argparse.ArgumentParser("CLEGO association joint_fair baseline", parents=[get_args_parser()])
    args = parser.parse_args()
    main_worker(args)


if __name__ == "__main__":
    main()


