import argparse
import os
import math
import time
import re
from copy import deepcopy
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.amp as amp
import torch.distributed as dist
from omegaconf import OmegaConf

import sys
# Ensure repo root on path (needed for importing `clego_cl` when running from this subdir).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from function.config import get_config
from function.logger import get_logger
from function import distributed as dist_utils
from function.utils import build_scheduler, resume_checkpoint, build_optimizer, build_val_transform
from models.builder import build_model
from models import model_utils
from models.tokenizer import generate_tokenizer
from data.ourdata_train_dataset import OurTrainDataset
from data.ourdata_dataset import OurDataset

import torch.backends.cudnn as cudnn
from function.random import random_seed as set_random_seed
from clego_cl.continual_algorithms import build_continual_algorithm
from clego_cl.continual_algorithms.lwf_generic import LwFGeneric, LwFGenericConfig
from clego_cl.vista import VISTAState, build_vista_router, vista_eval_router_grouped
from skill_benchmark.adapters import AdapterBank
from clego_cl.l2p import L2PPool

# Reuse training loop from original main
from main import train_one_epoch, get_args_parser as get_base_args_parser
import main as base_main

import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from data.video_transforms import Permute


def get_args_parser():
    # Extend the base parser with no changes (kept to allow future extensions)
    return get_base_args_parser()


def _load_video_to_task_map(path):
    arr = np.load(path, allow_pickle=True)
    mp = None
    # Try dict saved via np.save
    if hasattr(arr, 'item'):
        try:
            mp = arr.item()
        except Exception:
            mp = None
    # Try 2-col array
    if mp is None:
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2:
            mp = {str(a): int(b) for a, b in arr}
        elif isinstance(arr, dict):
            mp = arr
        else:
            raise ValueError(f"Unsupported video_to_task format: type={type(arr)}, shape={getattr(arr, 'shape', None)}")

    # Normalize keys/values: handle both 'task5' and '5' formats
    def parse_task_id(v):
        """Parse task ID from various formats: 5, '5', 'task5', 'Task5', etc."""
        if isinstance(v, int):
            return v
        v_str = str(v).lower()
        # Extract number from strings like 'task5', 'Task5', '5', etc.
        match = re.search(r'\d+', v_str)
        if match:
            return int(match.group())
        raise ValueError(f"Cannot parse task ID from: {v}")

    mp = {str(k): parse_task_id(v) for k, v in mp.items()}
    tasks = sorted(set(mp.values()))
    if len(tasks) == 0:
        raise ValueError("video_to_task map is empty")
    # Make tasks 1..N if starts from 0
    if min(tasks) == 0:
        mp = {k: (v + 1) for k, v in mp.items()}
    return mp


def _build_train_loader_for_task(args, cfg, tokenizer, allowed_video_uids):
    crop_size = 224 if '336PX' not in cfg.model.name else 336
    transforms_list = [
        Permute([3, 0, 1, 2]),    # T H W C -> C T H W
        transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
    ]
    if 'OPENAI' in cfg.model.name:
        transforms_list.append(transforms_video.NormalizeVideo(mean=[122.7709393, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]))
    else:
        transforms_list.append(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
    train_transform = transforms.Compose(transforms_list)

    train_dataset = OurTrainDataset(
        cfg=cfg.data, tokenizer=tokenizer, is_training=True, transform=train_transform,
        allowed_video_uids=allowed_video_uids,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    batch_size_per_gpu = cfg.train.batch_size // args.world_size
    if cfg.train.batch_size % args.world_size != 0 and dist_utils.is_main_process():
        print(f'[Warn] train.batch_size={cfg.train.batch_size} not divisible by world_size={args.world_size}. '
              f'Per-GPU batch_size={batch_size_per_gpu}. This may cause empty loaders when tasks are small.')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_per_gpu, shuffle=(train_sampler is None),
        num_workers=cfg.train.workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )
    return train_loader, train_sampler


def _build_val_loader(args, cfg, tokenizer):
    val_dataset = OurDataset(
        cfg=cfg.test.ourdata,
        transform=build_val_transform(cfg.test.ourdata, cfg.model.name),
        is_training=False,
        tokenizer=tokenizer,
    )
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank,
            shuffle=False, drop_last=False
        )
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    batch_size_per_gpu = cfg.test.batch_size // args.world_size
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_per_gpu, shuffle=False,
        num_workers=cfg.test.workers, pin_memory=True, sampler=val_sampler, drop_last=False,
    )
    return val_loader


def evaluate_mcq_grouped_by_task(val_loader, model, cfg, args, logger, video_to_task, seen_task_ids):
    """Evaluate MCQ and aggregate accuracy per task (by query video task).

    Note:
    - A_tj is averaged within task j (mean over its query samples)
    - bar_A_t is the weighted average over all seen samples (sample-count weighted)
    - Now also returns per-direction metrics (Ego->Exo and Exo->Ego)
    """
    model.eval()
    with torch.no_grad():
        if dist_utils.is_main_process():
            logger.info('=> validation (continual) start forwarding')
        all_preds, all_gts, all_indices, all_types = [], [], [], []
        end_time = time.time()
        for i, inputs in enumerate(val_loader):
            if dist_utils.is_main_process() and i % 10 == 0:
                logger.info('finish validation batch {}/{} in {:.1f} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()
            frame_query = inputs[0].cuda(non_blocking=True)
            frames_options = inputs[1].cuda(non_blocking=True)
            answer = inputs[2]
            q_type = inputs[3]  # Get q_type from dataset
            batch_size = frames_options.shape[0]
            frames_options = frames_options.view(-1, *frames_options.shape[2:])
            # encode
            image_query_features = dist_utils.get_model(model).encode_image(frame_query)
            image_options_features = dist_utils.get_model(model).encode_image(frames_options)
            image_options_features = image_options_features.view(batch_size, -1, *image_options_features.shape[1:])
            # ----------------------------------------------------
            # L2P hook (infer): select top-K adapters by key-query match
            # ----------------------------------------------------
            if getattr(base_main, "l2p_enabled", False) and getattr(base_main, "l2p_mode", "none") == "infer":
                sel, _ = base_main._l2p_select_from_query(image_query_features, training=False)
                if sel is not None:
                    image_query_features = base_main._l2p_apply_embed(image_query_features, sel)
                    image_options_features = base_main._l2p_apply_embed(
                        image_options_features, sel, repeat=image_options_features.shape[1]
                    )
            # ----------------------------------------------------
            # VISTA hook (infer): infer mixture from query embedding and apply to query/options
            # ----------------------------------------------------
            if getattr(base_main, "vista_enabled", False) and getattr(base_main, "vista_mode", "none") == "infer":
                mix = base_main._vista_infer_mix_from_query(image_query_features)
                if mix is not None:
                    image_query_features = base_main._vista_apply_mixture_embed(image_query_features, mix)
                    image_options_features = base_main._vista_apply_mixture_embed(
                        image_options_features, mix, repeat=image_options_features.shape[1]
                    )
            all_gts.append(answer)
            all_types.append(q_type)  # Store q_type
            # keep dataset indices to recover query video id later
            idx_tensor2 = inputs[-1]
            all_indices.append(idx_tensor2)
            for j in range(batch_size):
                similarity_matrix = torch.matmul(image_query_features[j], image_options_features[j].T)
                similarity_matrix = similarity_matrix.cpu().detach()
                all_preds.append(similarity_matrix)
        # gather across processes
        if dist_utils.is_dist_avail_and_initialized():
            dist.barrier()
        if len(all_indices) > 0:
            all_indices = torch.cat(all_indices)
            all_preds = torch.stack(all_preds)
            all_gts = torch.cat(all_gts)
            all_types = torch.cat(all_types)
        else:
            all_indices = torch.empty((0,), dtype=torch.long)
            all_preds = torch.empty((0, 1), dtype=torch.float32)
            all_gts = torch.empty((0,), dtype=torch.long)
            all_types = torch.empty((0,), dtype=torch.long)
        if dist_utils.is_dist_avail_and_initialized():
            idx_list = [all_indices.cpu() for _ in range(dist_utils.get_world_size())]
            preds_list = [all_preds.cpu() for _ in range(dist_utils.get_world_size())]
            gts_list = [all_gts.cpu() for _ in range(dist_utils.get_world_size())]
            types_list = [all_types.cpu() for _ in range(dist_utils.get_world_size())]
            dist.all_gather_object(idx_list, all_indices.cpu())
            dist.all_gather_object(preds_list, all_preds.cpu())
            dist.all_gather_object(gts_list, all_gts.cpu())
            dist.all_gather_object(types_list, all_types.cpu())
            if dist_utils.is_main_process():
                all_indices = torch.cat([u for u in idx_list if len(u) > 0])
                all_preds = torch.cat([p for p in preds_list if len(p) > 0])
                all_gts = torch.cat([g for g in gts_list if len(g) > 0])
                all_types = torch.cat([t for t in types_list if len(t) > 0])
        # compute metrics only on rank 0
        if dist_utils.is_main_process():
            ds = val_loader.dataset
            # Build query video id per sample index
            per_task_total = {}
            per_task_correct = {}
            total_seen = 0
            correct_seen = 0
            # Per-direction metrics
            direction_metrics = {
                'Ego->Exo': {'total': 0, 'correct': 0},
                'Exo->Ego': {'total': 0, 'correct': 0}
            }
            for idx_tensor, pred, label, q_type_val in zip(all_indices, all_preds, all_gts, all_types):
                idx_int = int(idx_tensor.item())
                item = ds.samples[str(idx_int)]
                query = item['query']
                qvid = str(query.get('video_uid') or query.get('video_id'))
                task_id = video_to_task.get(qvid, None)
                if task_id is None or task_id not in seen_task_ids:
                    continue
                total_seen += 1
                pred_idx = int(torch.argmax(pred).item())
                is_correct = 1 if pred_idx == int(label.item()) else 0
                correct_seen += is_correct
                per_task_total[task_id] = per_task_total.get(task_id, 0) + 1
                per_task_correct[task_id] = per_task_correct.get(task_id, 0) + is_correct

                # Update per-direction metrics
                q_type_int = int(q_type_val.item())
                direction_key = 'Ego->Exo' if q_type_int == 1 else 'Exo->Ego'
                direction_metrics[direction_key]['total'] += 1
                direction_metrics[direction_key]['correct'] += is_correct

            per_task_acc = {}
            for t in sorted(seen_task_ids):
                if per_task_total.get(t, 0) > 0:
                    per_task_acc[t] = 100.0 * per_task_correct.get(t, 0) / per_task_total[t]
                else:
                    per_task_acc[t] = float('nan')
            overall_acc = 100.0 * correct_seen / total_seen if total_seen > 0 else float('nan')

            # Compute per-direction accuracies
            direction_acc = {}
            for direction, metrics in direction_metrics.items():
                if metrics['total'] > 0:
                    direction_acc[direction] = 100.0 * metrics['correct'] / metrics['total']
                else:
                    direction_acc[direction] = float('nan')

            return per_task_acc, overall_acc, direction_acc

        # Non-main ranks return placeholders (metrics are computed/logged on rank 0).
        return {}, float('nan'), {'Ego->Exo': float('nan'), 'Exo->Ego': float('nan')}


def _vista_load_router_tasks_from_dir(*, router, router_dir: str, task_ids: List[int]) -> None:
    """Load whitened-subspace router_task_XX.npz files into an in-memory router."""
    import numpy as _np

    for tid in task_ids:
        p = os.path.join(router_dir, f"router_task_{int(tid):02d}.npz")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[association vista testonly] missing router task file: {p}")
        z = _np.load(p)
        router.add_task_stats(int(tid), mu=z["mu"], var=z["var"], Bw=z["Bw"])


def _vista_load_adapter_bank_from_dir(*, adapter_dir: str, device: torch.device) -> AdapterBank:
    """Load AdapterBank saved by AdapterBank.save()."""
    meta_path = os.path.join(adapter_dir, "adapter_bank_meta.pt")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"[association vista testonly] missing adapter bank meta: {meta_path}")
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    tasks = list(meta.get("tasks", []))
    bank = AdapterBank(
        input_dim=int(meta.get("input_dim", 256)),
        bottleneck=int(meta.get("bottleneck", 64)),
        use_layernorm=bool(meta.get("use_layernorm", True)),
    ).to(device)
    for tid in tasks:
        bank.add_task(int(tid), init_from_task=None)
        sd_path = os.path.join(adapter_dir, f"adapter_task_{int(tid):02d}.pt")
        if not os.path.isfile(sd_path):
            raise FileNotFoundError(f"[association vista testonly] missing adapter state: {sd_path}")
        sd = torch.load(sd_path, map_location="cpu", weights_only=False)
        bank.get(int(tid)).load_state_dict(sd, strict=True)
    return bank


def main_worker(args):
    # Prepare config and env
    cfg = get_config(args)
    os.makedirs(cfg.output, exist_ok=True)

    dist_utils.init_distributed_mode(args)
    logger = get_logger(cfg)

    if dist_utils.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.yml')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config save to {path}')
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility across processes
    set_random_seed(getattr(cfg.train, 'seed', 42), dist_utils.get_rank())
    cudnn.deterministic = True
    cudnn.benchmark = False

    if args.distributed:
        dist.barrier()

    # Continual settings
    continual_cfg = getattr(cfg, 'continual', None)
    assert continual_cfg is not None and getattr(continual_cfg, 'enabled', False), \
        'Please set cfg.continual.enabled: true in config to use continual_main.py'

    video_to_task_path = getattr(continual_cfg, 'video_to_task_path', None)
    assert video_to_task_path is not None, 'cfg.continual.video_to_task_path must be provided'
    video_to_task = _load_video_to_task_map(video_to_task_path)
    all_task_ids_sorted = sorted(set(video_to_task.values()))
    num_tasks = getattr(continual_cfg, 'num_tasks', len(all_task_ids_sorted))

    # Subset of tasks to operate on (limited by cfg.continual.num_tasks)
    task_ids_subset = all_task_ids_sorted[:num_tasks]

    # Build model and components
    logger.info(f'Creating model: {cfg.model.name}')
    model = build_model(cfg.model)
    if cfg.model.freeze_temperature and hasattr(model, 'logit_scale'):
        logger.info('Freeze logit temperature')
        model.logit_scale.requires_grad = False
    model.cuda(args.gpu)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=cfg.train.find_unused_parameters,
        )
    tokenizer = generate_tokenizer(cfg.model.name)
    criterion = model_utils.get_loss(cfg.model.name, args, cfg, tokenizer=tokenizer).cuda(args.gpu)
    optimizer = build_optimizer(cfg.train, model, criterion)
    scaler = amp.GradScaler('cuda', enabled=not cfg.train.disable_amp)

    # resume if any
    loaded_resume = resume_checkpoint(cfg, model, optimizer, scaler, criterion)
    _ = loaded_resume  # start_epoch not used in continual stage-wise training

    # Build a single validation loader (we will filter by tasks during aggregation)
    val_loader = _build_val_loader(args, cfg, deepcopy(tokenizer))

    # Build a global LR schedule across all tasks (do not reset per task)
    lr_schedule = build_scheduler(cfg, None)

    # Continual test-only evaluation branch
    if args.testonly or getattr(cfg.test, 'testonly', False):
        logger.info('=> continual test-only evaluation mode')
        final_A = np.full((num_tasks, num_tasks), np.nan, dtype=np.float32)
        final_bar_A = np.full((num_tasks,), np.nan, dtype=np.float32)
        final_direction_A = {'Ego->Exo': np.full((num_tasks,), np.nan, dtype=np.float32),
                             'Exo->Ego': np.full((num_tasks,), np.nan, dtype=np.float32)}

        # Determine checkpoint source directory
        # If cfg.continual.checkpoint_dir is set, load checkpoints from there (for test-only mode)
        ckpt_source_dir = getattr(cfg.continual, 'checkpoint_dir', None)
        if dist_utils.is_main_process():
            logger.info(f'[TestOnly] Loading checkpoints from: {ckpt_source_dir}')

        # Determine task order for evaluation (prefer order saved during training)
        import json
        task_order = None
        order_path = os.path.join(ckpt_source_dir, 'task_order.json') if ckpt_source_dir else None
        if order_path and os.path.isfile(order_path):
            try:
                with open(order_path, 'r') as f:
                    task_order = json.load(f)
            except Exception as e:
                if dist_utils.is_main_process():
                    logger.info(f'[TestOnly] Failed to load task_order.json from {order_path}: {e}')
        if task_order is None:
            # Fallback: derive from config/seed (so test-only can still run without saved order)
            # Default: randomize order (seed-controlled) unless config explicitly disables it.
            randomize = bool(getattr(continual_cfg, 'randomize_order', True))
            task_order = list(task_ids_subset)
            if randomize:
                seed_for_order = getattr(getattr(cfg, 'train', None), 'seed', 42)
                rng = np.random.RandomState(seed_for_order)
                rng.shuffle(task_order)
        if dist_utils.is_main_process():
            logger.info(f'[TestOnly] Task order: {task_order}')

        # ---- VISTA setup + load router from training output (test-only) ----
        vista_enabled_testonly = bool(getattr(cfg, "vista_enabled", False))
        vista_state_testonly = None
        if vista_enabled_testonly:
            embed_dim = int(getattr(getattr(cfg, "model", None), "project_embed_dim", 256))
            adapter_bank = AdapterBank(
                input_dim=embed_dim,
                bottleneck=int(getattr(cfg, "vista_adapter_bottleneck", 64)),
                use_layernorm=True,
            ).cuda()
            router = build_vista_router(
                router_M=int(getattr(cfg, "vista_router_M", 1)),
                subspace_k=int(getattr(cfg, "vista_subspace_k", 32)),
                eps=float(getattr(cfg, "vista_eps", 1e-6)),
            )
            vista_state_testonly = VISTAState(
                enabled=True,
                adapter_bank=adapter_bank,
                router=router,
                router_M=int(getattr(cfg, "vista_router_M", 1)),
                topL=int(getattr(cfg, "vista_topL", 2)),
                gamma=float(getattr(cfg, "vista_gamma", 10.0)),
                eps=float(getattr(cfg, "vista_eps", 1e-6)),
                apply_to_target=bool(getattr(cfg, "vista_apply_to_target", True)),
                train_backbone_after_task1=False,
            )
            base_main.vista_enabled = True
            base_main.vista_state = vista_state_testonly
            base_main.vista_mode = "none"

            # Load all router tasks from training output root router/
            if ckpt_source_dir is None:
                raise RuntimeError("[association vista testonly] continual.checkpoint_dir is required to load VISTA router/adapters.")
            router_root_dir = os.path.join(ckpt_source_dir, "router")
            _vista_load_router_tasks_from_dir(
                router=vista_state_testonly.router,
                router_dir=router_root_dir,
                task_ids=[int(t) for t in task_order],
            )
            try:
                vista_state_testonly.router.save_index(output_dir=os.path.join(cfg.output, "router"))
            except Exception:
                pass
        else:
            base_main.vista_enabled = False
            base_main.vista_state = None
            base_main.vista_mode = "none"

        for t_idx, task_id in enumerate(task_order, start=1):
            # Load checkpoint from ckpt_source_dir (training output)
            ckpt_task_dir = os.path.join(ckpt_source_dir, f'task_{t_idx:02d}')
            ckpt_path = os.path.join(ckpt_task_dir, f'checkpoint_{cfg.train.epochs:04d}.pt')
            if not os.path.isfile(ckpt_path):
                raise RuntimeError(
                    f"[association test-only] checkpoint not found for task {t_idx}: {ckpt_path}. "
                    "Training output is incomplete; refusing to silently skip tasks."
                )
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state = ckpt.get('state_dict', ckpt)
            res = model.load_state_dict(state, strict=False)
            if dist_utils.is_main_process():
                logger.info(f'[TestOnly] Loaded checkpoint for task {t_idx} from {ckpt_path}: {res}')
            seen_tasks = set(task_order[:t_idx])
            # Load adapters snapshot for this task (so inference mixes over seen tasks)
            if vista_state_testonly is not None:
                ad_dir = os.path.join(ckpt_task_dir, "adapters")
                vista_state_testonly.adapter_bank = _vista_load_adapter_bank_from_dir(adapter_dir=ad_dir, device=torch.device("cuda"))
                base_main.vista_state = vista_state_testonly
                base_main.vista_mode = "infer"
            else:
                base_main.vista_mode = "none"
            # Load L2P pool snapshot for this task (if present)
            if getattr(base_main, "l2p_enabled", False):
                try:
                    l2p_path = os.path.join(ckpt_task_dir, "l2p", "l2p_pool.pt")
                    if os.path.isfile(l2p_path):
                        base_main.l2p_pool = L2PPool.load(l2p_path, device=torch.device("cuda"))
                        base_main.l2p_mode = "infer"
                    else:
                        base_main.l2p_mode = "none"
                except Exception:
                    base_main.l2p_mode = "none"
            per_task_acc, overall_acc, direction_acc = evaluate_mcq_grouped_by_task(val_loader, model, cfg, args, logger, video_to_task, seen_tasks)
            A_row = [float(per_task_acc.get(tj, float('nan'))) for tj in task_order[:t_idx]]
            if dist_utils.is_main_process():
                logger.info(f'[Continual-TestOnly] Task {t_idx}: ' +
                            ' '.join([f'A_{t_idx}{jidx+1}={A_row[jidx]:.3f}' for jidx in range(len(A_row))]) +
                            f' | bar_A={overall_acc:.3f}' +
                            f' | Ego->Exo={direction_acc["Ego->Exo"]:.3f} | Exo->Ego={direction_acc["Exo->Ego"]:.3f}')
                import json
                # Save metrics to test output directory (cfg.output), not training directory
                metrics_path = os.path.join(cfg.output, f'metrics_testonly_t{t_idx:02d}.json')
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'task_index': t_idx,
                        'A_row': A_row,
                        'bar_A': float(overall_acc),
                        'Ego->Exo': float(direction_acc['Ego->Exo']),
                        'Exo->Ego': float(direction_acc['Exo->Ego'])
                    }, f)

                # ---- VISTA: router eval on this split (val/test depends on cfg.test.ourdata.metadata) ----
                if vista_state_testonly is not None and vista_state_testonly.router is not None:
                    try:
                        ds = val_loader.dataset
                        global_index = 0
                        sum_stats: Dict[int, Dict[str, float]] = {}
                        sum_hits: Dict[int, Dict[str, float]] = {}
                        sum_n: Dict[int, int] = {}
                        for inputs in val_loader:
                            frame_query = inputs[0].cuda(non_blocking=True)
                            idx_tensor = inputs[-1]
                            idx_list = [int(x) for x in idx_tensor.cpu().tolist()]
                            keep = []
                            gt_ids = []
                            for j, idx_int in enumerate(idx_list):
                                item = ds.samples[str(idx_int)]
                                query = item["query"]
                                qvid = str(query.get("video_uid") or query.get("video_id"))
                                tid = video_to_task.get(qvid, None)
                                if tid is None or int(tid) not in seen_tasks:
                                    continue
                                keep.append(int(j))
                                gt_ids.append(int(tid))
                            global_index += int(len(idx_list))
                            if len(keep) == 0:
                                continue
                            emb = dist_utils.get_model(model).encode_image(frame_query[keep])
                            x = emb.detach().to(dtype=torch.float32)
                            gt = torch.tensor(gt_ids, device=x.device, dtype=torch.long)
                            b_stats, b_hits = vista_eval_router_grouped(
                                router=vista_state_testonly.router,
                                x=x,
                                gt_task_ids=gt,
                                M=1,
                                topL=int(vista_state_testonly.topL),
                                gamma=float(vista_state_testonly.gamma),
                            )
                            for tid, hh in b_hits.items():
                                n = int(hh.get("n_samples", 0))
                                if n <= 0:
                                    continue
                                sum_n[tid] = sum_n.get(tid, 0) + n
                                acc = sum_hits.get(tid, {"top1": 0.0, "topL": 0.0, "prob": 0.0, "topL_cfg": int(hh.get("topL", 1))})
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
                        router_stats = {}
                        router_hits = {}
                        for tid in sorted(sum_n.keys()):
                            n = int(sum_n[tid])
                            if n <= 0:
                                continue
                            hs = sum_hits.get(tid, {})
                            ss = sum_stats.get(tid, {})
                            router_hits[int(tid)] = {
                                "top1_hit_rate": float(hs.get("top1", 0.0)) / float(n),
                                "topL_hit_rate": float(hs.get("topL", 0.0)) / float(n),
                                "topL": int(hs.get("topL_cfg", int(vista_state_testonly.topL))),
                                "n_samples": int(n),
                                "true_task_prob_mean": float(hs.get("prob", 0.0)) / float(n),
                            }
                            router_stats[int(tid)] = {
                                "res_best_mean": float(ss.get("res_best_mean", 0.0)) / float(n),
                                "res_gap_mean": float(ss.get("res_gap_mean", 0.0)) / float(n),
                                "entropy_mean": float(ss.get("entropy_mean", 0.0)) / float(n),
                            }
                        split_tag = "test" if "association_test" in str(getattr(cfg.test.ourdata, "metadata", "")) else "val"
                        out_path = os.path.join(cfg.output, f"router_eval_{split_tag}_t{t_idx:02d}.json")
                        with open(out_path, "w", encoding="utf-8") as rf:
                            json.dump(
                                {
                                    "task_index": int(t_idx),
                                    "task_id": int(task_id),
                                    "seen_tasks": [int(x) for x in task_order[:t_idx]],
                                    f"router_stats_{split_tag}": {str(k): v for k, v in router_stats.items()},
                                    f"router_hit_{split_tag}": {str(k): v for k, v in router_hits.items()},
                                },
                                rf,
                                indent=2,
                                ensure_ascii=False,
                            )
                    except Exception as e:
                        raise RuntimeError(f"[association vista testonly] router eval failed at t={t_idx}") from e
                for j_local, acc in enumerate(A_row, start=1):
                    final_A[t_idx - 1, j_local - 1] = acc
                final_bar_A[t_idx - 1] = float(overall_acc)
                final_direction_A['Ego->Exo'][t_idx - 1] = float(direction_acc['Ego->Exo'])
                final_direction_A['Exo->Ego'][t_idx - 1] = float(direction_acc['Exo->Ego'])
        if dist_utils.is_main_process():
            np.save(os.path.join(cfg.output, 'continual_A.npy'), final_A)
            np.save(os.path.join(cfg.output, 'continual_bar_A.npy'), final_bar_A)
            np.save(os.path.join(cfg.output, 'continual_direction_Ego2Exo.npy'), final_direction_A['Ego->Exo'])
            np.save(os.path.join(cfg.output, 'continual_direction_Exo2Ego.npy'), final_direction_A['Exo->Ego'])
            logger.info('Saved continual A matrix, bar_A, and direction-specific metrics to output directory (testonly)')
        return

    # Begin continual learning over tasks
    logger.info('=> beginning continual training over {} tasks'.format(num_tasks))

    # For saving results across tasks
    final_A = np.full((num_tasks, num_tasks), np.nan, dtype=np.float32)
    final_bar_A = np.full((num_tasks,), np.nan, dtype=np.float32)
    final_direction_A = {'Ego->Exo': np.full((num_tasks,), np.nan, dtype=np.float32),
                         'Exo->Ego': np.full((num_tasks,), np.nan, dtype=np.float32)}

    # Determine task order for training (and save for later test/val)
    # Default: randomize order (seed-controlled) unless config explicitly disables it.
    randomize = bool(getattr(continual_cfg, 'randomize_order', True))
    task_order = list(task_ids_subset)
    if randomize:
        seed_for_order = getattr(getattr(cfg, 'train', None), 'seed', 42)
        rng = np.random.RandomState(seed_for_order)
        rng.shuffle(task_order)
    if dist_utils.is_main_process():
        logger.info(f'[Continual] Task order: {task_order}')
        # Save chosen order so test/val can reproduce
        try:
            import json
            with open(os.path.join(cfg.output, 'task_order.json'), 'w') as f:
                json.dump(task_order, f)
        except Exception as e:
            logger.info(f'[Continual] Failed to save task_order.json: {e}')

    # ------------------------------
    # Continual algorithm (ER, etc.)
    # ------------------------------
    algo_name = str(getattr(cfg, "continual_algorithm", "none")).strip().lower()
    if algo_name == "lwf":
        algo = LwFGeneric(cfg=LwFGenericConfig(alpha=float(getattr(cfg, "lwf_alpha", 0.5))))
    else:
        algo = build_continual_algorithm(
            algo_name=getattr(cfg, "continual_algorithm", "none"),
            buffer_ratio=getattr(cfg, "continual_algorithm_buffer_ratio", 0.2),
            replay_batch_ratio=getattr(cfg, "continual_algorithm_replay_batch_ratio", 0.2),
            distill_alpha=getattr(cfg, "continual_algorithm_distill_alpha", 0.5),
            ewc_lambda=getattr(cfg, "ewc_lambda", 1e-2),
            ewc_gamma=getattr(cfg, "ewc_gamma", 1.0),
            ewc_fisher_batches=getattr(cfg, "ewc_fisher_batches", 50),
            lwf_alpha=getattr(cfg, "lwf_alpha", 0.5),
            seed=int(getattr(getattr(cfg, "train", None), "seed", 42)),
        )
    if algo is not None or bool(getattr(cfg, "vista_enabled", False)) or bool(getattr(cfg, "l2p_enabled", False)):
        if args.distributed and int(getattr(args, "world_size", 1)) > 1:
            raise RuntimeError("[association continual] Continual algorithms currently support only single-process (world_size=1) for strict reproducibility.")
    if algo is not None:
        # Disallow VISTA/L2P combinations with EWC/LwF (per protocol).
        algo_name_str = str(getattr(algo, "name", "")).strip().lower()
        if algo_name_str in ("ewc", "lwf") and (
            bool(getattr(cfg, "vista_enabled", False)) or bool(getattr(cfg, "l2p_enabled", False))
        ):
            raise ValueError("EWC/LwF are not supported with VISTA or L2P in association_benchmark (disable vista_enabled/l2p_enabled).")
        # Pre-compute total train samples across all tasks for strict capacity definition.
        if hasattr(algo, "configure_total_capacity"):
            total_train = 0
            for _tid in task_order:
                _allowed = {k for k, v in video_to_task.items() if v == _tid}
                _loader, _ = _build_train_loader_for_task(args, cfg, tokenizer, _allowed)
                total_train += int(len(_loader.dataset))
            algo.configure_total_capacity(total_train_samples=int(total_train))
        if hasattr(algo, "bind_models"):
            algo.bind_models(models={"main": model})
        base_main.continual_algo = algo

    # ------------------------------
    # VISTA setup (embedding space)
    # ------------------------------
    vista_enabled = bool(getattr(cfg, "vista_enabled", False))
    vista_state = None
    base_main.vista_adapter_optimizer = None
    if vista_enabled:
        embed_dim = int(getattr(getattr(cfg, "model", None), "project_embed_dim", 256))
        adapter_bank = AdapterBank(
            input_dim=embed_dim,
            bottleneck=int(getattr(cfg, "vista_adapter_bottleneck", 64)),
            use_layernorm=True,
        ).cuda()
        router = build_vista_router(
            router_M=int(getattr(cfg, "vista_router_M", 1)),
            subspace_k=int(getattr(cfg, "vista_subspace_k", 32)),
            eps=float(getattr(cfg, "vista_eps", 1e-6)),
        )
        vista_state = VISTAState(
            enabled=True,
            adapter_bank=adapter_bank,
            router=router,
            router_M=int(getattr(cfg, "vista_router_M", 1)),
            topL=int(getattr(cfg, "vista_topL", 2)),
            gamma=float(getattr(cfg, "vista_gamma", 10.0)),
            eps=float(getattr(cfg, "vista_eps", 1e-6)),
            apply_to_target=bool(getattr(cfg, "vista_apply_to_target", True)),
            train_backbone_after_task1=bool(getattr(cfg, "vista_train_backbone_after_task1", False)),
        )
        base_main.vista_enabled = True
        base_main.vista_state = vista_state
        base_main.vista_mode = "none"
    else:
        base_main.vista_enabled = False
        base_main.vista_state = None
        base_main.vista_mode = "none"

    # ------------------------------
    # L2P setup (embedding space)
    # ------------------------------
    l2p_pool = None
    if bool(getattr(cfg, "l2p_enabled", False)):
        embed_dim = int(getattr(getattr(cfg, "model", None), "project_embed_dim", 256))
        key_dim = int(getattr(cfg, "l2p_router_M", 1)) * int(embed_dim)
        pool_size = int(len(task_order))
        l2p_pool = L2PPool(
            pool_size=pool_size,
            topk=int(getattr(cfg, "l2p_topK", 2)),
            adapter_dim=int(embed_dim),
            key_dim=int(key_dim),
            adapter_bottleneck=int(getattr(cfg, "l2p_adapter_bottleneck", 64)),
            diversed_selection=bool(getattr(cfg, "l2p_diversed_selection", True)),
            batchwise_selection=bool(getattr(cfg, "l2p_batchwise_selection", False)),
        ).cuda()
        base_main.l2p_enabled = True
        base_main.l2p_pool = l2p_pool
        base_main.l2p_topk = int(getattr(cfg, "l2p_topK", 2))
        base_main.l2p_router_M = int(getattr(cfg, "l2p_router_M", 1))
        base_main.l2p_sim_lambda = float(getattr(cfg, "l2p_sim_lambda", 0.5))
        base_main.l2p_diversed_selection = bool(getattr(cfg, "l2p_diversed_selection", True))
        base_main.l2p_batchwise_selection = bool(getattr(cfg, "l2p_batchwise_selection", False))
        base_main.l2p_mode = "train"
        base_main.l2p_optimizer = torch.optim.Adam(l2p_pool.parameters(), lr=float(getattr(getattr(cfg, "train", None), "lr", 1e-4)))
    else:
        base_main.l2p_enabled = False
        base_main.l2p_pool = None
        base_main.l2p_mode = "none"
        base_main.l2p_optimizer = None

    def _vista_fit_router_for_task(*, task_id: int, train_loader) -> None:
        if vista_state is None or vista_state.router is None:
            return
        model.eval()
        max_samples = int(getattr(cfg, "vista_router_fit_max_samples", 0))
        n_samples = 0

        def _iter_fit_batches():
            nonlocal n_samples
            with torch.no_grad():
                for batch_idx, batch in enumerate(train_loader):
                    inputs = batch
                    if not isinstance(inputs, dict) or "video" not in inputs:
                        continue
                    frames = inputs["video"].cuda(non_blocking=True)
                    embeds = dist_utils.get_model(model).encode_image(frames)
                    embeds = embeds.detach().to(dtype=torch.float32, device="cpu")
                    if max_samples > 0:
                        remaining = int(max_samples - n_samples)
                        if remaining <= 0:
                            break
                        embeds = embeds[:remaining]
                    if embeds.numel() == 0:
                        continue
                    n_samples += int(embeds.shape[0])
                    yield (embeds, embeds)
                    if max_samples > 0 and n_samples >= max_samples:
                        break

        t0 = time.time()
        logger.info(f"[VISTA] Router fit start: task_id={int(task_id)} max_samples={max_samples if max_samples>0 else 'ALL'}")
        vista_state.router.fit_from_loader(task_id=int(task_id), loader=_iter_fit_batches(), device="cpu", verbose=False)
        dt = time.time() - t0
        if n_samples <= 0:
            raise RuntimeError(f"[association vista] Empty router-fit data for task_id={task_id}")
        logger.info(f"[VISTA] Router fit done: task_id={int(task_id)} n_samples={int(n_samples)} time_sec={dt:.1f}")

    def _build_memory_loader_for_task(args, cfg, tokenizer, allowed_video_uids, *, batch_size_total: Optional[int] = None):
        """Like _build_train_loader_for_task but deterministic & without drop_last for memory sampling.

        Note: This loader is shared by multiple continual algorithms (ER/DER++ memory update and EWC Fisher).
        Use `batch_size_total` to control per-algorithm memory behavior without affecting others.
        """
        crop_size = 224 if '336PX' not in cfg.model.name else 336
        transforms_list = [
            Permute([3, 0, 1, 2]),
            transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        ]
        if 'OPENAI' in cfg.model.name:
            transforms_list.append(transforms_video.NormalizeVideo(mean=[122.7709393, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]))
        else:
            transforms_list.append(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
        train_transform = transforms.Compose(transforms_list)
        ds = OurTrainDataset(cfg=cfg.data, tokenizer=tokenizer, is_training=True, transform=train_transform, allowed_video_uids=allowed_video_uids)
        # Force a deterministic sampler in memory collection to make reservoir sampling well-defined.
        sampler = torch.utils.data.SequentialSampler(ds)
        bs_total = int(batch_size_total) if batch_size_total is not None else int(cfg.train.batch_size)
        # ceil-div to avoid 0 when bs_total < world_size
        batch_size_per_gpu = max(1, (bs_total + int(args.world_size) - 1) // int(args.world_size))
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=cfg.train.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=False,
        )
        return loader

    for t_idx, task_id in enumerate(task_order, start=1):
        # Build train loader restricted to current task
        allowed_videos = {k for k, v in video_to_task.items() if v == task_id}
        train_loader, train_sampler = _build_train_loader_for_task(args, cfg, tokenizer, allowed_videos)
        # use global lr_schedule (do not reset per task)

        task_dir = os.path.join(cfg.output, f'task_{t_idx:02d}')
        if dist_utils.is_main_process():
            os.makedirs(task_dir, exist_ok=True)
        last_A_row = None
        last_overall_acc = None
        last_direction_acc = None

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
                lr=float(getattr(getattr(cfg, "train", None), "lr", 1e-4)),
            )
            base_main.vista_mode = "train"
        if l2p_pool is not None:
            base_main.l2p_mode = "train"

        # ---- VISTA: optionally freeze backbone after task 1 ----
        if vista_state is not None and int(t_idx) >= 2:
            freeze = not bool(getattr(cfg, "vista_train_backbone_after_task1", False))
            if freeze:
                for p in model.parameters():
                    p.requires_grad = False

        # Baseline parity: keep legacy skip behavior when no continual_algorithm is enabled.
        # Strict mode (when ER is enabled): do NOT skip.
        if len(train_loader) == 0:
            if algo is None:
                if dist_utils.is_main_process():
                    logger.info(f'[Continual] Task {t_idx}: len(train_loader) == 0, skipping training for this task')
                seen_tasks = set(task_order[:t_idx])
                base_main.vista_mode = "infer"
                per_task_acc, overall_acc, direction_acc = evaluate_mcq_grouped_by_task(val_loader, model, cfg, args, logger, video_to_task, seen_tasks)
                A_row = [float(per_task_acc.get(tj, float("nan"))) for tj in task_order[:t_idx]]
                if dist_utils.is_main_process():
                    import json
                    metrics_path = os.path.join(task_dir, 'metrics_skipped.json')
                    with open(metrics_path, 'w') as f:
                        json.dump({
                            'task_index': t_idx,
                            'epoch': 0,
                            'A_row': A_row,
                            'bar_A': float(overall_acc),
                            'Ego->Exo': float(direction_acc['Ego->Exo']),
                            'Exo->Ego': float(direction_acc['Exo->Ego']),
                            'skipped': True
                        }, f)
                    for j_local, acc in enumerate(A_row, start=1):
                        final_A[t_idx - 1, j_local - 1] = acc
                    final_bar_A[t_idx - 1] = float(overall_acc)
                    final_direction_A['Ego->Exo'][t_idx - 1] = float(direction_acc['Ego->Exo'])
                    final_direction_A['Exo->Ego'][t_idx - 1] = float(direction_acc['Exo->Ego'])
                continue
            raise RuntimeError(f"[association continual strict] len(train_loader)==0 for task_index={t_idx} task_id={task_id}. Check data filtering/video_to_task.")

        # epochs per task come from cfg.train.epochs
        for epoch in range(cfg.train.epochs):
            if args.distributed and train_sampler is not None:
                # make epoch unique across tasks for better shuffling
                train_sampler.set_epoch((t_idx - 1) * cfg.train.epochs + epoch)
            base_main.vista_mode = "train"
            train_stats = train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg, logger)
            if dist_utils.is_main_process():
                for k, v in train_stats.items():
                    logger.info(f'Task {t_idx} Epoch {epoch}: Train_{k}: {round(v, 3)}')

            # Save ONLY the final-epoch checkpoint for this task to reduce disk usage.
            # Keep filename convention (checkpoint_{epoch:04d}.pt) so test-only/val-only flows are unchanged.
            # Also disable writing checkpoint.pt (latest/resume) by default for continual runs.
            if (epoch + 1) == cfg.train.epochs:
                dist_utils.save_on_master({
                    'epoch': epoch + 1,
                    'task_index': t_idx,
                    'task_id': int(task_id) if isinstance(task_id, (int, np.integer)) else task_id,
                    'state_dict': model.state_dict(),
                    'criterion': criterion.state_dict(),
                    'optimizer': optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
                    'scaler': scaler.state_dict(),
                    'best_acc1': 0.0,
                    'cfg': cfg,
                }, False, task_dir, is_epoch=True, save_latest_checkpoint=False)

            # Evaluate on seen tasks (1..t_idx)
            seen_tasks = set(task_order[:t_idx])
            base_main.vista_mode = "infer"
            if l2p_pool is not None:
                base_main.l2p_mode = "infer"
            per_task_acc, overall_acc, direction_acc = evaluate_mcq_grouped_by_task(val_loader, model, cfg, args, logger, video_to_task, seen_tasks)
            # Compose A row for current t
            A_row = [float(per_task_acc.get(tj, float('nan'))) for tj in task_order[:t_idx]]
            last_A_row = A_row
            last_overall_acc = float(overall_acc)
            last_direction_acc = direction_acc
            if dist_utils.is_main_process():
                logger.info(f'[Continual] Task {t_idx} Epoch {epoch+1}: ' +
                            ' '.join([f'A_{t_idx}{jidx+1}={A_row[jidx]:.3f}' for jidx in range(len(A_row))]) +
                            f' | bar_A={overall_acc:.3f}' +
                            f' | Ego->Exo={direction_acc["Ego->Exo"]:.3f} | Exo->Ego={direction_acc["Exo->Ego"]:.3f}')
                # Save epoch metrics JSON
                metrics_path = os.path.join(task_dir, f'metrics_epoch_{epoch+1:02d}.json')
                import json
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'task_index': t_idx,
                        'epoch': epoch + 1,
                        'A_row': A_row,
                        'bar_A': float(overall_acc),
                        'Ego->Exo': float(direction_acc['Ego->Exo']),
                        'Exo->Ego': float(direction_acc['Exo->Ego']),
                    }, f)
                # If this is final epoch of the task, store into final matrices
                if epoch + 1 == cfg.train.epochs:
                    for j_local, acc in enumerate(A_row, start=1):
                        final_A[t_idx - 1, j_local - 1] = acc
                    final_bar_A[t_idx - 1] = float(overall_acc)
                    final_direction_A['Ego->Exo'][t_idx - 1] = float(direction_acc['Ego->Exo'])
                    final_direction_A['Exo->Ego'][t_idx - 1] = float(direction_acc['Exo->Ego'])

        # ---- EWC: estimate Fisher at task end ----
        if algo is not None and str(getattr(algo, "name", "")).strip().lower() == "ewc":
            # Fisher estimation does additional forward/backward passes and can easily OOM
            # after a full task training+validation. Try to release cached blocks first.
            try:
                import gc

                gc.collect()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # EWC Fisher estimation is memory-heavy; default to a smaller batch size unless overridden.
            _ewc_bs_total = int(getattr(cfg, "ewc_batch_size", 0) or 0)
            if _ewc_bs_total <= 0:
                _ewc_bs_total = min(int(cfg.train.batch_size), 8)
            mem_loader = _build_memory_loader_for_task(args, cfg, tokenizer, allowed_videos, batch_size_total=_ewc_bs_total)
            if len(mem_loader.dataset) <= 0:
                raise RuntimeError(f"[EWC strict] Empty memory dataset for task_id={task_id}")

            def _ewc_loss_from_batch(batch_in):
                inputs = batch_in
                if not isinstance(inputs, dict):
                    raise RuntimeError("Association EWC expects dict batch with keys 'video' and 'text'.")
                video = inputs["video"].cuda(non_blocking=True)
                text = inputs["text"].cuda(non_blocking=True)
                # Match training memory behavior: enable AMP during Fisher estimation.
                # Also enable checkpointing here to reduce peak memory for Fisher passes.
                with torch.cuda.amp.autocast(enabled=not bool(getattr(cfg.train, "disable_amp", False))):
                    outputs = model(
                        video,
                        text,
                        use_checkpoint=True,
                        norm_embed=cfg.model.norm_embed,
                    )
                    loss_dict = criterion(outputs)
                    loss = loss_dict["loss"]
                return loss

            max_batches = int(getattr(cfg, "ewc_fisher_batches", 0))
            max_batches = max_batches if max_batches > 0 else None
            algo.update_fisher_from_loader(loader=mem_loader, loss_fn=_ewc_loss_from_batch, max_batches=max_batches)

        # ---- LwF: update teacher snapshot at task end ----
        if algo is not None and str(getattr(algo, "name", "")).strip().lower() == "lwf":
            if hasattr(algo, "update_teacher"):
                algo.update_teacher(model)

        # ---- VISTA: fit router and save adapters/router at task end ----
        if vista_state is not None:
            _vista_fit_router_for_task(task_id=int(task_id), train_loader=train_loader)
            if dist_utils.is_main_process():
                vista_state.adapter_bank.save(os.path.join(task_dir, "adapters"))
                vista_state.router.save_task(output_dir=os.path.join(task_dir, "router"), task_id=int(task_id))
                vista_state.router.save_task(output_dir=os.path.join(cfg.output, "router"), task_id=int(task_id))
                import json
                # ---- VISTA: router eval on VAL split (current cfg.test.ourdata.metadata) ----
                router_stats_val = {}
                router_hit_val = {}
                try:
                    router_eval_enabled = bool(getattr(cfg, "vista_router_eval_enabled", True))
                    router_eval_max_samples = int(getattr(cfg, "vista_router_eval_max_samples", 0))
                    if not router_eval_enabled:
                        logger.info(f"[VISTA] Router eval skipped by config: task_index={t_idx} task_id={int(task_id)}")
                        raise StopIteration()

                    logger.info(
                        f"[VISTA] Router eval (val) start: task_index={t_idx} task_id={int(task_id)} "
                        f"max_samples={router_eval_max_samples if router_eval_max_samples>0 else 'ALL'}"
                    )
                    t0_eval = time.time()
                    ds = val_loader.dataset
                    sum_stats: Dict[int, Dict[str, float]] = {}
                    sum_hits: Dict[int, Dict[str, float]] = {}
                    sum_n: Dict[int, int] = {}
                    seen_tasks_list = sorted([int(x) for x in seen_tasks])
                    n_eval = 0
                    for inputs in val_loader:
                        frame_query = inputs[0].cuda(non_blocking=True)
                        idx_tensor = inputs[-1]
                        idx_list = [int(x) for x in idx_tensor.cpu().tolist()]
                        keep = []
                        gt_ids = []
                        for j, idx_int in enumerate(idx_list):
                            item = ds.samples[str(idx_int)]
                            query = item["query"]
                            qvid = str(query.get("video_uid") or query.get("video_id"))
                            tid = video_to_task.get(qvid, None)
                            if tid is None or int(tid) not in seen_tasks:
                                continue
                            keep.append(int(j))
                            gt_ids.append(int(tid))
                        if len(keep) == 0:
                            continue
                        emb = dist_utils.get_model(model).encode_image(frame_query[keep])
                        x = emb.detach().to(dtype=torch.float32)
                        gt = torch.tensor(gt_ids, device=x.device, dtype=torch.long)
                        n_eval += int(gt.numel())
                        b_stats, b_hits = vista_eval_router_grouped(
                            router=vista_state.router,
                            x=x,
                            gt_task_ids=gt,
                            M=1,
                            topL=int(vista_state.topL),
                            gamma=float(vista_state.gamma),
                        )
                        for tid, hh in b_hits.items():
                            n = int(hh.get("n_samples", 0))
                            if n <= 0:
                                continue
                            sum_n[tid] = sum_n.get(tid, 0) + n
                            acc = sum_hits.get(tid, {"top1": 0.0, "topL": 0.0, "prob": 0.0, "topL_cfg": int(hh.get("topL", 1))})
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
                        if router_eval_max_samples > 0 and n_eval >= router_eval_max_samples:
                            break
                    for tid in sorted(sum_n.keys()):
                        n = int(sum_n[tid])
                        if n <= 0:
                            continue
                        hs = sum_hits.get(tid, {})
                        ss = sum_stats.get(tid, {})
                        router_hit_val[int(tid)] = {
                            "top1_hit_rate": float(hs.get("top1", 0.0)) / float(n),
                            "topL_hit_rate": float(hs.get("topL", 0.0)) / float(n),
                            "topL": int(hs.get("topL_cfg", int(vista_state.topL))),
                            "n_samples": int(n),
                            "true_task_prob_mean": float(hs.get("prob", 0.0)) / float(n),
                        }
                        router_stats_val[int(tid)] = {
                            "res_best_mean": float(ss.get("res_best_mean", 0.0)) / float(n),
                            "res_gap_mean": float(ss.get("res_gap_mean", 0.0)) / float(n),
                            "entropy_mean": float(ss.get("entropy_mean", 0.0)) / float(n),
                        }
                    dt_eval = time.time() - t0_eval
                    logger.info(
                        f"[VISTA] Router eval (val) done: task_index={t_idx} task_id={int(task_id)} "
                        f"n_eval={int(n_eval)} time_sec={dt_eval:.1f} seen_tasks={seen_tasks_list}"
                    )
                except StopIteration:
                    # Router eval disabled; keep empty stats and continue training.
                    router_stats_val = {}
                    router_hit_val = {}
                except Exception as e:
                    # Router eval is diagnostic; do not block continual training. Record the error in logs.
                    logger.exception(f"[association vista] router eval on val failed at task_index={t_idx} task_id={task_id}: {e}")
                    router_stats_val = {}
                    router_hit_val = {}

                metrics_task_end = {
                    "task_index": t_idx,
                    "task_id": int(task_id),
                    "A_row": last_A_row,
                    "bar_A": last_overall_acc,
                    "Ego->Exo": float(last_direction_acc["Ego->Exo"]) if last_direction_acc is not None else None,
                    "Exo->Ego": float(last_direction_acc["Exo->Ego"]) if last_direction_acc is not None else None,
                    "vista": {
                        "enabled": True,
                        "router_M": int(vista_state.router_M),
                        "subspace_k": int(getattr(cfg, "vista_subspace_k", 32)),
                        "topL": int(vista_state.topL),
                        "gamma": float(vista_state.gamma),
                        "eps": float(vista_state.eps),
                        "apply_to_target": bool(vista_state.apply_to_target),
                        "router_fit_max_samples": int(getattr(cfg, "vista_router_fit_max_samples", 0)),
                        "router_eval_enabled": bool(getattr(cfg, "vista_router_eval_enabled", True)),
                        "router_eval_max_samples": int(getattr(cfg, "vista_router_eval_max_samples", 0)),
                        "router_stats_val": {str(k): v for k, v in router_stats_val.items()},
                        "router_hit_val": {str(k): v for k, v in router_hit_val.items()},
                    },
                }
                with open(os.path.join(task_dir, "metrics_task_end.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics_task_end, f, indent=2, ensure_ascii=False)

                # Save router eval separately (val)
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

                # Save router index (skill parity)
                try:
                    vista_state.router.save_index(output_dir=os.path.join(cfg.output, "router"))
                except Exception:
                    pass

        # ---- L2P: save pool snapshot at task end ----
        if l2p_pool is not None and dist_utils.is_main_process():
            try:
                l2p_dir = os.path.join(task_dir, "l2p")
                l2p_pool.save(l2p_dir, filename="l2p_pool.pt")
            except Exception:
                pass

        # ---- ER/DER++ memory update at task end (strict, task-balanced) ----
        if algo is not None and hasattr(algo, "capacity") and int(getattr(algo, "capacity", 0)) > 0:
            # Keep original batch sizing for ER/DER++ memory update (do not inherit EWC Fisher defaults).
            mem_loader = _build_memory_loader_for_task(args, cfg, tokenizer, allowed_videos, batch_size_total=int(cfg.train.batch_size))
            if len(mem_loader.dataset) <= 0:
                raise RuntimeError(f"[ER strict] Empty memory dataset for task_id={task_id}")
            if getattr(algo, "name", "") == "derpp":
                # Distill targets are per-sample embeddings: (image_embed, text_embed).
                def _distill_target_fn(batch_in, model_obj):
                    # batch_in is dict with tensors (video, text)
                    outputs = model_obj(
                        batch_in["video"],
                        batch_in["text"],
                        use_checkpoint=cfg.train.use_checkpoint,
                        norm_embed=cfg.model.norm_embed,
                    )
                    return (outputs["image_embed"], outputs["text_embed"])

                algo.update_memory_from_loader(task_id=int(task_id), loader=mem_loader, model=model, distill_target_fn=_distill_target_fn)
            else:
                algo.update_memory_from_loader(task_id=int(task_id), loader=mem_loader)

    # Save final A and bar_A
    if dist_utils.is_main_process():
        np.save(os.path.join(cfg.output, 'continual_A.npy'), final_A)
        np.save(os.path.join(cfg.output, 'continual_bar_A.npy'), final_bar_A)
        np.save(os.path.join(cfg.output, 'continual_direction_Ego2Exo.npy'), final_direction_A['Ego->Exo'])
        np.save(os.path.join(cfg.output, 'continual_direction_Exo2Ego.npy'), final_direction_A['Exo->Ego'])
        logger.info('Saved continual A matrix, bar_A, and direction-specific metrics to output directory')


def main():
    parser = argparse.ArgumentParser('EgoExoLearn Association Continual Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main_worker(args)


if __name__ == '__main__':
    main()

