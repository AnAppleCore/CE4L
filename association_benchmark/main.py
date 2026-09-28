# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import math
import os
import pandas as pd
import sys
import time
import pickle
import random 
from typing import Optional

import torch
import torch.backends.cudnn as cudnn
import torch.amp as amp
import torch.distributed as dist
import torch.nn.parallel
from omegaconf import OmegaConf
from copy import deepcopy
import numpy as np
import logging

from models.builder import build_model
from models import model_utils
from models.tokenizer import generate_tokenizer
from function.meter import AverageMeter, ProgressMeter
from function import distributed as dist_utils
from function.utils import build_train_loader, build_val_loader, build_optimizer, resume_checkpoint, build_scheduler
from function.config import get_config
from function.logger import get_logger

from clego_cl.vista import infer_vista_mix_from_inputs
from skill_benchmark.adapters import MixtureSpec

# Optional: continual algorithm plugin (e.g., ER) injected by continual runners.
# Must expose:
# - mix_in_replay(cur_batch=inputs_dict, cur_batch_size=int) -> merged inputs_dict
continual_algo = None

def get_args_parser():
    parser = argparse.ArgumentParser(description='EgoExoLearn Association training and evaluation', add_help=False)
    # Data
    parser.add_argument('--config', default='configs/default.yml', type=str)
    # Optional OmegaConf-style overrides
    # Example:
    #   --opts train.seed=0 output=./exps/foo/
    parser.add_argument('--opts', nargs='*', default=None,
                        help='Override config keys, e.g. train.seed=0 output=./exps/foo/')

    # System
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--testonly', action='store_true', help='whether to perform test only')
    return parser

# Optional: VISTA modules injected by continual runner.
vista_enabled = False
vista_state = None  # clego_cl.vista.VISTAState
vista_mode = "none"  # "train" | "infer" | "none"
vista_adapter_optimizer = None

# Optional: L2P modules injected by continual runner.
l2p_enabled = False
l2p_mode = "none"  # "train" | "infer" | "none"
l2p_pool = None  # clego_cl.l2p.L2PPool
l2p_topk = 2
l2p_router_M = 1
l2p_sim_lambda = 0.5
l2p_diversed_selection = True
l2p_batchwise_selection = False
l2p_optimizer = None


def _l2p_select_from_query(query_embed: torch.Tensor, *, training: bool):
    """Select top-k adapters using query embedding."""
    global l2p_enabled, l2p_mode, l2p_pool, l2p_router_M
    if not l2p_enabled or l2p_mode == "none" or l2p_pool is None:
        return None, torch.zeros((), device=query_embed.device, dtype=query_embed.dtype)
    from skill_benchmark.task_router import extract_r

    # treat embeddings as [B,1,D] for extract_r
    q3 = query_embed.unsqueeze(1)
    r = extract_r(q3, M=int(l2p_router_M))
    match = l2p_pool.cosine_match(r)
    sel = l2p_pool.select_topk(match, training=training)
    sim_loss = sel.match.mean()
    return sel, sim_loss


def _l2p_apply_embed(x: torch.Tensor, sel, *, repeat: Optional[int] = None) -> torch.Tensor:
    """Apply L2P adapters to embedding tensor [B,D] or [B,N,D]."""
    global l2p_pool
    if sel is None or l2p_pool is None:
        return x
    if x.dim() == 2:
        x3 = x.unsqueeze(1)
        y3 = l2p_pool.apply_adapters(x3, sel)
        return y3.squeeze(1)
    if x.dim() == 3 and repeat is not None:
        B, N, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        x3 = x.reshape(B * N, 1, D)
        task_ids = sel.indices.repeat_interleave(int(N), dim=0)
        match = sel.match.repeat_interleave(int(N), dim=0)
        sel_rep = type(sel)(indices=task_ids, match=match)
        y3 = l2p_pool.apply_adapters(x3, sel_rep)
        return y3.reshape(B, N, D)
    raise ValueError(f"Unsupported L2P embed shape: {tuple(x.shape)}")

def _vista_apply_train_embed(x: torch.Tensor) -> torch.Tensor:
    global vista_enabled, vista_mode, vista_state
    if not vista_enabled or vista_mode != "train" or vista_state is None or vista_state.adapter_bank is None:
        return x
    if x.dim() != 2:
        raise ValueError(f"VISTA expects embedding tensor [B,D], got shape={tuple(x.shape)}")
    x3 = x.unsqueeze(1)  # [B,1,D]
    y3 = vista_state.adapter_bank.forward_train(x3)
    return y3.squeeze(1)


@torch.no_grad()
def _vista_apply_mixture_embed(x: torch.Tensor, mix: MixtureSpec, *, repeat: Optional[int] = None) -> torch.Tensor:
    global vista_enabled, vista_state
    if not vista_enabled or vista_state is None or vista_state.adapter_bank is None:
        return x
    if x.dim() == 2:
        x3 = x.unsqueeze(1)
        y3 = vista_state.adapter_bank.forward_mixture(x3, mix)
        return y3.squeeze(1)
    if x.dim() == 3 and repeat is not None:
        B, N, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        x3 = x.reshape(B * N, 1, D)
        task_ids = mix.task_ids.repeat_interleave(int(N), dim=0)
        weights = mix.weights.repeat_interleave(int(N), dim=0)
        mix_rep = MixtureSpec(task_ids=task_ids, weights=weights)
        y3 = vista_state.adapter_bank.forward_mixture(x3, mix_rep)
        return y3.reshape(B, N, D)
    raise ValueError(f"Unsupported VISTA embed shape: {tuple(x.shape)}")


def _vista_infer_mix_from_query(query_embed: torch.Tensor) -> Optional[MixtureSpec]:
    global vista_enabled, vista_mode, vista_state
    if not vista_enabled or vista_mode != "infer" or vista_state is None or vista_state.router is None:
        return None
    if vista_state.router.num_tasks() <= 0 or vista_state.adapter_bank is None or vista_state.adapter_bank.num_tasks() <= 0:
        return None
    if query_embed.dim() != 2:
        raise ValueError(f"VISTA expects query embedding [B,D], got shape={tuple(query_embed.shape)}")
    if int(vista_state.router_M) != 1:
        raise ValueError("VISTA association uses embedding vectors; vista_router_M must be 1.")
    mix = infer_vista_mix_from_inputs(
        router=vista_state.router,
        x1=query_embed,
        x2=None,
        M=int(vista_state.router_M),
        topL=int(vista_state.topL),
        gamma=float(vista_state.gamma),
    )
    return mix


def random_seed(seed=42, rank=0):
    """Set random seed for reproducibility in distributed training."""
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main(args):
    ### Prepare env ###
    cfg = get_config(args)
    os.makedirs(cfg.output, exist_ok=True)
    
    dist_utils.init_distributed_mode(args)
    logger = get_logger(cfg)   
    ### save config file ###
    if dist_utils.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.yml')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config save to {path}')

    ### log config ###
    logger.info(OmegaConf.to_yaml(cfg))

    global best_acc1
    random_seed(cfg.train.seed, dist_utils.get_rank())
    logger.info(f'Creating model:{cfg.model.name}')
    model = build_model(cfg.model)

    if cfg.model.freeze_temperature:
        logger.info('Freeze logit temperature')
        if hasattr(model, 'logit_scale'):
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

    # optionally resume from a checkpoint (takes precedence over autoresume)
    loaded_resume = resume_checkpoint(cfg, model, optimizer, scaler, criterion)
    start_epoch, best_acc1 = loaded_resume['start_epoch'], loaded_resume['best_acc1']
    cudnn.benchmark = True

    logger.info("=> creating dataset")

    # Only load training data if not in test-only mode
    if not cfg.test.testonly:
        train_loader, train_sampler = build_train_loader(args, cfg, tokenizer)
        lr_schedule = build_scheduler(cfg, train_loader)
    else:
        lr_schedule = None

    egobridge_v2v_loader = build_val_loader(args, cfg, dataset_name='egobridge_v2v', tokenizer=deepcopy(tokenizer))

    # Synchronize all processes
    if args.distributed:
        dist.barrier()

    if cfg.test.testonly:
        ### V2V ###
        metrics = validate_v2v_mcq(egobridge_v2v_loader, model, use_half=False, cfg=cfg, args=args, logger=logger)
        logger.info(metrics)
        exit(0)

    best_metric = 0.
    logger.info("=> beginning training")
    for epoch in range(start_epoch, cfg.train.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg, logger)
        
        ### logging training stats ###
        for k, v in train_stats.items():
            logger.info(f'Epoch {epoch}: Train_{k}: {round(v, 3)}')

        ### saving per epoch model ckpt before evaluation ###
        logger.info('=> saving per-epoch checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
            'scaler': scaler.state_dict(),
            'best_acc1': best_metric,
            'cfg': cfg,
        }, False, cfg.output, is_epoch=True)

        logger.info('=> 0-shot on MCQ')
        v2v_metrics = validate_v2v_mcq(egobridge_v2v_loader, model, use_half=False, cfg=cfg, args=args, logger=logger)

        # Synchronize all processes before accessing metrics
        if dist_utils.is_dist_avail_and_initialized():
            dist.barrier()

        # Only main process has the full metrics, broadcast to all processes
        if dist_utils.is_main_process():
            logger.info('V2V Ego->Exo: {:.3f} | V2V Exo->Ego: {:.3f}'.format(v2v_metrics['Ego->Exo'], v2v_metrics['Exo->Ego']))
            avg_map = 0.5 * (v2v_metrics['Ego->Exo'] + v2v_metrics['Exo->Ego'])
        else:
            avg_map = 0.0

        # Broadcast avg_map to all processes
        if dist_utils.is_dist_avail_and_initialized():
            avg_map_tensor = torch.tensor([avg_map], dtype=torch.float32).cuda()
            dist.broadcast(avg_map_tensor, src=0)
            avg_map = avg_map_tensor.item()

        if avg_map > best_metric:
            is_best = True
            best_metric = avg_map
        else:
            is_best = False

        ### save checkpoint ###
        is_epoch = ((epoch + 1) % cfg.train.save_freq) == 0

        if args.distributed and cfg.train.use_zero:
            logger.info("=> consolidating state_dict before saving (due to ZeRO)")
            optimizer.consolidate_state_dict()

        logger.info('=> saving the best checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
            'scaler': scaler.state_dict(),
            'best_acc1': best_metric,
            'cfg': cfg,
        }, is_best, cfg.output, is_epoch=is_epoch)


def train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg, logger):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = model_utils.get_metric_names(cfg)
    
    iters_per_epoch = len(train_loader) // cfg.train.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        # ----------------------------------------------------
        # Continual algorithm hook: Experience Replay (inputs)
        # ----------------------------------------------------
        global continual_algo
        if continual_algo is not None and hasattr(continual_algo, "mix_in_replay"):
            try:
                # mix replay before using inputs['text'].size(0)
                cur_bs = int(inputs["text"].size(0)) if isinstance(inputs, dict) and "text" in inputs else None
                if cur_bs is None:
                    raise RuntimeError("Association train_one_epoch expects dict batch with key 'text' for ER.")
                inputs = continual_algo.mix_in_replay(cur_batch=inputs, cur_batch_size=cur_bs)
            except Exception as e:
                raise RuntimeError(f"[association train_one_epoch] continual_algorithm failed to mix replay at epoch={epoch} iter={data_iter}") from e
        optim_iter = data_iter // cfg.train.update_freq
                
        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]
        
        batch_size = inputs['text'].size(0)
        model_inputs = [inputs['video'].cuda(args.gpu), inputs['text'].cuda(args.gpu)]
        
        # compute output
        with amp.autocast('cuda', enabled=not cfg.train.disable_amp):
            outputs = model(
                *model_inputs,
                use_checkpoint=cfg.train.use_checkpoint,
                norm_embed=cfg.model.norm_embed
            )
            # ----------------------------------------------------
            # L2P hook (train): select top-K adapters by key-query match
            # ----------------------------------------------------
            l2p_sim_loss = torch.zeros((), device=outputs["image_embed"].device, dtype=outputs["image_embed"].dtype)
            if l2p_enabled and l2p_mode == "train":
                sel, l2p_sim_loss = _l2p_select_from_query(outputs["image_embed"], training=True)
                outputs["image_embed"] = _l2p_apply_embed(outputs["image_embed"], sel)
                outputs["text_embed"] = _l2p_apply_embed(outputs["text_embed"], sel)
            if vista_enabled and vista_mode == "train":
                outputs["image_embed"] = _vista_apply_train_embed(outputs["image_embed"])
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            if l2p_enabled and l2p_mode == "train":
                loss = loss + (float(l2p_sim_lambda) * l2p_sim_loss)
            # ----------------------------------------------------
            # Continual algorithm hook: DER++ distillation (embeds)
            # ----------------------------------------------------
            if continual_algo is not None and hasattr(continual_algo, "distill_loss"):
                try:
                    loss = loss + continual_algo.distill_loss((outputs["image_embed"], outputs["text_embed"]))
                except Exception as e:
                    raise RuntimeError(
                        f"[association train_one_epoch] continual_algorithm failed to compute distill loss at epoch={epoch} iter={data_iter}"
                    ) from e
            # ----------------------------------------------------
            # Continual algorithm hook: LwF distillation (embeds)
            # ----------------------------------------------------
            if continual_algo is not None and hasattr(continual_algo, "lwf_loss") and hasattr(continual_algo, "teacher"):
                try:
                    teacher = continual_algo.teacher()
                    if teacher is not None:
                        teacher = teacher.to(device=outputs["image_embed"].device)
                        with torch.no_grad():
                            t_outputs = teacher(
                                model_inputs[0],
                                model_inputs[1],
                                use_checkpoint=cfg.train.use_checkpoint,
                                norm_embed=cfg.model.norm_embed,
                            )
                        loss = loss + continual_algo.lwf_loss(outputs["image_embed"], t_outputs["image_embed"])
                        loss = loss + continual_algo.lwf_loss(outputs["text_embed"], t_outputs["text_embed"])
                except Exception as e:
                    raise RuntimeError(
                        f"[association train_one_epoch] continual_algorithm failed to compute LwF loss at epoch={epoch} iter={data_iter}"
                    ) from e
            # ----------------------------------------------------
            # Continual algorithm hook: EWC regularization
            # ----------------------------------------------------
            if continual_algo is not None and hasattr(continual_algo, "regularization_loss"):
                try:
                    loss = loss + continual_algo.regularization_loss()
                except Exception as e:
                    raise RuntimeError(
                        f"[association train_one_epoch] continual_algorithm failed to compute EWC loss at epoch={epoch} iter={data_iter}"
                    ) from e
            loss /= cfg.train.update_freq

        if not math.isfinite(loss.item()):
            logger.info("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % cfg.train.update_freq != 0:
            continue

        if cfg.train.clip_grad_value is not None:
            scaler.unscale_(optimizer)
            if cfg.train.clip_grad_type == 'norm':
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.clip_grad_value, norm_type=2.
                )
            elif cfg.train.clip_grad_type == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg.train.clip_grad_value)
            else:
                assert False, f"Unknown clip mode ({cfg.train.clip_grad_type})."
        # compute gradient and do SGD step
        # NOTE:
        # When VISTA freezes the backbone after task 1, the backbone optimizer may have
        # no grads at all. Newer PyTorch AMP asserts if `scaler.step(optimizer)` is
        # called without any recorded inf checks for that optimizer.
        def _optimizer_has_grads(opt) -> bool:
            for group in getattr(opt, "param_groups", []):
                for p in group.get("params", []):
                    if getattr(p, "grad", None) is not None:
                        return True
            return False

        if _optimizer_has_grads(optimizer):
            scaler.step(optimizer)
        if vista_adapter_optimizer is not None and _optimizer_has_grads(vista_adapter_optimizer):
            scaler.step(vista_adapter_optimizer)
        if l2p_optimizer is not None and _optimizer_has_grads(l2p_optimizer):
            scaler.step(l2p_optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)
        if vista_adapter_optimizer is not None:
            vista_adapter_optimizer.zero_grad(set_to_none=True)
        if l2p_optimizer is not None:
            l2p_optimizer.zero_grad(set_to_none=True)

        ### adjust logit scale ###
        if hasattr(dist_utils.get_model(model), 'logit_scale'):
            # clamp logit scale to [0, 100]
            dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = dist_utils.get_model(model).logit_scale.exp().item()
        else:
            logit_scale = torch.nan

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), cfg.train.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % cfg.train.print_freq == 0:
            if dist_utils.is_main_process():
                train_iter_log = {
                            'iter': data_iter,
                            **{k: round(v.item(), 3) for k, v in loss_dict.items()},
                           'scaler': round(scaler.get_scale(), 3), 
                           'logit': round(logit_scale, 3)}
                train_iter_log_str = ''
                for logk, logv in train_iter_log.items():
                    train_iter_log_str += f'{logk}:{logv}  '

                logger.info(train_iter_log_str)

    progress.synchronize()
    if l2p_enabled and l2p_mode == "train" and l2p_pool is not None:
        l2p_pool.update_frequency()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate_v2v_mcq(val_loader, model, use_half=False, cfg=None, args=None, logger=None):
    model.eval()
    if use_half:
        model.half()
    with torch.no_grad():
        if dist_utils.is_main_process():
            logger.info('=> validation v2v start forwarding')

        all_preds = []
        all_gts = []
        all_types = []
        all_uids = []
        end_time = time.time()

        for i, inputs in enumerate(val_loader):
            if dist_utils.is_main_process() and i % 10 == 0:
                logger.info('finish validation v2v batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()

            frame_query = inputs[0].cuda(non_blocking=True)
            frames_options = inputs[1].cuda(non_blocking=True)
            if use_half:
                frames_options = frames_options.half()

            answer = inputs[2]
            q_type = inputs[3]
            uid = inputs[-1]

            batch_size = frames_options.shape[0]
            frames_options = frames_options.view(-1, *frames_options.shape[2:])

            ### encode videos ###
            image_query_features = dist_utils.get_model(model).encode_image(frame_query)
            image_options_features = dist_utils.get_model(model).encode_image(frames_options)

            image_options_features = image_options_features.view(batch_size, -1, *image_options_features.shape[1:])

            all_gts.append(answer)
            # ----------------------------------------------------
            # L2P hook (infer): select top-K adapters by key-query match
            # ----------------------------------------------------
            sel, _ = _l2p_select_from_query(image_query_features, training=False)
            if sel is not None:
                image_query_features = _l2p_apply_embed(image_query_features, sel)
                image_options_features = _l2p_apply_embed(image_options_features, sel, repeat=image_options_features.shape[1])
            # ----------------------------------------------------
            # VISTA hook (infer): infer mixture from query embedding and apply to query/options
            # ----------------------------------------------------
            mix = _vista_infer_mix_from_query(image_query_features)
            if mix is not None:
                image_query_features = _vista_apply_mixture_embed(image_query_features, mix)
                image_options_features = _vista_apply_mixture_embed(image_options_features, mix, repeat=image_options_features.shape[1])
            all_types.append(q_type)
            all_uids.append(uid)
            for j in range(batch_size):
                similarity_matrix = torch.matmul(image_query_features[j], image_options_features[j].T)
                similarity_matrix = similarity_matrix.cpu().detach()
                all_preds.append(similarity_matrix)

        # Synchronize all processes before gathering results
        if dist_utils.is_dist_avail_and_initialized():
            dist.barrier()

        # Gather results from all processes
        if len(all_uids) > 0:
            all_uids = torch.cat(all_uids)
            all_preds = torch.stack(all_preds)
            all_gts = torch.cat(all_gts)
            all_types = torch.cat(all_types)

            # Gather from all processes
            if dist_utils.is_dist_avail_and_initialized():
                # Convert to tensors for gathering
                all_uids_list = [all_uids.cpu() for _ in range(dist_utils.get_world_size())]
                all_preds_list = [all_preds.cpu() for _ in range(dist_utils.get_world_size())]
                all_gts_list = [all_gts.cpu() for _ in range(dist_utils.get_world_size())]
                all_types_list = [all_types.cpu() for _ in range(dist_utils.get_world_size())]

                dist.all_gather_object(all_uids_list, all_uids.cpu())
                dist.all_gather_object(all_preds_list, all_preds.cpu())
                dist.all_gather_object(all_gts_list, all_gts.cpu())
                dist.all_gather_object(all_types_list, all_types.cpu())

                # Concatenate results from all processes on rank 0
                if dist_utils.is_main_process():
                    all_uids = torch.cat([u for u in all_uids_list if len(u) > 0])
                    all_preds = torch.cat([p for p in all_preds_list if len(p) > 0])
                    all_gts = torch.cat([g for g in all_gts_list if len(g) > 0])
                    all_types = torch.cat([t for t in all_types_list if len(t) > 0])

            # Compute metrics only on rank 0
            if dist_utils.is_main_process():
                metrics = egomcq_accuracy_metrics(all_preds, all_gts, all_types)
                logger.info(metrics)
                return metrics
            else:
                return {}
        else:
            return {}
    
def save_pred_results(uids, preds, gts):
    import pandas as pd
    all_data = []
    predictions = torch.max(preds, 1)[1]
    
    for i in range(len(uids)):
        uid = int(uids[i].cpu().numpy())
        prediction = int(predictions[i].cpu().numpy())
        gt = int(gts[i].cpu().numpy())
        all_data.append([uid, prediction, gt])
    
    df = pd.DataFrame(all_data, columns=['uid', 'pred', 'GT'])
    df.to_csv('pred.csv', index=0)

def egomcq_accuracy_metrics(preds, labels, types):
    metrics = {}
    type_list = torch.unique(types)
    group_list = ['Ego->Exo', 'Exo->Ego']
    for type_i, group_i in zip(type_list, group_list):
        correct = 0
        total = 0
        for pred, label, typer in zip(preds, labels, types):
            if typer == type_i:
                pred_ = torch.argmax(pred)
                if pred_.item() == label.item():
                    correct += 1
                total += 1
        accuracy = correct/total
        metrics[group_i] = accuracy * 100
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EgoExoLearn Association training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
