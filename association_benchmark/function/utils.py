import json
import os
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from ipdb import set_trace as st
from torch.distributed.optim import ZeroRedundancyOptimizer

from . import distributed as dist_utils
from .logger import get_logger
from .scheduler import cosine_scheduler
from data import *
from data.video_transforms import Permute, SpatialCrop, TemporalCrop

def gather(tensor, args):
    output = [torch.empty_like(tensor) for _ in range(args.world_size)]
    dist.all_gather(output, tensor)
    return torch.cat(output, 0)

def gather_obj(obj_list, args):
    output = [None for _ in range(args.world_size)]
    dist.all_gather_object(output, obj_list) 
    output = sum(output, []) ## convert the 2d list to 1d list
    return output

def gather_obj_debug(obj_list, gpus=1):
    output = [None for _ in range(gpus)]
    dist.all_gather_object(output, obj_list) 
    output = sum(output, []) ## convert the 2d list to 1d list
    return output
    
def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        #args.rank = int(os.environ['SLURM_PROCID'])
        #args.gpu = args.rank % torch.cuda.device_count()
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = os.environ['SLURM_NTASKS']
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list)
        )
        master_port = os.environ.get('MASTER_PORT', '29484')
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = int(ntasks)
        args.rank = int(proc_id)
        args.gpu = int(proc_id % num_gpus)
        print(f'SLURM MODE: proc_id: {proc_id}, ntasks: {ntasks}, node_list: {node_list}, num_gpus:{num_gpus}, addr:{addr}, master port:{master_port}' )
        
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_distributed_training_run() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )


def convert_to_multiview_checkpoint(model, checkpoint):
    new_checkpoint = {}
    loaded_checkpoint = checkpoint['state_dict']
    load_success = {}

    ### change them temporally ###
    blocks_map = {
        'blocks.0.': 'blocks.98.',
        'blocks.1.': 'blocks.99.',
    }

    for k, v in model.state_dict().items():
        old_k = k
        if old_k in loaded_checkpoint:
            new_checkpoint[k] = loaded_checkpoint[old_k]
            continue

        if '.0.' in k or '.1.' in k:
            flag = False
            if 'blocks.0.' in k or 'blocks.1.' in k:
                flag = True
                for temp_k, temp_v in blocks_map.items():
                    old_k = old_k.replace(temp_k, temp_v)
            
            old_k = old_k.replace('.0.','.')
            old_k = old_k.replace('.1.','.') 
            if flag:
                for temp_k, temp_v in blocks_map.items():
                    old_k = old_k.replace(temp_v, temp_k)   
        
        if 'cross_attn' in old_k or 'global_encoder' in old_k:
            continue

        new_checkpoint[k] = loaded_checkpoint[old_k]
        load_success[k] = True

    return new_checkpoint

def resume_checkpoint(cfg, model, optimizer, scaler, criterion):
    start_epoch = 0
    best_acc1 = 0.0
    latest = os.path.join(cfg.output, 'checkpoint.pt')
    use_latest = False
    if os.path.isfile(latest):
        use_latest = True
        
    logger = get_logger(cfg)
    if use_latest:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(cfg.output, 'checkpoint.pt')
        ### if checkpoint.pt does not exists, auto-resume the best-checkpoint ###
        latest = latest if os.path.isfile(latest) else latest.replace('checkpoint.pt', 'checkpoint_best.pt')

        if os.path.isfile(latest):
            logger.info("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
            start_epoch = int(latest_checkpoint['epoch'])
            res = model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            
            logger.info(f'loading latest checkpoint:\n{res}')
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            logger.info("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    elif cfg.resume:
        resume_path = cfg.resume
        # Support passing a directory (common in this repo's scripts/configs): load checkpoint within it.
        if os.path.isdir(resume_path):
            # Prefer best checkpoint for evaluation-style resumes; fall back to latest.
            ckpt_best = os.path.join(resume_path, 'checkpoint_best.pt')
            ckpt_latest = os.path.join(resume_path, 'checkpoint.pt')
            if os.path.isfile(ckpt_best):
                resume_path = ckpt_best
            elif os.path.isfile(ckpt_latest):
                resume_path = ckpt_latest
            else:
                # Keep the directory path so the error message points to the intended location.
                resume_path = cfg.resume

        if os.path.isfile(resume_path):
            logger.info("=> loading resume checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            ### this trains with xattn
            if 'MULTIVIEW' in cfg.model.name:
                checkpoint = convert_to_multiview_checkpoint(model, checkpoint)
           
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

            
            if 'state_dict' in checkpoint:
                result = model.load_state_dict(checkpoint['state_dict'], strict=False) 
            elif 'model' in checkpoint:
                result = model.load_state_dict(checkpoint['model'], strict=False)
            else:
                is_ddp_checkpoint = False
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        is_ddp_checkpoint = True
                    break
                result = model.load_state_dict(checkpoint, strict=False) if is_ddp_checkpoint else model.module.load_state_dict(checkpoint, strict=False)

            logger.info(result)  

            ##### for initializing model, we do not need to load these params #####
            # optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            # scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            # criterion.load_state_dict(checkpoint['criterion']) if 'criterion' in checkpoint else ()
            # best_acc1 = checkpoint['best_acc1'] if 'best_acc1' in checkpoint else .0
            logger.info("=> loaded resume checkpoint '{}' (epoch {})".format(resume_path, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))
    else:
        print("=>No resumed checkpoint and no trained checkpoints")  
        
    return {
        'start_epoch': start_epoch,
        'best_acc1': best_acc1,
    }

def build_optimizer(cfg, model, criterion):
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    for n, p in criterion.named_parameters():
        if not p.requires_grad:
            continue
        p_non_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": cfg.optimizer.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    if cfg.use_zero:
        optimizer = ZeroRedundancyOptimizer(
            optim_params, optimizer_class=torch.optim.AdamW,
            lr=cfg.lr, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.wd
        )
    else:
        optimizer = torch.optim.AdamW(optim_params, lr=cfg.lr, betas=cfg.optimizer.betas,
                                      eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.wd)
    return optimizer


def build_scheduler(cfg, train_loader=None):
    """Build learning rate scheduler.

    Args:
        cfg: Configuration object
        train_loader: Training data loader (optional, used to calculate iterations per epoch)

    Returns:
        Learning rate schedule or None
    """
    if cfg.train.fix_lr:
        lr_schedule = None
    else:
        if train_loader is None:
            # If train_loader not provided, use a default estimate
            iters_per_epoch = 1000  # Default fallback
        else:
            iters_per_epoch = len(train_loader) // cfg.train.update_freq

        lr_schedule = cosine_scheduler(
            cfg.train.lr, cfg.train.lr_end, cfg.train.epochs, iters_per_epoch,
            warmup_epochs=cfg.train.warmup_epochs, start_warmup_value=cfg.train.lr_start,
        )
    return lr_schedule


def build_train_loader(args, cfg, tokenizer):
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
    )
    print('loading data')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    batch_size_per_gpu = cfg.train.batch_size // args.world_size
    if cfg.train.batch_size % args.world_size != 0:
        print(f'Warning: batch_size ({cfg.train.batch_size}) is not divisible by world_size ({args.world_size})')
        print(f'Using batch_size_per_gpu = {batch_size_per_gpu}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_per_gpu, shuffle=(train_sampler is None), # collate_fn = collate,
        num_workers=cfg.train.workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )

    print(f'len(train_loader) = {len(train_loader)}, batch_size_per_gpu = {batch_size_per_gpu}, world_size = {args.world_size}')
    return train_loader, train_sampler


def build_val_transform(cfg, model_name):
    ### hard code this 224 for now ###
    crop_size = 224
    
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),  # T H W C -> C T H W
        transforms.Resize((crop_size, crop_size)),
        transforms_video.NormalizeVideo(mean=[122.7709393, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]),
    ])
    
    return val_transform

def build_val_loader(args, cfg, dataset_name='youcook', tokenizer=None):
    val_dataset = OurDataset(
        cfg = cfg.test.ourdata,
        transform=build_val_transform(cfg.test.ourdata, cfg.model.name),
        is_training=False,
        tokenizer=tokenizer,
    )

    # Use DistributedSampler for validation to avoid duplicate computation
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            drop_last=False
        )
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    batch_size_per_gpu = cfg.test.batch_size // args.world_size
    if cfg.test.batch_size % args.world_size != 0:
        print(f'Warning: val batch_size ({cfg.test.batch_size}) is not divisible by world_size ({args.world_size})')
        print(f'Using val batch_size_per_gpu = {batch_size_per_gpu}')

    dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_per_gpu, shuffle=False,
        num_workers=cfg.test.workers, pin_memory=True, sampler=val_sampler, drop_last=False,
    )

    print('{} ==> len(val_dataset)={},len(val_dataloader)={},batch_size_per_gpu={}'.format(dataset_name, len(val_dataset), len(dataloader), batch_size_per_gpu))
    return dataloader

        