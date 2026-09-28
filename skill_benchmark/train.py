import argparse
import torch
import time, os
import shutil
from typing import Optional, Tuple

from dataset import SkillDataSet
from model import RAAN
from opts import parser, update_paths_from_args
from losses import RankingAttentionLoss 

from tensorboardX import SummaryWriter

# Optional writer for reuse from continual runner
writer = None

best_prec = 0

# Optional: continual algorithm plugin (e.g., ER) injected by continual runners.
# Must expose:
# - mix_in_replay(cur_batch=inputs, cur_batch_size=int) -> merged inputs
continual_algo = None

# Optional: VISTA modules injected by continual runner.
# We keep this file benchmark-local to avoid cross-benchmark assumptions.
vista_enabled = False
vista_mode = "none"  # "train" | "infer" | "none"
vista_adapter_bank = None  # skill_benchmark.adapters.AdapterBank
vista_router = None  # TaskWhitenedSubspaceRouter
vista_topL = 2
vista_gamma = 10.0
vista_router_M = 1
vista_adapter_optimizer = None  # optimizer for current task adapter params only

# Optional: L2P modules injected by continual runner.
l2p_enabled = False
l2p_mode = "none"  # "train" | "infer" | "none"
l2p_pool = None  # skill_benchmark.l2p.L2PPool
l2p_topk = 2
l2p_router_M = 1
l2p_sim_lambda = 0.5
l2p_diversed_selection = True
l2p_batchwise_selection = False
l2p_optimizer = None  # optimizer for L2P params (keys + adapters)


def _vista_apply_train(x: torch.Tensor) -> torch.Tensor:
    global vista_enabled, vista_mode, vista_adapter_bank
    if not vista_enabled or vista_mode != "train" or vista_adapter_bank is None:
        return x
    return vista_adapter_bank.forward_train(x)


def _l2p_apply_pair(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    x_exo: Optional[torch.Tensor] = None,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Apply L2P adapter mixture using key-query matching (train/test)."""
    global l2p_enabled, l2p_mode, l2p_pool, l2p_router_M
    if not l2p_enabled or l2p_mode == "none" or l2p_pool is None:
        return x1, x2, x_exo, torch.zeros((), device=x1.device, dtype=x1.dtype)

    from task_router import extract_r

    r1 = extract_r(x1, M=int(l2p_router_M))
    r2 = extract_r(x2, M=int(l2p_router_M))
    match1 = l2p_pool.cosine_match(r1)
    match2 = l2p_pool.cosine_match(r2)
    # Keep original per-sample pair behavior when batches align.
    B1 = int(match1.shape[0])
    B2 = int(match2.shape[0])
    B3 = int(x_exo.shape[0]) if x_exo is not None else 0
    aligned = (B1 == B2) and (x_exo is None or B3 == B1)
    if aligned:
        match = 0.5 * (match1 + match2)
        sel = l2p_pool.select_topk(match, training=training)
        x1m = l2p_pool.apply_adapters(x1, sel)
        x2m = l2p_pool.apply_adapters(x2, sel)
        x3m = l2p_pool.apply_adapters(x_exo, sel) if x_exo is not None else None
        sim_loss = sel.match.mean()
        return x1m, x2m, x3m, sim_loss

    # Fallback: when batch sizes differ, derive a shared top-K from batch-mean scores
    # and apply the same selected adapters to all inputs.
    from clego_cl.l2p import L2PSelection

    pooled = 0.5 * (match1.mean(dim=0) + match2.mean(dim=0))  # [P]
    pooled_match = pooled.view(1, -1).expand(B1 + B2 + B3, -1)  # [B1+B2(+B3), P]
    sel_all = l2p_pool.select_topk(pooled_match, training=training)

    idx1 = sel_all.indices[:B1]
    idx2 = sel_all.indices[B1 : B1 + B2]
    m1 = match1.gather(1, idx1)
    m2 = match2.gather(1, idx2)
    sel1 = L2PSelection(indices=idx1, match=m1)
    sel2 = L2PSelection(indices=idx2, match=m2)

    x1m = l2p_pool.apply_adapters(x1, sel1)
    x2m = l2p_pool.apply_adapters(x2, sel2)
    x3m = None
    if x_exo is not None:
        idx3 = sel_all.indices[B1 + B2 : B1 + B2 + B3]
        # exo match is not defined (no match3); reuse pooled scores for a stable sim term.
        m3 = pooled.view(1, -1).gather(1, idx3[:1]).expand(B3, -1) if B3 > 0 else pooled.new_zeros((0, int(idx1.shape[1])))
        sel3 = L2PSelection(indices=idx3, match=m3)
        x3m = l2p_pool.apply_adapters(x_exo, sel3)

    sim_loss = 0.5 * (m1.mean() + m2.mean())
    return x1m, x2m, x3m, sim_loss


def _vista_apply_infer_pair(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    x_exo: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Inference-time adapter mixture using task router residuals.

    - Router uses ONLY ego vid1/vid2 (x1/x2) to infer weights.
    - The same weights are applied to x1, x2, and (if provided) x_exo.
    """
    global vista_enabled, vista_mode, vista_adapter_bank, vista_router, vista_topL, vista_gamma, vista_router_M
    if not vista_enabled or vista_mode != "infer" or vista_adapter_bank is None or vista_router is None:
        return x1, x2, x_exo
    if vista_router.num_tasks() <= 0 or vista_adapter_bank.num_tasks() <= 0:
        return x1, x2, x_exo

    from adapters import MixtureSpec  # local import to avoid circulars
    from task_router import extract_r

    device = x1.device
    r1 = extract_r(x1, M=int(vista_router_M))
    r2 = extract_r(x2, M=int(vista_router_M))

    # Whitened-subspace routing: residual ratios for each ego sample, averaged, then top-L.
    e1, tids = vista_router.augmented_residual_scores(r1, device=device)
    e2, _ = vista_router.augmented_residual_scores(r2, device=device)
    e = 0.5 * (e1 + e2)
    p = torch.softmax((-float(vista_gamma) * e), dim=1)
    L = int(min(int(vista_topL), int(p.shape[1])))
    vals, idx = torch.topk(p, k=L, dim=1)
    tid_tensor = torch.tensor(tids, device=device, dtype=torch.long)
    task_ids = tid_tensor[idx]
    weights = vals / (vals.sum(dim=1, keepdim=True).clamp(min=1e-12))
    mix = MixtureSpec(task_ids=task_ids, weights=weights)

    x1m = vista_adapter_bank.forward_mixture(x1, mix)
    x2m = vista_adapter_bank.forward_mixture(x2, mix)
    x3m = vista_adapter_bank.forward_mixture(x_exo, mix) if x_exo is not None else None
    return x1m, x2m, x3m

def main():
    global args, best_prec, writer
    args = parser.parse_args()
    
    # Update paths based on actual CLI arguments (triplet_loss, relation_network)
    update_paths_from_args(args)

    print(args.run_folder)
    print('use gpu:', args.use_gpu_num)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu_num

    # print()

    writer = SummaryWriter('_'.join((args.run_folder, 'attention', str(args.attention), 'filters',
                                     str(args.num_filters), 'diversity', str(args.diversity_loss),
                                     str(args.lambda_param), 'disparity', str(args.disparity_loss),
                                     'rank_aware', str(args.rank_aware_loss), 'lr', str(args.lr))))

    if args.rank_aware_loss:
        num_attention_branches = 2
        models = {'pos': None, 'neg': None}
    else:
        num_attention_branches = 1
        models = {'att': None}
    for k in models.keys():
        models[k] = RAAN(args.num_samples, args.attention, args.num_filters, args.input_size).cuda() # 10, True, 3
    if args.disparity_loss or args.rank_aware_loss:
        model_uniform = RAAN(args.num_samples, attention=False, num_filters=1, input_size=args.input_size).cuda() # 10, False, 1

    # resume training
    if args.resume_train:
        print('resume training from %s' % args.resume_ckpt)
        checkpoint = torch.load(args.resume_ckpt)
        # print('checkpoint', checkpoint.keys())
        # print('epoch', checkpoint['epoch'])
        # print('best_prec', checkpoint['best_prec'])
        # print('pos', len(checkpoint['state_dict_pos']), checkpoint['state_dict_pos'][0].keys())
        # print('uni', len(checkpoint['state_dict_uniform']), checkpoint['state_dict_uniform'][0].keys())

        # print(models['pos'].state_dict().keys())
        # print(models['neg'].state_dict().keys())
        # print(model_uniform.state_dict().keys())

        # models['pos'].load_state_dict(checkpoint['state_dict_pos'][0])
        # models['neg'].load_state_dict(checkpoint['state_dict_neg'][0])
        # model_uniform.load_state_dict(checkpoint['state_dict_uniform'][0])


        best_prec = checkpoint['best_prec']
        best_epoch = checkpoint['epoch']
        for k in models.keys():
            models[k].load_state_dict(checkpoint['state_dict_' + k][0])
        if args.disparity_loss or args.rank_aware_loss:
            model_uniform.load_state_dict(checkpoint['state_dict_uniform'][0])
        args.start_epoch = checkpoint['epoch']
        print('best_prec', best_prec)
        print('best_epoch', best_epoch)
    else:
        print('training from scratch')

    train_loader = torch.utils.data.DataLoader(
        SkillDataSet(args.root_path, args.train_list, ftr_tmpl='{}_{}.npz', action_select=args.action_select, use_exo = args.use_exo, exo_root_path=args.exo_root_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        SkillDataSet(args.root_path, args.val_list, ftr_tmpl='{}_{}.npz', action_select=args.action_select, use_exo = args.use_exo, exo_root_path=args.exo_root_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    criterion = torch.nn.MarginRankingLoss(margin=args.m1).cuda()

    if args.disparity_loss or args.rank_aware_loss:
        attention_params = []
        model_params = []
        for model in models.values():
            for name, param in model.named_parameters():
                if param.requires_grad and 'att' in name:
                    attention_params.append(param)
                else:
                    model_params.append(param)
        optimizer = torch.optim.Adam(list(model_uniform.parameters()) + model_params, args.lr)
        optimizer_attention = torch.optim.Adam(attention_params, args.lr*0.1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if args.evaluate:
        validate(val_loader, models, criterion, 0, use_exo=args.use_exo, use_RN = args.relation_network)
        exit()

    phase = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.disparity_loss:
            phase = train_with_uniform(train_loader, models, model_uniform, criterion,
                                       optimizer, optimizer_attention,
                                       epoch, phase=phase, use_exo=args.use_exo, use_triplet_loss=args.triplet_loss, use_RN = args.relation_network)
        else:
            print('not implemented yet')
            exit()
            train(train_loader, models, criterion, optimizer, epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec = validate(val_loader, models, criterion, (epoch + 1), use_exo=args.use_exo, use_RN = args.relation_network)
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
            checkpoint_dict = {'epoch': epoch + 1, 'best_prec': best_prec}
            for k in models.keys():
                checkpoint_dict['state_dict_' + k] = models[k].state_dict(),
            if args.disparity_loss or args.rank_aware_loss:
                checkpoint_dict['state_dict_uniform'] = model_uniform.state_dict(),
            save_checkpoint(checkpoint_dict, is_best)
    writer.close()

def train(train_loader, models, criterion, optimizer, epoch, shuffle=True, phase=0):
    av_meters = {'batch_time': AverageMeter(), 'data_time': AverageMeter(), 'losses': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'diversity_losses': AverageMeter(),
                 'acc': AverageMeter()}
    model = models[models.keys()[0]]
    model.train()

    end = time.time()
    
    optimizer.zero_grad()
    for i, (input1, input2) in enumerate(train_loader):
        # measure data loading time
        av_meters['data_time'].update(time.time() - end)
        input_var1 = torch.autograd.Variable(input1.cuda(), requires_grad=True)
        input_var2 = torch.autograd.Variable(input2.cuda(), requires_grad=True)
        ## add small amount of gaussian noise to features for data augmentation
        if args.transform:
            input_var1, input_var2 = data_augmentation(input_var1, input_var2)
            
        labels = torch.ones(input1.size(0)).cuda()
        target  = torch.autograd.Variable(labels, requires_grad=False)

        output1, att1 = model(input_var1)
        output2, att2 = model(input_var2)
        
        ranking_loss = criterion(output1, output2, target)
        all_losses = ranking_loss
        if args.diversity_loss:
            div_loss_att1 = diversity_loss(att1)
            div_loss_att2 = diversity_loss(att2)
            all_losses += args.lambda_param*(div_loss_att1 + div_loss_att2)
            
        # measure accuracy and backprop
        prec = accuracy(output1.data, output2.data)

        all_losses.backward()

        optimizer.step()
        optimizer.zero_grad()

        # record losses
        av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0))
        if args.diversity_loss:
            av_meters['diversity_losses'].update(div_loss_att1.item() + div_loss_att2.item(),
                                                input1.size(0)*2)
        av_meters['losses'].update(all_losses.data.item(), input1.size(0))
        av_meters['acc'].update(prec, input1.size(0))

        # measure elapsed time
        av_meters['batch_time'].update(time.time() - end)
        end = time.time()

        if i % (args.print_freq) == 0:
            console_log_train(av_meters, epoch, i, len(train_loader), )

    tensorboard_log(av_meters, 'train', epoch) 

def train_with_uniform(train_loader, models, model_uniform, criterion, optimizer, optimizer_attention, epoch, shuffle=True, phase=0, use_exo=False, use_triplet_loss=False, use_RN=False):
    av_meters = {'batch_time': AverageMeter(), 'data_time': AverageMeter(), 'losses': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'ranking_losses_uniform': AverageMeter(),
                 'diversity_losses': AverageMeter(), 'disparity_losses': AverageMeter(),
                 'rank_aware_losses': AverageMeter(), 'acc': AverageMeter(), 'acc_uniform': AverageMeter()}
    if use_triplet_loss:
        av_meters['triplet_losses'] = AverageMeter()
    
    for k in models.keys():
        models[k].train()
    model_uniform.train()
    
    end = time.time()
    
    optimizer.zero_grad()
    for i, inputs in enumerate(train_loader):
        # ----------------------------------------------------
        # Continual algorithm hook: Experience Replay (inputs)
        # ----------------------------------------------------
        global continual_algo
        if continual_algo is not None and hasattr(continual_algo, "mix_in_replay"):
            try:
                cur_bs = int(inputs[0].size(0)) if isinstance(inputs, (tuple, list)) and torch.is_tensor(inputs[0]) else None
                if cur_bs is None:
                    # fallback for (input1,input2,input_exo) etc
                    if use_exo and isinstance(inputs, (tuple, list)) and len(inputs) >= 1 and torch.is_tensor(inputs[0]):
                        cur_bs = int(inputs[0].size(0))
                    else:
                        raise RuntimeError("Unsupported batch structure for ER in skill_benchmark/train_with_uniform")
                inputs = continual_algo.mix_in_replay(cur_batch=inputs, cur_batch_size=int(cur_bs))
            except Exception as e:
                raise RuntimeError(f"[skill train_with_uniform] continual_algorithm failed to mix replay at epoch={epoch} iter={i}") from e
        if use_exo:
            input1, input2, input_exo = inputs
        else:
            input1, input2 = inputs
        # measure data loading time
        av_meters['data_time'].update(time.time() - end)
        input_var1 = torch.autograd.Variable(input1.cuda(), requires_grad=True)
        input_var2 = torch.autograd.Variable(input2.cuda(), requires_grad=True)
        # print gpu memory usage
        # print(1, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())

        if use_exo:
            input_exo_var = torch.autograd.Variable(input_exo.cuda(), requires_grad=True)
        else:
            input_exo_var = None

        # print(2, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())



        ## add small amount of gaussian noise to features for data augmentation
        if args.transform:
            input_var1, input_var2 = data_augmentation(input_var1, input_var2)

        # ----------------------------------------------------
        # L2P hook (train): select top-K adapters by key-query match
        # ----------------------------------------------------
        l2p_sim_loss = torch.zeros((), device=input_var1.device, dtype=input_var1.dtype)
        global l2p_enabled, l2p_mode
        if l2p_enabled and l2p_mode == "train":
            input_var1, input_var2, input_exo_var, l2p_sim_loss = _l2p_apply_pair(
                input_var1, input_var2, x_exo=input_exo_var if use_exo else None, training=True
            )

        # ----------------------------------------------------
        # VISTA hook (train): apply ONLY current task adapter (no routing)
        # ----------------------------------------------------
        input_var1 = _vista_apply_train(input_var1)
        input_var2 = _vista_apply_train(input_var2)
        if use_exo:
            input_exo_var = _vista_apply_train(input_exo_var)
            
        labels = torch.ones(input1.size(0)).cuda()
        target = torch.autograd.Variable(labels, requires_grad=False)

        if use_RN:
            # print('before concat', input_var1.shape, input_var2.shape)
            input_var1 = torch.cat((input_var1, input_exo_var), dim=1)
            input_var2 = torch.cat((input_var2, input_exo_var), dim=1)
            # print('after concat', input_var1.shape, input_var2.shape)
            # exit()

        all_output1, all_output2, output1, output2, att1, att2 = {}, {}, {}, {}, {}, {}
        middle_feature1, middle_feature2 = {}, {}
        # print(3, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        for k in models.keys():
            # print(3.1, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            all_output1[k], att1[k], middle_feature1[k] = models[k](input_var1)
            all_output2[k], att2[k], middle_feature2[k] = models[k](input_var2)
            output1[k] = all_output1[k].mean(dim=1)
            output2[k] = all_output2[k].mean(dim=1)
            # print(3.2, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        # print(4, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        output1_uniform, _, _ = model_uniform(input_var1)
        output2_uniform, _, _ = model_uniform(input_var2)
        # print(5, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        output1_uniform = output1_uniform.mean(dim=1)
        output2_uniform = output2_uniform.mean(dim=1)
        # print(6, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())

        if use_exo and use_triplet_loss:
            all_output_exo, output_exo, att_exo = {}, {}, {}
            middle_feature_exo = {}
            for k in models.keys():
                _, _, middle_feature_exo[k] = models[k](input_exo_var)
                # output_exo[k] = all_output_exo[k].mean(dim=1)
            # output_exo_uniform, _ = model_uniform(input_exo_var)
            # output_exo_uniform = output_exo_uniform.mean(dim=1)

            total_triplet_loss = 0
            for k in models.keys():
                total_triplet_loss += triplet_loss_func(input_ego_better=middle_feature1[k], input_ego_worse=middle_feature2[k], input_exo=middle_feature_exo[k])
            # total_triplet_loss_uniform = triplet_loss_func(input_ego_better=output1_uniform, input_ego_worse=output2_uniform, input_exo=output_exo_uniform)



        

        ranking_loss = 0
        disparity_loss = 0
        for k in models.keys():
            ranking_loss += criterion(output1[k], output2[k], target)
            disparity_loss += multi_rank_loss(all_output1[k], all_output2[k], output1_uniform, output2_uniform, target, args.m2)
        ranking_loss_uniform = criterion(output1_uniform, output2_uniform, target)
        if args.rank_aware_loss:
            rank_aware_loss = multi_rank_loss(all_output1['pos'], all_output2['neg'], output1_uniform, output2_uniform, target, args.m3)

        # print(7, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())

        if args.diversity_loss:
            div_loss_att1, div_loss_att2 = 0, 0
            for k in models.keys():
                div_loss_att1 += diversity_loss(att1[k])
                div_loss_att2 += diversity_loss(att2[k])

        # print(8, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())

        all_losses = 0
        if phase == 0:
            all_losses += ranking_loss
            all_losses += ranking_loss_uniform
        else:
            all_losses += disparity_loss
            if args.rank_aware_loss:
                all_losses += rank_aware_loss
            if args.diversity_loss:
                all_losses += args.lambda_param*(div_loss_att1 + div_loss_att2)
            if use_exo and use_triplet_loss:
                all_losses += total_triplet_loss*0.1
                # print('total_triplet_loss', total_triplet_loss, 'disparity_loss', disparity_loss, 'rank_aware_loss', rank_aware_loss, 'diversity_loss', args.lambda_param*(div_loss_att1 + div_loss_att2))
        # L2P similarity loss (fixed weight)
        global l2p_sim_lambda
        if l2p_enabled and l2p_mode == "train":
            all_losses = all_losses + (float(l2p_sim_lambda) * l2p_sim_loss)
        # measure accuracy and backprop
        output1_all = torch.zeros(output1[list(models.keys())[0]].shape).cuda()
        output2_all = torch.zeros(output2[list(models.keys())[0]].shape).cuda()
        for k in models.keys():
            output1_all += output1[k]
            output2_all += output2[k]

        # ----------------------------------------------------
        # Continual algorithm hook: EWC regularization (if available)
        # ----------------------------------------------------
        if continual_algo is not None and hasattr(continual_algo, "regularization_loss"):
            try:
                all_losses = all_losses + continual_algo.regularization_loss()
            except Exception as e:
                raise RuntimeError(f"[skill train_with_uniform] continual_algorithm failed to compute regularization loss at epoch={epoch} iter={i}") from e

        # ----------------------------------------------------
        # Continual algorithm hook: LwF distillation (scores)
        # ----------------------------------------------------
        if continual_algo is not None and hasattr(continual_algo, "lwf_loss"):
            try:
                all_losses = all_losses + continual_algo.lwf_loss(
                    inputs=(input_var1, input_var2, input_exo_var),
                    student_z=(output1_all, output2_all),
                    models=models,
                    model_uniform=model_uniform,
                    use_exo=use_exo,
                    use_RN=use_RN,
                )
            except Exception as e:
                raise RuntimeError(f"[skill train_with_uniform] continual_algorithm failed to compute LwF loss at epoch={epoch} iter={i}") from e

        # ----------------------------------------------------
        # Continual algorithm hook: DER++ distillation (scheme A)
        # ----------------------------------------------------
        if continual_algo is not None and hasattr(continual_algo, "distill_loss"):
            try:
                all_losses = all_losses + continual_algo.distill_loss((output1_all, output2_all))
            except Exception as e:
                raise RuntimeError(f"[skill train_with_uniform] continual_algorithm failed to compute distill loss at epoch={epoch} iter={i}") from e

        # ----------------------------------------------------
        # Inf/NaN guard: skip this step if loss is non-finite
        # (prevents corrupting weights; should be a no-op in normal training)
        # ----------------------------------------------------
        try:
            if not torch.isfinite(all_losses.detach()).all().item():
                # Best-effort: log minimal context (avoid touching undefined vars)
                tl = None
                if use_exo and use_triplet_loss and ("total_triplet_loss" in locals()):
                    try:
                        tl = float(total_triplet_loss.detach().cpu().item())  # type: ignore[name-defined]
                    except Exception:
                        tl = None
                print(f"[WARN][skip-step] non-finite loss at epoch={epoch} iter={i} phase={phase} loss={all_losses} triplet_loss={tl}")

                # Clear grads and skip updates for this iteration
                try:
                    optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    optimizer.zero_grad()
                if optimizer_attention is not None:
                    try:
                        optimizer_attention.zero_grad(set_to_none=True)
                    except TypeError:
                        optimizer_attention.zero_grad()
                po = globals().get("vista_adapter_optimizer", None)
                if po is not None:
                    po.zero_grad(set_to_none=True)
                continue
        except Exception:
            # If the guard itself fails, fall back to original behavior.
            pass
        prec = accuracy(output1_all, output2_all)
        prec_uniform = accuracy(output1_uniform.data, output2_uniform.data)

        # print(9, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())

        all_losses.backward()

        # ----------------------------------------------------
        # VISTA hook: step adapter optimizer every iteration (disjoint from baseline optimizers)
        # ----------------------------------------------------
        global vista_adapter_optimizer
        if vista_adapter_optimizer is not None:
            vista_adapter_optimizer.step()
            vista_adapter_optimizer.zero_grad(set_to_none=True)
        # ----------------------------------------------------
        # L2P hook: step optimizer every iteration
        # ----------------------------------------------------
        global l2p_optimizer
        if l2p_optimizer is not None and l2p_enabled and l2p_mode == "train":
            l2p_optimizer.step()
            l2p_optimizer.zero_grad(set_to_none=True)

        # print(10, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        
        if phase == 0:
            optimizer.step()
            optimizer.zero_grad()
            phase = 1
        else:
            optimizer_attention.step()
            optimizer_attention.zero_grad()
            phase = 0

        # print(11, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())

        # record losses
        av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0)*len(models.keys()))
        av_meters['ranking_losses_uniform'].update(ranking_loss_uniform.item(), input1.size(0))
        av_meters['disparity_losses'].update(disparity_loss.item(), input1.size(0*len(models.keys())))
        if args.diversity_loss:
            av_meters['diversity_losses'].update(div_loss_att1.item() + div_loss_att2.item(),input1.size(0)*2*len(models.keys()))

        if args.rank_aware_loss:
            av_meters['rank_aware_losses'].update(rank_aware_loss.item(), input1.size(0))

        if use_triplet_loss:
            av_meters['triplet_losses'].update(total_triplet_loss.item(), input1.size(0)*len(models.keys()))

        av_meters['losses'].update(all_losses.data.item(), input1.size(0))
        av_meters['acc'].update(prec, input1.size(0))
        av_meters['acc_uniform'].update(prec_uniform, input1.size(0))

        # measure elapsed time
        av_meters['batch_time'].update(time.time() - end)
        end = time.time()

        if i % (args.print_freq) == 0:
            console_log_train(av_meters, epoch, i, len(train_loader), )

        # print(12, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())

    tensorboard_log_with_uniform(av_meters, 'train', epoch)
    # L2P: update selection frequency stats after each epoch
    if l2p_enabled and l2p_mode == "train" and l2p_pool is not None:
        l2p_pool.update_frequency()
    return phase

def validate(val_loader, models, criterion, epoch, use_exo=False, use_RN=False):
    av_meters = {'batch_time': AverageMeter(), 'losses': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'diversity_losses': AverageMeter(),
                 'acc': AverageMeter()}

    # switch to evaluate mode
    for k in models.keys():
        models[k].eval()

    end = time.time()
    for i, inputs in enumerate(val_loader):
        if use_exo:
            input1, input2, input_exo = inputs
        else:
            input1, input2 = inputs

        input_var1 = torch.autograd.Variable(input1.cuda())
        input_var2 = torch.autograd.Variable(input2.cuda())
        input_exo_var = None
        if use_exo:
            input_exo_var = torch.autograd.Variable(input_exo.cuda())

        # ----------------------------------------------------
        # L2P hook (infer): select top-K adapters by key-query match
        # ----------------------------------------------------
        global l2p_enabled, l2p_mode
        if l2p_enabled and l2p_mode == "infer":
            input_var1, input_var2, input_exo_var, _ = _l2p_apply_pair(
                input_var1, input_var2, x_exo=input_exo_var if use_exo else None, training=False
            )

        # ----------------------------------------------------
        # VISTA hook (infer): infer mixture weights from ego pair and mix adapters (task-id unknown)
        # ----------------------------------------------------
        input_var1, input_var2, input_exo_var = _vista_apply_infer_pair(input_var1, input_var2, x_exo=input_exo_var)

        if use_exo and use_RN:
            # print('before concat', input_var1.shape, input_var2.shape)
            input_var1 = torch.cat((input_var1, input_exo_var), dim=1)
            input_var2 = torch.cat((input_var2, input_exo_var), dim=1)
            # print('after concat', input_var1.shape, input_var2.shape)
            # exit()


        all_output1, all_output2, output1, output2, att1, att2 = {}, {}, {}, {}, {}, {}
        all_output_exo, output_exo, att_exo = {}, {}, {}
        for k in models.keys():
            all_output1[k], att1[k], _ = models[k](input_var1)
            all_output2[k], att2[k], _ = models[k](input_var2)
            output1[k] = all_output1[k].mean(dim=1)
            output2[k] = all_output2[k].mean(dim=1)



        labels = torch.ones(input1.size(0)).cuda()
        target = torch.autograd.Variable(labels)

        ranking_loss = 0
        for k in models.keys():
            ranking_loss += criterion(output1[k], output2[k], target)
        all_losses = ranking_loss
        if args.diversity_loss:
            div_loss_att1, div_loss_att2 = 0, 0
            for k in models.keys():
                div_loss_att1 += diversity_loss(att1[k])
                div_loss_att2 += diversity_loss(att2[k])
            all_losses += args.lambda_param*(div_loss_att1 + div_loss_att2)
        
        # measure accuracy
        # measure accuracy and backprop
        output1_all = torch.zeros(output1[list(models.keys())[0]].data.shape).cuda()
        output2_all = torch.zeros(output2[list(models.keys())[0]].data.shape).cuda()
        for k in models.keys():
            output1_all += output1[k].data
            output2_all += output2[k].data
        prec = accuracy(output1_all, output2_all)

        # record losses
        av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0))
        if args.diversity_loss:
            av_meters['diversity_losses'].update(div_loss_att1.item() + div_loss_att2.item(), input1.size(0)*2)
        av_meters['losses'].update(all_losses.data.item(), input1.size(0))
        av_meters['acc'].update(prec, input1.size(0))

        # measure elapsed time
        av_meters['batch_time'].update(time.time() - end)
        end = time.time()

        if i % (args.print_freq) == 0:
            console_log_test(av_meters, i, len(val_loader))

    print(('Testing Results: Acc {acc.avg:.4f} Loss {loss.avg:.5f}'
           .format(acc=av_meters['acc'], loss=av_meters['losses'])))
    tensorboard_log(av_meters, 'val', epoch)
    
    return av_meters['acc'].avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, 'attention', str(args.attention), 'filters',
                         str(args.num_filters), 'diversity', str(args.diversity_loss), 'disparity',
                         str(args.disparity_loss), 'rank_aware', str(args.rank_aware_loss),
                         str(args.lambda_param), 'lr', str(args.lr), filename))
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

class AverageMeter(object):
    """Compute and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output1, output2):
    """Computes the % of correctly ordered pairs"""
    pred1 = output1
    pred2 = output2
    correct = torch.gt(pred1, pred2)
    return float(correct.sum())/correct.size(0)

def diversity_loss(attention):
    attention_t = torch.transpose(attention, 1, 2)
    num_features = attention.shape[1]
    res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).cuda()
    res = res.view(-1, args.num_filters*args.num_filters)
    return torch.norm(res, p=2, dim=1).sum() / attention.size(0)

def multi_rank_loss(input_a_1, input_a_2, input_b_1, input_b_2, target, margin):
    inter1, _ = torch.min((input_a_1 - input_a_2), dim=1)
    inter2 = (input_b_1 - input_b_2)
    inter = -target * (inter1.view(-1) - inter2.view(-1)) + torch.ones(input_a_1.size(0)).cuda()*margin
    losses = torch.max(torch.zeros(input_a_1.size(0)).cuda(), inter)
    return losses.sum()/input_a_1.size(0)

def triplet_loss_func(input_ego_better, input_ego_worse, input_exo, margin=None):
    anchor = input_exo
    pos = input_ego_better
    neg = input_ego_worse

    if margin is None:  # if no margin assigned, use soft-margin
        Loss = torch.nn.SoftMarginLoss()

        if len(anchor.shape) == 1:
            num_samples = 1
            print('should not reach here')
            exit()
        else:
            num_samples = anchor.shape[0]
        y = torch.ones((num_samples, 1)).view(-1)
        if anchor.is_cuda: y = y.cuda()

        ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
        an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
        
        loss = Loss(an_dist - ap_dist, y)
    else:
        Loss = torch.nn.TripletMarginLoss(margin=margin, p=2)
        loss = Loss(anchor, pos, neg)

    return loss
    
def console_log_train(av_meters, epoch, iter, epoch_len):
    print(('Epoch: [{0}][{1}/{2}]\t'
           'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
           'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
           'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
           'Prec: {acc.val:.3f} ({acc.avg:.3f})'.format(
               epoch, iter, epoch_len, batch_time=av_meters['batch_time'],
               data_time=av_meters['data_time'], loss=av_meters['losses'],
               acc=av_meters['acc'])))

def console_log_test(av_meters, iter, test_len):
    print(('Test: [{0}/{1}\t'
           'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
           'Prec {acc.val} ({acc.avg:.3f})'.format(
               iter, test_len, batch_time=av_meters['batch_time'], loss=av_meters['losses'],
               acc=av_meters['acc'])))

def tensorboard_log(av_meters, mode, epoch):
    if writer is None:
        return
    writer.add_scalar(mode+'/total_loss', av_meters['losses'].avg, epoch)
    writer.add_scalar(mode+'/ranking_loss', av_meters['ranking_losses'].avg, epoch)
    writer.add_scalar(mode+'/diversity_loss', av_meters['diversity_losses'].avg, epoch)
    writer.add_scalar(mode+'/acc', av_meters['acc'].avg, epoch)

def tensorboard_log_with_uniform(av_meters, mode, epoch):
    tensorboard_log(av_meters, mode, epoch)
    if writer is None:
        return
    writer.add_scalar(mode+'/disparity_loss', av_meters['disparity_losses'].avg, epoch)
    writer.add_scalar(mode+'/ranking_loss_uniform', av_meters['ranking_losses_uniform'].avg, epoch)
    writer.add_scalar(mode+'/acc_uniform', av_meters['acc_uniform'].avg, epoch)
    writer.add_scalar(mode+'/rank_aware_loss', av_meters['rank_aware_losses'].avg, epoch)

def data_augmentation(input_var1, input_var2):
    noise = torch.autograd.Variable(torch.normal(torch.zeros(input_var1.size()[1],
                                                             input_var1.size()[2]),
                                                 0.01)).cuda()
    input_var1 = torch.add(input_var1, noise)
    input_var2 = torch.add(input_var2, noise)
    return input_var1, input_var2
    
if __name__ == '__main__':
    main()


    
