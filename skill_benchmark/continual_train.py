import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

# Ensure repo root on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.continual_recorder import ContinualRecorder
from clego_cl.task_stats import UnknownTracker, save_task_stats, save_unknown_ids
from clego_cl.task_order import save_task_order
from clego_cl.continual_algorithms import build_continual_algorithm

from dataset import SkillDataSet
from model import RAAN
from opts import parser as base_parser, update_paths_from_args

import train as base

from clego_cl.config_utils import load_yaml_config, split_argv_config, apply_config_as_defaults

from adapters import AdapterBank
from l2p import L2PPool
from task_router import TaskWhitenedSubspaceRouter, extract_r


def main() -> None:
    cfg_path, remaining = split_argv_config(sys.argv[1:])
    parser = argparse.ArgumentParser(parents=[base_parser], add_help=False)
    parser.add_argument("--config", type=str, default=None, help="YAML config file (optional). CLI args override config.")
    p = parser
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--dry_run", action="store_true", help="Only write task_order/task_stats then exit.")
    p.add_argument("--epochs_per_task", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_tasks", type=int, default=None, help="Optional cap on number of tasks (after ordering).")
    # Default: randomize task order (but keep the fixed task set).
    # Use --no_randomize_order to reproduce the legacy fixed order behavior.
    p.add_argument("--randomize_order", action="store_true", default=True, help="Shuffle task order using --seed.")
    p.add_argument("--no_randomize_order", action="store_true", help="Disable task order randomization (legacy fixed order).")
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
    # VISTA: task subspace router + per-task adapters (task-id unknown at inference)
    # ------------------------------
    p.add_argument("--vista_enabled", action="store_true", default=False, help="Enable VISTA (whitened-subspace router + per-task adapters).")
    p.add_argument("--vista_router_M", type=int, default=1, help="Time-chunk pooling segments M for router representation r(x). Default 1 (global mean).")
    p.add_argument("--vista_subspace_k", type=int, default=32, help="Whitened PCA subspace dimension per task.")
    p.add_argument("--vista_topL", type=int, default=2, help="Inference-time top-L tasks to mix adapters (default: 2).")
    p.add_argument("--vista_gamma", type=float, default=10.0, help="Softmax temperature for residual-based routing (default: 10).")
    p.add_argument("--vista_eps", type=float, default=1e-6, help="Epsilon for normalized residuals (default: 1e-6).")
    p.add_argument("--vista_adapter_bottleneck", type=int, default=64, help="Adapter bottleneck dim r (default: 64).")
    p.add_argument(
        "--vista_train_raan_after_task1",
        action="store_true",
        default=False,
        help="If set, keep training RAAN after task 1. Default false => freeze RAAN after task 1 when VISTA is enabled.",
    )
    p.add_argument("--vista_save_router_stats", action="store_true", default=True, help="Save router confidence stats for gamma tuning (default: true).")
    # ------------------------------
    # L2P: fixed prompt pool (adapters + learnable keys)
    # ------------------------------
    p.add_argument("--l2p_enabled", action="store_true", default=False, help="Enable L2P (fixed adapter pool + key-query selection).")
    p.add_argument("--l2p_pool_size", type=int, default=4, help="L2P: prompt pool size (default: 4).")
    p.add_argument("--l2p_topK", type=int, default=2, help="L2P: top-K adapters selected per sample (default: 2).")
    p.add_argument("--l2p_router_M", type=int, default=1, help="L2P: time-chunk pooling segments M for query representation (default: 1).")
    p.add_argument("--l2p_adapter_bottleneck", type=int, default=64, help="L2P: adapter bottleneck dim r (default: 64).")
    p.add_argument("--l2p_sim_lambda", type=float, default=0.5, help="L2P: similarity loss weight (default fixed: 0.5).")
    p.add_argument("--l2p_diversed_selection", action="store_true", default=True, help="L2P: enable frequency-based diversified selection during training.")
    p.add_argument("--l2p_batchwise_selection", action="store_true", default=False, help="L2P: batch-wise top-K selection (optional).")
    if cfg_path is not None:
        cfg = load_yaml_config(cfg_path)
        apply_config_as_defaults(parser, cfg)
    args = p.parse_args(remaining)
    if getattr(args, "no_randomize_order", False):
        args.randomize_order = False
    if not args.output_root:
        raise ValueError("--output_root must be provided via CLI or --config")

    # Make train.py helper functions use our args and disable tensorboard in continual runner
    base.args = args
    base.writer = None

    # skill task definition (action-incremental)
    all_task_ids: List[int] = [1, 2, 3, 4]
    task_to_actions = {1: "18", 2: "06", 3: "20", 4: "13,14,15"}
    # Task order can be randomized to support multi-seed continual stats (like association benchmark).
    from clego_cl.task_order import make_task_order
    task_order: List[int] = make_task_order(
        all_task_ids=all_task_ids,
        num_tasks=args.num_tasks,
        randomize=bool(getattr(args, "randomize_order", False)),
        seed=int(args.seed),
    )

    os.makedirs(args.output_root, exist_ok=True)
    save_task_order(os.path.join(args.output_root, "task_order.json"), task_order)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Keep baseline path updates for internal flags (does not affect our output_root usage)
    update_paths_from_args(args)

    # Build models
    if args.rank_aware_loss:
        models = {"pos": None, "neg": None}
    else:
        models = {"att": None}

    for k in models.keys():
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

    # Stats + recorder
    unknown = UnknownTracker(max_examples=200)
    stats = {
        "benchmark": "skill_benchmark",
        "task_order": task_order,
        "task_to_actions": {str(k): v for k, v in task_to_actions.items()},
        "micro_weight_unit": "num_pairs_in_eval_split",
        "splits": {"train": {}, "val": {}},
    }

    rec = ContinualRecorder(task_order=task_order)

    # ------------------------------
    # VISTA: initialize adapter bank + router bank (built progressively per task)
    # ------------------------------
    vista_adapter_bank = None
    vista_router = None
    if bool(getattr(args, "vista_enabled", False)):
        # Determine input_dim: in RN mode, ego and exo are concatenated along time dim, but channel dim stays input_size.
        input_dim = int(getattr(args, "input_size", 1024))
        vista_adapter_bank = AdapterBank(input_dim=input_dim, bottleneck=int(getattr(args, "vista_adapter_bottleneck", 64)), use_layernorm=True).cuda()
        vista_router = TaskWhitenedSubspaceRouter(
            M=int(getattr(args, "vista_router_M", 1)),
            k=int(getattr(args, "vista_subspace_k", 32)),
            eps=float(getattr(args, "vista_eps", 1e-6)),
        )
        # Inject into baseline train.py as module-globals
        base.vista_enabled = True
        base.vista_adapter_bank = vista_adapter_bank
        base.vista_router = vista_router
        base.vista_topL = int(getattr(args, "vista_topL", 2))
        base.vista_gamma = float(getattr(args, "vista_gamma", 10.0))
        base.vista_router_M = int(getattr(args, "vista_router_M", 1))

    # ------------------------------
    # L2P: initialize fixed adapter+key pool
    # ------------------------------
    l2p_pool = None
    if bool(getattr(args, "l2p_enabled", False)):
        input_dim = int(getattr(args, "input_size", 1024))
        key_dim = int(getattr(args, "l2p_router_M", 1)) * int(input_dim)
        # Bind pool size to total number of tasks
        pool_size = int(len(task_order))
        l2p_pool = L2PPool(
            pool_size=pool_size,
            topk=int(getattr(args, "l2p_topK", 2)),
            adapter_dim=int(input_dim),
            key_dim=int(key_dim),
            adapter_bottleneck=int(getattr(args, "l2p_adapter_bottleneck", 64)),
            diversed_selection=bool(getattr(args, "l2p_diversed_selection", True)),
            batchwise_selection=bool(getattr(args, "l2p_batchwise_selection", False)),
        ).cuda()
        base.l2p_enabled = True
        base.l2p_pool = l2p_pool
        base.l2p_topk = int(getattr(args, "l2p_topK", 2))
        base.l2p_router_M = int(getattr(args, "l2p_router_M", 1))
        base.l2p_sim_lambda = float(getattr(args, "l2p_sim_lambda", 0.5))
        base.l2p_diversed_selection = bool(getattr(args, "l2p_diversed_selection", True))
        base.l2p_batchwise_selection = bool(getattr(args, "l2p_batchwise_selection", False))
        base.l2p_mode = "train"
        base.l2p_optimizer = torch.optim.Adam(l2p_pool.parameters(), lr=float(getattr(args, "lr", 1e-4)))
    else:
        base.l2p_enabled = False
        base.l2p_pool = None
        base.l2p_mode = "none"
        base.l2p_optimizer = None

    # ------------------------------
    # Continual algorithm (ER/DERPP/EWC/LwF)
    # ------------------------------
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
        if algo_name in ("ewc", "lwf") and (bool(getattr(args, "vista_enabled", False)) or bool(getattr(args, "l2p_enabled", False))):
            raise ValueError("EWC/LwF are not supported with VISTA or L2P in skill_benchmark (disable vista_enabled/l2p_enabled).")
        # Capacity based on total #pairs in the task-filtered training set across the selected task_order.
        if hasattr(algo, "configure_total_capacity"):
            total_train = 0
            for tid in task_order:
                act_sel = task_to_actions[int(tid)]
                ds = SkillDataSet(
                    args.root_path,
                    args.train_list,
                    action_select=act_sel,
                    use_exo=args.use_exo,
                    exo_root_path=args.exo_root_path,
                )
                total_train += int(len(ds))
            algo.configure_total_capacity(total_train_samples=int(total_train))
        # Bind models if needed (EWC uses this for regularization).
        if hasattr(algo, "bind_models"):
            algo.bind_models(models=models, model_uniform=model_uniform)
        # Inject into baseline train.py (train_with_uniform reads module-global)
        base.continual_algo = algo

    if args.dry_run:
        stats.update(unknown.to_dict())
        stats["unknown_ratio"] = 0.0
        save_task_stats(args.output_root, stats)
        save_unknown_ids(args.output_root, unknown)
        return

    def build_loader(list_path: str, action_select: str, shuffle: bool) -> torch.utils.data.DataLoader:
        ds = SkillDataSet(
            args.root_path,
            list_path,
            ftr_tmpl="{}_{}.npz",
            action_select=action_select,
            use_exo=args.use_exo,
            exo_root_path=args.exo_root_path,
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
        )

    def _ewc_loss_from_batch(inputs) -> torch.Tensor:
        """Compute phase0/phase1 averaged loss for EWC Fisher estimation."""
        if args.use_exo:
            input1, input2, input_exo = inputs
        else:
            input1, input2 = inputs
            input_exo = None
        input_var1 = torch.autograd.Variable(input1.cuda(non_blocking=True), requires_grad=True)
        input_var2 = torch.autograd.Variable(input2.cuda(non_blocking=True), requires_grad=True)
        input_exo_var = None
        if args.use_exo and input_exo is not None:
            input_exo_var = torch.autograd.Variable(input_exo.cuda(non_blocking=True), requires_grad=True)

        if args.transform:
            input_var1, input_var2 = base.data_augmentation(input_var1, input_var2)

        labels = torch.ones(input1.size(0)).cuda()
        target = torch.autograd.Variable(labels, requires_grad=False)

        if args.relation_network and args.use_exo and input_exo_var is not None:
            input_var1 = torch.cat((input_var1, input_exo_var), dim=1)
            input_var2 = torch.cat((input_var2, input_exo_var), dim=1)

        all_output1, all_output2, output1, output2, att1, att2 = {}, {}, {}, {}, {}, {}
        middle_feature1, middle_feature2 = {}, {}
        for k in models.keys():
            all_output1[k], att1[k], middle_feature1[k] = models[k](input_var1)
            all_output2[k], att2[k], middle_feature2[k] = models[k](input_var2)
            output1[k] = all_output1[k].mean(dim=1)
            output2[k] = all_output2[k].mean(dim=1)

        output1_uniform, _, _ = model_uniform(input_var1)
        output2_uniform, _, _ = model_uniform(input_var2)
        output1_uniform = output1_uniform.mean(dim=1)
        output2_uniform = output2_uniform.mean(dim=1)

        total_triplet_loss = None
        if args.use_exo and args.triplet_loss and input_exo_var is not None:
            middle_feature_exo = {}
            for k in models.keys():
                _, _, middle_feature_exo[k] = models[k](input_exo_var)
            total_triplet_loss = 0
            for k in models.keys():
                total_triplet_loss += base.triplet_loss_func(
                    input_ego_better=middle_feature1[k],
                    input_ego_worse=middle_feature2[k],
                    input_exo=middle_feature_exo[k],
                )

        ranking_loss = 0
        disparity_loss = 0
        for k in models.keys():
            ranking_loss += criterion(output1[k], output2[k], target)
            disparity_loss += base.multi_rank_loss(all_output1[k], all_output2[k], output1_uniform, output2_uniform, target, args.m2)
        ranking_loss_uniform = criterion(output1_uniform, output2_uniform, target)

        rank_aware_loss = None
        if args.rank_aware_loss:
            rank_aware_loss = base.multi_rank_loss(all_output1["pos"], all_output2["neg"], output1_uniform, output2_uniform, target, args.m3)

        div_loss_att1, div_loss_att2 = 0, 0
        if args.diversity_loss:
            for k in models.keys():
                div_loss_att1 += base.diversity_loss(att1[k])
                div_loss_att2 += base.diversity_loss(att2[k])

        loss_phase0 = ranking_loss + ranking_loss_uniform
        loss_phase1 = disparity_loss
        if args.rank_aware_loss and rank_aware_loss is not None:
            loss_phase1 += rank_aware_loss
        if args.diversity_loss:
            loss_phase1 += args.lambda_param * (div_loss_att1 + div_loss_att2)
        if total_triplet_loss is not None:
            loss_phase1 += total_triplet_loss * 0.1

        return 0.5 * (loss_phase0 + loss_phase1)

    # A0: eval before training for each task val subset
    a0_vals: Dict[int, float] = {}
    a0_weights: Dict[int, float] = {}
    for tid in task_order:
        act_sel = task_to_actions[tid]
        val_loader = build_loader(args.val_list, act_sel, shuffle=False)
        stats["splits"]["val"][str(tid)] = int(len(val_loader.dataset))
        a0_weights[tid] = float(len(val_loader.dataset))
        if len(val_loader.dataset) == 0:
            a0_vals[tid] = float("nan")
        else:
            a0_vals[tid] = float(base.validate(val_loader, models, criterion, epoch=0, use_exo=args.use_exo, use_RN=args.relation_network))
    rec.set_A0("ranking_acc", a0_vals, a0_weights)

    # Continual training
    phase = 0
    for t_idx, tid in enumerate(task_order, start=1):
        act_sel = task_to_actions[tid]
        task_dir = os.path.join(args.output_root, f"task_{t_idx:02d}")
        os.makedirs(os.path.join(task_dir, "checkpoints"), exist_ok=True)

        # ---- VISTA: create current task adapter (hot-start from previous) ----
        if vista_adapter_bank is not None:
            init_from = None
            if t_idx >= 2:
                # previous task in sequence
                init_from = int(task_order[t_idx - 2])
            vista_adapter_bank.add_task(int(tid), init_from_task=init_from)
            vista_adapter_bank.set_current_task(int(tid))
            # Train only current adapter parameters
            vista_adapter_bank.freeze_all_except(int(tid))
            # Adapter optimizer (always step each iteration, independent from RAAN optimizers)
            base.vista_adapter_optimizer = torch.optim.Adam(
                [p for p in vista_adapter_bank.get(int(tid)).parameters() if p.requires_grad],
                lr=float(getattr(args, "lr", 1e-4)),
            )
            base.vista_mode = "train"
        if l2p_pool is not None:
            base.l2p_mode = "train"

        # ---- VISTA: freeze RAAN after task 1 (optional) ----
        if bool(getattr(args, "vista_enabled", False)) and int(t_idx) >= 2:
            freeze = not bool(getattr(args, "vista_train_raan_after_task1", False))
            if bool(getattr(args, "vista_enabled", False)) and freeze:
                for m in models.values():
                    for p in m.parameters():
                        p.requires_grad = False
                if model_uniform is not None:
                    for p in model_uniform.parameters():
                        p.requires_grad = False

        train_loader = build_loader(args.train_list, act_sel, shuffle=True)
        val_loader = build_loader(args.val_list, act_sel, shuffle=False)
        stats["splits"]["train"][str(tid)] = int(len(train_loader.dataset))
        stats["splits"]["val"][str(tid)] = int(len(val_loader.dataset))

        start = time.time()
        for ep in range(int(args.epochs_per_task)):
            if model_uniform is None:
                raise ValueError("This continual runner expects disparity_loss/rank_aware_loss enabled (baseline default).")
            phase = base.train_with_uniform(
                train_loader,
                models,
                model_uniform,
                criterion,
                optimizer,
                optimizer_attention,
                epoch=(t_idx - 1) * int(args.epochs_per_task) + ep,
                phase=phase,
                use_exo=args.use_exo,
                use_triplet_loss=args.triplet_loss,
                use_RN=args.relation_network,
            )

        # ---- EWC: estimate Fisher at task end ----
        if algo is not None and str(getattr(algo, "name", "")).strip().lower() == "ewc":
            fisher_loader = build_loader(args.train_list, act_sel, shuffle=False)
            max_batches = int(getattr(args, "ewc_fisher_batches", 0))
            max_batches = max_batches if max_batches > 0 else None
            algo.update_fisher_from_loader(loader=fisher_loader, loss_fn=_ewc_loss_from_batch, max_batches=max_batches)

        # ---- LwF: update teacher snapshot at task end ----
        if algo is not None and str(getattr(algo, "name", "")).strip().lower() == "lwf":
            algo.update_teacher_from_models(models=models, model_uniform=model_uniform)

        # ---- VISTA: fit router subspace for this task at task end (offline pass, no grads) ----
        router_task_dir = os.path.join(task_dir, "router")
        router_space = None
        if vista_router is not None:
            fit_loader = build_loader(args.train_list, act_sel, shuffle=False)
            torch.set_grad_enabled(False)
            router_space = vista_router.fit_from_loader(task_id=int(tid), loader=fit_loader, device="cpu", verbose=False)
            vista_router.save_task(output_dir=router_task_dir, task_id=int(tid))
            vista_router.save_index(output_dir=os.path.join(args.output_root, "router"))
            # Save adapter bank snapshots per task for reproducibility
            vista_adapter_bank.save(os.path.join(task_dir, "adapters"))
            torch.set_grad_enabled(True)

        # ---- ER/DER++ memory update at task end (strict, task-balanced) ----
        # EWC/LwF do not use replay memory; guard by capability.
        if algo is not None and hasattr(algo, "capacity") and int(getattr(algo, "capacity", 0)) > 0:
            mem_loader = build_loader(args.train_list, act_sel, shuffle=False)
            if len(mem_loader.dataset) <= 0:
                raise RuntimeError(f"[ER strict] Empty memory dataset for task_id={tid}")
            if getattr(algo, "name", "") == "derpp":
                # Distill target: aggregate ranking scores (scheme A).
                def _distill_target_fn(batch_in, _model_ignored):
                    if args.use_exo:
                        input1, input2, input_exo = batch_in
                    else:
                        input1, input2 = batch_in
                        input_exo = None
                    input_var1 = input1.cuda(non_blocking=True)
                    input_var2 = input2.cuda(non_blocking=True)
                    if args.use_exo and input_exo is not None:
                        input_exo_var = input_exo.cuda(non_blocking=True)
                    else:
                        input_exo_var = None
                    # replicate RN concat semantics
                    if args.relation_network and input_exo_var is not None:
                        input_var1 = torch.cat((input_var1, input_exo_var), dim=1)
                        input_var2 = torch.cat((input_var2, input_exo_var), dim=1)
                    # aggregate across model branches
                    out1_all = None
                    out2_all = None
                    for k in models.keys():
                        all1, _, _ = models[k](input_var1)
                        all2, _, _ = models[k](input_var2)
                        o1 = all1.mean(dim=1)
                        o2 = all2.mean(dim=1)
                        out1_all = o1 if out1_all is None else (out1_all + o1)
                        out2_all = o2 if out2_all is None else (out2_all + o2)
                    return (out1_all, out2_all)

                any_model = models[list(models.keys())[0]]
                algo.update_memory_from_loader(
                    task_id=int(tid),
                    loader=mem_loader,
                    model=any_model,
                    distill_target_fn=_distill_target_fn,
                )
            else:
                algo.update_memory_from_loader(task_id=int(tid), loader=mem_loader)

        # Evaluate on all seen tasks (val subsets per task)
        if bool(getattr(args, "vista_enabled", False)) and vista_adapter_bank is not None and vista_router is not None:
            # Switch train.py validate() to inference mode: use router+mixture (task-id unknown)
            base.vista_mode = "infer"
        if l2p_pool is not None:
            base.l2p_mode = "infer"
        seen = task_order[:t_idx]
        per_task_values: Dict[int, float] = {}
        weights: Dict[int, float] = {}
        router_stats: Dict[int, Dict[str, float]] = {}
        router_hits: Dict[int, Dict[str, float]] = {}
        for j_tid in seen:
            j_act_sel = task_to_actions[j_tid]
            j_val_loader = build_loader(args.val_list, j_act_sel, shuffle=False)
            weights[j_tid] = float(len(j_val_loader.dataset))
            if len(j_val_loader.dataset) == 0:
                per_task_values[j_tid] = float("nan")
            else:
                per_task_values[j_tid] = float(
                    base.validate(j_val_loader, models, criterion, epoch=0, use_exo=args.use_exo, use_RN=args.relation_network)
                )
                # Optional: router confidence stats for gamma tuning (computed without using labels in inference).
                if bool(getattr(args, "vista_save_router_stats", True)) and vista_router is not None and vista_router.num_tasks() > 0:
                    # Compute stats + hit rates from this eval split (using ego pair only) with current router bank.
                    try:
                        stats_accum = {"res_best_mean": 0.0, "res_gap_mean": 0.0, "entropy_mean": 0.0}
                        n_batches = 0
                        n_samples = 0
                        top1_hits = 0
                        topL_hits = 0
                        true_prob_sum = 0.0
                        topL_cfg = int(getattr(args, "vista_topL", 2))
                        gamma_cfg = float(getattr(args, "vista_gamma", 10.0))
                        M_cfg = int(getattr(args, "vista_router_M", 1))
                        for batch in j_val_loader:
                            x1 = batch[0].cuda(non_blocking=True)
                            x2 = batch[1].cuda(non_blocking=True)
                            r1 = extract_r(x1, M=M_cfg)
                            r2 = extract_r(x2, M=M_cfg)

                            e1, tids = vista_router.augmented_residual_scores(r1, device=r1.device)
                            e2, _ = vista_router.augmented_residual_scores(r2, device=r2.device)
                            e = 0.5 * (e1 + e2)
                            e_all = e1
                            with torch.no_grad():
                                e_sorted, _ = torch.sort(e_all, dim=1)
                                best = e_sorted[:, 0]
                                second = e_sorted[:, 1] if e_sorted.shape[1] >= 2 else best
                                gap = (second - best).clamp(min=0)
                                p_full = torch.softmax((-float(gamma_cfg) * e_all), dim=1)
                                ent = -(p_full * (p_full.clamp(min=1e-12)).log()).sum(dim=1)
                                stats_accum["res_best_mean"] += float(best.mean().item())
                                stats_accum["res_gap_mean"] += float(gap.mean().item())
                                stats_accum["entropy_mean"] += float(ent.mean().item())
                            n_batches += 1

                            p = torch.softmax((-float(gamma_cfg) * e), dim=1)
                            L = int(min(int(topL_cfg), int(p.shape[1])))
                            _, idx = torch.topk(p, k=L, dim=1)
                            tid_tensor = torch.tensor(tids, device=p.device, dtype=torch.long)
                            task_ids = tid_tensor[idx]

                            B = int(task_ids.shape[0])
                            gt = int(j_tid)
                            gt_tensor = torch.full((B,), gt, device=task_ids.device, dtype=task_ids.dtype)
                            top1_hits += int((task_ids[:, 0] == gt_tensor).sum().item())
                            topL_hits += int((task_ids == gt_tensor[:, None]).any(dim=1).sum().item())
                            n_samples += B

                            # Optional: mean probability mass assigned to GT task (full softmax over all tasks)
                            try:
                                if gt in tids:
                                    gt_col = int(tids.index(gt))
                                    true_prob_sum += float(p[:, gt_col].sum().item())
                            except Exception:
                                pass
                        if n_batches > 0:
                            router_stats[int(j_tid)] = {k: v / float(n_batches) for k, v in stats_accum.items()}
                        if n_samples > 0:
                            router_hits[int(j_tid)] = {
                                "top1_hit_rate": float(top1_hits) / float(n_samples),
                                "topL_hit_rate": float(topL_hits) / float(n_samples),
                                "topL": int(min(int(topL_cfg), int(vista_router.num_tasks()))),
                                "n_samples": int(n_samples),
                                "true_task_prob_mean": float(true_prob_sum) / float(n_samples) if true_prob_sum > 0.0 else 0.0,
                            }
                    except Exception:
                        pass

        rec.update_after_task(t_idx=t_idx, metrics={"ranking_acc": per_task_values}, weights=weights)
        rec.save(args.output_root)

        torch.save(
            {
                "task_index": t_idx,
                "task_id": tid,
                "models": {k: m.state_dict() for k, m in models.items()},
                "model_uniform": None if model_uniform is None else model_uniform.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_attention": None if optimizer_attention is None else optimizer_attention.state_dict(),
                "phase": phase,
            },
            os.path.join(task_dir, "checkpoints", "task_end.pth"),
        )

        with open(os.path.join(task_dir, "metrics_task_end.json"), "w", encoding="utf-8") as f:
            import json

            json.dump(
                {
                    "task_index": t_idx,
                    "task_id": tid,
                    "seen_tasks": [int(x) for x in seen],
                    "train_time_sec": float(time.time() - start),
                    "weights": {str(k): float(v) for k, v in weights.items()},
                    "metrics": {"ranking_acc": {str(k): float(v) for k, v in per_task_values.items()}},
                    "vista": {
                        "enabled": bool(getattr(args, "vista_enabled", False)),
                        "router_M": int(getattr(args, "vista_router_M", 1)),
                        "subspace_k": int(getattr(args, "vista_subspace_k", 32)),
                        "topL": int(getattr(args, "vista_topL", 2)),
                        "gamma": float(getattr(args, "vista_gamma", 10.0)),
                        "eps": float(getattr(args, "vista_eps", 1e-6)),
                        "train_raan_after_task1": bool(getattr(args, "vista_train_raan_after_task1", False)),
                        "router_stats_val": {str(k): v for k, v in router_stats.items()},
                        "router_hit_val": {str(k): v for k, v in router_hits.items()},
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    stats.update(unknown.to_dict())
    stats["unknown_ratio"] = 0.0
    save_task_stats(args.output_root, stats)
    save_unknown_ids(args.output_root, unknown)


if __name__ == "__main__":
    main()


