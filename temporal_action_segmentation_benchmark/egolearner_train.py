import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from loss import *
from tensorboardX import SummaryWriter
from typing import Callable, Optional, Tuple

# Optional: continual algorithm plugin (e.g., ER) injected by continual runners.
# Must expose:
# - mix_in_replay(cur_batch=(x,y,mask), cur_batch_size=int) -> merged (x,y,mask)
continual_algo = None

# Optional: VISTA modules injected by continual runners.
vista_enabled = False
vista_state = None  # clego_cl.vista.VISTAState
vista_mode = "none"  # "train" | "infer" | "none"
vista_adapter_optimizer = None

# Optional: L2P modules injected by continual runners.
l2p_enabled = False
l2p_mode = "none"  # "train" | "infer" | "none"
l2p_pool = None  # clego_cl.l2p.L2PPool
l2p_topk = 2
l2p_router_M = 1
l2p_sim_lambda = 0.5
l2p_diversed_selection = True
l2p_batchwise_selection = False
l2p_optimizer = None


def _vista_to_btc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"VISTA expects [B,C,T], got shape={tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


def _vista_to_bct(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"VISTA expects [B,T,C], got shape={tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


def _vista_apply_train_bct(x: torch.Tensor) -> torch.Tensor:
    global vista_enabled, vista_mode, vista_state
    if not vista_enabled or vista_mode != "train" or vista_state is None or vista_state.adapter_bank is None:
        return x
    return vista_state.adapter_bank.forward_train(x)


def _vista_apply_infer_pair_bct(x_source: torch.Tensor, x_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    global vista_enabled, vista_mode, vista_state
    if not vista_enabled or vista_mode != "infer" or vista_state is None or vista_state.router is None:
        return x_source, x_target
    if vista_state.router.num_tasks() <= 0 or vista_state.adapter_bank is None or vista_state.adapter_bank.num_tasks() <= 0:
        return x_source, x_target
    from clego_cl.vista import infer_vista_mix_from_inputs

    mix = infer_vista_mix_from_inputs(
        router=vista_state.router,
        x1=x_source,
        x2=None,
        M=int(vista_state.router_M),
        topL=int(vista_state.topL),
        gamma=float(vista_state.gamma),
    )
    xs = vista_state.adapter_bank.forward_mixture(x_source, mix)
    if bool(getattr(vista_state, "apply_to_target", True)):
        xt = vista_state.adapter_bank.forward_mixture(x_target, mix)
    else:
        xt = x_target
    return xs, xt


def _l2p_apply_bct(x_source: torch.Tensor, x_target: torch.Tensor, *, training: bool):
    """Apply L2P adapters to (B,C,T) inputs by converting to (B,T,C)."""
    global l2p_enabled, l2p_mode, l2p_pool, l2p_router_M
    if not l2p_enabled or l2p_mode == "none" or l2p_pool is None:
        return x_source, x_target, torch.zeros((), device=x_source.device, dtype=x_source.dtype)
    from skill_benchmark.task_router import extract_r

    src_btc = _vista_to_btc(x_source)
    tgt_btc = _vista_to_btc(x_target)
    r1 = extract_r(src_btc, M=int(l2p_router_M))
    r2 = extract_r(tgt_btc, M=int(l2p_router_M))
    match1 = l2p_pool.cosine_match(r1)
    match2 = l2p_pool.cosine_match(r2)
    if int(match1.shape[0]) == int(match2.shape[0]):
        match = 0.5 * (match1 + match2)
        sel = l2p_pool.select_topk(match, training=training)
        src_btc = l2p_pool.apply_adapters(src_btc, sel)
        tgt_btc = l2p_pool.apply_adapters(tgt_btc, sel)
        sim_loss = sel.match.mean()
        return _vista_to_bct(src_btc), _vista_to_bct(tgt_btc), sim_loss

    from clego_cl.l2p import L2PSelection

    B1 = int(match1.shape[0])
    B2 = int(match2.shape[0])
    pooled = 0.5 * (match1.mean(dim=0) + match2.mean(dim=0))  # [P]
    pooled_match = pooled.view(1, -1).expand(B1 + B2, -1)  # [B1+B2, P]
    sel_all = l2p_pool.select_topk(pooled_match, training=training)

    idx_src = sel_all.indices[:B1]
    idx_tgt = sel_all.indices[B1 : B1 + B2]
    m_src = match1.gather(1, idx_src)
    m_tgt = match2.gather(1, idx_tgt)
    sel_src = L2PSelection(indices=idx_src, match=m_src)
    sel_tgt = L2PSelection(indices=idx_tgt, match=m_tgt)

    src_btc = l2p_pool.apply_adapters(src_btc, sel_src)
    tgt_btc = l2p_pool.apply_adapters(tgt_btc, sel_tgt)
    sim_loss = 0.5 * (m_src.mean() + m_tgt.mean())
    return _vista_to_bct(src_btc), _vista_to_bct(tgt_btc), sim_loss


class Trainer:

    def __init__(self, num_classes):
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_d = nn.CrossEntropyLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def adapt_weight(self,
                     iter_now,
                     iter_max_default,
                     iter_max_input,
                     weight_loss,
                     weight_value=10.0,
                     high_value=1.0,
                     low_value=0.0):
        # affect adaptive weight value
        iter_max = iter_max_default
        if weight_loss < -1:
            iter_max = iter_max_input

        high = high_value
        low = low_value
        weight = weight_value
        p = float(iter_now) / iter_max
        adaptive_weight = (2. /
                           (1. + np.exp(-weight * p)) - 1) * (high - low) + low
        return adaptive_weight

    def train(
        self,
        model,
        model_dir,
        results_dir,
        batch_gen_source_train,
        batch_gen_target_train,
        batch_gen_source_test,
        batch_gen_target_test,
        device,
        args,
        epoch_end_callback: Optional[Callable[[int, object, object], None]] = None,
        save_epoch_checkpoints: bool = True,
    ):
        # ====== collect arguments ====== #
        verbose = args.verbose
        num_epochs = args.num_epochs
        batch_size = args.bS
        num_f_maps = args.num_f_maps
        learning_rate = args.lr
        alpha = args.alpha
        tau = args.tau
        use_target = args.use_target
        ratio_source = args.ratio_source
        ratio_label_source = args.ratio_label_source
        resume_epoch = args.resume_epoch
        # tensorboard
        use_tensorboard = args.use_tensorboard
        epoch_embedding = args.epoch_embedding
        stage_embedding = args.stage_embedding
        num_frame_video_embedding = args.num_frame_video_embedding
        # adversarial loss
        DA_adv = args.DA_adv
        DA_adv_video = args.DA_adv_video
        iter_max_beta_user = args.iter_max_beta
        place_adv = args.place_adv
        beta = args.beta
        # multi-class adversarial loss
        multi_adv = args.multi_adv
        weighted_domain_loss = args.weighted_domain_loss
        ps_lb = args.ps_lb
        # semantic loss
        method_centroid = args.method_centroid
        DA_sem = args.DA_sem
        place_sem = args.place_sem
        ratio_ma = args.ratio_ma
        gamma = args.gamma
        iter_max_gamma_user = args.iter_max_gamma
        # entropy loss
        DA_ent = args.DA_ent
        place_ent = args.place_ent
        mu = args.mu
        # discrepancy loss
        DA_dis = args.DA_dis
        place_dis = args.place_dis
        nu = args.nu
        iter_max_nu_user = args.iter_max_nu
        # ensemble loss
        DA_ens = args.DA_ens
        place_ens = args.place_ens
        dim_proj = args.dim_proj
        # self-supervised learning for videos
        SS_video = args.SS_video
        place_ss = args.place_ss
        eta = args.eta

        # multi-GPU
        if args.multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.train()
        model.to(device)
        if resume_epoch > 0:
            model.load_state_dict(
                torch.load(model_dir + "/epoch-" + str(resume_epoch) +
                           ".model"))

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Improve resume behavior: also restore optimizer state if present.
        if resume_epoch > 0:
            opt_path = model_dir + "/epoch-" + str(resume_epoch) + ".opt"
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            except Exception:
                pass

        # determine batch size
        batch_size_source = batch_size
        src_n = int(getattr(batch_gen_source_train, "num_examples", 0) or 0)
        tgt_n = int(getattr(batch_gen_target_train, "num_examples", 0) or 0)
        if src_n <= 0 or tgt_n <= 0:
            # Nothing to train on (this can happen if a task is missing in one domain after filtering).
            if verbose:
                print(f"[Trainer.train] Skip training: source_n={src_n}, target_n={tgt_n}", flush=True)
            return

        batch_size_target = max(int(tgt_n / src_n * batch_size_source), 1)
        num_iter_epoch = np.ceil(src_n / batch_size_source)

        acc_best_source = 0.0  # store the best source acc
        acc_best_target = 0.0  # store the best target acc
        if use_tensorboard:
            writer = SummaryWriter(results_dir +
                                   '/tensorboard')  # for tensorboardX

        for epoch in range(resume_epoch, num_epochs):
            epoch_loss = 0
            correct_source = 0
            total_source = 0
            correct_target = 0
            total_target = 0
            iter_batch = 0

            start_iter = epoch * num_iter_epoch
            iter_max_default = num_epochs * num_iter_epoch  # affect adaptive weight value

            # initialize the embedding (for tensorboardX)
            if use_tensorboard and (epoch_embedding == epoch + 1
                                    or epoch_embedding == -1):
                feat_source_display = None
                label_source_display = None

                feat_target_display = None
                label_target_display = None

            # start training
            while batch_gen_source_train.has_next():
                # adaptive weight for adversarial loss
                iter_now = iter_batch + start_iter
                adaptive_beta_0 = self.adapt_weight(iter_now, iter_max_default,
                                                    iter_max_beta_user[0],
                                                    beta[0])
                adaptive_beta_1 = self.adapt_weight(iter_now, iter_max_default,
                                                    iter_max_beta_user[1],
                                                    beta[1]) / 10.0
                adaptive_gamma = self.adapt_weight(iter_now, iter_max_default,
                                                   iter_max_gamma_user, gamma)
                adaptive_nu = self.adapt_weight(iter_now, iter_max_default,
                                                iter_max_nu_user, nu)
                beta_in_0 = adaptive_beta_0 if beta[0] < 0 else beta[0]
                beta_in_1 = adaptive_beta_1 if beta[1] < 0 else beta[1]
                beta_in = [beta_in_0, beta_in_1]
                gamma_in = adaptive_gamma if gamma < 0 else gamma
                nu_in = adaptive_nu if nu < 0 else nu

                # ====== Feed-forward data ====== #
                # prepare inputs
                input_source, label_source, mask_source = batch_gen_source_train.next_batch(batch_size_source, 'source')

                # ----------------------------------------------------
                # Continual algorithm hook: Experience Replay (source)
                # ----------------------------------------------------
                global continual_algo
                if continual_algo is not None and hasattr(continual_algo, "mix_in_replay"):
                    try:
                        merged = continual_algo.mix_in_replay(
                            cur_batch=(input_source, label_source, mask_source),
                            cur_batch_size=int(input_source.size(0)),
                        )
                        input_source, label_source, mask_source = merged
                    except Exception as e:
                        raise RuntimeError(f"[TAS Trainer.train] continual_algorithm failed to mix replay at epoch={epoch} iter={iter_batch}") from e

                input_source, label_source, mask_source = input_source.to(device), label_source.to(device), mask_source.to(device)

                # drop some source frames (including labels) for semi-supervised learning
                input_source, label_source, mask_source = self.ctrl_video_length(
                    input_source, label_source, mask_source, ratio_source)

                # drop source labels only
                label_source_new, mask_source_new = self.ctrl_video_label_length(
                    label_source, mask_source, ratio_label_source)

                input_target, label_target, mask_target = batch_gen_target_train.next_batch(
                    batch_size_target, 'target')
                input_target, label_target, mask_target = input_target.to(
                    device), label_target.to(device), mask_target.to(device)

                # ----------------------------------------------------
                # L2P hook (train): apply adapter selection
                # ----------------------------------------------------
                l2p_sim_loss = torch.zeros((), device=input_source.device, dtype=input_source.dtype)
                if l2p_enabled and l2p_mode == "train":
                    input_source, input_target, l2p_sim_loss = _l2p_apply_bct(input_source, input_target, training=True)

                # ----------------------------------------------------
                # VISTA hook (train): apply current-task adapter
                # ----------------------------------------------------
                if vista_enabled and vista_mode == "train":
                    src_btc = _vista_to_btc(input_source)
                    tgt_btc = _vista_to_btc(input_target)
                    src_btc = _vista_apply_train_bct(src_btc)
                    if vista_state is not None and bool(getattr(vista_state, "apply_to_target", True)):
                        tgt_btc = _vista_apply_train_bct(tgt_btc)
                    input_source = _vista_to_bct(src_btc)
                    input_target = _vista_to_bct(tgt_btc)

                # crop longer ego video
                # if input_target.shape[2] > input_source.shape[2]:
                #     crop_start = random.randint(0, input_target.shape[2] - input_source.shape[2])
                #     input_target = input_target[:, :, crop_start:crop_start + input_source.shape[2]]
                #     label_target = label_target[:, crop_start:crop_start + input_source.shape[2]]
                #     mask_target = mask_target[:, :, crop_start:crop_start + input_source.shape[2]]

                # print("input_source:", input_source.shape, label_source.shape, mask_source.shape, flush=True)
                # print(torch.unique(label_source), flush=True)

                # print("input_target:", input_target.shape, label_target.shape, mask_target.shape, flush=True)
                # print(torch.unique(label_target), flush=True)

                # forward-pass data
                # label: (batch, frame#)
                # pred: (batch, stage#, class#, frame#)
                # feat: (batch, stage#, dim, frame#)
                # pred_d: (batch x frame#, stage#, class#, 2)
                # pred_d_video: (batch x seg#, stage#, 2)
                pred_source, prob_source, feat_source, pred_target, prob_target, feat_target, \
                    pred_d, pred_d_video, label_d, label_d_video, \
                    pred_source_2, prob_source_2, pred_target_2, prob_target_2 \
                    = model(input_source, input_target, mask_source, mask_target, beta_in, reverse=False)

                num_stages = pred_source.shape[1]

                # ------ store the embedding ------ #
                # only store the frame-level features ==> need to reshape
                if use_tensorboard and (epoch_embedding == epoch + 1
                                        or epoch_embedding == -1):
                    id_source = self.select_id_embedding(
                        mask_source,
                        num_frame_video_embedding)  # sample frame indices

                    feat_source_reshape = feat_source[:, stage_embedding, :,
                                                      id_source].detach(
                                                      ).transpose(
                                                          1, 2).reshape(
                                                              -1, num_f_maps)
                    feat_source_display = feat_source_reshape if iter_batch == 0 else torch.cat(
                        (feat_source_display, feat_source_reshape), 0)
                    label_source_reshape = label_source[:, id_source].detach(
                    ).reshape(-1)
                    label_source_display = label_source_reshape if iter_batch == 0 else torch.cat(
                        (label_source_display, label_source_reshape), 0)

                    id_target = self.select_id_embedding(
                        mask_target,
                        num_frame_video_embedding)  # sample frame indices

                    feat_target_reshape = feat_target[:, stage_embedding, :,
                                                      id_target].detach(
                                                      ).transpose(
                                                          1, 2).reshape(
                                                              -1, num_f_maps)
                    feat_target_display = feat_target_reshape if iter_batch == 0 else torch.cat(
                        (feat_target_display, feat_target_reshape), 0)
                    label_target_reshape = label_target[:, id_target].detach(
                    ).reshape(-1)
                    label_target_display = label_target_reshape if iter_batch == 0 else torch.cat(
                        (label_target_display, label_target_reshape), 0)

                # ------ Classification loss ------ #
                loss = 0
                for s in range(num_stages):
                    p = pred_source[:,
                                    s, :, :]  # select one stage --> (batch, class#, frame#)
                    loss += self.ce(
                        p.transpose(2,
                                    1).contiguous().view(-1, self.num_classes),
                        label_source_new.view(-1))
                    loss += alpha * torch.mean(
                        torch.clamp(self.mse(
                            F.log_softmax(p[:, :, 1:], dim=1),
                            F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                    min=0,
                                    max=tau**2) * mask_source_new[:, :, 1:])
                    if DA_ens == 'MCD' or DA_ens == 'SWD' and use_target != 'none':
                        p_2 = pred_source_2[:,
                                            s, :, :]  # select one stage --> (batch, class#, frame#)
                        loss += self.ce(
                            p_2.transpose(2, 1).contiguous().view(
                                -1, self.num_classes),
                            label_source_new.view(-1))
                        loss += alpha * torch.mean(
                            torch.clamp(self.mse(
                                F.log_softmax(p_2[:, :, 1:], dim=1),
                                F.log_softmax(p_2.detach()[:, :, :-1], dim=1)),
                                        min=0,
                                        max=tau**2) *
                            mask_source_new[:, :, 1:])

                if l2p_enabled and l2p_mode == "train":
                    loss = loss + (float(l2p_sim_lambda) * l2p_sim_loss)

                # ----------------------------------------------------
                # Continual algorithm hook: LwF distillation (source logits)
                # ----------------------------------------------------
                if continual_algo is not None and hasattr(continual_algo, "lwf_loss") and hasattr(continual_algo, "teacher"):
                    try:
                        teacher = continual_algo.teacher()
                        if teacher is not None:
                            teacher = teacher.to(device=input_source.device)
                            with torch.no_grad():
                                t_pred, _t_prob, _t_feat, _t_feat_video, _t_pred_d, _t_pred_d_video, _t_lb_d, _t_lb_d_video, _t_pred2, _t_prob2 = teacher.forward_domain(
                                    input_source,
                                    mask_source,
                                    0,
                                    beta=[0.0, 0.0],
                                    reverse=False,
                                )
                                t_z = t_pred[:, -1, :, :]
                                t_z = t_z * (mask_source[:, :1, :] > 0).to(t_z.dtype)
                            z_cur = pred_source[:, -1, :, :] * (mask_source[:, :1, :] > 0).to(pred_source.dtype)
                            loss = loss + continual_algo.lwf_loss(z_cur, t_z)
                    except Exception as e:
                        raise RuntimeError(
                            f"[TAS Trainer.train] continual_algorithm lwf_loss failed at epoch={epoch} iter={iter_batch}"
                        ) from e

                # ----------------------------------------------------
                # Continual algorithm hook: DERPP distillation loss
                # - distill on last-stage source logits only
                # - mask invalid/padded frames using mask_source
                # ----------------------------------------------------
                if continual_algo is not None and getattr(continual_algo, "name", "") == "derpp":
                    try:
                        z_cur = pred_source[:, -1, :, :]  # (B, C, T)
                        frame_mask = (mask_source[:, :1, :] > 0).to(z_cur.dtype)  # (B,1,T)
                        z_cur = z_cur * frame_mask
                        loss = loss + continual_algo.distill_loss(z_cur)
                    except Exception as e:
                        raise RuntimeError(
                            f"[TAS Trainer.train] continual_algorithm derpp distill_loss failed at epoch={epoch} iter={iter_batch}"
                        ) from e
                # ----------------------------------------------------
                # Continual algorithm hook: EWC regularization
                # ----------------------------------------------------
                if continual_algo is not None and hasattr(continual_algo, "regularization_loss"):
                    try:
                        loss = loss + continual_algo.regularization_loss()
                    except Exception as e:
                        raise RuntimeError(
                            f"[TAS Trainer.train] continual_algorithm failed to compute EWC loss at epoch={epoch} iter={iter_batch}"
                        ) from e

                # ====== Domain Adaptation ====== #
                if use_target != 'none':
                    num_class_domain = pred_d.size(2)

                    if DA_ens != 'none':  # get multiple target outputs
                        _, _, _, _, prob_target, _, _, _, _, _, _, _, _, prob_target_2 \
                            = model(input_source, input_target, mask_source, mask_target, beta_in, reverse=True)

                    for s in range(num_stages):
                        # --- select data for th current staege --- #
                        # masking class prediction
                        pred_select_source, prob_select_source, prob_select_source_2, feat_select_source, label_select_source, classweight_stage_select_source \
                            = self.select_data_stage(s, pred_source, prob_source, prob_source_2, feat_source,
                                                     label_source)

                        pred_select_target, prob_select_target, prob_select_target_2, feat_select_target, label_select_target, classweight_stage_select_target \
                            = self.select_data_stage(s, pred_target, prob_target, prob_target_2, feat_target,
                                                     label_target)

                        # masking domain prediction
                        pred_d_stage, pred_d_video_stage, label_d_stage, label_d_video_stage \
                            = self.select_data_domain_stage(s, pred_d, pred_d_video, label_d, label_d_video)

                        # concatenate class probability masks
                        classweight_stage = torch.cat(
                            (classweight_stage_select_source,
                             classweight_stage_select_target), 0)
                        classweight_stage_hardmask = classweight_stage == classweight_stage.max(
                            dim=1,
                            keepdim=True)[0]  # highest prob: 1, others: 0
                        classweight_stage_hardmask = classweight_stage_hardmask.float(
                        )

                        # ------ Adversarial loss ------ #
                        if DA_adv == 'rev_grad':
                            if place_adv[s] == 'Y':
                                # calculate loss
                                loss_adv = 0
                                for c in range(num_class_domain):
                                    pred_d_class = pred_d_stage[:,
                                                                c, :]  # (batch x frame#, 2)
                                    label_d_class = label_d_stage[:,
                                                                  c]  # (batch x frame#)

                                    loss_adv_class = self.ce_d(
                                        pred_d_class, label_d_class)
                                    if weighted_domain_loss == 'Y' and multi_adv[
                                            1] == 'Y':  # weighted by class prediction
                                        if ps_lb == 'soft':
                                            loss_adv_class *= classweight_stage[:,
                                                                                c].detach(
                                                                                )
                                        elif ps_lb == 'hard':
                                            loss_adv_class *= classweight_stage_hardmask[:,
                                                                                         c].detach(
                                                                                         )

                                    loss_adv += loss_adv_class.mean()

                                loss += loss_adv

                                if 'rev_grad' in DA_adv_video:
                                    loss_adv_video = self.ce_d(
                                        pred_d_video_stage,
                                        label_d_video_stage)
                                    loss += loss_adv_video.mean()

                        # ------ Discrepancy loss ------ #
                        if DA_dis == 'JAN':
                            if place_dis[s] == 'Y':
                                # calculate loss
                                size_loss = min(
                                    prob_select_source.size(0),
                                    prob_select_target.size(
                                        0))  # choose the smaller number
                                size_loss = min(
                                    512,
                                    size_loss)  # avoid "out of memory" issue
                                # random indices
                                id_rand_source = torch.randperm(
                                    prob_select_source.size(0))
                                id_rand_target = torch.randperm(
                                    prob_select_target.size(0))
                                feat_source_sel = [
                                    feat_select_source[
                                        id_rand_source[:size_loss]],
                                    prob_select_source[
                                        id_rand_source[:size_loss]]
                                ]
                                feat_target_sel = [
                                    feat_select_target[
                                        id_rand_target[:size_loss]],
                                    prob_select_target[
                                        id_rand_target[:size_loss]]
                                ]

                                loss_dis = loss_jan(feat_source_sel,
                                                    feat_target_sel)

                                loss += nu_in * loss_dis

                        # ------ Semantic loss between centroids ------ #
                        if method_centroid != 'none':
                            if place_sem[s] == 'Y':
                                # update centroids: (num_classes, num_f_maps)
                                centroid_source, centroid_target \
                                    = model.centroids[s].update_centroids(feat_select_source, feat_select_target,
                                                                          label_select_source,
                                                                          prob_select_target, method_centroid, ratio_ma)

                                # calculate semantic loss from centroids
                                if DA_sem == 'mse':
                                    loss_sem = self.mse(
                                        centroid_target,
                                        centroid_source).mean()
                                    loss += gamma_in * loss_sem

                                model.centroids[
                                    s].centroid_s = centroid_source.detach()
                                model.centroids[
                                    s].centroid_t = centroid_target.detach()

                        # ------ Ensemble loss ------ #
                        if DA_ens != 'none':
                            if place_ens[s] == 'Y':
                                loss_ens = 0
                                # calculate loss
                                if DA_ens == 'MCD':
                                    loss_ens = -dis_mcd(
                                        prob_select_target,
                                        prob_select_target_2)
                                elif DA_ens == 'SWD':
                                    loss_ens = -dis_swd(
                                        prob_select_target,
                                        prob_select_target_2, dim_proj)

                                loss += loss_ens

                        # ------ Entropy loss ------ #
                        if DA_ent == 'target':
                            if place_ent[s] == 'Y':
                                # calculate loss
                                loss_ent = cross_entropy_soft(
                                    pred_select_target)
                                loss += mu * loss_ent
                        elif DA_ent == 'attn':
                            if place_ent[s] == 'Y':
                                # calculate loss
                                loss_ent = 0
                                for c in range(num_class_domain):
                                    pred_d_class = pred_d_stage[:,
                                                                c, :]  # (batch x frame#, 2)

                                    loss_ent_class = attentive_entropy(
                                        torch.cat((pred_select_source,
                                                   pred_select_target), 0),
                                        pred_d_class)
                                    if weighted_domain_loss == 'Y' and multi_adv[
                                            1] == 'Y':  # weighted by class prediction
                                        if ps_lb == 'soft':
                                            loss_ent_class *= classweight_stage[:,
                                                                                c].detach(
                                                                                )
                                        elif ps_lb == 'hard':
                                            loss_ent_class *= classweight_stage_hardmask[:,
                                                                                         c].detach(
                                                                                         )

                                    loss_ent += loss_ent_class.mean()
                                loss += mu * loss_ent

                        # ------ Adversarial loss ------ #
                        if SS_video == 'VCOP':
                            if place_ss[s] == 'Y':
                                loss_ss_video = self.ce_d(
                                    pred_d_video_stage, label_d_video_stage)
                                loss += eta * loss_ss_video.mean()

                # training
                optimizer.zero_grad()
                global vista_adapter_optimizer
                if vista_adapter_optimizer is not None:
                    vista_adapter_optimizer.zero_grad(set_to_none=True)
                if l2p_optimizer is not None:
                    l2p_optimizer.zero_grad(set_to_none=True)

                loss.backward()
                optimizer.step()
                if vista_adapter_optimizer is not None:
                    vista_adapter_optimizer.step()
                    vista_adapter_optimizer.zero_grad(set_to_none=True)
                if l2p_optimizer is not None and l2p_enabled and l2p_mode == "train":
                    l2p_optimizer.step()
                    l2p_optimizer.zero_grad(set_to_none=True)
                # print("Loss:", loss, flush=True)
                epoch_loss += loss.item()  # record the epoch loss

                # prediction
                _, pred_id_source = torch.max(
                    pred_source[:, -1, :, :].data,
                    1)  # predicted indices (from the last stage)
                correct_source += (
                    (pred_id_source == label_source_new).float() *
                    mask_source_new[:, 0, :].squeeze(1)).sum().item()
                total_source += torch.sum(mask_source_new[:, 0, :]).item()
                _, pred_id_target = torch.max(
                    pred_target[:, -1, :, :].data,
                    1)  # predicted indices (from the last stage)
                correct_target += (
                    (pred_id_target == label_target).float() *
                    mask_target[:, 0, :].squeeze(1)).sum().item()
                total_target += torch.sum(mask_target[:, 0, :]).item()

                iter_batch += 1

            batch_gen_source_train.reset()
            batch_gen_target_train.reset()
            random.shuffle(batch_gen_source_train.list_of_examples)
            random.shuffle(batch_gen_target_train.list_of_examples
                           )  # use it only when using target data

            acc_epoch_source = float(correct_source) / total_source
            acc_epoch_target = float(correct_target) / total_target
            # Optional epoch-end callback (e.g., continual runner's online val-best tracking).
            if epoch_end_callback is not None:
                try:
                    epoch_end_callback(int(epoch + 1), model, optimizer)
                except Exception as e:
                    raise RuntimeError(f"[Trainer.train] epoch_end_callback failed at epoch={epoch + 1}") from e

            if l2p_enabled and l2p_mode == "train" and l2p_pool is not None:
                l2p_pool.update_frequency()

            if save_epoch_checkpoints:
                torch.save(model.state_dict(),
                           model_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(),
                           model_dir + "/epoch-" + str(epoch + 1) + ".opt")

                # update the "best" model (best training acc)
                if acc_epoch_source > acc_best_source:
                    acc_best_source = acc_epoch_source
                    torch.save(model.state_dict(),
                               model_dir + "/acc_best_source.model")
                    torch.save(optimizer.state_dict(),
                               model_dir + "/acc_best_source.opt")

                if acc_epoch_target > acc_best_target:
                    acc_best_target = acc_epoch_target
                    torch.save(model.state_dict(),
                               model_dir + "/acc_best_target.model")
                    torch.save(optimizer.state_dict(),
                               model_dir + "/acc_best_target.opt")

            if verbose:
                print(
                    "[epoch %d]: epoch loss = %f,   acc (S) = %f,   acc (T) = %f,   beta = (%f, %f),   nu = %f"
                    %
                    (epoch + 1, epoch_loss / num_iter_epoch, acc_epoch_source,
                     acc_epoch_target, beta_in[0], beta_in[1], nu_in),
                    flush=True)  # uncomment for debugging

            # ------ update the embedding every epoch ------ #
            if use_tensorboard and (epoch_embedding == epoch + 1
                                    or epoch_embedding == -1):
                # generate domain labels
                label_source_domain_display = torch.full_like(
                    feat_source_display[:, 0], 0)
                label_target_domain_display = torch.full_like(
                    feat_target_display[:, 0], 1)

                # mix source and target
                feat_all_display = torch.cat(
                    (feat_source_display, feat_target_display), 0)
                label_all_class_display = torch.cat(
                    (label_source_display, label_target_display), 0)
                label_all_domain_display = torch.cat(
                    (label_source_domain_display, label_target_domain_display),
                    0)
                label_all_display = list(
                    zip(label_all_class_display, label_all_domain_display))
                writer.add_embedding(feat_all_display,
                                     metadata=label_all_display,
                                     metadata_header=['class', 'domain'],
                                     global_step=iter_now)

        if use_tensorboard:
            writer.close()

    def select_data_stage(self, s, pred, prob, prob_2, feat, label):
        dim_feat = feat.size(2)

        # features & prediction
        feat_stage = feat[:,
                          s, :, :]  # select one stage --> (batch, dim, frame#)
        feat_frame = feat_stage.transpose(1, 2).reshape(-1, dim_feat)
        pred_stage = pred[:,
                          s, :, :]  # select one stage --> (batch, class#, frame#)
        pred_frame = pred_stage.transpose(1, 2).reshape(-1, self.num_classes)
        prob_stage = prob[:,
                          s, :, :]  # select one stage --> (batch, class#, frame#)
        prob_frame = prob_stage.transpose(1, 2).reshape(-1, self.num_classes)
        prob_2_stage = prob_2[:,
                              s, :, :]  # select one stage --> (batch, class#, frame#)
        prob_2_frame = prob_2_stage.transpose(1,
                                              2).reshape(-1, self.num_classes)

        # select the masked frames & labels
        label_vector = label.reshape(-1).clone()
        feat_select = feat_frame[label_vector != -100]
        pred_select = pred_frame[label_vector != -100]
        label_select = label_vector[label_vector != -100]
        prob_select = prob_frame[label_vector != -100]
        prob_2_select = prob_2_frame[label_vector != -100]

        # class probability as class weights
        classweight_stage = prob[:,
                                 s, :, :]  # select one stage --> (batch, class#, frame#)
        classweight_stage = classweight_stage.transpose(1, 2).reshape(
            -1, self.num_classes)  # (batch x frame#, class#)

        # mask frames
        classweight_stage_select = classweight_stage[label_vector != -100]

        return pred_select, prob_select, prob_2_select, feat_select, label_select, classweight_stage_select

    def select_data_domain_stage(self, s, pred_d, pred_d_video, label_d,
                                 label_d_video):

        # domain predictions & labels (frame-level)
        pred_d_select = pred_d[:,
                               s, :, :]  # select one stage --> (batch x frame#, class#, 2)
        label_d_select = label_d[:,
                                 s, :]  # select one stage --> (batch x frame#, class#)

        # domain predictions & labels (video-level)
        pred_d_select_seg = pred_d_video[:,
                                         s, :]  # select one stage --> (batch x seg#, 2)
        label_d_select_video = label_d_video[:,
                                             s]  # select one stage --> (batch x seg#)

        return pred_d_select, pred_d_select_seg, label_d_select, label_d_select_video

    def select_id_embedding(self, mask, num_frame_select):
        # sample frame indices
        num_frame_min = mask[:, 0, :].sum(-1).min()  # length of shortest video
        if num_frame_min.item() < num_frame_select:
            raise ValueError('space between frames should be at least 1!')
        index = torch.tensor(
            np.linspace(0,
                        num_frame_min.item() - 1,
                        num_frame_select).tolist()).long()
        if mask.get_device() >= 0:
            index = index.to(mask.get_device())

        return index

    def ctrl_video_length(self, input_data, label, mask, ratio_length):
        # shapes:
        # input_data: (batch, dim, frame#)
        # label: (batch, frame#)
        # mask: (batch, class#, frame#)

        # get the indices of the frames to keep
        num_frame = input_data.size(-1)  # length of video
        num_frame_drop = int(round((1 - ratio_length) * num_frame))
        if num_frame_drop <= 0:
            return input_data, label, mask
        id_drop = np.floor(np.linspace(0, num_frame - 1, num_frame_drop)).tolist()
        id_keep = list(set(range(num_frame)) - set(id_drop))
        id_keep = torch.tensor(id_keep).long()
        if input_data.get_device() >= 0:
            id_keep = id_keep.to(input_data.get_device())

        # filter the inputs w/ the above indices
        input_data_filtered = input_data[:, :, id_keep]
        label_filtered = label[:, id_keep]
        mask_filtered = mask[:, :, id_keep]

        return input_data_filtered, label_filtered, mask_filtered

    def ctrl_video_label_length(self, label, mask, ratio_length):
        # shapes:
        # label: (batch, frame#)
        # mask: (batch, class#, frame#)
        mask_new = mask.clone()
        label_new = label.clone()

        # get the indices of the frames to keep
        num_frame = mask.size(-1)  # length of video
        num_frame_drop = int(round((1 - ratio_length) * num_frame))
        if num_frame_drop <= 0:
            return label_new, mask_new
        id_drop = np.floor(np.linspace(0, num_frame - 1, num_frame_drop)).tolist()
        id_drop = torch.tensor(id_drop).long()
        if mask.get_device() >= 0:
            id_drop = id_drop.to(mask.get_device())

        # assign 0 to the above indices
        mask_new[:, :, id_drop] = 0
        label_new[:,
                  id_drop] = -100  # class id -100 won't be calculated in cross-entropy

        return label_new, mask_new
