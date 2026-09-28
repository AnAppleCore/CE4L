import torch
import torch.nn as nn
import numpy as np

from clego_cl.vista import infer_vista_mix_from_inputs
from clego_cl.task_map import normalize_video_id

# Optional: VISTA modules injected by continual runners.
vista_enabled = False
vista_state = None  # clego_cl.vista.VISTAState
vista_mode = "none"  # "train" | "infer" | "none"

# Optional: L2P modules injected by continual runners.
l2p_enabled = False
l2p_mode = "none"  # "train" | "infer" | "none"
l2p_pool = None  # clego_cl.l2p.L2PPool
l2p_topk = 2
l2p_router_M = 1
l2p_sim_lambda = 0.5
l2p_diversed_selection = True
l2p_batchwise_selection = False


def _l2p_apply_bct(x_source: torch.Tensor, x_target: torch.Tensor):
    """Apply L2P adapters to (B,C,T) inputs by converting to (B,T,C)."""
    global l2p_enabled, l2p_mode, l2p_pool, l2p_router_M
    if not l2p_enabled or l2p_mode == "none" or l2p_pool is None:
        return x_source, x_target
    from skill_benchmark.task_router import extract_r

    src_btc = x_source.transpose(1, 2).contiguous()
    tgt_btc = x_target.transpose(1, 2).contiguous()
    r1 = extract_r(src_btc, M=int(l2p_router_M))
    r2 = extract_r(tgt_btc, M=int(l2p_router_M))
    match1 = l2p_pool.cosine_match(r1)
    match2 = l2p_pool.cosine_match(r2)
    if int(match1.shape[0]) == int(match2.shape[0]):
        match = 0.5 * (match1 + match2)
        sel = l2p_pool.select_topk(match, training=False)
        src_btc = l2p_pool.apply_adapters(src_btc, sel)
        tgt_btc = l2p_pool.apply_adapters(tgt_btc, sel)
        return src_btc.transpose(1, 2).contiguous(), tgt_btc.transpose(1, 2).contiguous()

    from clego_cl.l2p import L2PSelection

    B1 = int(match1.shape[0])
    B2 = int(match2.shape[0])
    pooled = 0.5 * (match1.mean(dim=0) + match2.mean(dim=0))  # [P]
    pooled_match = pooled.view(1, -1).expand(B1 + B2, -1)  # [B1+B2, P]
    sel_all = l2p_pool.select_topk(pooled_match, training=False)

    idx_src = sel_all.indices[:B1]
    idx_tgt = sel_all.indices[B1 : B1 + B2]
    m_src = match1.gather(1, idx_src)
    m_tgt = match2.gather(1, idx_tgt)
    sel_src = L2PSelection(indices=idx_src, match=m_src)
    sel_tgt = L2PSelection(indices=idx_tgt, match=m_tgt)

    src_btc = l2p_pool.apply_adapters(src_btc, sel_src)
    tgt_btc = l2p_pool.apply_adapters(tgt_btc, sel_tgt)
    return src_btc.transpose(1, 2).contiguous(), tgt_btc.transpose(1, 2).contiguous()

def predict(model, model_dir, results_dir, features_path, vid_list_file,
            feat_suffix, feat_sample_rate, all_sample_rate, epoch,
            actions_dict, device, args, load_model: bool = True):
    # Optional: VISTA modules injected by continual runners.
    global vista_enabled, vista_state, vista_mode
    # collect arguments
    verbose = args.verbose
    use_best_model = args.use_best_model

    # multi-GPU
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    with torch.no_grad():
        model.to(device)
        if load_model:
            if use_best_model == 'source':
                model.load_state_dict(
                    torch.load(model_dir + "/acc_best_source.model"))
                print("load best source model")
            elif use_best_model == 'target':
                model.load_state_dict(
                    torch.load(model_dir + "/acc_best_target.model"))
                print("load best target model")
            else:
                model.load_state_dict(
                    torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
                print("load epoch-" + str(epoch) + " model")

        list_of_videos = []
        if isinstance(vid_list_file, str):
            vid_list_file = [vid_list_file]
        # print("file_list:", vid_list_file)
        # print("feat_suffix:", feat_suffix)
        assert len(feat_suffix) == len(vid_list_file)
        for i, file in enumerate(vid_list_file):
            file_ptr = open(file, 'r')
            list_of_examples = file_ptr.read().strip().split('\n')
            file_ptr.close()
            # print(feat_suffix[i])
            list_of_examples = [
                dict(vid=x,
                     feat_file=
                     f"{features_path}{x.split('.')[0]}{feat_suffix[i]}.pt")
                for x in list_of_examples
            ]
            list_of_videos.extend(list_of_examples)
            # print(file, list_of_examples)

        for vid in list_of_videos:
            if verbose:
                print(vid)
            feat_file = vid["feat_file"]
            # print(feat_file)
            features = torch.load(feat_file)
            features = features.transpose(1, 0)
            features = features[:, ::feat_sample_rate]
            features = features[:, ::all_sample_rate]
            # `features` may already be a Tensor; `torch.tensor(tensor)` emits a warning and can copy.
            # `as_tensor` preserves tensors and avoids unnecessary copies, while keeping behavior for numpy/array-likes.
            input_x = torch.as_tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            mask = torch.ones_like(input_x)
            # Ensure `input_target` is always defined before optional adapters run.
            # This also allows L2P-only inference to keep its adapted target, instead of being
            # overwritten when VISTA is disabled.
            input_target = input_x
            if l2p_enabled and l2p_mode == "infer":
                input_x, input_target = _l2p_apply_bct(input_x, input_target)
            if (
                vista_enabled
                and vista_mode == "infer"
                and vista_state is not None
                and vista_state.router is not None
                and vista_state.adapter_bank is not None
            ):
                src_btc = input_x.transpose(1, 2).contiguous()
                src_btc_orig = src_btc
                mix = infer_vista_mix_from_inputs(
                    router=vista_state.router,
                    x1=src_btc,
                    x2=None,
                    M=int(vista_state.router_M),
                    topL=int(vista_state.topL),
                    gamma=float(vista_state.gamma),
                )
                src_btc = vista_state.adapter_bank.forward_mixture(src_btc, mix)
                if bool(getattr(vista_state, "apply_to_target", True)):
                    tgt_btc = vista_state.adapter_bank.forward_mixture(src_btc_orig, mix)
                else:
                    tgt_btc = src_btc_orig
                input_x = src_btc.transpose(1, 2).contiguous()
                input_target = tgt_btc.transpose(1, 2).contiguous()
            predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(
                input_x, input_target, mask, mask, [0, 0], reverse=False)
            _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
            predicted = predicted.squeeze()
            recognition = []
            # print(all_sample_rate,feat_sample_rate)
            for i in range(predicted.size(0)):
                recognition = np.concatenate((recognition, [
                    list(actions_dict.keys())[list(
                        actions_dict.values()).index(predicted[i].item())]
                ] * all_sample_rate * feat_sample_rate))
            f_name = vid["vid"].split('/')[-1].split('.')[0]
            f_ptr = open(results_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
