import os
import sys
from typing import Dict, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score

# Ensure repo root on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.task_map import normalize_video_id
from anticipation_main import mean_average_precision
import anticipation_main as base_main


@torch.no_grad()
def eval_grouped(
    val_loader,
    model,
    criterion,
    num_class: int,
    args,
    video_to_task: Dict[str, int],
    seen_tasks: Set[int],
    ema_model=None,
) -> Tuple[Dict[str, Dict[int, float]], Dict[int, float]]:
    """Return per-task anticipation metrics on current val_loader.

    Metrics:
    - recall_top5_macro
    - precision_top5_macro
    - map_macro
    """
    model.eval()
    if ema_model is not None:
        ema_model.eval()
    gpu_count = torch.cuda.device_count()

    # Collect per-sample scores/labels and their task ids
    all_scores = []
    all_labels = []
    all_task_ids = []

    global_index = 0
    ds = val_loader.dataset

    for i, (val_data, val_label) in enumerate(val_loader):
        val_size_ori = val_data.size()
        batch_val_ori = val_size_ori[0]

        # add dummy tensors to keep the same batch size for each epoch (for the last epoch)
        if batch_val_ori < args.batch_size[2]:
            val_data_dummy = torch.zeros(args.batch_size[2] - batch_val_ori, val_size_ori[1], val_size_ori[2])
            val_data = torch.cat((val_data, val_data_dummy))

        # add dummy tensors to make sure batch size can be divided by gpu #
        if val_data.size(0) % gpu_count != 0:
            val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1), val_data.size(2))
            val_data = torch.cat((val_data, val_data_dummy))

        val_label = val_label.cuda(non_blocking=True)
        val_data = val_data.cuda(non_blocking=True)

        if args.baseline_type == "frame":
            label = val_label.unsqueeze(1).repeat(1, args.num_segments).view(-1)
        else:
            label = val_label

        val_source = val_data
        val_target = val_data
        if getattr(base_main, "vista_enabled", False) and getattr(base_main, "vista_mode", "none") == "infer":
            val_source, val_target = base_main._vista_apply_infer_pair(val_source, val_target)
        _, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val = model(
            val_source, val_target, [0] * len(args.beta), 0, is_train=False, reverse=False
        )
        if ema_model is not None:
            _a, _b, _c, _d, _e, _attn_e, out_val_ema, _out_val_2e, _pd_e, _feat_e = ema_model(
                val_source, val_target, [0] * len(args.beta), 0, is_train=False, reverse=False
            )
            out_val = 0.5 * (out_val + out_val_ema)

        # ignore dummy tensors
        out_val = out_val[:batch_val_ori]
        label = label[:batch_val_ori]

        pred = out_val
        if args.baseline_type == "tsn":
            pred = pred.view(batch_val_ori, -1, num_class).mean(dim=1)

        # bookkeeping for grouping
        for j in range(batch_val_ori):
            rec = ds.video_list[global_index + j]
            uid = normalize_video_id(rec.path)
            tid = video_to_task.get(uid, None)
            if tid is None or int(tid) not in seen_tasks:
                all_task_ids.append(None)
            else:
                all_task_ids.append(int(tid))
        global_index += batch_val_ori

        all_scores.append(pred.detach().cpu())
        all_labels.append(label.detach().cpu())

    if len(all_scores) == 0:
        return {"recall_top5_macro": {}, "precision_top5_macro": {}, "map_macro": {}}, {}

    scores = torch.cat(all_scores, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    all_task_ids = np.array(all_task_ids, dtype=object)

    per_task_metrics: Dict[str, Dict[int, float]] = {
        "recall_top5_macro": {},
        "precision_top5_macro": {},
        "map_macro": {},
    }
    weights: Dict[int, float] = {}

    for tid in sorted(seen_tasks):
        mask = all_task_ids == int(tid)
        if mask.sum() == 0:
            continue
        scores_t = scores[mask]
        labels_t = labels[mask]

        pred_top5 = np.zeros_like(scores_t)
        top5_idx = np.argsort(scores_t, axis=1)[:, ::-1][:, :5]
        for r in range(scores_t.shape[0]):
            pred_top5[r, top5_idx[r]] = 1.0

        # Many tasks have classes with no positive samples / no predicted samples in the eval subset.
        # sklearn's default behavior is to set those undefined values to 0 and emit warnings.
        # We make that behavior explicit for reproducibility and to avoid noisy logs.
        recall = recall_score(labels_t, pred_top5, average="macro", zero_division=0) * 100.0
        prec = precision_score(labels_t, pred_top5, average="macro", zero_division=0) * 100.0
        map_macro = mean_average_precision(list(scores_t), list(labels_t)) * 100.0

        per_task_metrics["recall_top5_macro"][int(tid)] = float(recall)
        per_task_metrics["precision_top5_macro"][int(tid)] = float(prec)
        per_task_metrics["map_macro"][int(tid)] = float(map_macro)
        weights[int(tid)] = float(mask.sum())

    return per_task_metrics, weights


