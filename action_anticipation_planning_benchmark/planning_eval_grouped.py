import os
import sys
from typing import Dict, Set, Tuple

import numpy as np
import torch

# Ensure repo root on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.task_map import normalize_video_id
import planning_main as base


@torch.no_grad()
def eval_grouped(
    val_loader,
    model,
    args,
    video_to_task: Dict[str, int],
    seen_tasks: Set[int],
    ema_model=None,
) -> Tuple[Dict[str, Dict[int, object]], Dict[int, float]]:
    """Planning grouped eval.

    Returns:
      per_task_metrics:
        - 'ed_final' -> float (scalar summary used for continual statistics)
      weights[task_id] = num_samples
    """
    model.eval()
    if ema_model is not None:
        ema_model.eval()
    gpu_count = torch.cuda.device_count()

    ds = val_loader.dataset
    global_index = 0
    all_preds = []
    all_labels = []
    all_task_ids = []

    for i, (val_data, val_label) in enumerate(val_loader):
        val_size_ori = val_data.size()
        batch_val_ori = val_size_ori[0]

        # dummy padding (match base.validate behavior)
        if batch_val_ori < args.batch_size[2]:
            val_data_dummy = torch.zeros(args.batch_size[2] - batch_val_ori, val_size_ori[1], val_size_ori[2])
            val_data = torch.cat((val_data, val_data_dummy))
        if val_data.size(0) % gpu_count != 0:
            val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1), val_data.size(2))
            val_data = torch.cat((val_data, val_data_dummy))

        val_label = val_label.cuda(non_blocking=True)
        val_data = val_data.cuda(non_blocking=True)

        label = val_label.unsqueeze(1).repeat(1, args.num_segments).view(-1) if args.baseline_type == "frame" else val_label

        val_source = val_data
        val_target = val_data
        if getattr(base, "vista_enabled", False) and getattr(base, "vista_mode", "none") == "infer":
            val_source, val_target = base._vista_apply_infer_pair(val_source, val_target)
        _, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val = model(
            val_source, val_target, [0] * len(args.beta), 0, is_train=False, reverse=False
        )
        if ema_model is not None:
            _a, _b, _c, _d, _e, _attn_e, out_val_ema, _out_val_2e, _pd_e, _feat_e = ema_model(
                val_source, val_target, [0] * len(args.beta), 0, is_train=False, reverse=False
            )
            out_val = 0.5 * (out_val + out_val_ema)

        out_val = out_val[:batch_val_ori]
        label = label[:batch_val_ori]

        pred = out_val
        if args.baseline_type == "tsn":
            pred = pred.view(batch_val_ori, -1, base.ACTION_NUM_CLASSES * base.FUTURE_LENGTH).mean(dim=1)

        pred = pred.reshape(-1, base.FUTURE_LENGTH, (base.ACTION_NUM_CLASSES * base.FUTURE_LENGTH) // base.FUTURE_LENGTH)

        # grouping ids
        for j in range(batch_val_ori):
            rec = ds.video_list[global_index + j]
            uid = normalize_video_id(rec.path)
            tid = video_to_task.get(uid, None)
            all_task_ids.append(int(tid) if tid is not None and int(tid) in seen_tasks else None)
        global_index += batch_val_ori

        all_preds.append(pred.detach().cpu())
        all_labels.append(label.detach().cpu())

    if len(all_preds) == 0:
        return {"ed_final": {}}, {}

    all_preds = torch.cat(all_preds, dim=0)  # (N,8,C)
    all_labels = torch.cat(all_labels, dim=0)  # (N,8)
    all_task_ids = np.array(all_task_ids, dtype=object)

    # ContinualRecorder assumes scalar metrics. Planning has rich step-wise metrics (top1/top5 by step),
    # but for continual statistics we only return the scalar summary (ed_final).
    per_task_metrics: Dict[str, Dict[int, object]] = {"ed_final": {}}
    weights: Dict[int, float] = {}

    for tid in sorted(seen_tasks):
        mask = all_task_ids == int(tid)
        if mask.sum() == 0:
            continue
        preds_t = all_preds[mask]
        labels_t = all_labels[mask]

        auedit, _ = base.calc_ed(preds_t, labels_t, k=5, logits=True)
        ed_final = 1.0 - float(auedit["action_AUED"])

        per_task_metrics["ed_final"][int(tid)] = float(ed_final)
        weights[int(tid)] = float(mask.sum())

    return per_task_metrics, weights


