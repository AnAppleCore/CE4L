from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from skill_benchmark.adapters import AdapterBank, MixtureSpec
from skill_benchmark.task_router import TaskWhitenedSubspaceRouter, extract_r


def vista_router_posterior_from_r(
    *,
    router: TaskWhitenedSubspaceRouter,
    r: torch.Tensor,
    gamma: float,
) -> Tuple[list[int], torch.Tensor, torch.Tensor]:
    """Dense posterior over tasks from the whitened-subspace router.

    Returns:
      task_ids: length T, column order of the scores
      e: (B, T) residual ratios (lower is better)
      p: (B, T) softmax(-gamma * e)
    """
    if r.dim() != 2:
        raise ValueError(f"vista_router_posterior_from_r expects r [B,D], got shape={tuple(r.shape)}")
    e, tids = router.augmented_residual_scores(r, device=r.device)
    p = torch.softmax((-float(gamma) * e), dim=1)
    task_ids = [int(x) for x in tids]
    return task_ids, e, p


def vista_router_topk_from_posterior(
    *,
    task_ids: list[int],
    p: torch.Tensor,
    topL: int,
) -> torch.Tensor:
    """Return top-L predicted task ids for each sample given dense posterior p."""
    if p.dim() != 2:
        raise ValueError(f"vista_router_topk_from_posterior expects p [B,T], got {tuple(p.shape)}")
    L = int(min(int(topL), int(p.shape[1])))
    vals, idx = torch.topk(p, k=L, dim=1)
    _ = vals
    tid_tensor = torch.tensor(task_ids, device=p.device, dtype=torch.long)
    return tid_tensor[idx]


def vista_eval_router_grouped(
    *,
    router: TaskWhitenedSubspaceRouter,
    x: torch.Tensor,
    gt_task_ids: torch.Tensor,
    M: int,
    topL: int,
    gamma: float,
) -> Tuple[dict, dict]:
    """Evaluate router hit rates and confidence stats grouped by ground-truth task."""
    if gt_task_ids.dim() != 1:
        raise ValueError(f"vista_eval_router_grouped expects gt_task_ids [B], got {tuple(gt_task_ids.shape)}")
    r = extract_r(x, M=int(M))
    task_ids, e, p = vista_router_posterior_from_r(router=router, r=r, gamma=gamma)
    top_ids = vista_router_topk_from_posterior(task_ids=task_ids, p=p, topL=topL)

    stats_sum = {}
    stats_cnt = {}
    top1_hits = {}
    topL_hits = {}
    prob_sum = {}
    n_samp = {}

    e_sorted, _ = torch.sort(e, dim=1)
    best = e_sorted[:, 0]
    second = e_sorted[:, 1] if e_sorted.shape[1] >= 2 else best
    gap = (second - best).clamp(min=0)
    ent = -(p * (p.clamp(min=1e-12)).log()).sum(dim=1)
    col_index = {int(t): int(i) for i, t in enumerate(task_ids)}

    gt_list = gt_task_ids.detach().to(device=top_ids.device).to(dtype=torch.long)
    for i in range(int(gt_list.shape[0])):
        gt = int(gt_list[i].item())
        n_samp[gt] = n_samp.get(gt, 0) + 1
        if int(top_ids[i, 0].item()) == gt:
            top1_hits[gt] = top1_hits.get(gt, 0) + 1
        if bool((top_ids[i] == gt).any().item()):
            topL_hits[gt] = topL_hits.get(gt, 0) + 1
        if gt in col_index:
            prob_sum[gt] = prob_sum.get(gt, 0.0) + float(p[i, col_index[gt]].item())
        else:
            prob_sum[gt] = prob_sum.get(gt, 0.0)
        st = stats_sum.get(gt, {"res_best_mean": 0.0, "res_gap_mean": 0.0, "entropy_mean": 0.0})
        st["res_best_mean"] += float(best[i].item())
        st["res_gap_mean"] += float(gap[i].item())
        st["entropy_mean"] += float(ent[i].item())
        stats_sum[gt] = st
        stats_cnt[gt] = stats_cnt.get(gt, 0) + 1

    router_stats = {}
    router_hits = {}
    for gt, n in sorted(n_samp.items(), key=lambda kv: kv[0]):
        cnt = int(stats_cnt.get(gt, 0))
        if cnt > 0:
            router_stats[int(gt)] = {k: float(v) / float(cnt) for k, v in stats_sum[int(gt)].items()}
        else:
            router_stats[int(gt)] = {}
        router_hits[int(gt)] = {
            "top1_hit_rate": float(top1_hits.get(gt, 0)) / float(n),
            "topL_hit_rate": float(topL_hits.get(gt, 0)) / float(n),
            "topL": int(min(int(topL), int(len(task_ids)))),
            "n_samples": int(n),
            "true_task_prob_mean": float(prob_sum.get(gt, 0.0)) / float(n),
        }
    return router_stats, router_hits


@dataclass
class VISTAState:
    enabled: bool
    adapter_bank: Optional[AdapterBank]
    router: Optional[TaskWhitenedSubspaceRouter]
    router_M: int
    topL: int
    gamma: float
    eps: float
    apply_to_target: bool
    train_backbone_after_task1: bool


def build_vista_router(
    *,
    router_M: int,
    subspace_k: int,
    eps: float,
) -> TaskWhitenedSubspaceRouter:
    """VISTA router: diagonal-whitened subspace residual, mixed over top-L tasks."""
    return TaskWhitenedSubspaceRouter(M=int(router_M), k=int(subspace_k), eps=float(eps))


def _infer_mix_from_r(
    *,
    router: TaskWhitenedSubspaceRouter,
    r1: torch.Tensor,
    r2: Optional[torch.Tensor],
    topL: int,
    gamma: float,
) -> MixtureSpec:
    device = r1.device
    e1, tids = router.augmented_residual_scores(r1, device=device)
    if r2 is None:
        e = e1
    else:
        e2, _ = router.augmented_residual_scores(r2, device=device)
        e = 0.5 * (e1 + e2)
    p = torch.softmax((-float(gamma) * e), dim=1)
    L = int(min(int(topL), int(p.shape[1])))
    vals, idx = torch.topk(p, k=L, dim=1)
    tid_tensor = torch.tensor(tids, device=device, dtype=torch.long)
    task_ids = tid_tensor[idx]
    weights = vals / (vals.sum(dim=1, keepdim=True).clamp(min=1e-12))
    return MixtureSpec(task_ids=task_ids, weights=weights)


def infer_vista_mix_from_inputs(
    *,
    router: TaskWhitenedSubspaceRouter,
    x1: torch.Tensor,
    x2: Optional[torch.Tensor],
    M: int,
    topL: int,
    gamma: float,
) -> MixtureSpec:
    r1 = extract_r(x1, M=int(M))
    r2 = extract_r(x2, M=int(M)) if x2 is not None else None
    return _infer_mix_from_r(router=router, r1=r1, r2=r2, topL=topL, gamma=gamma)
