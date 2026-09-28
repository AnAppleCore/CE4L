#!/usr/bin/env python3
"""
Extract Skill benchmark experiment results under ./exps/ and write CSVs into ./exps/:
  - results_seed.csv            (joint_fair + continual final results, seed-level, long format)
  - results_summary.csv         (mean/std across seeds grouped by experiment keys)
  - results_continual_curve.csv (continual curve, unified schema with AAP/Association)

Authoritative sources:
  - joint_fair: exps/joint_fair/{i3d_baseline|i3d_rn|i3d_tl}/seedX/joint_fair_eval.json
      keys: best_micro_avg, best_macro_avg, best_epoch
  - continual: exps/continual/{i3d_baseline|i3d_rn|i3d_tl}/seedX/
      continual_A_ranking_acc.npy + continual_bar_A_{micro,macro}_ranking_acc.npy + task_order.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR = os.path.join(HERE, "exps")
MIN_VALID_SEEDS_FOR_SUMMARY = 3
MAX_WARN_LINES = 1e5
_WARN_LINES_PRINTED = 0
_WARN_LINES_SUPPRESSED = 0

def _discover_run_types(exps_root: str) -> List[str]:
    """Discover available continual run type folders under exps/.

    Important: some methods have variants encoded in folder names, e.g.
      continual_derpp_br0p1_rr0p1
    We keep the folder name as run_type to preserve variant identity.
    """
    out: List[str] = []
    try:
        for name in os.listdir(exps_root):
            p = os.path.join(exps_root, name)
            if os.path.isdir(p) and str(name).startswith("continual"):
                out.append(str(name))
    except Exception:
        pass
    return sorted(list(dict.fromkeys(out)))


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json_compact(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _nanmean(xs: Iterable[Optional[float]]) -> Optional[float]:
    arr = np.array([x for x in xs if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _nanstd_pop(xs: Iterable[Optional[float]]) -> Optional[float]:
    arr = np.array([x for x in xs if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.std(arr, ddof=0))


def _vals_by_seed(rows: Iterable["SeedRow"], attr: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in rows:
        v = _safe_float(getattr(r, attr))
        if v is None:
            continue
        if str(r.seed) != "":
            out[str(r.seed)] = float(v)
    return out


def _warn_insufficient_summary(*, kind: str, key: Tuple[Any, ...], details: str) -> None:
    global _WARN_LINES_PRINTED, _WARN_LINES_SUPPRESSED
    if _WARN_LINES_PRINTED >= MAX_WARN_LINES:
        _WARN_LINES_SUPPRESSED += 1
        return
    _WARN_LINES_PRINTED += 1
    print(f"[WARN][skill][{kind}] <{MIN_VALID_SEEDS_FOR_SUMMARY} valid seeds; key={key}; {details}")


def _warn_incomplete_run(*, run_type: str, variant: str, seed: str, t_max: int, t_total: int, seed_path: str) -> None:
    global _WARN_LINES_PRINTED, _WARN_LINES_SUPPRESSED
    if _WARN_LINES_PRINTED >= MAX_WARN_LINES:
        _WARN_LINES_SUPPRESSED += 1
        return
    _WARN_LINES_PRINTED += 1
    print(
        f"[WARN][skill][incomplete_run] skip (t_max={t_max} != T={t_total}); "
        f"run_type={run_type}; variant={variant}; seed={seed}; dir={seed_path}"
    )


def _max_finite_t(arr: np.ndarray) -> int:
    # 1-based max index with finite value; 0 if none.
    try:
        m = 0
        for i, v in enumerate(arr.tolist()):
            try:
                if np.isfinite(float(v)):
                    m = i + 1
            except Exception:
                continue
        return int(m)
    except Exception:
        return 0


def _parse_json_list(s: Any) -> List[Any]:
    if s is None:
        return []
    if isinstance(s, list):
        return s
    try:
        return json.loads(str(s))
    except Exception:
        return []


def _parse_A_row_json(s: Any) -> List[Optional[float]]:
    arr = _parse_json_list(s)
    out: List[Optional[float]] = []
    for v in arr:
        out.append(_safe_float(v))
    return out


def _compute_forgetting_final(*, rows_by_t: Dict[int, List[Optional[float]]]) -> Optional[float]:
    """Final-session forgetting for scalar metrics (higher-is-better)."""
    if not rows_by_t:
        return None
    t_end = max(int(t) for t in rows_by_t.keys())
    if t_end <= 1:
        return None
    last_row = rows_by_t.get(t_end)
    if not last_row or len(last_row) != t_end:
        return None

    diffs: List[float] = []
    for j in range(0, t_end - 1):
        cur = last_row[j] if j < len(last_row) else None
        if cur is None:
            continue
        prev_vals: List[float] = []
        for k in range(j + 1, t_end):
            row_k = rows_by_t.get(k)
            if not row_k or j >= len(row_k):
                continue
            vv = row_k[j]
            if vv is None:
                continue
            prev_vals.append(float(vv))
        if not prev_vals:
            continue
        diffs.append(max(prev_vals) - float(cur))
    if not diffs:
        return None
    return float(np.mean(np.asarray(diffs, dtype=np.float64)))


def _build_forgetting_rows_from_curve(curve_rows: List["CurveRow"]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List["CurveRow"]] = {}
    for r in curve_rows:
        key = (r.exp_family, r.variant, r.seed, r.metric_key)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda x: x[0]):
        exp_family, variant, seed, metric_key = key
        rows_by_t: Dict[int, List[Optional[float]]] = {}
        t_end = 0
        task_order_json = ""
        source_dir = ""
        split = "val"
        for r in rs:
            t = int(r.t)
            t_end = max(t_end, t)
            rows_by_t[t] = _parse_A_row_json(r.A_row_json)
            split = str(r.split)
            if not task_order_json:
                task_order_json = str(r.task_order_json)
            if not source_dir:
                source_dir = str(r.source_dir)
        out.append(
            {
                "benchmark": "skill",
                "exp_family": exp_family,
                "kind": "",
                "head": "",
                "variant": variant,
                "split": split,
                "mode": "",
                "seed": seed,
                "metric_key": metric_key,
                "t_end": t_end,
                "forgetting_final": _compute_forgetting_final(rows_by_t=rows_by_t),
                "task_order_json": task_order_json,
                "source_dir": source_dir,
            }
        )
    return out


def _summarize_forgetting_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (
            str(r.get("benchmark", "")),
            str(r.get("exp_family", "")),
            str(r.get("variant", "")),
            str(r.get("split", "")),
            str(r.get("metric_key", "")),
        )
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda x: x[0]):
        seed_to_val: Dict[str, float] = {}
        for x in rs:
            s = str(x.get("seed", ""))
            v = _safe_float(x.get("forgetting_final"))
            if s == "" or v is None:
                continue
            seed_to_val[s] = float(v)
        seeds = sorted(seed_to_val.keys())
        if len(seeds) < MIN_VALID_SEEDS_FOR_SUMMARY:
            all_seeds = sorted({str(x.get("seed", "")) for x in rs if str(x.get("seed", "")) != ""})
            _warn_insufficient_summary(
                kind="forgetting_summary",
                key=key,
                details=f"valid_seeds={seeds}; all_seeds={all_seeds}",
            )
            continue
        vals: List[Optional[float]] = [seed_to_val[s] for s in seeds]
        out.append(
            {
                "benchmark": key[0],
                "exp_family": key[1],
                "variant": key[2],
                "split": key[3],
                "metric_key": key[4],
                "n_seeds": len(seeds),
                "seeds": _dump_json_compact(seeds),
                "forgetting_mean": _nanmean(vals),
                "forgetting_std": _nanstd_pop(vals),
            }
        )
    return out


@dataclass
class SeedRow:
    benchmark: str
    run_type: str  # joint_fair | continual
    exp_family: str  # joint_fair | continual
    kind: str  # ranking
    head: str
    variant: str  # baseline/rn/tl or i3d_baseline/i3d_rn/i3d_tl
    mode: str
    split: str  # val
    seed: str
    metric_key: str  # ranking_acc
    value_micro: Optional[float]
    value_macro: Optional[float]
    ego2exo: Optional[float]
    exo2ego: Optional[float]
    source_path: str


@dataclass
class CurveRow:
    benchmark: str
    exp_family: str
    kind: str
    head: str
    variant: str
    split: str
    mode: str
    seed: str
    t: int
    metric_key: str
    task_order_json: str
    A_row_json: str
    bar_A_micro: Optional[float]
    bar_A_macro: Optional[float]
    ego2exo: Optional[float]
    exo2ego: Optional[float]
    source_dir: str


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _summarize_seed_rows(rows: List[SeedRow]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, str, str], List[SeedRow]] = {}
    for r in rows:
        key = (r.benchmark, r.run_type, r.exp_family, r.variant, r.split, r.metric_key)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda x: x[0]):
        micro_by_seed = _vals_by_seed(rs, "value_micro")
        macro_by_seed = _vals_by_seed(rs, "value_macro")
        micro_seeds = sorted(micro_by_seed.keys())
        if len(micro_seeds) < MIN_VALID_SEEDS_FOR_SUMMARY:
            all_seeds = sorted({str(r.seed) for r in rs if str(r.seed) != ""})
            _warn_insufficient_summary(
                kind="results_summary",
                key=key,
                details=(
                    f"micro_valid={micro_seeds}; macro_valid={sorted(macro_by_seed.keys())}; "
                    f"all_seeds={all_seeds}"
                ),
            )
            continue
        micro_vals: List[Optional[float]] = [micro_by_seed[s] for s in micro_seeds]
        macro_vals_all = [macro_by_seed.get(s) for s in micro_seeds]
        macro_valid_n = len([v for v in macro_vals_all if v is not None])
        if 0 < macro_valid_n < MIN_VALID_SEEDS_FOR_SUMMARY:
            _warn_insufficient_summary(
                kind="results_summary_macro",
                key=key,
                details=f"macro_valid_n={macro_valid_n}; micro_seeds={micro_seeds}; macro_valid={sorted(macro_by_seed.keys())}",
            )
        macro_mean = _nanmean(macro_vals_all) if macro_valid_n >= MIN_VALID_SEEDS_FOR_SUMMARY else None
        macro_std = _nanstd_pop(macro_vals_all) if macro_valid_n >= MIN_VALID_SEEDS_FOR_SUMMARY else None
        out.append(
            {
                "benchmark": key[0],
                "run_type": key[1],
                "exp_family": key[2],
                "variant": key[3],
                "split": key[4],
                "metric_key": key[5],
                "n_seeds": len(micro_seeds),
                "seeds": _dump_json_compact(micro_seeds),
                "micro_mean": _nanmean(micro_vals),
                "micro_std": _nanstd_pop(micro_vals),
                "macro_mean": macro_mean,
                "macro_std": macro_std,
            }
        )
    return out


def _scan_joint_fair_seed_rows(exps_root: str) -> List[SeedRow]:
    out: List[SeedRow] = []
    jf_root = os.path.join(exps_root, "joint_fair")
    if not os.path.isdir(jf_root):
        return out

    # structure: exps/joint_fair/{i3d_baseline|i3d_rn|i3d_tl}/seedX/joint_fair_eval.json
    for variant in sorted(os.listdir(jf_root)):
        vdir = os.path.join(jf_root, variant)
        if not os.path.isdir(vdir):
            continue
        for seed_dir in sorted(os.listdir(vdir)):
            m_seed = re.match(r"seed(\d+)$", seed_dir)
            if not m_seed:
                continue
            seed = m_seed.group(1)
            p = os.path.join(vdir, seed_dir, "joint_fair_eval.json")
            if not os.path.isfile(p):
                continue
            d = _read_json(p)
            micro = _safe_float(d.get("best_micro_avg"))
            macro = _safe_float(d.get("best_macro_avg"))
            out.append(
                SeedRow(
                    benchmark="skill",
                    run_type="joint_fair",
                    exp_family="joint_fair",
                    kind="ranking",
                    head="",
                    variant=variant,
                    mode="",
                    split="val",
                    seed=seed,
                    metric_key="ranking_acc",
                    value_micro=micro,
                    value_macro=macro,
                    ego2exo=None,
                    exo2ego=None,
                    source_path=p,
                )
            )
    return out


def _scan_continual_seed_and_curve_rows(exps_root: str) -> Tuple[List[SeedRow], List[CurveRow]]:
    seed_rows: List[SeedRow] = []
    curve_rows: List[CurveRow] = []
    for run_type in _discover_run_types(exps_root):
        cont_root = os.path.join(exps_root, run_type)
        if not os.path.isdir(cont_root):
            continue

        for variant in sorted(os.listdir(cont_root)):
            variant_dir = os.path.join(cont_root, variant)
            if not os.path.isdir(variant_dir):
                continue
            for seed_dir in sorted(os.listdir(variant_dir)):
                m_seed = re.match(r"seed(\d+)$", seed_dir)
                if not m_seed:
                    continue
                seed = m_seed.group(1)
                seed_path = os.path.join(variant_dir, seed_dir)
                task_order_path = os.path.join(seed_path, "task_order.json")
                if not os.path.isfile(task_order_path):
                    continue
                task_order = _read_json(task_order_path)
                T = int(len(task_order))
                task_order_json = _dump_json_compact(task_order)

                A_path = os.path.join(seed_path, "continual_A_ranking_acc.npy")
                bar_micro_path = os.path.join(seed_path, "continual_bar_A_micro_ranking_acc.npy")
                bar_macro_path = os.path.join(seed_path, "continual_bar_A_macro_ranking_acc.npy")
                if not (os.path.isfile(A_path) and os.path.isfile(bar_micro_path) and os.path.isfile(bar_macro_path)):
                    continue

                A = np.load(A_path)
                bar_micro = np.load(bar_micro_path)
                bar_macro = np.load(bar_macro_path)
                if A.ndim != 2 or A.shape[0] < T or A.shape[1] < T:
                    continue

                t_max = min(_max_finite_t(bar_micro), _max_finite_t(bar_macro))
                if T > 0 and t_max != T:
                    _warn_incomplete_run(
                        run_type=run_type,
                        variant=variant,
                        seed=seed,
                        t_max=t_max,
                        t_total=T,
                        seed_path=seed_path,
                    )
                    continue

                for t in range(1, T + 1):
                    row_vals = A[t - 1, :t].astype(np.float64).tolist()
                    row_vals = [None if (v is None or not np.isfinite(float(v))) else float(v) for v in row_vals]
                    curve_rows.append(
                        CurveRow(
                            benchmark="skill",
                            exp_family=run_type,
                            kind="ranking",
                            head="",
                            variant=variant,
                            split="val",
                            mode="",
                            seed=seed,
                            t=t,
                            metric_key="ranking_acc",
                            task_order_json=task_order_json,
                            A_row_json=_dump_json_compact(row_vals),
                            bar_A_micro=_safe_float(bar_micro[t - 1]) if t - 1 < len(bar_micro) else None,
                            bar_A_macro=_safe_float(bar_macro[t - 1]) if t - 1 < len(bar_macro) else None,
                            ego2exo=None,
                            exo2ego=None,
                            source_dir=seed_path,
                        )
                    )

                seed_rows.append(
                    SeedRow(
                        benchmark="skill",
                        run_type=run_type,
                        exp_family=run_type,
                        kind="ranking",
                        head="",
                        variant=variant,
                        mode="",
                        split="val",
                        seed=seed,
                        metric_key="ranking_acc",
                        value_micro=_safe_float(bar_micro[T - 1]) if T - 1 < len(bar_micro) else None,
                        value_macro=_safe_float(bar_macro[T - 1]) if T - 1 < len(bar_macro) else None,
                        ego2exo=None,
                        exo2ego=None,
                        source_path=seed_path,
                    )
                )

    return seed_rows, curve_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps_dir", type=str, default=EXPS_DIR, help="Skill exps/ directory")
    args = ap.parse_args()

    exps_dir = os.path.abspath(args.exps_dir)
    if not os.path.isdir(exps_dir):
        raise SystemExit(f"exps_dir not found: {exps_dir}")

    seed_rows: List[SeedRow] = []
    curve_rows: List[CurveRow] = []

    seed_rows.extend(_scan_joint_fair_seed_rows(exps_dir))
    cont_seed, cont_curve = _scan_continual_seed_and_curve_rows(exps_dir)
    seed_rows.extend(cont_seed)
    curve_rows.extend(cont_curve)

    # stable ordering
    def _seed_key(r: SeedRow) -> Tuple:
        seed_num = int(r.seed) if r.seed.isdigit() else -1
        return (r.run_type, r.exp_family, r.variant, seed_num, r.metric_key)

    seed_rows.sort(key=_seed_key)
    curve_rows.sort(key=lambda r: (r.variant, int(r.seed), r.metric_key, r.t))

    seed_csv = os.path.join(exps_dir, "results_seed.csv")
    _write_csv(
        seed_csv,
        [
            {
                "benchmark": r.benchmark,
                "run_type": r.run_type,
                "exp_family": r.exp_family,
                "kind": r.kind,
                "head": r.head,
                "variant": r.variant,
                "mode": r.mode,
                "split": r.split,
                "seed": r.seed,
                "metric_key": r.metric_key,
                "value_micro": r.value_micro,
                "value_macro": r.value_macro,
                "ego2exo": r.ego2exo,
                "exo2ego": r.exo2ego,
                "source_path": r.source_path,
            }
            for r in seed_rows
        ],
        fieldnames=[
            "benchmark",
            "run_type",
            "exp_family",
            "kind",
            "head",
            "variant",
            "mode",
            "split",
            "seed",
            "metric_key",
            "value_micro",
            "value_macro",
            "ego2exo",
            "exo2ego",
            "source_path",
        ],
    )

    summary_csv = os.path.join(exps_dir, "results_summary.csv")
    _write_csv(
        summary_csv,
        _summarize_seed_rows(seed_rows),
        fieldnames=[
            "benchmark",
            "run_type",
            "exp_family",
            "variant",
            "split",
            "metric_key",
            "n_seeds",
            "seeds",
            "micro_mean",
            "micro_std",
            "macro_mean",
            "macro_std",
        ],
    )

    curve_csv = os.path.join(exps_dir, "results_continual_curve.csv")
    _write_csv(
        curve_csv,
        [
            {
                "benchmark": r.benchmark,
                "exp_family": r.exp_family,
                "kind": r.kind,
                "head": r.head,
                "variant": r.variant,
                "split": r.split,
                "mode": r.mode,
                "seed": r.seed,
                "t": r.t,
                "metric_key": r.metric_key,
                "task_order_json": r.task_order_json,
                "A_row_json": r.A_row_json,
                "bar_A_micro": r.bar_A_micro,
                "bar_A_macro": r.bar_A_macro,
                "ego2exo": r.ego2exo,
                "exo2ego": r.exo2ego,
                "source_dir": r.source_dir,
            }
            for r in curve_rows
        ],
        fieldnames=[
            "benchmark",
            "exp_family",
            "kind",
            "head",
            "variant",
            "split",
            "mode",
            "seed",
            "t",
            "metric_key",
            "task_order_json",
            "A_row_json",
            "bar_A_micro",
            "bar_A_macro",
            "ego2exo",
            "exo2ego",
            "source_dir",
        ],
    )

    print(f"[Skill] wrote: {seed_csv}")
    print(f"[Skill] wrote: {summary_csv}")
    print(f"[Skill] wrote: {curve_csv}")

    # write forgetting (final session) CSV
    forgetting_csv = os.path.join(exps_dir, "results_forgetting_seed.csv")
    forgetting_rows = _build_forgetting_rows_from_curve(curve_rows)
    _write_csv(
        forgetting_csv,
        forgetting_rows,
        fieldnames=[
            "benchmark",
            "exp_family",
            "kind",
            "head",
            "variant",
            "split",
            "mode",
            "seed",
            "metric_key",
            "t_end",
            "forgetting_final",
            "task_order_json",
            "source_dir",
        ],
    )
    print(f"[Skill] wrote: {forgetting_csv}")

    forgetting_summary_csv = os.path.join(exps_dir, "results_forgetting_summary.csv")
    _write_csv(
        forgetting_summary_csv,
        _summarize_forgetting_rows(forgetting_rows),
        fieldnames=[
            "benchmark",
            "exp_family",
            "variant",
            "split",
            "metric_key",
            "n_seeds",
            "seeds",
            "forgetting_mean",
            "forgetting_std",
        ],
    )
    print(f"[Skill] wrote: {forgetting_summary_csv}")
    if _WARN_LINES_SUPPRESSED > 0:
        print(f"[WARN][skill] suppressed {_WARN_LINES_SUPPRESSED} additional warnings (increase MAX_WARN_LINES to show more)")


if __name__ == "__main__":
    main()

