#!/usr/bin/env python3
"""
Extract AAP (action anticipation & planning) experiment results under ./exps/
and write CSVs into ./exps/:
  - results_seed.csv           (joint_fair + continual final results, seed-level, long format)
  - results_summary.csv        (mean/std across seeds grouped by experiment keys)
  - results_continual_curve.csv (continual curve, unified schema with other benchmarks)

Authoritative sources:
  - joint_fair: seed*/joint_fair_eval.json
  - continual: continual_A_*.npy + continual_bar_A_{micro,macro}_*.npy + task_order.json
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


HERE = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR = os.path.join(HERE, "exps")
MIN_VALID_SEEDS_FOR_SUMMARY = 3
MAX_WARN_LINES = 1e5
_WARN_LINES_PRINTED = 0
_WARN_LINES_SUPPRESSED = 0


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
    # Keep as a single line for easy grep.
    global _WARN_LINES_PRINTED, _WARN_LINES_SUPPRESSED
    if _WARN_LINES_PRINTED >= MAX_WARN_LINES:
        _WARN_LINES_SUPPRESSED += 1
        return
    _WARN_LINES_PRINTED += 1
    print(f"[WARN][aap][{kind}] <{MIN_VALID_SEEDS_FOR_SUMMARY} valid seeds; key={key}; {details}")


def _warn_incomplete_run(*, run_type: str, exp_name: str, seed: str, metric_key: str, t_max: int, t_total: int, seed_path: str) -> None:
    global _WARN_LINES_PRINTED, _WARN_LINES_SUPPRESSED
    if _WARN_LINES_PRINTED >= MAX_WARN_LINES:
        _WARN_LINES_SUPPRESSED += 1
        return
    _WARN_LINES_PRINTED += 1
    print(
        f"[WARN][aap][incomplete_run] skip (t_max={t_max} != T={t_total}); "
        f"run_type={run_type}; exp={exp_name}; seed={seed}; metric={metric_key}; dir={seed_path}"
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


def _macro_over_tasks(per_task: Dict[str, Any]) -> Optional[float]:
    # Task-equal average over task values (ignore NaN/None).
    vals = []
    for _, v in per_task.items():
        fv = _safe_float(v)
        if fv is not None:
            vals.append(fv)
    return _nanmean(vals)


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
    # past tasks j are 0..t_end-2
    for j in range(0, t_end - 1):
        cur = last_row[j] if j < len(last_row) else None
        if cur is None:
            continue
        prev_vals: List[float] = []
        # best_prev over k=j..t_end-2 (0-based) => 1-based k+1 in [j+1, t_end-1]
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
    # group by identifiers note: AAP split is always val-like but we keep split in the key.
    groups: Dict[Tuple[str, str, str, str, str, str, str], List["CurveRow"]] = {}
    for r in curve_rows:
        key = (r.exp_family, r.kind, r.head, r.mode, r.seed, r.metric_key, r.split)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda x: x[0]):
        exp_family, kind, head, mode, seed, metric_key, split = key
        rows_by_t: Dict[int, List[Optional[float]]] = {}
        t_end = 0
        task_order_json = ""
        source_dir = ""
        for r in rs:
            t = int(r.t)
            t_end = max(t_end, t)
            rows_by_t[t] = _parse_A_row_json(r.A_row_json)
            if not task_order_json:
                task_order_json = str(r.task_order_json)
            if not source_dir:
                source_dir = str(r.source_dir)
        out.append(
            {
                "benchmark": "aap",
                "exp_family": exp_family,
                "kind": kind,
                "head": head,
                "variant": "",
                "split": split,
                "mode": mode,
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
    # Group by identifiers excluding seed/source_dir and compute mean/std of forgetting_final.
    groups: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (
            str(r.get("benchmark", "")),
            str(r.get("exp_family", "")),
            str(r.get("kind", "")),
            str(r.get("head", "")),
            str(r.get("mode", "")),
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
                "kind": key[2],
                "head": key[3],
                "mode": key[4],
                "split": key[5],
                "metric_key": key[6],
                "n_seeds": len(seeds),
                "seeds": _dump_json_compact(seeds),
                "forgetting_mean": _nanmean(vals),
                "forgetting_std": _nanstd_pop(vals),
            }
        )
    return out


@dataclass
class SeedRow:
    # common identifiers
    benchmark: str
    run_type: str  # joint_fair | continual
    kind: str  # anticipation | planning
    head: str  # noun | verb | ""
    mode: str
    split: str  # for AAP, joint_fair/continual are val-like; keep "val"
    seed: str
    metric_key: str
    # values
    value_micro: Optional[float]
    value_macro: Optional[float]
    # provenance
    source_path: str


@dataclass
class CurveRow:
    benchmark: str
    exp_family: str  # continual
    kind: str
    head: str
    variant: str  # unused for AAP
    split: str  # val
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


def _seed_rows_to_csv_dicts(rows: List[SeedRow]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "benchmark": r.benchmark,
                "run_type": r.run_type,
                "kind": r.kind,
                "head": r.head,
                "mode": r.mode,
                "split": r.split,
                "seed": r.seed,
                "metric_key": r.metric_key,
                "value_micro": r.value_micro,
                "value_macro": r.value_macro,
                "source_path": r.source_path,
            }
        )
    return out


def _curve_rows_to_csv_dicts(rows: List[CurveRow]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
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
        )
    return out


def _summarize_seed_rows(rows: List[SeedRow]) -> List[Dict[str, Any]]:
    # Group by identifiers excluding seed/source_path and compute mean/std for micro/macro.
    groups: Dict[Tuple[str, str, str, str, str, str, str], List[SeedRow]] = {}
    for r in rows:
        key = (r.benchmark, r.run_type, r.kind, r.head, r.mode, r.split, r.metric_key)
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
                "kind": key[2],
                "head": key[3],
                "mode": key[4],
                "split": key[5],
                "metric_key": key[6],
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

    for dirpath, _, filenames in os.walk(jf_root):
        if "joint_fair_eval.json" not in filenames:
            continue
        p = os.path.join(dirpath, "joint_fair_eval.json")
        rel = os.path.relpath(p, jf_root)
        parts = rel.split(os.sep)
        if len(parts) < 3:
            continue
        exp_name = parts[0]  # anticipation_noun_ego_ego | planning_ego_ego
        seed_dir = parts[1]  # seed0
        m_seed = re.match(r"seed(\d+)$", seed_dir)
        if not m_seed:
            continue
        seed = m_seed.group(1)

        data = _read_json(p)

        if exp_name.startswith("anticipation_"):
            # anticipation_{head}_{mode}
            toks = exp_name.split("_")
            if len(toks) < 4:
                continue
            head = toks[1]
            mode = "_".join(toks[2:])
            metrics = data.get("metrics", {})
            micro = data.get("micro_avg", {})
            # metrics: {metric_key: {task: value}}
            for metric_key, per_task in metrics.items():
                v_micro = _safe_float(micro.get(metric_key))
                # derive macro-over-tasks for seed-level (task-equal average), for completeness
                v_macro = _macro_over_tasks(per_task or {})
                out.append(
                    SeedRow(
                        benchmark="aap",
                        run_type="joint_fair",
                        kind="anticipation",
                        head=head,
                        mode=mode,
                        split="val",
                        seed=seed,
                        metric_key=str(metric_key),
                        value_micro=v_micro,
                        value_macro=v_macro,
                        source_path=p,
                    )
                )
        elif exp_name.startswith("planning_"):
            mode = exp_name.split("_", 1)[1]
            # planning JSON stores per_task dict under 'per_task' with metric name 'ed_final'
            metric_key = str(data.get("metric", "ed_final"))
            per_task = data.get("per_task", {})
            v_micro = _safe_float(data.get("micro_avg"))
            v_macro = _macro_over_tasks(per_task or {})
            out.append(
                SeedRow(
                    benchmark="aap",
                    run_type="joint_fair",
                    kind="planning",
                    head="",
                    mode=mode,
                    split="val",
                    seed=seed,
                    metric_key=metric_key,
                    value_micro=v_micro,
                    value_macro=v_macro,
                    source_path=p,
                )
            )
    return out


def _scan_continual_seed_and_curve_rows(exps_root: str) -> Tuple[List[SeedRow], List[CurveRow]]:
    seed_rows: List[SeedRow] = []
    curve_rows: List[CurveRow] = []
    # Support continual families including ER/DERPP variants encoded in folder name, e.g.:
    #   exps/continual_er_br0p1_rr0p1/<exp_name>/seedX/...
    # We discover all folders matching "continual" or "continual_*" to avoid missing variants.
    run_types: List[str] = []
    try:
        for name in os.listdir(exps_root):
            p = os.path.join(exps_root, name)
            if not os.path.isdir(p):
                continue
            if str(name) == "continual" or str(name).startswith("continual_"):
                run_types.append(str(name))
    except Exception:
        pass
    run_types = sorted(list(dict.fromkeys(run_types)))

    for run_type in run_types:
        cont_root = os.path.join(exps_root, run_type)
        if not os.path.isdir(cont_root):
            continue

        # Scan seed dirs like: <run_type>/anticipation_noun_ego_ego/seed0/...
        for exp_name in sorted(os.listdir(cont_root)):
            exp_dir = os.path.join(cont_root, exp_name)
            if not os.path.isdir(exp_dir):
                continue
            for seed_dir in sorted(os.listdir(exp_dir)):
                m_seed = re.match(r"seed(\d+)$", seed_dir)
                if not m_seed:
                    continue
                seed = m_seed.group(1)
                seed_path = os.path.join(exp_dir, seed_dir)
                if not os.path.isdir(seed_path):
                    continue

                task_order_path = os.path.join(seed_path, "task_order.json")
                if not os.path.isfile(task_order_path):
                    # Without task_order we can't safely interpret A matrix ordering.
                    continue
                task_order = _read_json(task_order_path)
                T = int(len(task_order))
                task_order_json = _dump_json_compact(task_order)

                # Determine kind/head/mode
                kind = ""
                head = ""
                mode = ""
                if exp_name.startswith("anticipation_"):
                    kind = "anticipation"
                    toks = exp_name.split("_")
                    if len(toks) >= 4:
                        head = toks[1]
                        mode = "_".join(toks[2:])
                    else:
                        continue
                elif exp_name.startswith("planning_"):
                    kind = "planning"
                    mode = exp_name.split("_", 1)[1]
                else:
                    continue

                # Find available metric keys from continual_A_*.npy files
                metric_keys = []
                for fn in os.listdir(seed_path):
                    if fn.startswith("continual_A_") and fn.endswith(".npy") and not fn.startswith("continual_A0_"):
                        metric_keys.append(fn[len("continual_A_") : -len(".npy")])
                metric_keys = sorted(metric_keys)
                if not metric_keys:
                    continue

                # For each metric, load A + bar arrays and write:
                # - final seed row: t=T, bar values at t_end
                # - curve rows: t=1..T
                for metric_key in metric_keys:
                    A_path = os.path.join(seed_path, f"continual_A_{metric_key}.npy")
                    bar_micro_path = os.path.join(seed_path, f"continual_bar_A_micro_{metric_key}.npy")
                    bar_macro_path = os.path.join(seed_path, f"continual_bar_A_macro_{metric_key}.npy")

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
                            exp_name=exp_name,
                            seed=seed,
                            metric_key=metric_key,
                            t_max=t_max,
                            t_total=T,
                            seed_path=seed_path,
                        )
                        continue

                    # curve rows
                    for t in range(1, T + 1):
                        row_vals = A[t - 1, :t].astype(np.float64).tolist()
                        # convert nan to None for JSON
                        row_vals = [None if (v is None or not np.isfinite(float(v))) else float(v) for v in row_vals]
                        curve_rows.append(
                            CurveRow(
                                benchmark="aap",
                                exp_family=run_type,
                                kind=kind,
                                head=head,
                                variant="",
                                split="val",
                                mode=mode,
                                seed=seed,
                                t=t,
                                metric_key=metric_key,
                                task_order_json=task_order_json,
                                A_row_json=_dump_json_compact(row_vals),
                                bar_A_micro=_safe_float(bar_micro[t - 1]) if t - 1 < len(bar_micro) else None,
                                bar_A_macro=_safe_float(bar_macro[t - 1]) if t - 1 < len(bar_macro) else None,
                                ego2exo=None,
                                exo2ego=None,
                                source_dir=seed_path,
                            )
                        )

                    # final seed row uses t=T
                    seed_rows.append(
                        SeedRow(
                            benchmark="aap",
                            run_type=run_type,
                            kind=kind,
                            head=head,
                            mode=mode,
                            split="val",
                            seed=seed,
                            metric_key=metric_key,
                            value_micro=_safe_float(bar_micro[T - 1]) if T - 1 < len(bar_micro) else None,
                            value_macro=_safe_float(bar_macro[T - 1]) if T - 1 < len(bar_macro) else None,
                            source_path=seed_path,
                        )
                    )

    return seed_rows, curve_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps_dir", type=str, default=EXPS_DIR, help="AAP exps/ directory")
    args = ap.parse_args()

    exps_dir = os.path.abspath(args.exps_dir)
    if not os.path.isdir(exps_dir):
        raise SystemExit(f"exps_dir not found: {exps_dir}")

    seed_rows: List[SeedRow] = []
    curve_rows: List[CurveRow] = []

    # joint_fair seed rows
    seed_rows.extend(_scan_joint_fair_seed_rows(exps_dir))

    # continual seed + curve rows
    cont_seed, cont_curve = _scan_continual_seed_and_curve_rows(exps_dir)
    seed_rows.extend(cont_seed)
    curve_rows.extend(cont_curve)

    # sort for stable output
    seed_rows.sort(key=lambda r: (r.run_type, r.kind, r.head, r.mode, int(r.seed) if r.seed.isdigit() else r.seed, r.metric_key))
    curve_rows.sort(key=lambda r: (r.kind, r.head, r.mode, int(r.seed) if r.seed.isdigit() else r.seed, r.metric_key, r.t))

    # write seed-level CSV
    seed_csv = os.path.join(exps_dir, "results_seed.csv")
    seed_dicts = _seed_rows_to_csv_dicts(seed_rows)
    _write_csv(
        seed_csv,
        seed_dicts,
        fieldnames=[
            "benchmark",
            "run_type",
            "kind",
            "head",
            "mode",
            "split",
            "seed",
            "metric_key",
            "value_micro",
            "value_macro",
            "source_path",
        ],
    )

    # write summary CSV
    summary_csv = os.path.join(exps_dir, "results_summary.csv")
    summary_dicts = _summarize_seed_rows(seed_rows)
    _write_csv(
        summary_csv,
        summary_dicts,
        fieldnames=[
            "benchmark",
            "run_type",
            "kind",
            "head",
            "mode",
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

    # write continual curve CSV (unified schema)
    curve_csv = os.path.join(exps_dir, "results_continual_curve.csv")
    curve_dicts = _curve_rows_to_csv_dicts(curve_rows)
    _write_csv(
        curve_csv,
        curve_dicts,
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

    print(f"[AAP] wrote: {seed_csv}")
    print(f"[AAP] wrote: {summary_csv}")
    print(f"[AAP] wrote: {curve_csv}")

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
    print(f"[AAP] wrote: {forgetting_csv}")

    forgetting_summary_csv = os.path.join(exps_dir, "results_forgetting_summary.csv")
    _write_csv(
        forgetting_summary_csv,
        _summarize_forgetting_rows(forgetting_rows),
        fieldnames=[
            "benchmark",
            "exp_family",
            "kind",
            "head",
            "mode",
            "split",
            "metric_key",
            "n_seeds",
            "seeds",
            "forgetting_mean",
            "forgetting_std",
        ],
    )
    print(f"[AAP] wrote: {forgetting_summary_csv}")
    if _WARN_LINES_SUPPRESSED > 0:
        print(f"[WARN][aap] suppressed {_WARN_LINES_SUPPRESSED} additional warnings (increase MAX_WARN_LINES to show more)")


if __name__ == "__main__":
    main()

