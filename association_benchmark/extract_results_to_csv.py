#!/usr/bin/env python3
"""
Extract Association benchmark experiment results under ./exps/ and write CSVs into ./exps/:
  - results_seed.csv            (joint_fair + continual final results, seed-level, long format)
  - results_summary.csv         (mean/std across seeds grouped by experiment keys)
  - results_continual_curve.csv (continual curve, unified schema with other benchmarks)

Authoritative sources:
  - joint_fair: exps/joint_fair/{val|test}_{mode}_seedX/joint_fair_testonly.json
  - continual families: exps/multi_seed_cl*/{val|test}_{mode}_seedX/metrics_testonly_tXX.json
      task_order is taken from: exps/<family>/train_{mode}_seedX/task_order.json

In this benchmark, bar_A is a micro (sample-count weighted) accuracy over all seen samples.
We additionally compute bar_A_macro = mean(A_row) for curve CSV (requested for unified output).
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


def _nanmean_min_n(xs: Iterable[Optional[float]], min_n: int) -> Optional[float]:
    vals = [float(v) for v in xs if v is not None and np.isfinite(float(v))]
    if len(vals) < int(min_n):
        return None
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _nanstd_pop_min_n(xs: Iterable[Optional[float]], min_n: int) -> Optional[float]:
    vals = [float(v) for v in xs if v is not None and np.isfinite(float(v))]
    if len(vals) < int(min_n):
        return None
    return float(np.std(np.asarray(vals, dtype=np.float64), ddof=0))


def _warn_insufficient_summary(*, kind: str, key: Tuple[Any, ...], details: str) -> None:
    global _WARN_LINES_PRINTED, _WARN_LINES_SUPPRESSED
    if _WARN_LINES_PRINTED >= MAX_WARN_LINES:
        _WARN_LINES_SUPPRESSED += 1
        return
    _WARN_LINES_PRINTED += 1
    print(f"[WARN][association][{kind}] <{MIN_VALID_SEEDS_FOR_SUMMARY} valid seeds; key={key}; {details}")


def _warn_incomplete_run(*, run_type: str, exp_family: str, split: str, mode: str, seed: str, t_max: int, t_total: int, source_dir: str) -> None:
    global _WARN_LINES_PRINTED, _WARN_LINES_SUPPRESSED
    if _WARN_LINES_PRINTED >= MAX_WARN_LINES:
        _WARN_LINES_SUPPRESSED += 1
        return
    _WARN_LINES_PRINTED += 1
    print(
        f"[WARN][association][incomplete_run] skip (t_max={t_max} != T={t_total}); "
        f"run_type={run_type}; exp_family={exp_family}; split={split}; mode={mode}; seed={seed}; dir={source_dir}"
    )

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


def _compute_scalar_forgetting_final(*, values_by_t: Dict[int, Optional[float]]) -> Optional[float]:
    """Final-session forgetting for a scalar per-session series (higher-is-better).

    values_by_t: 1-based session t -> value at session t.
    """
    if not values_by_t:
        return None
    t_end = max(int(t) for t in values_by_t.keys())
    if t_end <= 1:
        return None
    last = values_by_t.get(t_end)
    if last is None:
        return None
    prev = []
    for t in range(1, t_end):
        v = values_by_t.get(t)
        if v is None:
            continue
        prev.append(float(v))
    if not prev:
        return None
    return float(max(prev) - float(last))


def _build_forgetting_rows_from_curve(curve_rows: List["CurveRow"]) -> List[Dict[str, Any]]:
    # group by identifiers for association continual curve
    groups: Dict[Tuple[str, str, str, str, str], List["CurveRow"]] = {}
    for r in curve_rows:
        key = (r.exp_family, r.split, r.mode, r.seed, r.metric_key)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda x: x[0]):
        exp_family, split, mode, seed, metric_key = key
        rows_by_t: Dict[int, List[Optional[float]]] = {}
        ego2exo_by_t: Dict[int, Optional[float]] = {}
        exo2ego_by_t: Dict[int, Optional[float]] = {}
        t_end = 0
        task_order_json = ""
        source_dir = ""
        for r in rs:
            t = int(r.t)
            t_end = max(t_end, t)
            rows_by_t[t] = _parse_A_row_json(r.A_row_json)
            ego2exo_by_t[t] = _safe_float(r.ego2exo)
            exo2ego_by_t[t] = _safe_float(r.exo2ego)
            if not task_order_json:
                task_order_json = str(r.task_order_json)
            if not source_dir:
                source_dir = str(r.source_dir)
        out.append(
            {
                "benchmark": "association",
                "exp_family": exp_family,
                "kind": "",
                "head": "",
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

        # Association-specific: for acc metric, also export direction and avg forgetting.
        # - acc_ego2exo / acc_exo2ego: scalar series over sessions from curve row columns.
        # - acc_avg: per-session average of the two directions, then scalar forgetting.
        if str(metric_key) == "acc":
            avg_by_t: Dict[int, Optional[float]] = {}
            for t in range(1, t_end + 1):
                a = ego2exo_by_t.get(t)
                b = exo2ego_by_t.get(t)
                if a is None or b is None:
                    avg_by_t[t] = None
                else:
                    avg_by_t[t] = (float(a) + float(b)) / 2.0

            out.append(
                {
                    "benchmark": "association",
                    "exp_family": exp_family,
                    "kind": "",
                    "head": "",
                    "variant": "",
                    "split": split,
                    "mode": mode,
                    "seed": seed,
                    "metric_key": "acc_ego2exo",
                    "t_end": t_end,
                    "forgetting_final": _compute_scalar_forgetting_final(values_by_t=ego2exo_by_t),
                    "task_order_json": task_order_json,
                    "source_dir": source_dir,
                }
            )
            out.append(
                {
                    "benchmark": "association",
                    "exp_family": exp_family,
                    "kind": "",
                    "head": "",
                    "variant": "",
                    "split": split,
                    "mode": mode,
                    "seed": seed,
                    "metric_key": "acc_exo2ego",
                    "t_end": t_end,
                    "forgetting_final": _compute_scalar_forgetting_final(values_by_t=exo2ego_by_t),
                    "task_order_json": task_order_json,
                    "source_dir": source_dir,
                }
            )
            out.append(
                {
                    "benchmark": "association",
                    "exp_family": exp_family,
                    "kind": "",
                    "head": "",
                    "variant": "",
                    "split": split,
                    "mode": mode,
                    "seed": seed,
                    "metric_key": "acc_avg",
                    "t_end": t_end,
                    "forgetting_final": _compute_scalar_forgetting_final(values_by_t=avg_by_t),
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
            str(r.get("split", "")),
            str(r.get("mode", "")),
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
                "split": key[2],
                "mode": key[3],
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
    exp_family: str  # joint_fair | multi_seed_cl*
    kind: str  # mcq
    head: str  # unused
    variant: str  # unused
    mode: str
    split: str  # val/test
    seed: str
    metric_key: str  # acc
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
    # Group by identifiers excluding seed/source_path and compute mean/std for micro/macro (and directions).
    groups: Dict[Tuple[str, str, str, str, str, str, str], List[SeedRow]] = {}
    for r in rows:
        key = (r.benchmark, r.run_type, r.exp_family, r.split, r.mode, r.kind, r.metric_key)
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

        ego2exo_vals = [_safe_float(getattr(r, "ego2exo")) for r in rs if str(r.seed) in set(micro_seeds)]
        exo2ego_vals = [_safe_float(getattr(r, "exo2ego")) for r in rs if str(r.seed) in set(micro_seeds)]
        ego2exo_valid_n = len([v for v in ego2exo_vals if v is not None])
        exo2ego_valid_n = len([v for v in exo2ego_vals if v is not None])
        if 0 < ego2exo_valid_n < MIN_VALID_SEEDS_FOR_SUMMARY:
            _warn_insufficient_summary(
                kind="direction_summary",
                key=key,
                details=f"ego2exo_valid_n={ego2exo_valid_n}; micro_seeds={micro_seeds}",
            )
        if 0 < exo2ego_valid_n < MIN_VALID_SEEDS_FOR_SUMMARY:
            _warn_insufficient_summary(
                kind="direction_summary",
                key=key,
                details=f"exo2ego_valid_n={exo2ego_valid_n}; micro_seeds={micro_seeds}",
            )
        out.append(
            {
                "benchmark": key[0],
                "run_type": key[1],
                "exp_family": key[2],
                "split": key[3],
                "mode": key[4],
                "kind": key[5],
                "metric_key": key[6],
                "n_seeds": len(micro_seeds),
                "seeds": _dump_json_compact(micro_seeds),
                "micro_mean": _nanmean(micro_vals),
                "micro_std": _nanstd_pop(micro_vals),
                "macro_mean": macro_mean,
                "macro_std": macro_std,
                # Directions may be missing for some runs; only summarize when enough values exist.
                "ego2exo_mean": _nanmean_min_n(ego2exo_vals, MIN_VALID_SEEDS_FOR_SUMMARY),
                "ego2exo_std": _nanstd_pop_min_n(ego2exo_vals, MIN_VALID_SEEDS_FOR_SUMMARY),
                "exo2ego_mean": _nanmean_min_n(exo2ego_vals, MIN_VALID_SEEDS_FOR_SUMMARY),
                "exo2ego_std": _nanstd_pop_min_n(exo2ego_vals, MIN_VALID_SEEDS_FOR_SUMMARY),
            }
        )
    return out


def _scan_joint_fair_seed_rows(exps_root: str) -> List[SeedRow]:
    out: List[SeedRow] = []
    jf_root = os.path.join(exps_root, "joint_fair")
    if not os.path.isdir(jf_root):
        return out

    for exp in sorted(os.listdir(jf_root)):
        m = re.match(r"^(val|test)_(egoonly|exoonly|egoexo)_seed(\d+)$", exp)
        if not m:
            continue
        split, mode, seed = m.group(1), m.group(2), m.group(3)
        p = os.path.join(jf_root, exp, "joint_fair_testonly.json")
        if not os.path.isfile(p):
            continue
        d = _read_json(p)
        overall = _safe_float(d.get("overall_acc"))
        per_task = d.get("per_task_acc", {})
        # macro over tasks (equal-weight over tasks)
        macro = _nanmean([_safe_float(v) for v in per_task.values()])
        direction = d.get("direction_acc", {}) or {}
        ego2exo = _safe_float(direction.get("Ego->Exo"))
        exo2ego = _safe_float(direction.get("Exo->Ego"))
        out.append(
            SeedRow(
                benchmark="association",
                run_type="joint_fair",
                exp_family="joint_fair",
                kind="mcq",
                head="",
                variant="",
                mode=mode,
                split=split,
                seed=seed,
                metric_key="acc",
                value_micro=overall,
                value_macro=macro,
                ego2exo=ego2exo,
                exo2ego=exo2ego,
                source_path=p,
            )
        )
    return out


def _scan_continual_seed_and_curve_rows(exps_root: str) -> Tuple[List[SeedRow], List[CurveRow]]:
    seed_rows: List[SeedRow] = []
    curve_rows: List[CurveRow] = []

    # families: (exp_family_label, family_dir, run_type)
    families: List[Tuple[str, str, str]] = []

    # Standard continual families: exps/multi_seed_cl*
    for name in os.listdir(exps_root):
        if name.startswith("multi_seed_cl"):
            p = os.path.join(exps_root, name)
            if os.path.isdir(p):
                families.append((name, p, "continual"))

    # Continual_* families that use the common multi_seed layout, including:
    #   - continual_ewc/multi_seed
    #   - continual_lwf/multi_seed
    #   - continual_er*/multi_seed (incl. variants like br0p1/rr0p1)
    #   - continual_derpp*/multi_seed (incl. variants)
    # We keep the folder name as run_type so CSV distinguishes variants.
    for name in os.listdir(exps_root):
        if not str(name).startswith("continual_"):
            continue
        multi_seed_dir = os.path.join(exps_root, name, "multi_seed")
        if os.path.isdir(multi_seed_dir):
            families.append((f"{name}_multi_seed", multi_seed_dir, str(name)))

    families.sort(key=lambda x: x[0])

    for fam, fam_dir, run_type in families:
        # val/test dirs contain metrics_testonly_tXX.json
        for exp in sorted(os.listdir(fam_dir)):
            m = re.match(r"^(val|test)_(egoonly|exoonly|egoexo)_seed(\d+)$", exp)
            if not m:
                continue
            split, mode, seed = m.group(1), m.group(2), m.group(3)
            exp_dir = os.path.join(fam_dir, exp)
            if not os.path.isdir(exp_dir):
                continue

            # task_order from corresponding train dir (as requested)
            train_dir = os.path.join(fam_dir, f"train_{mode}_seed{seed}")
            task_order_path = os.path.join(train_dir, "task_order.json")
            if not os.path.isfile(task_order_path):
                # cannot unify without task order
                continue
            task_order = _read_json(task_order_path)
            task_order_json = _dump_json_compact(task_order)
            T_total = int(len(task_order))

            # gather all t files
            t_files = []
            for fn in os.listdir(exp_dir):
                mm = re.match(r"^metrics_testonly_t(\d+)\.json$", fn)
                if mm:
                    t_files.append((int(mm.group(1)), os.path.join(exp_dir, fn)))
            t_files.sort(key=lambda x: x[0])
            if not t_files:
                continue

            t_max = int(t_files[-1][0])
            if T_total > 0 and t_max != T_total:
                _warn_incomplete_run(
                    run_type=run_type,
                    exp_family=fam,
                    split=split,
                    mode=mode,
                    seed=seed,
                    t_max=t_max,
                    t_total=T_total,
                    source_dir=exp_dir,
                )
                continue

            # curve rows
            last = None
            for t_idx, p in t_files:
                d = _read_json(p)
                A_row = d.get("A_row", [])
                # sanitize row vals for JSON (nan -> None)
                row_vals = []
                for v in A_row:
                    fv = _safe_float(v)
                    row_vals.append(None if fv is None else fv)
                bar_micro = _safe_float(d.get("bar_A"))
                bar_macro = _nanmean(row_vals)
                ego2exo = _safe_float(d.get("Ego->Exo"))
                exo2ego = _safe_float(d.get("Exo->Ego"))
                curve_rows.append(
                    CurveRow(
                        benchmark="association",
                        exp_family=fam,
                        kind="mcq",
                        head="",
                        variant="",
                        split=split,
                        mode=mode,
                        seed=seed,
                        t=int(t_idx),
                        metric_key="acc",
                        task_order_json=task_order_json,
                        A_row_json=_dump_json_compact(row_vals),
                        bar_A_micro=bar_micro,
                        bar_A_macro=bar_macro,
                        ego2exo=ego2exo,
                        exo2ego=exo2ego,
                        source_dir=exp_dir,
                    )
                )
                last = (t_idx, bar_micro, bar_macro, ego2exo, exo2ego, exp_dir)

            # final seed row uses last t
            if last is not None:
                seed_rows.append(
                    SeedRow(
                        benchmark="association",
                        run_type=run_type,
                        exp_family=fam,
                        kind="mcq",
                        head="",
                        variant="",
                        mode=mode,
                        split=split,
                        seed=seed,
                        metric_key="acc",
                        value_micro=last[1],
                        value_macro=last[2],
                        ego2exo=last[3],
                        exo2ego=last[4],
                        source_path=last[5],
                    )
                )

    return seed_rows, curve_rows


def _scan_continual_vista_seed_and_curve_rows(exps_root: str) -> Tuple[List[SeedRow], List[CurveRow]]:
    """Scan VISTA continual experiments.

    Layout:
      exps/continual_vista_<router>/<mode>/seedX/{train,val,test}/...
        - task_order: train/task_order.json
        - split metrics: {val|test}/metrics_testonly_tXX.json
    """
    seed_rows: List[SeedRow] = []
    curve_rows: List[CurveRow] = []

    if not os.path.isdir(exps_root):
        return seed_rows, curve_rows

    prefix = "continual_vista_"
    router_roots: List[Tuple[str, str]] = []
    for name in os.listdir(exps_root):
        if not name.startswith(prefix):
            continue
        p = os.path.join(exps_root, name)
        if os.path.isdir(p):
            router_name = name[len(prefix) :]
            if router_name:
                router_roots.append((router_name, p))
    router_roots.sort(key=lambda x: x[0])

    for router_name, router_dir in router_roots:
        exp_family = f"continual_vista_{router_name}"
        for mode in sorted(os.listdir(router_dir)):
            mode_dir = os.path.join(router_dir, mode)
            if not os.path.isdir(mode_dir):
                continue
            for seed_dir in sorted(os.listdir(mode_dir)):
                m_seed = re.match(r"^seed(\d+)$", seed_dir)
                if not m_seed:
                    continue
                seed = m_seed.group(1)
                seed_root = os.path.join(mode_dir, seed_dir)
                task_order_path = os.path.join(seed_root, "train", "task_order.json")
                if not os.path.isfile(task_order_path):
                    continue
                task_order = _read_json(task_order_path)
                task_order_json = _dump_json_compact(task_order)
                T_total = int(len(task_order))

                for split in ["val", "test"]:
                    split_dir = os.path.join(seed_root, split)
                    if not os.path.isdir(split_dir):
                        continue
                    t_files = []
                    for fn in os.listdir(split_dir):
                        mm = re.match(r"^metrics_testonly_t(\d+)\.json$", fn)
                        if mm:
                            t_files.append((int(mm.group(1)), os.path.join(split_dir, fn)))
                    t_files.sort(key=lambda x: x[0])
                    if not t_files:
                        continue

                    t_max = int(t_files[-1][0])
                    if T_total > 0 and t_max != T_total:
                        _warn_incomplete_run(
                            run_type="continual_vista",
                            exp_family=exp_family,
                            split=split,
                            mode=mode,
                            seed=seed,
                            t_max=t_max,
                            t_total=T_total,
                            source_dir=split_dir,
                        )
                        continue

                    last = None
                    for t_idx, p in t_files:
                        d = _read_json(p)
                        A_row = d.get("A_row", [])
                        row_vals = []
                        for v in (A_row if isinstance(A_row, list) else []):
                            fv = _safe_float(v)
                            row_vals.append(None if fv is None else fv)
                        bar_micro = _safe_float(d.get("bar_A"))
                        bar_macro = _nanmean(row_vals)
                        ego2exo = _safe_float(d.get("Ego->Exo"))
                        exo2ego = _safe_float(d.get("Exo->Ego"))
                        curve_rows.append(
                            CurveRow(
                                benchmark="association",
                                exp_family=exp_family,
                                kind="mcq",
                                head="",
                                variant="",
                                split=split,
                                mode=mode,
                                seed=seed,
                                t=int(t_idx),
                                metric_key="acc",
                                task_order_json=task_order_json,
                                A_row_json=_dump_json_compact(row_vals),
                                bar_A_micro=bar_micro,
                                bar_A_macro=bar_macro,
                                ego2exo=ego2exo,
                                exo2ego=exo2ego,
                                source_dir=split_dir,
                            )
                        )
                        last = (bar_micro, bar_macro, ego2exo, exo2ego, p)

                    if last is not None:
                        seed_rows.append(
                            SeedRow(
                                benchmark="association",
                                run_type="continual_vista",
                                exp_family=exp_family,
                                kind="mcq",
                                head="",
                                variant="",
                                mode=mode,
                                split=split,
                                seed=seed,
                                metric_key="acc",
                                value_micro=last[0],
                                value_macro=last[1],
                                ego2exo=last[2],
                                exo2ego=last[3],
                                source_path=last[4],
                            )
                        )

    return seed_rows, curve_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps_dir", type=str, default=EXPS_DIR, help="Association exps/ directory")
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
    vista_seed, vista_curve = _scan_continual_vista_seed_and_curve_rows(exps_dir)
    seed_rows.extend(vista_seed)
    curve_rows.extend(vista_curve)

    seed_rows.sort(key=lambda r: (r.run_type, r.exp_family, r.split, r.mode, int(r.seed), r.metric_key))
    curve_rows.sort(key=lambda r: (r.exp_family, r.split, r.mode, int(r.seed), r.metric_key, r.t))

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
            "split",
            "mode",
            "kind",
            "metric_key",
            "n_seeds",
            "seeds",
            "micro_mean",
            "micro_std",
            "macro_mean",
            "macro_std",
            "ego2exo_mean",
            "ego2exo_std",
            "exo2ego_mean",
            "exo2ego_std",
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

    print(f"[Association] wrote: {seed_csv}")
    print(f"[Association] wrote: {summary_csv}")
    print(f"[Association] wrote: {curve_csv}")

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
    print(f"[Association] wrote: {forgetting_csv}")

    forgetting_summary_csv = os.path.join(exps_dir, "results_forgetting_summary.csv")
    _write_csv(
        forgetting_summary_csv,
        _summarize_forgetting_rows(forgetting_rows),
        fieldnames=[
            "benchmark",
            "exp_family",
            "split",
            "mode",
            "metric_key",
            "n_seeds",
            "seeds",
            "forgetting_mean",
            "forgetting_std",
        ],
    )
    print(f"[Association] wrote: {forgetting_summary_csv}")
    if _WARN_LINES_SUPPRESSED > 0:
        print(f"[WARN][association] suppressed {_WARN_LINES_SUPPRESSED} additional warnings (increase MAX_WARN_LINES to show more)")


if __name__ == "__main__":
    main()

