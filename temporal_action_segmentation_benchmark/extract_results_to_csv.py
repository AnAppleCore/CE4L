#!/usr/bin/env python3
"""
Extract Temporal Action Segmentation (TAS) benchmark results under ./exps/ and write CSVs into ./exps/:
  - results_seed.csv            (joint_fair + continual final results, seed-level, long format)
  - results_summary.csv         (mean/std across seeds grouped by experiment keys)
  - results_continual_curve.csv (continual curve, unified schema with other benchmarks)

Key design choices (aligned with AAP/Association/Skill extractors):
  - Long format: one metric per row via metric_key (no wide-column reshaping).
  - Include derived metrics as additional metric_key values (f1_avg, avg) without changing schema.
  - For continual, prefer recorder outputs (.npy) as authoritative; support both VAL (seed root)
    and TEST (seed_root/test_metrics) curves.
  - For joint_fair, parse results/best_result_{val|test}.txt (authoritative line with Best Epoch).

Metrics:
  Base metrics present in both joint_fair and continual recorder:
    acc, edit, f1_010, f1_025, f1_050
  Derived metrics (added for alignment):
    f1_avg = mean(f1_010, f1_025, f1_050)
    avg    = mean(acc, edit, f1_010, f1_025, f1_050)
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


BASE_METRICS = ["acc", "edit", "f1_010", "f1_025", "f1_050"]
DERIVED_METRICS = ["f1_avg", "avg"]
ALL_METRICS = BASE_METRICS + DERIVED_METRICS


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
    print(f"[WARN][tas][{kind}] <{MIN_VALID_SEEDS_FOR_SUMMARY} valid seeds; key={key}; {details}")


def _warn_incomplete_seed(*, run_type: str, mode: str, seed: str, split: str, t_max: int, t_total: int, seed_root: str) -> None:
    global _WARN_LINES_PRINTED, _WARN_LINES_SUPPRESSED
    if _WARN_LINES_PRINTED >= MAX_WARN_LINES:
        _WARN_LINES_SUPPRESSED += 1
        return
    _WARN_LINES_PRINTED += 1
    print(
        f"[WARN][tas][incomplete_run] skip (t_max={t_max} != T={t_total}); "
        f"run_type={run_type}; mode={mode}; seed={seed}; split={split}; dir={seed_root}"
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


def _min_max_finite_t(arrs: Iterable[np.ndarray]) -> int:
    ts = []
    for a in arrs:
        ts.append(_max_finite_t(a))
    if not ts:
        return 0
    return int(min(ts))


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
    groups: Dict[Tuple[str, str, str, str, str], List["CurveRow"]] = {}
    for r in curve_rows:
        key = (r.exp_family, r.split, r.mode, r.seed, r.metric_key)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda x: x[0]):
        exp_family, split, mode, seed, metric_key = key
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
                "benchmark": "tas",
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


def _derive_f1_avg(vals: Dict[str, Optional[float]]) -> Optional[float]:
    return _nanmean([vals.get("f1_010"), vals.get("f1_025"), vals.get("f1_050")])


def _derive_avg(vals: Dict[str, Optional[float]]) -> Optional[float]:
    return _nanmean([vals.get("acc"), vals.get("edit"), vals.get("f1_010"), vals.get("f1_025"), vals.get("f1_050")])


@dataclass
class SeedRow:
    benchmark: str
    run_type: str  # joint_fair | continual
    exp_family: str  # joint_fair | continual
    kind: str  # segmentation
    head: str  # unused
    variant: str  # unused
    mode: str
    split: str  # val/test
    seed: str
    metric_key: str
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
        key = (r.benchmark, r.run_type, r.split, r.mode, r.metric_key, r.exp_family)
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
                "split": key[2],
                "mode": key[3],
                "metric_key": key[4],
                "exp_family": key[5],
                "n_seeds": len(micro_seeds),
                "seeds": _dump_json_compact(micro_seeds),
                "micro_mean": _nanmean(micro_vals),
                "micro_std": _nanstd_pop(micro_vals),
                "macro_mean": macro_mean,
                "macro_std": macro_std,
            }
        )
    return out


def _parse_best_result_line(line: str) -> Dict[str, Optional[float]]:
    # Example:
    # Best Epoch 122: Acc: 66.1497, Edit: 46.9666, F1@0.10: 48.6317, F1@0.25: 43.5798, F1@0.50: 32.9033, F1@Avg: 41.7049, Avg: 47.6462
    # We parse the numeric values; ignore epoch except for provenance.
    out: Dict[str, Optional[float]] = {
        "acc": None,
        "edit": None,
        "f1_010": None,
        "f1_025": None,
        "f1_050": None,
        "f1_avg": None,
        "avg": None,
    }
    # normalize separators
    s = line.strip()
    # Acc/Edit
    m = re.search(r"Acc:\s*([0-9.]+)", s)
    if m:
        out["acc"] = _safe_float(m.group(1))
    m = re.search(r"Edit:\s*([0-9.]+)", s)
    if m:
        out["edit"] = _safe_float(m.group(1))
    # F1s
    m = re.search(r"F1@0\.10:\s*([0-9.]+)", s)
    if m:
        out["f1_010"] = _safe_float(m.group(1))
    m = re.search(r"F1@0\.25:\s*([0-9.]+)", s)
    if m:
        out["f1_025"] = _safe_float(m.group(1))
    m = re.search(r"F1@0\.50:\s*([0-9.]+)", s)
    if m:
        out["f1_050"] = _safe_float(m.group(1))
    # Derived already present in txt
    m = re.search(r"F1@Avg:\s*([0-9.]+)", s)
    if m:
        out["f1_avg"] = _safe_float(m.group(1))
    # Use rsplit to avoid F1@Avg
    if "Avg:" in s:
        try:
            out["avg"] = _safe_float(s.rsplit("Avg:", 1)[1].strip())
        except Exception:
            pass

    # Ensure derived present even if missing in txt
    if out["f1_avg"] is None:
        out["f1_avg"] = _derive_f1_avg(out)
    if out["avg"] is None:
        out["avg"] = _derive_avg(out)
    return out


def _scan_joint_fair_seed_rows(exps_root: str) -> List[SeedRow]:
    out: List[SeedRow] = []
    jf_root = os.path.join(exps_root, "joint_fair")
    if not os.path.isdir(jf_root):
        return out

    # structure: joint_fair/{mode}/seed{0,1,2}/results/best_result_{val|test}.txt
    for mode in sorted(os.listdir(jf_root)):
        mode_dir = os.path.join(jf_root, mode)
        if not os.path.isdir(mode_dir):
            continue
        for seed_dir in sorted(os.listdir(mode_dir)):
            m_seed = re.match(r"seed(\d+)$", seed_dir)
            if not m_seed:
                continue
            seed = m_seed.group(1)
            seed_path = os.path.join(mode_dir, seed_dir)
            res_dir = os.path.join(seed_path, "results")
            if not os.path.isdir(res_dir):
                continue
            for split in ["val", "test"]:
                p = os.path.join(res_dir, f"best_result_{split}.txt")
                if not os.path.isfile(p):
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    line = f.read().strip().splitlines()[0].strip() if f.readable() else ""
                if not line:
                    continue
                vals = _parse_best_result_line(line)
                for mk in ALL_METRICS:
                    out.append(
                        SeedRow(
                            benchmark="tas",
                            run_type="joint_fair",
                            exp_family="joint_fair",
                            kind="segmentation",
                            head="",
                            variant="",
                            mode=mode,
                            split=split,
                            seed=seed,
                            metric_key=mk,
                            value_micro=vals.get(mk),
                            value_macro=None,
                            ego2exo=None,
                            exo2ego=None,
                            source_path=p,
                        )
                    )
    return out


def _load_task_order(seed_root: str) -> Optional[List[int]]:
    p = os.path.join(seed_root, "task_order.json")
    if not os.path.isfile(p):
        return None
    try:
        arr = _read_json(p)
        return [int(x) for x in arr]
    except Exception:
        return None


def _load_A_and_bar_arrays(root: str, metric_key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    A_path = os.path.join(root, f"continual_A_{metric_key}.npy")
    bar_micro_path = os.path.join(root, f"continual_bar_A_micro_{metric_key}.npy")
    bar_macro_path = os.path.join(root, f"continual_bar_A_macro_{metric_key}.npy")
    if not (os.path.isfile(A_path) and os.path.isfile(bar_micro_path) and os.path.isfile(bar_macro_path)):
        return None, None, None
    try:
        A = np.load(A_path)
        bar_micro = np.load(bar_micro_path)
        bar_macro = np.load(bar_macro_path)
        return A, bar_micro, bar_macro
    except Exception:
        return None, None, None


def _derive_arrays_for_metric(
    *,
    A_by_metric: Dict[str, np.ndarray],
    bar_micro_by_metric: Dict[str, np.ndarray],
    bar_macro_by_metric: Dict[str, np.ndarray],
    metric_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # All arrays are (T,T) for A and (T,) for bars.
    if metric_key in BASE_METRICS:
        return A_by_metric[metric_key], bar_micro_by_metric[metric_key], bar_macro_by_metric[metric_key]
    if metric_key == "f1_avg":
        A = (A_by_metric["f1_010"] + A_by_metric["f1_025"] + A_by_metric["f1_050"]) / 3.0
        bar_micro = (bar_micro_by_metric["f1_010"] + bar_micro_by_metric["f1_025"] + bar_micro_by_metric["f1_050"]) / 3.0
        bar_macro = (bar_macro_by_metric["f1_010"] + bar_macro_by_metric["f1_025"] + bar_macro_by_metric["f1_050"]) / 3.0
        return A, bar_micro, bar_macro
    if metric_key == "avg":
        A = (A_by_metric["acc"] + A_by_metric["edit"] + A_by_metric["f1_010"] + A_by_metric["f1_025"] + A_by_metric["f1_050"]) / 5.0
        bar_micro = (
            bar_micro_by_metric["acc"]
            + bar_micro_by_metric["edit"]
            + bar_micro_by_metric["f1_010"]
            + bar_micro_by_metric["f1_025"]
            + bar_micro_by_metric["f1_050"]
        ) / 5.0
        bar_macro = (
            bar_macro_by_metric["acc"]
            + bar_macro_by_metric["edit"]
            + bar_macro_by_metric["f1_010"]
            + bar_macro_by_metric["f1_025"]
            + bar_macro_by_metric["f1_050"]
        ) / 5.0
        return A, bar_micro, bar_macro
    raise ValueError(f"Unknown metric_key: {metric_key}")


def _scan_continual_seed_and_curve_rows(exps_root: str) -> Tuple[List[SeedRow], List[CurveRow]]:
    seed_rows: List[SeedRow] = []
    curve_rows: List[CurveRow] = []
    # Support continual families including ER/DERPP variants encoded in folder name, e.g.:
    #   exps/continual_er_br0p1_rr0p1/<mode>/seedX/...
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

        for mode in sorted(os.listdir(cont_root)):
            mode_dir = os.path.join(cont_root, mode)
            if not os.path.isdir(mode_dir):
                continue
            for seed_dir in sorted(os.listdir(mode_dir)):
                m_seed = re.match(r"seed(\d+)$", seed_dir)
                if not m_seed:
                    continue
                seed = m_seed.group(1)
                seed_root = os.path.join(mode_dir, seed_dir)

                task_order = _load_task_order(seed_root)
                if not task_order:
                    continue
                T = int(len(task_order))
                task_order_json = _dump_json_compact(task_order)

                # Load base arrays for VAL (seed_root) and TEST (seed_root/test_metrics)
                for split, root in [("val", seed_root), ("test", os.path.join(seed_root, "test_metrics"))]:
                    if not os.path.isdir(root):
                        continue

                    A_by_metric: Dict[str, np.ndarray] = {}
                    bar_micro_by_metric: Dict[str, np.ndarray] = {}
                    bar_macro_by_metric: Dict[str, np.ndarray] = {}
                    ok = True
                    for mk in BASE_METRICS:
                        A, bm, bM = _load_A_and_bar_arrays(root, mk)
                        if A is None or bm is None or bM is None:
                            ok = False
                            break
                        if A.ndim != 2:
                            ok = False
                            break
                        A_by_metric[mk] = A.astype(np.float32)
                        bar_micro_by_metric[mk] = bm.astype(np.float32)
                        bar_macro_by_metric[mk] = bM.astype(np.float32)
                    if not ok:
                        continue

                    # Completion check ONCE per seed+split (avoid per-metric spam).
                    t_max = _min_max_finite_t(list(bar_micro_by_metric.values()) + list(bar_macro_by_metric.values()))
                    if T > 0 and t_max != T:
                        _warn_incomplete_seed(
                            run_type=run_type,
                            mode=mode,
                            seed=seed,
                            split=split,
                            t_max=t_max,
                            t_total=T,
                            seed_root=seed_root,
                        )
                        continue

                    # Produce curve rows for ALL_METRICS at each t
                    for mk in ALL_METRICS:
                        A, bm, bM = _derive_arrays_for_metric(
                            A_by_metric=A_by_metric,
                            bar_micro_by_metric=bar_micro_by_metric,
                            bar_macro_by_metric=bar_macro_by_metric,
                            metric_key=mk,
                        )
                        # shape sanity
                        if A.shape[0] < T or A.shape[1] < T:
                            continue
                        for t in range(1, T + 1):
                            row_vals = A[t - 1, :t].astype(np.float64).tolist()
                            row_vals = [None if (v is None or not np.isfinite(float(v))) else float(v) for v in row_vals]
                            curve_rows.append(
                                CurveRow(
                                    benchmark="tas",
                                    exp_family=run_type,
                                    kind="segmentation",
                                    head="",
                                    variant="",
                                    split=split,
                                    mode=mode,
                                    seed=seed,
                                    t=t,
                                    metric_key=mk,
                                    task_order_json=task_order_json,
                                    A_row_json=_dump_json_compact(row_vals),
                                    bar_A_micro=_safe_float(bm[t - 1]) if t - 1 < len(bm) else None,
                                    bar_A_macro=_safe_float(bM[t - 1]) if t - 1 < len(bM) else None,
                                    ego2exo=None,
                                    exo2ego=None,
                                    source_dir=root,
                                )
                            )

                        # final seed rows for this split/metric
                        seed_rows.append(
                            SeedRow(
                                benchmark="tas",
                                run_type=run_type,
                                exp_family=run_type,
                                kind="segmentation",
                                head="",
                                variant="",
                                mode=mode,
                                split=split,
                                seed=seed,
                                metric_key=mk,
                                value_micro=_safe_float(bm[T - 1]) if T - 1 < len(bm) else None,
                                value_macro=_safe_float(bM[T - 1]) if T - 1 < len(bM) else None,
                                ego2exo=None,
                                exo2ego=None,
                                source_path=root,
                            )
                        )

    return seed_rows, curve_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps_dir", type=str, default=EXPS_DIR, help="TAS exps/ directory")
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

    seed_rows.sort(key=lambda r: (r.run_type, r.split, r.mode, int(r.seed), r.metric_key))
    curve_rows.sort(key=lambda r: (r.split, r.mode, int(r.seed), r.metric_key, r.t))

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
            "split",
            "mode",
            "metric_key",
            "exp_family",
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

    print(f"[TAS] wrote: {seed_csv}")
    print(f"[TAS] wrote: {summary_csv}")
    print(f"[TAS] wrote: {curve_csv}")

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
    print(f"[TAS] wrote: {forgetting_csv}")

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
    print(f"[TAS] wrote: {forgetting_summary_csv}")
    if _WARN_LINES_SUPPRESSED > 0:
        print(f"[WARN][tas] suppressed {_WARN_LINES_SUPPRESSED} additional warnings (increase MAX_WARN_LINES to show more)")


if __name__ == "__main__":
    main()

