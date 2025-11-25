#!/usr/bin/env python3
"""
analyze_ubc_baselines.py

Scan baseline runs (MLP / Transformer) for UBC Boolean tasks and produce:

  - meta_runs.csv
      One row per run_dir (i.e., per summary.json).
      Includes:
        * config meta: variant, baseline, match_mode, gate_set, optimizer, route, seed, l_strategy
        * metrics: avg_row_acc, em_rate, n_instances
        * param stats: avg_model_params, avg_ubc_soft_params, avg_ubc_total_params

  - meta_groups.csv
      Aggregated by (folder_root, variant, baseline, match_mode,
                     gate_set, optimizer, route).
      For each group:
        * num_runs, num_seeds, seeds
        * em_mean/std, avg_row_acc_mean/std
        * avg_model_params_mean/std, avg_ubc_soft_params_mean/std, avg_ubc_total_params_mean/std
        * n_instances_total

  - best_overall.csv
      Single best run overall (by EM, then row_acc).

  - best_by_setup.csv
      Best run per setup, where setup key =
        (variant, baseline, match_mode, gate_set, optimizer, route).

Usage:
  python analyze_ubc_baselines.py ROOT [ROOT2 ...] [--out OUT_DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Could not read {p}: {e}")
        return None


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


def _extract_metrics_baseline(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metrics & param stats from a baseline summary.json.

    Expected shape from run_baseline_dataset():
      {
        "config": cfg_eff,
        "baseline": baseline,
        "match_mode": match_mode,
        "l_strategy": l_strategy,
        "avg_row_acc": ...,
        "em_rate": ...,
        "hist": {...},
        "params": {
          "avg_model": ...,
          "avg_ubc_soft": ...,
          "avg_ubc_total": ...
        },
        "results": [...]
      }
    """
    out: Dict[str, Any] = {}

    out["avg_row_acc"] = float(summary.get("avg_row_acc", float("nan")))
    out["em_rate"] = float(summary.get("em_rate", float("nan")))

    results = summary.get("results", [])
    out["n_instances"] = int(len(results)) if isinstance(results, list) else None

    params = summary.get("params", {})
    if isinstance(params, dict):
        out["avg_model_params"] = float(params.get("avg_model", float("nan")))
        out["avg_ubc_soft_params"] = float(params.get("avg_ubc_soft", float("nan")))
        out["avg_ubc_total_params"] = float(params.get("avg_ubc_total", float("nan")))
    else:
        out["avg_model_params"] = float("nan")
        out["avg_ubc_soft_params"] = float("nan")
        out["avg_ubc_total_params"] = float("nan")

    return out


def _infer_baseline_meta(run_dir: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer config meta for baselines.

    Returns dict with:
      variant, baseline, match_mode, gate_set, optimizer, route,
      seed, l_strategy
    """
    parts = run_dir.parts

    # --- variant from path ---
    # try to distinguish "baselines/full_grid" vs other future variants
    if "baselines" in parts:
        if "full_grid" in parts:
            variant = "baselines_full_grid"
        else:
            variant = "baselines"
    else:
        variant = "unknown"

    # --- from top-level summary keys ---
    baseline = summary.get("baseline", None)
    match_mode = summary.get("match_mode", None)
    l_strategy = summary.get("l_strategy", None)

    cfg = summary.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}

    # --- basic config fields ---
    # seed
    seed = None
    if "seed" in cfg:
        try:
            seed = int(cfg["seed"])
        except Exception:
            seed = None

    # gate_set
    gate_set = None
    if "gate_set" in cfg:
        try:
            gate_set = str(cfg["gate_set"])
        except Exception:
            gate_set = None

    # optimizer
    optimizer = None
    if "optimizer" in cfg:
        try:
            optimizer = str(cfg["optimizer"]).lower()
        except Exception:
            optimizer = None

    # route (from pair sub-config)
    route = None
    pair = cfg.get("pair", {})
    if isinstance(pair, dict) and "route" in pair:
        try:
            route = str(pair["route"])
        except Exception:
            route = None

    # fallback seed from dir name: *_seed3_*
    if seed is None:
        import re

        m = re.search(r"seed(\d+)", run_dir.name)
        if m:
            try:
                seed = int(m.group(1))
            except Exception:
                seed = None

    return {
        "variant": variant,
        "baseline": baseline,
        "match_mode": match_mode,
        "gate_set": gate_set,
        "optimizer": optimizer,
        "route": route,
        "seed": seed,
        "l_strategy": l_strategy,
    }


def _score_tuple(run: Dict[str, Any]) -> Tuple[float, float]:
    """
    Ranking key for 'best' selection:
      1) higher em_rate
      2) then higher avg_row_acc
    """
    em = run.get("em_rate")
    acc = run.get("avg_row_acc")
    em = float(em) if isinstance(em, (int, float)) else float("-inf")
    acc = float(acc) if isinstance(acc, (int, float)) else float("-inf")
    return (em, acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "roots",
        nargs="+",
        help="Root directories to scan (e.g., ~/scratch/UBC-Results/baselines/full_grid ...)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: first root)",
    )
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in args.roots]
    out_dir = Path(args.out).expanduser() if args.out else roots[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    # -------- scan all summary.json --------
    for root in roots:
        for p in root.rglob("summary.json"):
            s = _read_json(p)
            if not s:
                continue

            run_dir = p.parent
            folder_root = str(root)

            meta = _infer_baseline_meta(run_dir, s)
            mets = _extract_metrics_baseline(s)

            row = {
                "folder_root": folder_root,
                "variant": meta["variant"],
                "baseline": meta["baseline"],
                "match_mode": meta["match_mode"],
                "gate_set": meta["gate_set"],
                "optimizer": meta["optimizer"],
                "route": meta["route"],
                "l_strategy": meta["l_strategy"],
                "seed": meta["seed"],
                "run_dir": str(run_dir),
                "avg_row_acc": mets["avg_row_acc"],
                "em_rate": mets["em_rate"],
                "n_instances": mets["n_instances"],
                "avg_model_params": mets["avg_model_params"],
                "avg_ubc_soft_params": mets["avg_ubc_soft_params"],
                "avg_ubc_total_params": mets["avg_ubc_total_params"],
            }
            rows.append(row)

    if not rows:
        print("[warn] No baseline runs found (no summary.json under given roots).")
        return

    # -------- per-run CSV --------
    runs_csv = out_dir / "meta_runs.csv"
    fieldnames_runs = [
        "folder_root",
        "variant",
        "baseline",
        "match_mode",
        "gate_set",
        "optimizer",
        "route",
        "l_strategy",
        "seed",
        "run_dir",
        "avg_row_acc",
        "em_rate",
        "n_instances",
        "avg_model_params",
        "avg_ubc_soft_params",
        "avg_ubc_total_params",
    ]
    with runs_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_runs)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames_runs})
    print(f"[ok] Wrote per-run: {runs_csv}")

    # -------- grouped CSV --------
    GroupKey = Tuple[
        str,
        str,
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
    ]
    grouped: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        key: GroupKey = (
            r["folder_root"],
            r["variant"],
            r.get("baseline"),
            r.get("match_mode"),
            r.get("gate_set"),
            r.get("optimizer"),
            r.get("route"),
        )
        grouped[key].append(r)

    group_rows: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        (
            folder_root,
            variant,
            baseline,
            match_mode,
            gate_set,
            optimizer,
            route,
        ) = key

        seeds = sorted(
            set(it.get("seed") for it in items if it.get("seed") is not None)
        )

        ems = [it["em_rate"] for it in items]
        accs = [it["avg_row_acc"] for it in items]
        params_model = [it["avg_model_params"] for it in items]
        params_soft = [it["avg_ubc_soft_params"] for it in items]
        params_total = [it["avg_ubc_total_params"] for it in items]
        n_instances_total = sum(int(it.get("n_instances") or 0) for it in items)

        em_mean, em_std = _mean_std(ems)
        acc_mean, acc_std = _mean_std(accs)
        pm_mean, pm_std = _mean_std(params_model)
        ps_mean, ps_std = _mean_std(params_soft)
        pt_mean, pt_std = _mean_std(params_total)

        group_rows.append(
            {
                "folder_root": folder_root,
                "variant": variant,
                "baseline": baseline,
                "match_mode": match_mode,
                "gate_set": gate_set,
                "optimizer": optimizer,
                "route": route,
                "num_runs": len(items),
                "num_seeds": len(seeds),
                "seeds": ",".join(map(str, seeds)) if seeds else "",
                "em_mean": em_mean,
                "em_std": em_std,
                "avg_row_acc_mean": acc_mean,
                "avg_row_acc_std": acc_std,
                "avg_model_params_mean": pm_mean,
                "avg_model_params_std": pm_std,
                "avg_ubc_soft_params_mean": ps_mean,
                "avg_ubc_soft_params_std": ps_std,
                "avg_ubc_total_params_mean": pt_mean,
                "avg_ubc_total_params_std": pt_std,
                "n_instances_total": n_instances_total,
            }
        )

    groups_csv = out_dir / "meta_groups.csv"
    with groups_csv.open("w", newline="") as f:
        fieldnames_groups = [
            "folder_root",
            "variant",
            "baseline",
            "match_mode",
            "gate_set",
            "optimizer",
            "route",
            "num_runs",
            "num_seeds",
            "seeds",
            "em_mean",
            "em_std",
            "avg_row_acc_mean",
            "avg_row_acc_std",
            "avg_model_params_mean",
            "avg_model_params_std",
            "avg_ubc_soft_params_mean",
            "avg_ubc_soft_params_std",
            "avg_ubc_total_params_mean",
            "avg_ubc_total_params_std",
            "n_instances_total",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames_groups)
        w.writeheader()
        for r in sorted(
            group_rows,
            key=lambda x: (
                x["folder_root"],
                str(x.get("variant") or ""),
                str(x.get("baseline") or ""),
                str(x.get("match_mode") or ""),
                str(x.get("gate_set") or ""),
                str(x.get("optimizer") or ""),
                str(x.get("route") or ""),
            ),
        ):
            w.writerow({k: r.get(k, "") for k in fieldnames_groups})
    print(f"[ok] Wrote grouped: {groups_csv}")

    # -------- best overall --------
    best_overall = max(rows, key=_score_tuple)
    best_overall_csv = out_dir / "best_overall.csv"
    with best_overall_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_dir",
                "baseline",
                "match_mode",
                "gate_set",
                "optimizer",
                "route",
                "seed",
                "em_rate",
                "avg_row_acc",
                "avg_model_params",
                "avg_ubc_soft_params",
                "avg_ubc_total_params",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "run_dir": best_overall.get("run_dir", ""),
                "baseline": best_overall.get("baseline", ""),
                "match_mode": best_overall.get("match_mode", ""),
                "gate_set": best_overall.get("gate_set", ""),
                "optimizer": best_overall.get("optimizer", ""),
                "route": best_overall.get("route", ""),
                "seed": best_overall.get("seed", ""),
                "em_rate": best_overall.get("em_rate", ""),
                "avg_row_acc": best_overall.get("avg_row_acc", ""),
                "avg_model_params": best_overall.get("avg_model_params", ""),
                "avg_ubc_soft_params": best_overall.get("avg_ubc_soft_params", ""),
                "avg_ubc_total_params": best_overall.get("avg_ubc_total_params", ""),
            }
        )
    print(f"[ok] Wrote best overall: {best_overall_csv}")

    # -------- best by setup --------
    # setup key: (variant, baseline, match_mode, gate_set, optimizer, route)
    SetupKey = Tuple[
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
    ]
    setup_groups: Dict[SetupKey, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key: SetupKey = (
            r.get("variant"),
            r.get("baseline"),
            r.get("match_mode"),
            r.get("gate_set"),
            r.get("optimizer"),
            r.get("route"),
        )
        setup_groups[key].append(r)

    best_by_setup_rows: List[Dict[str, Any]] = []
    for key, items in setup_groups.items():
        variant, baseline, match_mode, gate_set, optimizer, route = key
        winner = max(items, key=_score_tuple)
        best_by_setup_rows.append(
            {
                "variant": variant,
                "baseline": baseline,
                "match_mode": match_mode,
                "gate_set": gate_set,
                "optimizer": optimizer,
                "route": route,
                "run_dir": winner.get("run_dir", ""),
                "seed": winner.get("seed", ""),
                "em_rate": winner.get("em_rate", ""),
                "avg_row_acc": winner.get("avg_row_acc", ""),
                "avg_model_params": winner.get("avg_model_params", ""),
                "avg_ubc_soft_params": winner.get("avg_ubc_soft_params", ""),
                "avg_ubc_total_params": winner.get("avg_ubc_total_params", ""),
            }
        )

    best_by_setup_csv = out_dir / "best_by_setup.csv"
    with best_by_setup_csv.open("w", newline="") as f:
        fieldnames = [
            "variant",
            "baseline",
            "match_mode",
            "gate_set",
            "optimizer",
            "route",
            "run_dir",
            "seed",
            "em_rate",
            "avg_row_acc",
            "avg_model_params",
            "avg_ubc_soft_params",
            "avg_ubc_total_params",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(
            best_by_setup_rows,
            key=lambda x: (
                str(x.get("variant") or ""),
                str(x.get("baseline") or ""),
                str(x.get("match_mode") or ""),
                str(x.get("gate_set") or ""),
                str(x.get("optimizer") or ""),
                str(x.get("route") or ""),
            ),
        ):
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[ok] Wrote best-by-setup: {best_by_setup_csv}")


if __name__ == "__main__":
    main()
