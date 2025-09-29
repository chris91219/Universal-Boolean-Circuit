#!/usr/bin/env python3
"""
analyze_ubc_results.py

Scan UBC-Results directories for run summary.json files and produce:
  - meta_runs.csv   : per-run rows
  - meta_groups.csv : aggregated by (folder_root, variant, mode, eta)
  - pair_sweep_pivot_em.csv (if mode/eta present): eta x mode pivot of mean EM

Usage examples:
  python analyze_ubc_results.py \
      ~/scratch/UBC-Results/ablations \
      ~/scratch/UBC-Results/bench_pair_sweep

  python analyze_ubc_results.py \
      ~/scratch/UBC-Results/ablations_opt \
      --out ~/scratch/UBC-Results/ablations_opt

No external deps (stdlib only).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Could not read {p}: {e}")
        return None


def _infer_variant_mode_eta_seed(run_dir: Path, summary: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[float], Optional[int]]:
    """
    Infer:
      - variant: e.g., 'base', 'anneal_td', 'td_div_rows', 'bench_default', 'base_adam', ...
      - mode:    for pair-repel sweep (e.g., 'log' or 'mul'), else None
      - eta:     for pair-repel sweep (float), else None
      - seed:    prefer summary['config']['seed'], fallback parse from dir name
    """
    parts = run_dir.parts
    variant = None
    mode = None
    eta = None
    seed = None

    # Prefer seed from saved config
    cfg = summary.get("config", {})
    if isinstance(cfg, dict) and "seed" in cfg:
        try:
            seed = int(cfg["seed"])
        except Exception:
            pass

    # Try to infer from path shape
    # .../UBC-Results/ablations/<variant>/<timestamp>_jobXXXX_seedY/summary.json
    if "ablations" in parts:
        i = parts.index("ablations")
        if i + 1 < len(parts):
            variant = parts[i + 1]

    # .../UBC-Results/ablations_opt/<variant>/<timestamp>_jobXXXX_seedY/summary.json
    if variant is None and "ablations_opt" in parts:
        i = parts.index("ablations_opt")
        if i + 1 < len(parts):
            variant = parts[i + 1]

    # .../UBC-Results/bench_pair_sweep/<mode_etaX>/<timestamp>_job.../summary.json
    if "bench_pair_sweep" in parts:
        i = parts.index("bench_pair_sweep")
        if i + 1 < len(parts):
            variant_folder = parts[i + 1]  # e.g., 'log_eta0.5'
            variant = "pair_sweep"
            m = re.match(r"(log|mul)_eta([0-9.]+)", variant_folder)
            if m:
                mode = m.group(1)
                try:
                    eta = float(m.group(2))
                except Exception:
                    eta = None

    # .../UBC-Results/bench_default/<timestamp>_jobXXXX_seed0/summary.json
    if variant is None and "bench_default" in parts:
        variant = "bench_default"

    # If still None, try to get something from config
    if variant is None:
        # maybe it was run_single or a custom folder; fall back to 'unknown'
        variant = "unknown"

    # If seed not in config, parse from tail
    if seed is None:
        m = re.search(r"seed(\d+)", run_dir.name)
        if m:
            try:
                seed = int(m.group(1))
            except Exception:
                seed = None

    # Also try config.pair for mode/eta if not parsed from path
    if mode is None or eta is None:
        pair = cfg.get("pair", {}) if isinstance(cfg, dict) else {}
        if isinstance(pair, dict):
            if mode is None and "mode" in pair:
                try:
                    mode = str(pair["mode"])
                except Exception:
                    pass
            if eta is None and "eta" in pair:
                try:
                    eta = float(pair["eta"])
                except Exception:
                    pass

    return variant, mode, eta, seed


def _extract_metrics(summary: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """
    Support dataset (run_dataset) and single-task (run_single) summaries.
    """
    out: Dict[str, Any] = {}
    # dataset-style
    if "avg_row_acc" in summary:
        out["avg_row_acc"] = float(summary.get("avg_row_acc", float("nan")))
        out["em_rate"] = float(summary.get("em_rate", float("nan")))
        out["equiv_rate"] = float(summary.get("equiv_rate", float("nan")))
        results = summary.get("results", [])
        out["n_instances"] = int(len(results)) if isinstance(results, list) else None

        simpler = summary.get("simpler", {})
        counts = simpler.get("counts", {}) if isinstance(simpler, dict) else {}
        ratios = simpler.get("ratios", {}) if isinstance(simpler, dict) else {}

        # counts
        out["simpler_pred"] = int(counts.get("pred", 0))
        out["simpler_label"] = int(counts.get("label", 0))
        out["simpler_tie"] = int(counts.get("tie", 0))
        out["simpler_same"] = int(counts.get("same", 0))
        # ratios
        out["ratio_pred"] = float(ratios.get("pred", 0.0))
        out["ratio_label"] = float(ratios.get("label", 0.0))
        out["ratio_tie"] = float(ratios.get("tie", 0.0))
        out["ratio_same"] = float(ratios.get("same", 0.0))

    # single-task fallback (run_single) -> synthesize dataset-like fields
    else:
        out["n_instances"] = 1
        out["avg_row_acc"] = float(summary.get("row_acc", float("nan")))
        em = summary.get("em", None)
        if em is None:
            out["em_rate"] = float("nan")
        else:
            try:
                out["em_rate"] = float(em)
            except Exception:
                out["em_rate"] = float("nan")
        out["equiv_rate"] = out["em_rate"]  # same meaning in your code
        out["simpler_pred"] = out["simpler_label"] = out["simpler_tie"] = out["simpler_same"] = 0
        out["ratio_pred"] = out["ratio_label"] = out["ratio_tie"] = out["ratio_same"] = 0.0

    return out


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="Root directories to scan (e.g., ~/scratch/UBC-Results/ablations ...)")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: first root)")
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in args.roots]
    out_dir = Path(args.out).expanduser() if args.out else roots[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for root in roots:
        for p in root.rglob("summary.json"):
            s = _read_json(p)
            if not s:
                continue
            run_dir = p.parent
            folder_root = str(root)
            variant, mode, eta, seed = _infer_variant_mode_eta_seed(run_dir, s)
            mets = _extract_metrics(s, run_dir)

            row = {
                "folder_root": folder_root,
                "variant": variant,
                "mode": mode,
                "eta": eta,
                "seed": seed,
                "run_dir": str(run_dir),
                **mets,
            }
            rows.append(row)

    # Write per-run CSV
    runs_csv = out_dir / "meta_runs.csv"
    if rows:
        fieldnames = [
            "folder_root", "variant", "mode", "eta", "seed", "run_dir",
            "avg_row_acc", "em_rate", "equiv_rate", "n_instances",
            "simpler_pred", "simpler_label", "simpler_tie", "simpler_same",
            "ratio_pred", "ratio_label", "ratio_tie", "ratio_same",
        ]
        with runs_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"[ok] Wrote per-run: {runs_csv}")
    else:
        print("[warn] No runs found (no summary.json under given roots).")
        return

    # Grouped stats by (folder_root, variant, mode, eta)
    grouped: Dict[Tuple[str, str, Optional[str], Optional[float]], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["folder_root"], r["variant"], r["mode"], r["eta"])
        grouped[key].append(r)

    group_rows: List[Dict[str, Any]] = []
    for (folder_root, variant, mode, eta), items in grouped.items():
        seeds = sorted(set(it.get("seed") for it in items if it.get("seed") is not None))
        ems = [it["em_rate"] for it in items]
        accs = [it["avg_row_acc"] for it in items]
        eqvs = [it["equiv_rate"] for it in items]

        em_mean, em_std = _mean_std(ems)
        acc_mean, acc_std = _mean_std(accs)
        eqv_mean, eqv_std = _mean_std(eqvs)

        # Sum counts across runs (these are already dataset totals per run)
        simpler_pred = sum(int(it.get("simpler_pred", 0)) for it in items)
        simpler_label = sum(int(it.get("simpler_label", 0)) for it in items)
        simpler_tie = sum(int(it.get("simpler_tie", 0)) for it in items)
        simpler_same = sum(int(it.get("simpler_same", 0)) for it in items)
        n_instances_total = sum(int(it.get("n_instances", 0) or 0) for it in items)

        group_rows.append({
            "folder_root": folder_root,
            "variant": variant,
            "mode": mode,
            "eta": eta,
            "num_runs": len(items),
            "num_seeds": len(seeds),
            "seeds": ",".join(map(str, seeds)) if seeds else "",
            "em_mean": em_mean,
            "em_std": em_std,
            "avg_row_acc_mean": acc_mean,
            "avg_row_acc_std": acc_std,
            "equiv_rate_mean": eqv_mean,
            "equiv_rate_std": eqv_std,
            "n_instances_total": n_instances_total,
            "simpler_pred_sum": simpler_pred,
            "simpler_label_sum": simpler_label,
            "simpler_tie_sum": simpler_tie,
            "simpler_same_sum": simpler_same,
        })

    groups_csv = out_dir / "meta_groups.csv"
    with groups_csv.open("w", newline="") as f:
        fieldnames = [
            "folder_root", "variant", "mode", "eta",
            "num_runs", "num_seeds", "seeds",
            "em_mean", "em_std",
            "avg_row_acc_mean", "avg_row_acc_std",
            "equiv_rate_mean", "equiv_rate_std",
            "n_instances_total",
            "simpler_pred_sum", "simpler_label_sum", "simpler_tie_sum", "simpler_same_sum",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(group_rows, key=lambda x: (x["folder_root"], x["variant"], str(x["mode"]), x["eta"] if x["eta"] is not None else -1)):
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[ok] Wrote grouped: {groups_csv}")

    # Optional: EM pivot for pair sweep (eta rows x mode columns)
    has_pair = any((r.get("mode") is not None and r.get("eta") is not None) for r in rows)
    if has_pair:
        # Build mean EM per (mode, eta) across all roots/variants where mode/eta exist
        em_by = defaultdict(list)  # (mode, eta) -> list of EMs
        for r in rows:
            m = r.get("mode")
            e = r.get("eta")
            em = r.get("em_rate")
            if m is not None and e is not None and isinstance(em, (int, float)) and not math.isnan(em):
                em_by[(m, e)].append(em)
        # Collect unique sorted axes
        modes = sorted({m for (m, _e) in em_by.keys()})
        etas = sorted({e for (_m, e) in em_by.keys()})

        pivot_csv = out_dir / "pair_sweep_pivot_em.csv"
        with pivot_csv.open("w", newline="") as f:
            w = csv.writer(f)
            header = ["eta"] + modes
            w.writerow(header)
            for e in etas:
                row = [e]
                for m in modes:
                    xs = em_by.get((m, e), [])
                    mean = sum(xs) / len(xs) if xs else ""
                    row.append(mean)
                w.writerow(row)
        print(f"[ok] Wrote pair-sweep EM pivot: {pivot_csv}")


if __name__ == "__main__":
    main()
