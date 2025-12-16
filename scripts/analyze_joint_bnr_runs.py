#!/usr/bin/env python3
"""
Analyze joint_bnr runs (UBC + MLP) across seeds and match modes.

Outputs:
  - meta_runs.csv      (one row per run dir)
  - meta_groups.csv    (grouped by match mode: mean/std across seeds)
  - gate_overlap.csv   (per-run interpretability overlap summaries)

Focus metrics:
  1) EM performance: UBC vs MLP
  2) Interpretability + gate correctness:
       - MLP BNR exact (L1 + avg over layers)
       - MLP exact primitive hit rate (L1)
       - Gate overlap between MLP recovered primitives and UBC path gates (multiset Jaccard)
       - UBC pred_expr == label_expr rate (normalized strings)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import math

import pandas as pd


def find_run_dirs(root: Path):
    # run dir contains summary.json AND results.jsonl
    for p in root.rglob("summary.json"):
        run_dir = p.parent
        if (run_dir / "results.jsonl").exists():
            yield run_dir


def multiset_jaccard(a: dict, b: dict) -> float:
    # a,b map item -> count
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    inter = 0
    union = 0
    for k in keys:
        ca = int(a.get(k, 0))
        cb = int(b.get(k, 0))
        inter += min(ca, cb)
        union += max(ca, cb)
    return float(inter) / float(union) if union > 0 else 0.0


def safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / max(1, len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Root directory containing joint_bnr run folders")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: <root>/analysis_joint_bnr)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else (root / "analysis_joint_bnr")
    out.mkdir(parents=True, exist_ok=True)

    rows_runs = []
    rows_overlap = []

    for run_dir in find_run_dirs(root):
        summary_path = run_dir / "summary.json"
        results_path = run_dir / "results.jsonl"

        summ = json.loads(summary_path.read_text())
        mlp_match = summ.get("mlp_match", summ.get("config", {}).get("mlp_match", "unknown"))

        # Heuristic parse seed from folder name
        seed = None
        name = run_dir.name
        if "_seed" in name:
            try:
                seed = int(name.split("_seed")[-1].split("_")[0])
            except Exception:
                seed = None

        # Load per-instance rows
        ubc_em = []
        mlp_em = []
        ubc_expr_eq = []
        bnr_L1 = []
        bnr_avg = []
        prim_hit_L1 = []
        gate_overlap = []

        with results_path.open("r") as f:
            for line in f:
                r = json.loads(line)
                u = r["ubc"]
                m = r["mlp"]

                ubc_em.append(float(u["em"]))
                mlp_em.append(float(m["em"]))

                # pred_expr equality (normalized strings already in file)
                ubc_expr_eq.append(1.0 if (u.get("pred_expr","") == r.get("label_expr","")) else 0.0)

                # BNR stats
                bnr_list = m.get("bnr_exact_per_layer", [])
                bnr_L1.append(float(bnr_list[0]) if bnr_list else 0.0)
                bnr_avg.append(float(sum(bnr_list)/max(1,len(bnr_list))) if bnr_list else 0.0)

                prim = m.get("prim_interp_first_layer", {}) or {}
                prim_hit_L1.append(float(prim.get("exact_primitive_hit_rate", 0.0)))

                # overlap: UBC path_counts vs MLP gate_hist (exact hits only)
                ubc_path = (u.get("gate_usage", {}) or {}).get("path_counts", {}) or {}
                mlp_hist = prim.get("gate_hist", {}) or {}
                gate_overlap.append(multiset_jaccard(ubc_path, mlp_hist))

        run_row = {
            "run_dir": str(run_dir),
            "match_mode": mlp_match,
            "seed": seed,
            "n_instances": len(ubc_em),

            "ubc_em_rate": safe_mean(ubc_em),
            "mlp_em_rate": safe_mean(mlp_em),
            "ubc_expr_eq_rate": safe_mean(ubc_expr_eq),

            "mlp_bnr_exact_L1": safe_mean(bnr_L1),
            "mlp_bnr_exact_avg": safe_mean(bnr_avg),
            "mlp_prim_hit_L1": safe_mean(prim_hit_L1),
            "gate_overlap_mset_jaccard": safe_mean(gate_overlap),

            "avg_mlp_params": float(summ.get("means", {}).get("mlp_params", float("nan"))),
            "avg_ubc_soft_params": float(summ.get("means", {}).get("ubc_soft_params", float("nan"))),
            "avg_ubc_total_params": float(summ.get("means", {}).get("ubc_total_params", float("nan"))),
        }
        rows_runs.append(run_row)

        rows_overlap.append({
            "run_dir": str(run_dir),
            "match_mode": mlp_match,
            "seed": seed,
            "gate_overlap_mset_jaccard": safe_mean(gate_overlap),
            "mlp_prim_hit_L1": safe_mean(prim_hit_L1),
            "mlp_bnr_exact_L1": safe_mean(bnr_L1),
            "ubc_expr_eq_rate": safe_mean(ubc_expr_eq),
        })

    df = pd.DataFrame(rows_runs)
    if df.empty:
        print("[error] No runs found. Expected folders containing summary.json + results.jsonl.")
        return

    df.to_csv(out / "meta_runs.csv", index=False)

    # Group by match mode
    g = df.groupby("match_mode").agg(
        n_runs=("run_dir","count"),
        ubc_em_mean=("ubc_em_rate","mean"),
        ubc_em_std=("ubc_em_rate","std"),
        mlp_em_mean=("mlp_em_rate","mean"),
        mlp_em_std=("mlp_em_rate","std"),
        ubc_expr_eq_mean=("ubc_expr_eq_rate","mean"),
        mlp_bnr_L1_mean=("mlp_bnr_exact_L1","mean"),
        mlp_bnr_L1_std=("mlp_bnr_exact_L1","std"),
        mlp_prim_hit_L1_mean=("mlp_prim_hit_L1","mean"),
        mlp_prim_hit_L1_std=("mlp_prim_hit_L1","std"),
        gate_overlap_mean=("gate_overlap_mset_jaccard","mean"),
        gate_overlap_std=("gate_overlap_mset_jaccard","std"),
        avg_mlp_params_mean=("avg_mlp_params","mean"),
    ).reset_index()

    g.to_csv(out / "meta_groups.csv", index=False)

    pd.DataFrame(rows_overlap).to_csv(out / "gate_overlap.csv", index=False)

    # Console summary
    print("\n=== Joint BNR Analysis (by match_mode) ===")
    print(g.sort_values("match_mode").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n[ok] wrote:\n  {out/'meta_runs.csv'}\n  {out/'meta_groups.csv'}\n  {out/'gate_overlap.csv'}")


if __name__ == "__main__":
    main()
