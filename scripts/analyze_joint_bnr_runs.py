#!/usr/bin/env python3
"""
Analyze joint_bnr runs (UBC + MLP) across seeds and match modes.

Expected per run_dir:
  - summary.json
  - results.jsonl
  - expr_table.csv (optional; results.jsonl is the source of truth)

Outputs:
  - meta_runs.csv
  - meta_groups.csv
  - expr_shortest.csv
  - gate_overlap.csv

Focus:
  1) EM performance: UBC vs MLP (raw MLP)
  2) Interpretability:
      - MLP BNR exact (L1 + avg over layers)
      - MLP BNR eps (L1)
      - MLP primitive hit rate L1 + mean best primitive acc L1
      - Gate overlap: UBC path gate histogram vs MLP exact-hit gate histogram (multiset Jaccard)
  3) Rough expression simplicity:
      - who is shortest (token then char) among {label, ubc, mlp}
      - average token lengths for label/ubc/mlp
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


def find_run_dirs(root: Path):
    for p in root.rglob("summary.json"):
        run_dir = p.parent
        if (run_dir / "results.jsonl").exists():
            yield run_dir


def multiset_jaccard(a: dict, b: dict) -> float:
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


def parse_seed_from_path(run_dir: Path) -> int | None:
    s = str(run_dir)
    # your run dirs include "..._seed{SEED}"
    if "_seed" in s:
        try:
            tail = s.split("_seed")[-1]
            num = ""
            for ch in tail:
                if ch.isdigit():
                    num += ch
                else:
                    break
            return int(num) if num else None
        except Exception:
            return None
    return None


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
    rows_shortest = []

    for run_dir in find_run_dirs(root):
        summary_path = run_dir / "summary.json"
        results_path = run_dir / "results.jsonl"

        summ = json.loads(summary_path.read_text())
        match_mode = summ.get("mlp_match", "unknown")
        seed = parse_seed_from_path(run_dir)

        ubc_em = []
        mlp_em = []

        ubc_expr_eq = []

        bnr_exact_L1 = []
        bnr_exact_avg = []
        bnr_eps_L1 = []

        prim_hit_L1 = []
        prim_best_acc_L1 = []

        gate_overlap = []

        # expr length stats
        shortest_counts = Counter()
        label_tok = []
        ubc_tok = []
        mlp_tok = []

        with results_path.open("r") as f:
            for line in f:
                r = json.loads(line)
                u = r.get("ubc", {})
                m = r.get("mlp", {})

                ubc_em.append(float(u.get("em", 0)))
                mlp_em.append(float(m.get("em", 0)))

                # pred_expr == label_expr (normalized strings) -- only a syntactic rate
                ubc_expr_eq.append(1.0 if (u.get("pred_expr", "") == r.get("label_expr", "")) else 0.0)

                # BNR
                bnr_list = m.get("bnr_exact_per_layer", []) or []
                bnr_exact_L1.append(float(bnr_list[0]) if bnr_list else 0.0)
                bnr_exact_avg.append(float(sum(bnr_list) / max(1, len(bnr_list))) if bnr_list else 0.0)

                eps_list = m.get("bnr_eps_per_layer", []) or []
                bnr_eps_L1.append(float(eps_list[0]) if eps_list else 0.0)

                prim_hit_L1.append(float(m.get("primitive_hit_rate_L1", 0.0)))
                prim_best_acc_L1.append(float(m.get("mean_best_primitive_acc_L1", 0.0)))

                # overlap: UBC path_counts vs MLP gate_hist_exact_L1 (exact hits only)
                ubc_path = (u.get("gate_usage", {}) or {}).get("path_counts", {}) or {}
                mlp_hist = m.get("gate_hist_exact_L1", {}) or {}
                gate_overlap.append(multiset_jaccard(ubc_path, mlp_hist))

                # shortest
                el = r.get("expr_lengths", {}) or {}
                sh = el.get("shortest", None)
                if sh in {"label", "ubc", "mlp"}:
                    shortest_counts[sh] += 1

                # token lengths
                try:
                    label_tok.append(int(el.get("label", {}).get("tok", 0)))
                    ubc_tok.append(int(el.get("ubc", {}).get("tok", 0)))
                    mlp_tok.append(int(el.get("mlp", {}).get("tok", 0)))
                except Exception:
                    pass

        n_inst = len(ubc_em)
        run_row = {
            "run_dir": str(run_dir),
            "match_mode": match_mode,
            "seed": seed,
            "n_instances": n_inst,

            "ubc_em_rate": safe_mean(ubc_em),
            "mlp_em_rate": safe_mean(mlp_em),
            "ubc_expr_eq_rate": safe_mean(ubc_expr_eq),

            "mlp_bnr_exact_L1": safe_mean(bnr_exact_L1),
            "mlp_bnr_exact_avg": safe_mean(bnr_exact_avg),
            "mlp_bnr_eps_L1": safe_mean(bnr_eps_L1),

            "mlp_prim_hit_L1": safe_mean(prim_hit_L1),
            "mlp_prim_best_acc_L1": safe_mean(prim_best_acc_L1),

            "gate_overlap_mset_jaccard": safe_mean(gate_overlap),

            "avg_mlp_params": float(summ.get("means", {}).get("mlp_params", float("nan"))),
            "avg_ubc_soft_params": float(summ.get("means", {}).get("ubc_soft_params", float("nan"))),
            "avg_ubc_total_params": float(summ.get("means", {}).get("ubc_total_params", float("nan"))),

            "mean_label_tok": safe_mean(label_tok),
            "mean_ubc_tok": safe_mean(ubc_tok),
            "mean_mlp_tok": safe_mean(mlp_tok),
        }
        rows_runs.append(run_row)

        rows_overlap.append({
            "run_dir": str(run_dir),
            "match_mode": match_mode,
            "seed": seed,
            "gate_overlap_mset_jaccard": safe_mean(gate_overlap),
            "mlp_prim_hit_L1": safe_mean(prim_hit_L1),
            "mlp_bnr_exact_L1": safe_mean(bnr_exact_L1),
            "mlp_bnr_eps_L1": safe_mean(bnr_eps_L1),
            "ubc_expr_eq_rate": safe_mean(ubc_expr_eq),
        })

        # shortest distribution per run
        denom = max(1, n_inst)
        rows_shortest.append({
            "run_dir": str(run_dir),
            "match_mode": match_mode,
            "seed": seed,
            "n_instances": n_inst,
            "shortest_label_rate": shortest_counts["label"] / denom,
            "shortest_ubc_rate": shortest_counts["ubc"] / denom,
            "shortest_mlp_rate": shortest_counts["mlp"] / denom,
        })

    df = pd.DataFrame(rows_runs)
    if df.empty:
        print("[error] No runs found. Expected folders containing summary.json + results.jsonl.")
        return

    df.to_csv(out / "meta_runs.csv", index=False)

    # Group by match mode
    g = df.groupby("match_mode").agg(
        n_runs=("run_dir", "count"),

        ubc_em_mean=("ubc_em_rate", "mean"),
        ubc_em_std=("ubc_em_rate", "std"),

        mlp_em_mean=("mlp_em_rate", "mean"),
        mlp_em_std=("mlp_em_rate", "std"),

        mlp_bnr_exact_L1_mean=("mlp_bnr_exact_L1", "mean"),
        mlp_bnr_exact_L1_std=("mlp_bnr_exact_L1", "std"),

        mlp_bnr_eps_L1_mean=("mlp_bnr_eps_L1", "mean"),
        mlp_bnr_eps_L1_std=("mlp_bnr_eps_L1", "std"),

        mlp_prim_hit_L1_mean=("mlp_prim_hit_L1", "mean"),
        mlp_prim_hit_L1_std=("mlp_prim_hit_L1", "std"),

        mlp_prim_best_acc_L1_mean=("mlp_prim_best_acc_L1", "mean"),
        mlp_prim_best_acc_L1_std=("mlp_prim_best_acc_L1", "std"),

        gate_overlap_mean=("gate_overlap_mset_jaccard", "mean"),
        gate_overlap_std=("gate_overlap_mset_jaccard", "std"),

        mean_label_tok=("mean_label_tok", "mean"),
        mean_ubc_tok=("mean_ubc_tok", "mean"),
        mean_mlp_tok=("mean_mlp_tok", "mean"),

        avg_mlp_params_mean=("avg_mlp_params", "mean"),
    ).reset_index()

    g.to_csv(out / "meta_groups.csv", index=False)

    pd.DataFrame(rows_overlap).to_csv(out / "gate_overlap.csv", index=False)
    pd.DataFrame(rows_shortest).to_csv(out / "expr_shortest.csv", index=False)

    # Console summary
    print("\n=== Joint BNR Analysis (by match_mode) ===")
    print(g.sort_values("match_mode").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n[ok] wrote:\n  {out/'meta_runs.csv'}\n  {out/'meta_groups.csv'}\n  {out/'gate_overlap.csv'}\n  {out/'expr_shortest.csv'}")


if __name__ == "__main__":
    main()
