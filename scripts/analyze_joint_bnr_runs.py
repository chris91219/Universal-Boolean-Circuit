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
  - gate_stats_long.csv
  - gate_stats_by_mode.csv

Adds:
  (1) Across-layers primitive scores from m["layer_stats"]:
      - mlp_prim_exact_avg_layers  (mean over layers of exact_fit_rate)
      - mlp_prim_best_acc_avg_layers (mean over layers of mean_best_fit_acc)

  (2) Gate stats:
      - UBC path gates (path_counts)
      - UBC all-unit gates (all_unit_counts)
      - MLP exact-hit L1 gates (gate_hist_exact_L1)
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


def _add_gate_stats_rows(rows: list, *, run_dir: str, match_mode: str, seed: int | None,
                         source: str, gate_counter: Counter):
    total = sum(gate_counter.values())
    for gate, cnt in gate_counter.items():
        rows.append({
            "run_dir": run_dir,
            "match_mode": match_mode,
            "seed": seed,
            "source": source,            # ubc_path | ubc_all | mlp_L1_exact
            "gate": str(gate),
            "count": int(cnt),
            "frac": float(cnt) / float(total) if total > 0 else 0.0,
            "total": int(total),
        })


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
    rows_gate_stats_long = []

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

        # NEW: across-layer primitive scores (from layer_stats)
        prim_exact_avg_layers = []
        prim_best_acc_avg_layers = []

        # overlaps
        gate_overlap_path = []
        gate_overlap_all = []

        # expr length stats
        shortest_counts = Counter()
        label_tok = []
        ubc_tok = []
        mlp_tok = []

        # gate hist totals across instances (for per-run gate-stats)
        ubc_path_tot = Counter()
        ubc_all_tot = Counter()
        mlp_L1_exact_tot = Counter()

        with results_path.open("r") as f:
            for line in f:
                r = json.loads(line)
                u = r.get("ubc", {}) or {}
                m = r.get("mlp", {}) or {}

                ubc_em.append(float(u.get("em", 0)))
                mlp_em.append(float(m.get("em", 0)))

                ubc_expr_eq.append(1.0 if (u.get("pred_expr", "") == r.get("label_expr", "")) else 0.0)

                # BNR
                bnr_list = m.get("bnr_exact_per_layer", []) or []
                bnr_exact_L1.append(float(bnr_list[0]) if bnr_list else 0.0)
                bnr_exact_avg.append(float(sum(bnr_list) / max(1, len(bnr_list))) if bnr_list else 0.0)

                eps_list = m.get("bnr_eps_per_layer", []) or []
                bnr_eps_L1.append(float(eps_list[0]) if eps_list else 0.0)

                prim_hit_L1.append(float(m.get("primitive_hit_rate_L1", 0.0)))
                prim_best_acc_L1.append(float(m.get("mean_best_primitive_acc_L1", 0.0)))

                # NEW: across-layers primitive scores
                layer_stats = m.get("layer_stats", []) or []
                if layer_stats:
                    exact_rates = [float(ls.get("exact_fit_rate", 0.0)) for ls in layer_stats]
                    best_accs = [float(ls.get("mean_best_fit_acc", 0.0)) for ls in layer_stats]
                    prim_exact_avg_layers.append(sum(exact_rates) / max(1, len(exact_rates)))
                    prim_best_acc_avg_layers.append(sum(best_accs) / max(1, len(best_accs)))
                else:
                    prim_exact_avg_layers.append(0.0)
                    prim_best_acc_avg_layers.append(0.0)

                # gate usage
                gu = u.get("gate_usage", {}) or {}
                ubc_path = gu.get("path_counts", {}) or {}
                ubc_all = gu.get("all_unit_counts", {}) or {}

                mlp_hist = m.get("gate_hist_exact_L1", {}) or {}

                # per-instance overlaps
                gate_overlap_path.append(multiset_jaccard(ubc_path, mlp_hist))
                gate_overlap_all.append(multiset_jaccard(ubc_all, mlp_hist))

                # accumulate totals for gate stats
                ubc_path_tot.update({k: int(v) for k, v in ubc_path.items()})
                ubc_all_tot.update({k: int(v) for k, v in ubc_all.items()})
                mlp_L1_exact_tot.update({k: int(v) for k, v in mlp_hist.items()})

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

            # NEW across layers
            "mlp_prim_exact_avg_layers": safe_mean(prim_exact_avg_layers),
            "mlp_prim_best_acc_avg_layers": safe_mean(prim_best_acc_avg_layers),

            # overlaps
            "gate_overlap_path_vs_mlpL1": safe_mean(gate_overlap_path),
            "gate_overlap_all_vs_mlpL1": safe_mean(gate_overlap_all),

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
            "gate_overlap_path_vs_mlpL1": safe_mean(gate_overlap_path),
            "gate_overlap_all_vs_mlpL1": safe_mean(gate_overlap_all),
            "mlp_prim_hit_L1": safe_mean(prim_hit_L1),
            "mlp_bnr_exact_L1": safe_mean(bnr_exact_L1),
            "mlp_bnr_eps_L1": safe_mean(bnr_eps_L1),
            "ubc_expr_eq_rate": safe_mean(ubc_expr_eq),
        })

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

        # gate stats rows (per run)
        _add_gate_stats_rows(
            rows_gate_stats_long,
            run_dir=str(run_dir), match_mode=match_mode, seed=seed,
            source="ubc_path", gate_counter=ubc_path_tot
        )
        _add_gate_stats_rows(
            rows_gate_stats_long,
            run_dir=str(run_dir), match_mode=match_mode, seed=seed,
            source="ubc_all", gate_counter=ubc_all_tot
        )
        _add_gate_stats_rows(
            rows_gate_stats_long,
            run_dir=str(run_dir), match_mode=match_mode, seed=seed,
            source="mlp_L1_exact", gate_counter=mlp_L1_exact_tot
        )

    df = pd.DataFrame(rows_runs)
    if df.empty:
        print("[error] No runs found. Expected folders containing summary.json + results.jsonl.")
        return

    df.to_csv(out / "meta_runs.csv", index=False)
    pd.DataFrame(rows_overlap).to_csv(out / "gate_overlap.csv", index=False)
    pd.DataFrame(rows_shortest).to_csv(out / "expr_shortest.csv", index=False)

    # Group by match mode
    g = df.groupby("match_mode").agg(
        n_runs=("run_dir", "count"),

        ubc_em_mean=("ubc_em_rate", "mean"),
        ubc_em_std=("ubc_em_rate", "std"),

        mlp_em_mean=("mlp_em_rate", "mean"),
        mlp_em_std=("mlp_em_rate", "std"),

        mlp_bnr_exact_L1_mean=("mlp_bnr_exact_L1", "mean"),
        mlp_bnr_exact_L1_std=("mlp_bnr_exact_L1", "std"),

        mlp_bnr_exact_avg_mean=("mlp_bnr_exact_avg", "mean"),
        mlp_bnr_exact_avg_std=("mlp_bnr_exact_avg", "std"),

        mlp_bnr_eps_L1_mean=("mlp_bnr_eps_L1", "mean"),
        mlp_bnr_eps_L1_std=("mlp_bnr_eps_L1", "std"),

        mlp_prim_hit_L1_mean=("mlp_prim_hit_L1", "mean"),
        mlp_prim_hit_L1_std=("mlp_prim_hit_L1", "std"),

        mlp_prim_best_acc_L1_mean=("mlp_prim_best_acc_L1", "mean"),
        mlp_prim_best_acc_L1_std=("mlp_prim_best_acc_L1", "std"),

        # NEW across layers
        mlp_prim_exact_avg_layers_mean=("mlp_prim_exact_avg_layers", "mean"),
        mlp_prim_exact_avg_layers_std=("mlp_prim_exact_avg_layers", "std"),
        mlp_prim_best_acc_avg_layers_mean=("mlp_prim_best_acc_avg_layers", "mean"),
        mlp_prim_best_acc_avg_layers_std=("mlp_prim_best_acc_avg_layers", "std"),

        # overlaps
        gate_overlap_path_mean=("gate_overlap_path_vs_mlpL1", "mean"),
        gate_overlap_path_std=("gate_overlap_path_vs_mlpL1", "std"),
        gate_overlap_all_mean=("gate_overlap_all_vs_mlpL1", "mean"),
        gate_overlap_all_std=("gate_overlap_all_vs_mlpL1", "std"),

        mean_label_tok=("mean_label_tok", "mean"),
        mean_ubc_tok=("mean_ubc_tok", "mean"),
        mean_mlp_tok=("mean_mlp_tok", "mean"),

        avg_mlp_params_mean=("avg_mlp_params", "mean"),
    ).reset_index()

    g.to_csv(out / "meta_groups.csv", index=False)

    # Gate stats CSVs
    df_gates = pd.DataFrame(rows_gate_stats_long)
    df_gates.to_csv(out / "gate_stats_long.csv", index=False)

    # Aggregate gate stats by match_mode + source + gate
    if not df_gates.empty:
        df_gm = df_gates.groupby(["match_mode", "source", "gate"], as_index=False)["count"].sum()
        # recompute fraction within (match_mode, source)
        df_gm["total"] = df_gm.groupby(["match_mode", "source"])["count"].transform("sum")
        df_gm["frac"] = df_gm["count"] / df_gm["total"].clip(lower=1)
        df_gm = df_gm.sort_values(["match_mode", "source", "frac"], ascending=[True, True, False])
        df_gm.to_csv(out / "gate_stats_by_mode.csv", index=False)
    else:
        (out / "gate_stats_by_mode.csv").write_text("")

    # Console summary
    print("\n=== Joint BNR Analysis (by match_mode) ===")
    print(g.sort_values("match_mode").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n[ok] wrote:\n"
          f"  {out/'meta_runs.csv'}\n"
          f"  {out/'meta_groups.csv'}\n"
          f"  {out/'gate_overlap.csv'}\n"
          f"  {out/'expr_shortest.csv'}\n"
          f"  {out/'gate_stats_long.csv'}\n"
          f"  {out/'gate_stats_by_mode.csv'}")


if __name__ == "__main__":
    main()
