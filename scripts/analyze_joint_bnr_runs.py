#!/usr/bin/env python3
"""
Analyze joint_bnr runs (UBC + MLP) across seeds and match modes.

Expected per run_dir:
  - summary.json
  - results.jsonl

Outputs:
  - meta_runs.csv
  - meta_groups.csv
  - expr_shortest.csv
  - gate_overlap.csv
  - gate_stats_long.csv
  - gate_stats_by_mode.csv

Updated to match new joint script outputs:
MLP fields used:
  - bnr_exact_per_layer, bnr_eps_per_layer
  - primitive_hit_rate_L1, mean_best_primitive_acc_L1 (input-bit primitive probe)
  - prim_exact_per_layer, prim_best_acc_per_layer, prim_exact_avg, prim_best_acc_avg (layerwise primitive probe)
  - gate_hist_exact_L1, gate_hist_exact_all, gate_hist_exact_path
UBC fields used:
  - gate_usage.path_counts, gate_usage.all_unit_counts
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from collections import Counter

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


def _counter_from_dict(d: dict) -> Counter:
    c = Counter()
    for k, v in (d or {}).items():
        try:
            c[str(k)] += int(v)
        except Exception:
            pass
    return c


def _add_gate_stats_rows(rows: list, *, run_dir: str, match_mode: str, seed: int | None,
                         source: str, gate_counter: Counter):
    total = sum(gate_counter.values())
    if total <= 0:
        return
    for gate, cnt in gate_counter.items():
        rows.append({
            "run_dir": run_dir,
            "match_mode": match_mode,
            "seed": seed,
            "source": source,            # ubc_path | ubc_all | mlp_L1 | mlp_all | mlp_path
            "gate": str(gate),
            "count": int(cnt),
            "frac": float(cnt) / float(total),
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
        summ = json.loads((run_dir / "summary.json").read_text())
        match_mode = summ.get("mlp_match", "unknown")
        seed = parse_seed_from_path(run_dir)

        ubc_em = []
        mlp_em = []

        # BNR (L1 + avg)
        bnr_exact_L1 = []
        bnr_eps_L1 = []
        bnr_exact_avg = []
        bnr_eps_avg = []

        # Input-bit primitive probe (L1 only)
        prim_hit_input_L1 = []
        prim_best_input_L1 = []

        # Layerwise primitive probe (L1 + avg)
        prim_exact_L1 = []
        prim_best_L1 = []
        prim_exact_avg = []
        prim_best_avg = []

        # overlaps (various)
        overlap_path_vs_mlpAll = []
        overlap_all_vs_mlpAll = []
        overlap_path_vs_mlpPath = []
        overlap_all_vs_mlpPath = []

        # expr length
        shortest_counts = Counter()
        label_tok = []
        ubc_tok = []
        mlp_tok = []

        # gate stats totals across instances
        ubc_path_tot = Counter()
        ubc_all_tot = Counter()
        mlp_L1_tot = Counter()
        mlp_all_tot = Counter()
        mlp_path_tot = Counter()

        with (run_dir / "results.jsonl").open("r") as f:
            for line in f:
                r = json.loads(line)
                u = r.get("ubc", {}) or {}
                m = r.get("mlp", {}) or {}

                ubc_em.append(float(u.get("em", 0)))
                mlp_em.append(float(m.get("em", 0)))

                # ---- BNR ----
                bnr_e = m.get("bnr_exact_per_layer", []) or []
                bnr_p = m.get("bnr_eps_per_layer", []) or []
                if bnr_e:
                    bnr_exact_L1.append(float(bnr_e[0]))
                    bnr_exact_avg.append(float(sum(bnr_e) / max(1, len(bnr_e))))
                else:
                    bnr_exact_L1.append(0.0); bnr_exact_avg.append(0.0)
                if bnr_p:
                    bnr_eps_L1.append(float(bnr_p[0]))
                    bnr_eps_avg.append(float(sum(bnr_p) / max(1, len(bnr_p))))
                else:
                    bnr_eps_L1.append(0.0); bnr_eps_avg.append(0.0)

                # ---- input-bit primitive probe (your original L1 search on inputs) ----
                prim_hit_input_L1.append(float(m.get("primitive_hit_rate_L1", 0.0)))
                prim_best_input_L1.append(float(m.get("mean_best_primitive_acc_L1", 0.0)))

                # ---- layerwise primitive probe (from new fields) ----
                pe = m.get("prim_exact_per_layer", []) or []
                pb = m.get("prim_best_acc_per_layer", []) or []
                prim_exact_L1.append(float(pe[0]) if pe else 0.0)
                prim_best_L1.append(float(pb[0]) if pb else 0.0)
                prim_exact_avg.append(float(m.get("prim_exact_avg", 0.0)))
                prim_best_avg.append(float(m.get("prim_best_acc_avg", 0.0)))

                # ---- gate usage ----
                gu = u.get("gate_usage", {}) or {}
                ubc_path = gu.get("path_counts", {}) or {}
                ubc_all  = gu.get("all_unit_counts", {}) or {}

                mlp_L1   = m.get("gate_hist_exact_L1", {}) or {}
                mlp_all  = m.get("gate_hist_exact_all", {}) or {}
                mlp_path = m.get("gate_hist_exact_path", {}) or {}

                overlap_path_vs_mlpAll.append(multiset_jaccard(ubc_path, mlp_all))
                overlap_all_vs_mlpAll.append(multiset_jaccard(ubc_all, mlp_all))
                overlap_path_vs_mlpPath.append(multiset_jaccard(ubc_path, mlp_path))
                overlap_all_vs_mlpPath.append(multiset_jaccard(ubc_all, mlp_path))

                ubc_path_tot.update(_counter_from_dict(ubc_path))
                ubc_all_tot.update(_counter_from_dict(ubc_all))
                mlp_L1_tot.update(_counter_from_dict(mlp_L1))
                mlp_all_tot.update(_counter_from_dict(mlp_all))
                mlp_path_tot.update(_counter_from_dict(mlp_path))

                # ---- expression shortest + token lengths ----
                el = r.get("expr_lengths", {}) or {}
                sh = el.get("shortest", None)
                if sh in {"label", "ubc", "mlp"}:
                    shortest_counts[sh] += 1
                try:
                    label_tok.append(int(el.get("label", {}).get("tok", 0)))
                    ubc_tok.append(int(el.get("ubc", {}).get("tok", 0)))
                    mlp_tok.append(int(el.get("mlp", {}).get("tok", 0)))
                except Exception:
                    pass

        n_inst = len(ubc_em)

        rows_runs.append({
            "run_dir": str(run_dir),
            "match_mode": match_mode,
            "seed": seed,
            "n_instances": n_inst,

            "ubc_em_rate": safe_mean(ubc_em),
            "mlp_em_rate": safe_mean(mlp_em),

            # BNR
            "mlp_bnr_exact_L1": safe_mean(bnr_exact_L1),
            "mlp_bnr_eps_L1": safe_mean(bnr_eps_L1),
            "mlp_bnr_exact_avg": safe_mean(bnr_exact_avg),
            "mlp_bnr_eps_avg": safe_mean(bnr_eps_avg),

            # input-bit primitive probe (L1 only)
            "mlp_prim_hit_input_L1": safe_mean(prim_hit_input_L1),
            "mlp_prim_best_input_L1": safe_mean(prim_best_input_L1),

            # layerwise primitive probe
            "mlp_prim_exact_L1": safe_mean(prim_exact_L1),
            "mlp_prim_best_L1": safe_mean(prim_best_L1),
            "mlp_prim_exact_avg": safe_mean(prim_exact_avg),
            "mlp_prim_best_avg": safe_mean(prim_best_avg),

            # overlaps
            "overlap_path_vs_mlpAll": safe_mean(overlap_path_vs_mlpAll),
            "overlap_all_vs_mlpAll": safe_mean(overlap_all_vs_mlpAll),
            "overlap_path_vs_mlpPath": safe_mean(overlap_path_vs_mlpPath),
            "overlap_all_vs_mlpPath": safe_mean(overlap_all_vs_mlpPath),

            # params
            "avg_mlp_params": float(summ.get("means", {}).get("mlp_params", float("nan"))),
            "avg_ubc_soft_params": float(summ.get("means", {}).get("ubc_soft_params", float("nan"))),
            "avg_ubc_total_params": float(summ.get("means", {}).get("ubc_total_params", float("nan"))),

            # expr lengths
            "mean_label_tok": safe_mean(label_tok),
            "mean_ubc_tok": safe_mean(ubc_tok),
            "mean_mlp_tok": safe_mean(mlp_tok),
        })

        rows_overlap.append({
            "run_dir": str(run_dir),
            "match_mode": match_mode,
            "seed": seed,
            "overlap_path_vs_mlpAll": safe_mean(overlap_path_vs_mlpAll),
            "overlap_all_vs_mlpAll": safe_mean(overlap_all_vs_mlpAll),
            "overlap_path_vs_mlpPath": safe_mean(overlap_path_vs_mlpPath),
            "overlap_all_vs_mlpPath": safe_mean(overlap_all_vs_mlpPath),
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

        # gate stats per run
        _add_gate_stats_rows(rows_gate_stats_long, run_dir=str(run_dir), match_mode=match_mode, seed=seed,
                             source="ubc_path", gate_counter=ubc_path_tot)
        _add_gate_stats_rows(rows_gate_stats_long, run_dir=str(run_dir), match_mode=match_mode, seed=seed,
                             source="ubc_all", gate_counter=ubc_all_tot)
        _add_gate_stats_rows(rows_gate_stats_long, run_dir=str(run_dir), match_mode=match_mode, seed=seed,
                             source="mlp_L1", gate_counter=mlp_L1_tot)
        _add_gate_stats_rows(rows_gate_stats_long, run_dir=str(run_dir), match_mode=match_mode, seed=seed,
                             source="mlp_all", gate_counter=mlp_all_tot)
        _add_gate_stats_rows(rows_gate_stats_long, run_dir=str(run_dir), match_mode=match_mode, seed=seed,
                             source="mlp_path", gate_counter=mlp_path_tot)

    df = pd.DataFrame(rows_runs)
    if df.empty:
        print("[error] No runs found.")
        return

    df.to_csv(out / "meta_runs.csv", index=False)
    pd.DataFrame(rows_overlap).to_csv(out / "gate_overlap.csv", index=False)
    pd.DataFrame(rows_shortest).to_csv(out / "expr_shortest.csv", index=False)

    # group summary
    g = df.groupby("match_mode").agg(
        n_runs=("run_dir", "count"),
        ubc_em_mean=("ubc_em_rate", "mean"),
        ubc_em_std=("ubc_em_rate", "std"),
        mlp_em_mean=("mlp_em_rate", "mean"),
        mlp_em_std=("mlp_em_rate", "std"),

        mlp_bnr_exact_L1_mean=("mlp_bnr_exact_L1", "mean"),
        mlp_bnr_eps_L1_mean=("mlp_bnr_eps_L1", "mean"),
        mlp_bnr_exact_avg_mean=("mlp_bnr_exact_avg", "mean"),
        mlp_bnr_eps_avg_mean=("mlp_bnr_eps_avg", "mean"),

        mlp_prim_hit_input_L1_mean=("mlp_prim_hit_input_L1", "mean"),
        mlp_prim_best_input_L1_mean=("mlp_prim_best_input_L1", "mean"),

        mlp_prim_exact_L1_mean=("mlp_prim_exact_L1", "mean"),
        mlp_prim_best_L1_mean=("mlp_prim_best_L1", "mean"),
        mlp_prim_exact_avg_mean=("mlp_prim_exact_avg", "mean"),
        mlp_prim_best_avg_mean=("mlp_prim_best_avg", "mean"),

        overlap_path_vs_mlpAll_mean=("overlap_path_vs_mlpAll", "mean"),
        overlap_all_vs_mlpAll_mean=("overlap_all_vs_mlpAll", "mean"),
        overlap_path_vs_mlpPath_mean=("overlap_path_vs_mlpPath", "mean"),
        overlap_all_vs_mlpPath_mean=("overlap_all_vs_mlpPath", "mean"),

        avg_mlp_params_mean=("avg_mlp_params", "mean"),
    ).reset_index()

    g.to_csv(out / "meta_groups.csv", index=False)

    # gate stats CSVs
    df_gates = pd.DataFrame(rows_gate_stats_long)
    df_gates.to_csv(out / "gate_stats_long.csv", index=False)

    if not df_gates.empty:
        df_gm = df_gates.groupby(["match_mode", "source", "gate"], as_index=False)["count"].sum()
        df_gm["total"] = df_gm.groupby(["match_mode", "source"])["count"].transform("sum")
        df_gm["frac"] = df_gm["count"] / df_gm["total"].clip(lower=1)
        df_gm = df_gm.sort_values(["match_mode", "source", "frac"], ascending=[True, True, False])
        df_gm.to_csv(out / "gate_stats_by_mode.csv", index=False)
    else:
        (out / "gate_stats_by_mode.csv").write_text("")

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
