#!/usr/bin/env python3
# scripts/collate_meta_summary.py
# Scan run dirs (e.g., ~/scratch/UBC-Results/ablations/*/*/summary.json)
# and write a variant-level meta_summary.json (+ CSV).

import argparse, json, csv
from pathlib import Path
from collections import defaultdict, Counter

def load_summary(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def ensure_simpler_hist(summary):
    # If train.py already wrote "simpler", use it.
    if "simpler" in summary and "counts" in summary["simpler"]:
        return summary["simpler"]["counts"]
    # Otherwise compute from per-instance results.
    c = Counter()
    for r in summary.get("results", []):
        c[r.get("simpler", "tie")] += 1
    for k in ["pred","label","tie","same"]:
        c.setdefault(k, 0)
    return dict(c)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root of runs, e.g. ~/scratch/UBC-Results/ablations")
    ap.add_argument("--outfile", default="meta_summary.json")
    ap.add_argument("--csv", default="meta_summary.csv")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    summaries = list(root.glob("*/*/summary.json"))  # <variant>/<stamp_job_seed>/summary.json
    if not summaries:
        print(f"No summary.json found under {root}")
        return

    per_variant = defaultdict(lambda: {
        "runs": 0,
        "avg_row_acc_sum": 0.0,
        "em_rate_sum": 0.0,
        "equiv_rate_sum": 0.0,
        "n_instances_sum": 0,
        "simpler_counts": Counter(),  # across runs (sum of per-instance)
    })

    rows_csv = []
    for smp in summaries:
        variant = smp.parent.parent.name  # ablations/<variant>/<run>/summary.json
        s = load_summary(smp)
        if not s: 
            continue
        n_instances = len(s.get("results", []))
        simpler_counts = ensure_simpler_hist(s)

        pv = per_variant[variant]
        pv["runs"] += 1
        pv["avg_row_acc_sum"] += float(s.get("avg_row_acc", 0.0))
        pv["em_rate_sum"]     += float(s.get("em_rate", 0.0))
        pv["equiv_rate_sum"]  += float(s.get("equiv_rate", 0.0))
        pv["n_instances_sum"] += n_instances
        pv["simpler_counts"].update(simpler_counts)

        rows_csv.append({
            "variant": variant,
            "run_dir": str(smp.parent),
            "avg_row_acc": s.get("avg_row_acc", None),
            "em_rate": s.get("em_rate", None),
            "equiv_rate": s.get("equiv_rate", None),
            "n_instances": n_instances,
            "simpler_pred": simpler_counts.get("pred", 0),
            "simpler_label": simpler_counts.get("label", 0),
            "simpler_tie": simpler_counts.get("tie", 0),
            "simpler_same": simpler_counts.get("same", 0),
        })

    meta = {"variants": {}, "overall": {}}
    # Variant-level aggregates
    for v, pv in per_variant.items():
        runs = max(1, pv["runs"])
        total_inst = max(1, pv["n_instances_sum"])
        sc = pv["simpler_counts"]
        meta["variants"][v] = {
            "runs": pv["runs"],
            "avg_row_acc_macro": pv["avg_row_acc_sum"] / runs,
            "em_rate_macro": pv["em_rate_sum"] / runs,
            "equiv_rate_macro": pv["equiv_rate_sum"] / runs,
            "instances_total": pv["n_instances_sum"],
            "simpler": {
                "counts": {k: sc.get(k, 0) for k in ["pred","label","tie","same"]},
                "ratios": {
                    k: (sc.get(k, 0) / total_inst) for k in ["pred","label","tie","same"]
                }
            }
        }

    # Overall (pool all variants)
    runs_all = sum(pv["runs"] for pv in per_variant.values()) or 1
    inst_all = sum(pv["n_instances_sum"] for pv in per_variant.values()) or 1
    simpler_all = Counter()
    for pv in per_variant.values():
        simpler_all.update(pv["simpler_counts"])
    meta["overall"] = {
        "runs": runs_all,
        "instances_total": inst_all,
        "simpler": {
            "counts": {k: simpler_all.get(k, 0) for k in ["pred","label","tie","same"]},
            "ratios": {k: simpler_all.get(k, 0)/inst_all for k in ["pred","label","tie","same"]}
        }
    }

    # Write JSON
    out_json = root / args.outfile
    out_json.write_text(json.dumps(meta, indent=2))
    print(f"Wrote meta summary to {out_json}")

    # Write CSV (one row per run)
    out_csv = root / args.csv
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"Wrote run table to {out_csv}")

if __name__ == "__main__":
    main()
