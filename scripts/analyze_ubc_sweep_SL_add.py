#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def fmt_ms(m: float, s: float) -> str:
    if math.isnan(m):
        return "nan"
    if math.isnan(s):
        return f"{m:.4f}"
    return f"{m:.4f} ± {s:.4f}"


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Root directory containing sweep runs (e.g., .../UBC-Results/sweep_SL_add)")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: <root>/analysis)")
    ap.add_argument("--topk", type=int, default=10, help="Top-k setups to print in report")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser() if args.out else (root / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    n_bad = 0

    for p in root.rglob("summary.json"):
        s = read_json(p)
        if not s:
            n_bad += 1
            continue

        cfg = s.get("config", {})
        if not isinstance(cfg, dict):
            cfg = {}

        scale = cfg.get("scale", {})
        if not isinstance(scale, dict):
            scale = {}

        # we expect add/add sweep, but keep generic
        S_op = str(scale.get("S_op", "identity"))
        S_k  = int(scale.get("S_k", 0))
        L_op = str(scale.get("L_op", "identity"))
        L_k  = int(scale.get("L_k", 0))

        seed = cfg.get("seed", None)
        gate_set = cfg.get("gate_set", None)
        pair = cfg.get("pair", {})
        route = pair.get("route", None) if isinstance(pair, dict) else None

        avg_row_acc = float(s.get("avg_row_acc", float("nan")))
        em_rate     = float(s.get("em_rate", float("nan")))

        runs.append({
            "run_dir": str(p.parent),
            "seed": seed,
            "gate_set": gate_set,
            "route": route,
            "S_op": S_op, "S_add": S_k,
            "L_op": L_op, "L_add": L_k,
            "avg_row_acc": avg_row_acc,
            "em_rate": em_rate,
        })

    if not runs:
        print("[warn] No usable summary.json found under", root)
        return

    # --- runs.csv ---
    runs_csv = out / "runs.csv"
    write_csv(runs_csv, runs)
    print("[ok] wrote", runs_csv)

    # --- groups by (S_add, L_add), only add/add ---
    groups = defaultdict(list)
    for r in runs:
        if r["S_op"] == "add" and r["L_op"] == "add":
            key = (int(r["S_add"]), int(r["L_add"]))
            groups[key].append(r)

    group_rows: List[Dict[str, Any]] = []
    for (sadd, ladd), items in groups.items():
        ems = [float(it["em_rate"]) for it in items]
        accs = [float(it["avg_row_acc"]) for it in items]
        em_m, em_s = mean_std(ems)
        acc_m, acc_s = mean_std(accs)
        seeds = sorted({it["seed"] for it in items if it["seed"] is not None})
        group_rows.append({
            "S_add": sadd,
            "L_add": ladd,
            "n": len(items),
            "seeds": ",".join(map(str, seeds)),
            "em_mean": em_m, "em_std": em_s,
            "acc_mean": acc_m, "acc_std": acc_s,
        })

    group_rows.sort(key=lambda d: (d["S_add"], d["L_add"]))

    groups_csv = out / "groups.csv"
    write_csv(groups_csv, group_rows)
    print("[ok] wrote", groups_csv)

    # --- text report ---
    # rank by em_mean then acc_mean
    by_em = sorted(group_rows, key=lambda r: (-(r["em_mean"] if not math.isnan(r["em_mean"]) else -1e9),
                                             -(r["acc_mean"] if not math.isnan(r["acc_mean"]) else -1e9)))
    by_acc = sorted(group_rows, key=lambda r: (-(r["acc_mean"] if not math.isnan(r["acc_mean"]) else -1e9),
                                              -(r["em_mean"] if not math.isnan(r["em_mean"]) else -1e9)))

    best_em = by_em[0] if by_em else None
    best_acc = by_acc[0] if by_acc else None

    report = out / "report.txt"
    with report.open("w") as f:
        f.write(f"UBC Sweep (S_add, L_add) Report\n")
        f.write(f"Root: {root}\n")
        f.write(f"Out : {out}\n\n")
        f.write(f"Runs found: {len(runs)}\n")
        f.write(f"Bad/unreadable summaries: {n_bad}\n")
        f.write(f"Add/Add groups: {len(group_rows)}\n\n")

        if best_em:
            f.write("Best by EM:\n")
            f.write(f"  S_add={best_em['S_add']}  L_add={best_em['L_add']}  "
                    f"EM={fmt_ms(best_em['em_mean'], best_em['em_std'])}  "
                    f"Acc={fmt_ms(best_em['acc_mean'], best_em['acc_std'])}  "
                    f"n={best_em['n']} seeds={best_em['seeds']}\n\n")
        if best_acc:
            f.write("Best by Row-Acc:\n")
            f.write(f"  S_add={best_acc['S_add']}  L_add={best_acc['L_add']}  "
                    f"Acc={fmt_ms(best_acc['acc_mean'], best_acc['acc_std'])}  "
                    f"EM={fmt_ms(best_acc['em_mean'], best_acc['em_std'])}  "
                    f"n={best_acc['n']} seeds={best_acc['seeds']}\n\n")

        topk = int(args.topk)
        f.write(f"Top-{topk} by EM (tie-break Acc):\n")
        for i, r in enumerate(by_em[:topk], 1):
            f.write(f"{i:02d}. S_add={r['S_add']:<2d} L_add={r['L_add']:<2d}  "
                    f"EM={fmt_ms(r['em_mean'], r['em_std'])}  "
                    f"Acc={fmt_ms(r['acc_mean'], r['acc_std'])}  "
                    f"n={r['n']} seeds={r['seeds']}\n")
        f.write("\n")

        f.write(f"Top-{topk} by Row-Acc (tie-break EM):\n")
        for i, r in enumerate(by_acc[:topk], 1):
            f.write(f"{i:02d}. S_add={r['S_add']:<2d} L_add={r['L_add']:<2d}  "
                    f"Acc={fmt_ms(r['acc_mean'], r['acc_std'])}  "
                    f"EM={fmt_ms(r['em_mean'], r['em_std'])}  "
                    f"n={r['n']} seeds={r['seeds']}\n")
        f.write("\n")

        # full table (compact)
        f.write("All settings (S_add, L_add):\n")
        for r in group_rows:
            f.write(f"  S_add={r['S_add']:<2d} L_add={r['L_add']:<2d}  "
                    f"EM={fmt_ms(r['em_mean'], r['em_std'])}  "
                    f"Acc={fmt_ms(r['acc_mean'], r['acc_std'])}  "
                    f"n={r['n']} seeds={r['seeds']}\n")

    print("[ok] wrote", report)

    # --- plots (optional) ---
    plt = try_import_matplotlib()
    if plt is None:
        print("[warn] matplotlib not available; skipping plots. (CSV + report.txt were written.)")
        return

    Sadds = sorted({r["S_add"] for r in group_rows})
    Ladds = sorted({r["L_add"] for r in group_rows})
    em_map = {(r["S_add"], r["L_add"]): r["em_mean"] for r in group_rows}
    acc_map = {(r["S_add"], r["L_add"]): r["acc_mean"] for r in group_rows}

    # EM vs S_add (lines = L_add)
    plt.figure()
    for ladd in Ladds:
        ys = [em_map.get((sadd, ladd), float("nan")) for sadd in Sadds]
        plt.plot(Sadds, ys, marker="o", label=f"L_add={ladd}")
    plt.xlabel("S_add (S_used = S_base + S_add)")
    plt.ylabel("Mean EM rate")
    plt.title("EM vs S_add (lines = L_add)")
    plt.legend()
    plt.tight_layout()
    p = out / "em_vs_Sadd.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print("[ok] wrote", p)

    # Acc vs S_add
    plt.figure()
    for ladd in Ladds:
        ys = [acc_map.get((sadd, ladd), float("nan")) for sadd in Sadds]
        plt.plot(Sadds, ys, marker="o", label=f"L_add={ladd}")
    plt.xlabel("S_add (S_used = S_base + S_add)")
    plt.ylabel("Mean row accuracy")
    plt.title("Row-Acc vs S_add (lines = L_add)")
    plt.legend()
    plt.tight_layout()
    p = out / "acc_vs_Sadd.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print("[ok] wrote", p)

    # Heatmap EM
    Z = []
    for sadd in Sadds:
        Z.append([em_map.get((sadd, ladd), float("nan")) for ladd in Ladds])

    plt.figure()
    plt.imshow(Z, aspect="auto")
    plt.xticks(range(len(Ladds)), [str(x) for x in Ladds])
    plt.yticks(range(len(Sadds)), [str(x) for x in Sadds])
    plt.xlabel("L_add (L_used = L_base + L_add)")
    plt.ylabel("S_add (S_used = S_base + S_add)")
    plt.title("Mean EM heatmap")
    plt.colorbar()
    plt.tight_layout()
    p = out / "heatmap_em.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print("[ok] wrote", p)


if __name__ == "__main__":
    main()
