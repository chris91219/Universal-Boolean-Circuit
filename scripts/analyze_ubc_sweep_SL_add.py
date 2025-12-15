#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def read_json(p: Path) -> Dict[str, Any] | None:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Root directory containing sweep runs (e.g., .../UBC-Results/sweep_SL_add)")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: <root>/analysis)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser() if args.out else (root / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []

    for p in root.rglob("summary.json"):
        s = read_json(p)
        if not s:
            continue
        cfg = s.get("config", {}) if isinstance(s.get("config", {}), dict) else {}
        scale = cfg.get("scale", {})
        if not isinstance(scale, dict):
            scale = {}

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
        print("[warn] No summary.json found under", root)
        return

    # --- runs.csv ---
    runs_csv = out / "runs.csv"
    with runs_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        for r in runs:
            w.writerow(r)
    print("[ok] wrote", runs_csv)

    # --- groups by (S_add, L_add) but only for add/add ---
    groups = defaultdict(list)
    for r in runs:
        if r["S_op"] == "add" and r["L_op"] == "add":
            key = (int(r["S_add"]), int(r["L_add"]))
            groups[key].append(r)

    group_rows = []
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
    with groups_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        w.writeheader()
        for r in group_rows:
            w.writerow(r)
    print("[ok] wrote", groups_csv)

    # --- plots ---
    Sadds = sorted({k[0] for k in groups.keys()})
    Ladds = sorted({k[1] for k in groups.keys()})

    em_map = {(s, l): next(gr["em_mean"] for gr in group_rows if gr["S_add"] == s and gr["L_add"] == l)
              for (s, l) in groups.keys()}
    acc_map = {(s, l): next(gr["acc_mean"] for gr in group_rows if gr["S_add"] == s and gr["L_add"] == l)
               for (s, l) in groups.keys()}

    # EM vs S_add, line per L_add
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
