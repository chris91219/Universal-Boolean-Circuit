#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------- IO helpers ----------------------

def read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


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


# ---------------------- gate_usage extraction ----------------------

def extract_gate_usage(summary: Dict[str, Any]) -> Tuple[Counter, Counter]:
    """
    Returns (path_counts, all_unit_counts). Empty counters if missing.
    Supports:
      1) summary["gate_usage"]["path_counts"], ["all_unit_counts"]
      2) sum over summary["results"][i]["gate_usage"][...]
    """
    gu = summary.get("gate_usage", None)
    if isinstance(gu, dict):
        pc = gu.get("path_counts", None)
        ac = gu.get("all_unit_counts", None)
        if isinstance(pc, dict) or isinstance(ac, dict):
            path = Counter({k: int(v) for k, v in (pc or {}).items()})
            allu = Counter({k: int(v) for k, v in (ac or {}).items()})
            return path, allu

    results = summary.get("results", None)
    if isinstance(results, list):
        path = Counter()
        allu = Counter()
        for r in results:
            if not isinstance(r, dict):
                continue
            rgu = r.get("gate_usage", None)
            if not isinstance(rgu, dict):
                continue
            pc = rgu.get("path_counts", {})
            ac = rgu.get("all_unit_counts", {})
            if isinstance(pc, dict):
                path.update({k: int(v) for k, v in pc.items()})
            if isinstance(ac, dict):
                allu.update({k: int(v) for k, v in ac.items()})
        return path, allu

    return Counter(), Counter()


def counter_to_rows(c: Counter) -> List[Dict[str, Any]]:
    tot = sum(c.values())
    rows = []
    for g, v in c.most_common():
        rows.append({"gate": g, "count": int(v), "frac": (float(v) / tot) if tot > 0 else 0.0})
    return rows


def plot_gate_bar(plt, rows: List[Dict[str, Any]], title: str, out_png: Path, topk: int = 20) -> None:
    if not rows:
        return
    rows = rows[:topk]
    labels = [r["gate"] for r in rows]
    vals = [r["frac"] for r in rows]

    plt.figure(figsize=(10, max(3.5, 0.28 * len(labels))))
    y = list(range(len(labels)))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Fraction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------- grouping + ranking ----------------------

def _nan_safe(x: float, default: float = -1e9) -> float:
    return x if not math.isnan(x) else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Root dir of sweep (e.g., .../UBC-Results/sweep_SL_addmul)")
    ap.add_argument("--out", type=str, default=None, help="Output dir (default: <root>/analysis)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--gate_topk", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser() if args.out else (root / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    run_summaries: Dict[str, Dict[str, Any]] = {}
    n_bad = 0

    # ---- collect runs ----
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

        # S scaling mode: add or mul (from your sweep)
        S_op = str(scale.get("S_op", "identity")).lower()
        S_k = int(scale.get("S_k", 0))

        # L scaling is always add in your sweep; still parse generically
        L_op = str(scale.get("L_op", "identity")).lower()
        L_k = int(scale.get("L_k", 0))

        seed = cfg.get("seed", None)
        gate_set = cfg.get("gate_set", None)
        pair = cfg.get("pair", {})
        route = pair.get("route", None) if isinstance(pair, dict) else None

        avg_row_acc = float(s.get("avg_row_acc", float("nan")))
        em_rate     = float(s.get("em_rate", float("nan")))

        run_dir = str(p.parent)
        runs.append({
            "run_dir": run_dir,
            "seed": seed,
            "gate_set": gate_set,
            "route": route,
            "S_op": S_op, "S_k": S_k,
            "L_op": L_op, "L_k": L_k,
            "avg_row_acc": avg_row_acc,
            "em_rate": em_rate,
        })
        run_summaries[run_dir] = s

    if not runs:
        print("[warn] No usable summary.json found under", root)
        return

    # ---- runs.csv ----
    runs_csv = out / "runs.csv"
    write_csv(runs_csv, runs)
    print("[ok] wrote", runs_csv)

    # ---- groups by (S_op,S_k,L_k) (we assume L_op=add, but keep it explicit) ----
    groups: Dict[Tuple[str, int, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        key = (str(r["S_op"]), int(r["S_k"]), str(r["L_op"]), int(r["L_k"]))
        groups[key].append(r)

    group_rows: List[Dict[str, Any]] = []
    for key, items in groups.items():
        ems = [float(it["em_rate"]) for it in items]
        accs = [float(it["avg_row_acc"]) for it in items]
        em_m, em_s = mean_std(ems)
        acc_m, acc_s = mean_std(accs)
        seeds = sorted({it["seed"] for it in items if it["seed"] is not None})
        group_rows.append({
            "S_op": key[0], "S_k": key[1],
            "L_op": key[2], "L_k": key[3],
            "n": len(items),
            "seeds": ",".join(map(str, seeds)),
            "em_mean": em_m, "em_std": em_s,
            "acc_mean": acc_m, "acc_std": acc_s,
        })

    group_rows.sort(key=lambda d: (d["S_op"], d["L_k"], d["S_k"]))

    groups_csv = out / "groups.csv"
    write_csv(groups_csv, group_rows)
    print("[ok] wrote", groups_csv)

    # ---- best group overall (by em_mean, tie acc_mean) ----
    def gkey(g):
        return (_nan_safe(g["em_mean"]), _nan_safe(g["acc_mean"]))

    best_group = max(group_rows, key=gkey) if group_rows else None

    best_group_runs: List[Dict[str, Any]] = []
    best_run: Optional[Dict[str, Any]] = None
    best_run_summary: Optional[Dict[str, Any]] = None

    if best_group is not None:
        key = (best_group["S_op"], int(best_group["S_k"]), best_group["L_op"], int(best_group["L_k"]))
        best_group_runs = groups.get(key, [])

        def rkey(r):
            return (_nan_safe(float(r["em_rate"])), _nan_safe(float(r["avg_row_acc"])))

        best_run = max(best_group_runs, key=rkey) if best_group_runs else None
        if best_run is not None:
            best_run_summary = run_summaries.get(best_run["run_dir"])

    # ---- report.txt ----
    topk = int(args.topk)
    ranked = sorted(group_rows, key=lambda g: (-gkey(g)[0], -gkey(g)[1]))
    report = out / "report.txt"
    with report.open("w") as f:
        f.write("UBC Sweep (S_op/S_k + L_add) Report\n")
        f.write(f"Root: {root}\nOut : {out}\n\n")
        f.write(f"Runs found: {len(runs)}\n")
        f.write(f"Bad/unreadable summaries: {n_bad}\n")
        f.write(f"Groups: {len(group_rows)}\n\n")

        if best_group is not None:
            f.write("Best group overall (by EM, tie-break Acc):\n")
            f.write(f"  S_op={best_group['S_op']} S_k={best_group['S_k']}  "
                    f"L_op={best_group['L_op']} L_k={best_group['L_k']}  "
                    f"EM={fmt_ms(best_group['em_mean'], best_group['em_std'])}  "
                    f"Acc={fmt_ms(best_group['acc_mean'], best_group['acc_std'])}  "
                    f"n={best_group['n']} seeds={best_group['seeds']}\n\n")

        if best_run is not None:
            f.write("Best run within best group:\n")
            f.write(f"  run_dir={best_run['run_dir']}\n")
            f.write(f"  seed={best_run['seed']}  EM={best_run['em_rate']:.4f}  Acc={best_run['avg_row_acc']:.4f}\n\n")

        f.write(f"Top-{topk} groups by EM:\n")
        for i, g in enumerate(ranked[:topk], 1):
            f.write(f"{i:02d}. S_op={g['S_op']:<3s} S_k={g['S_k']:<2d}  "
                    f"L_k={g['L_k']:<2d}  "
                    f"EM={fmt_ms(g['em_mean'], g['em_std'])}  "
                    f"Acc={fmt_ms(g['acc_mean'], g['acc_std'])}  "
                    f"n={g['n']} seeds={g['seeds']}\n")
        f.write("\n")

    print("[ok] wrote", report)

    # ---- plots ----
    plt = try_import_matplotlib()
    if plt is None:
        print("[warn] matplotlib not available; skipping plots. (CSV + report.txt were written.)")
    else:
        # Plot ADD mode: EM vs S_add for each L_add
        add_rows = [g for g in group_rows if g["S_op"] == "add" and g["L_op"] == "add"]
        if add_rows:
            Sadds = sorted({int(g["S_k"]) for g in add_rows})
            Ladds = sorted({int(g["L_k"]) for g in add_rows})
            em_map = {(int(g["S_k"]), int(g["L_k"])): float(g["em_mean"]) for g in add_rows}
            acc_map = {(int(g["S_k"]), int(g["L_k"])): float(g["acc_mean"]) for g in add_rows}

            plt.figure()
            for ladd in Ladds:
                ys = [em_map.get((sadd, ladd), float("nan")) for sadd in Sadds]
                plt.plot(Sadds, ys, marker="o", label=f"L_add={ladd}")
            plt.xlabel("S_add (S_used = S_base + S_add)")
            plt.ylabel("Mean EM rate")
            plt.title("ADD mode: EM vs S_add (lines = L_add)")
            plt.legend()
            plt.tight_layout()
            p = out / "em_vs_Sadd_addmode.png"
            plt.savefig(p, dpi=200)
            plt.close()
            print("[ok] wrote", p)

            plt.figure()
            for ladd in Ladds:
                ys = [acc_map.get((sadd, ladd), float("nan")) for sadd in Sadds]
                plt.plot(Sadds, ys, marker="o", label=f"L_add={ladd}")
            plt.xlabel("S_add (S_used = S_base + S_add)")
            plt.ylabel("Mean row accuracy")
            plt.title("ADD mode: Row-Acc vs S_add (lines = L_add)")
            plt.legend()
            plt.tight_layout()
            p = out / "acc_vs_Sadd_addmode.png"
            plt.savefig(p, dpi=200)
            plt.close()
            print("[ok] wrote", p)

        # Plot MUL mode: EM vs S_mult for each L_add
        mul_rows = [g for g in group_rows if g["S_op"] == "mul" and g["L_op"] == "add"]
        if mul_rows:
            Smuls = sorted({int(g["S_k"]) for g in mul_rows})
            Ladds = sorted({int(g["L_k"]) for g in mul_rows})
            em_map = {(int(g["S_k"]), int(g["L_k"])): float(g["em_mean"]) for g in mul_rows}
            acc_map = {(int(g["S_k"]), int(g["L_k"])): float(g["acc_mean"]) for g in mul_rows}

            plt.figure()
            for ladd in Ladds:
                ys = [em_map.get((smul, ladd), float("nan")) for smul in Smuls]
                plt.plot(Smuls, ys, marker="o", label=f"L_add={ladd}")
            plt.xlabel("S_mult (S_used = S_base * S_mult)")
            plt.ylabel("Mean EM rate")
            plt.title("MUL mode: EM vs S_mult (lines = L_add)")
            plt.legend()
            plt.tight_layout()
            p = out / "em_vs_Smult_mulmode.png"
            plt.savefig(p, dpi=200)
            plt.close()
            print("[ok] wrote", p)

            plt.figure()
            for ladd in Ladds:
                ys = [acc_map.get((smul, ladd), float("nan")) for smul in Smuls]
                plt.plot(Smuls, ys, marker="o", label=f"L_add={ladd}")
            plt.xlabel("S_mult (S_used = S_base * S_mult)")
            plt.ylabel("Mean row accuracy")
            plt.title("MUL mode: Row-Acc vs S_mult (lines = L_add)")
            plt.legend()
            plt.tight_layout()
            p = out / "acc_vs_Smult_mulmode.png"
            plt.savefig(p, dpi=200)
            plt.close()
            print("[ok] wrote", p)

    # ---- best model gate usage (best run + best group aggregated) ----
    if best_run is None or best_run_summary is None or best_group is None:
        print("[warn] Could not identify best run/group for gate usage plots.")
        return

    # best run
    best_path, best_all = extract_gate_usage(best_run_summary)
    best_path_rows = counter_to_rows(best_path)
    best_all_rows = counter_to_rows(best_all)

    write_csv(out / "best_gate_usage_path.csv", best_path_rows)
    write_csv(out / "best_gate_usage_all.csv", best_all_rows)
    print("[ok] wrote best_gate_usage_{path,all}.csv")

    # best group aggregated (sum across seeds)
    grp_path = Counter()
    grp_all = Counter()
    for r in best_group_runs:
        s = run_summaries.get(r["run_dir"])
        if not s:
            continue
        pc, ac = extract_gate_usage(s)
        grp_path.update(pc)
        grp_all.update(ac)

    grp_path_rows = counter_to_rows(grp_path)
    grp_all_rows = counter_to_rows(grp_all)

    write_csv(out / "best_group_gate_usage_path.csv", grp_path_rows)
    write_csv(out / "best_group_gate_usage_all.csv", grp_all_rows)
    print("[ok] wrote best_group_gate_usage_{path,all}.csv")

    # append gate usage to report
    with (out / "report.txt").open("a") as f:
        f.write("\nGate usage (best run):\n")
        f.write(f"  run_dir={best_run['run_dir']}\n")
        f.write("  Top path gates:\n")
        for r in best_path_rows[:15]:
            f.write(f"    {r['gate']:<12s}  count={r['count']:<6d}  frac={r['frac']:.4f}\n")
        f.write("  Top all-unit gates:\n")
        for r in best_all_rows[:15]:
            f.write(f"    {r['gate']:<12s}  count={r['count']:<6d}  frac={r['frac']:.4f}\n")

        f.write("\nGate usage (best group aggregated over seeds):\n")
        f.write(f"  S_op={best_group['S_op']} S_k={best_group['S_k']}  L_k={best_group['L_k']}  seeds={best_group['seeds']}\n")
        f.write("  Top path gates:\n")
        for r in grp_path_rows[:15]:
            f.write(f"    {r['gate']:<12s}  count={r['count']:<6d}  frac={r['frac']:.4f}\n")
        f.write("  Top all-unit gates:\n")
        for r in grp_all_rows[:15]:
            f.write(f"    {r['gate']:<12s}  count={r['count']:<6d}  frac={r['frac']:.4f}\n")

    print("[ok] appended gate usage to report.txt")

    # gate plots
    plt = try_import_matplotlib()
    if plt is None:
        print("[warn] matplotlib not available; skipping gate usage plots.")
        return

    topk_g = int(args.gate_topk)

    plot_gate_bar(
        plt, best_path_rows,
        title=f"Best run: path gate usage (top {topk_g})",
        out_png=(out / "best_gate_usage_path.png"),
        topk=topk_g,
    )
    plot_gate_bar(
        plt, best_all_rows,
        title=f"Best run: all-unit gate usage (top {topk_g})",
        out_png=(out / "best_gate_usage_all.png"),
        topk=topk_g,
    )
    plot_gate_bar(
        plt, grp_path_rows,
        title=f"Best group: path gate usage (top {topk_g})",
        out_png=(out / "best_group_gate_usage_path.png"),
        topk=topk_g,
    )
    plot_gate_bar(
        plt, grp_all_rows,
        title=f"Best group: all-unit gate usage (top {topk_g})",
        out_png=(out / "best_group_gate_usage_all.png"),
        topk=topk_g,
    )

    print("[ok] wrote gate usage plots (best run + best group)")


if __name__ == "__main__":
    main()
