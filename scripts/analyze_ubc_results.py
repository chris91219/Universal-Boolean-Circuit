#!/usr/bin/env python3
"""
analyze_ubc_results.py  (updated)

Scans UBC-Results roots for run summary.json files and produces:
  - meta_runs.csv            : per-run rows with rich config metadata
  - meta_groups.csv          : aggregated by (folder_root, variant, gate_set, route, repel, mode, lam_const16, eta)
  - best_overall.csv         : single best run overall (by EM, then row_acc, then ratio_pred)
  - best_by_setup.csv        : best run per setup (same grouping key as meta_groups)
  - pair_sweep_pivot_em.csv  : (unchanged) eta x mode pivot of mean EM if both mode & eta exist

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


def _infer_variant_meta(run_dir: Path, summary: Dict[str, Any]) -> Tuple[
    str, Optional[str], Optional[float], Optional[int], Optional[str], Optional[bool], Optional[str], Optional[float]
]:
    """
    Infer:
      - variant     : high-level folder variant ('ablations', 'bench_default', 'mi_ablation', 'repel_const_ablation', ...)
      - mode        : pair repulsion mode ('log', 'mul', 'hard-log', 'hard-mul') or None
      - eta         : repulsion eta (float) or None
      - seed        : prefer summary['config']['seed'], else parse from dir name
      - gate_set    : "6" | "16" | None
      - repel       : bool or None
      - route       : 'learned' | 'mi_soft' | 'mi_hard' | None
      - lam_const16 : float or None
    """
    parts = run_dir.parts
    variant = None
    mode = None
    eta = None
    seed = None
    gate_set = None
    repel = None
    route = None
    lam_const16 = None

    cfg = summary.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}

    # Seed from config preferred
    if "seed" in cfg:
        try:
            seed = int(cfg["seed"])
        except Exception:
            seed = None

    # Try to infer a high-level "variant" from the path
    for key in (
        "ablations",
        "ablations_opt",
        "bench_pair_sweep",
        "bench_default_best_main",
        "bench_default_main_sadd_ladd_grid",
        "bench_default_gumbel_grid",
        "bench_default",
        "bench_default_g16",
        "mi_ablation",
        "repel_const_ablation",
        "mi_ablation_cpu",
        "repel_const_ablation_cpu",
    ):
        if key in parts:
            variant = key
            break
    if variant is None:
        variant = "unknown"

    # Parse bench_pair_sweep folder names like: log_eta0.5
    if "bench_pair_sweep" in parts:
        i = parts.index("bench_pair_sweep")
        if i + 1 < len(parts):
            m = re.match(r"(log|mul)_eta([0-9.]+)", parts[i + 1])
            if m:
                mode = m.group(1)
                try:
                    eta = float(m.group(2))
                except Exception:
                    eta = None

    # If seed not in config, parse from trail like *_seed7
    if seed is None:
        m = re.search(r"seed(\d+)", run_dir.name)
        if m:
            try:
                seed = int(m.group(1))
            except Exception:
                seed = None

    # Pull new fields from saved config
    if cfg:
        # gate_set
        if "gate_set" in cfg:
            try:
                gate_set = str(cfg["gate_set"])
            except Exception:
                gate_set = None

        # regs.lam_const16
        regs = cfg.get("regs", {})
        if isinstance(regs, dict) and "lam_const16" in regs:
            try:
                lam_const16 = float(regs["lam_const16"])
            except Exception:
                lam_const16 = None

        # pair sub-config
        pair = cfg.get("pair", {})
        if isinstance(pair, dict):
            # route
            if "route" in pair:
                try:
                    route = str(pair["route"])
                except Exception:
                    route = None
            # repel
            if "repel" in pair:
                try:
                    repel = bool(pair["repel"])
                except Exception:
                    repel = None
            # mode and eta (if not deduced yet)
            if mode is None and "mode" in pair:
                try:
                    mode = str(pair["mode"])
                except Exception:
                    mode = None
            if eta is None and "eta" in pair:
                try:
                    eta = float(pair["eta"])
                except Exception:
                    eta = None

    return variant, mode, eta, seed, gate_set, repel, route, lam_const16


def _extract_metrics(summary: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """
    Support dataset (run_dataset) and single-task (run_single) summaries.
    """
    out: Dict[str, Any] = {}
    # dataset-style
    if "avg_row_acc" in summary:
        out["avg_row_acc"] = float(summary.get("avg_row_acc", float("nan")))
        out["em_rate"] = float(summary.get("em_rate", float("nan")))
        out["avg_decoded_row_acc"] = float(summary.get("avg_decoded_row_acc", float("nan")))
        out["decoded_em_rate"] = float(summary.get("decoded_em_rate", float("nan")))
        out["equiv_rate"] = float(summary.get("equiv_rate", float("nan")))
        results = summary.get("results", [])
        out["n_instances"] = int(len(results)) if isinstance(results, list) else None
        out["avg_train_steps"] = _safe_nested_mean(results, ["train_steps"])

        avg_diag = summary.get("avg_diagnostics", {})
        if isinstance(avg_diag, dict):
            for k, v in avg_diag.items():
                if isinstance(v, (int, float)):
                    out[f"diag_{k}"] = float(v)

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
        out["em_rate"] = float(em) if isinstance(em, (int, float)) else float("nan")
        decoded_em = summary.get("decoded_em", None)
        out["decoded_em_rate"] = float(decoded_em) if isinstance(decoded_em, (int, float)) else float("nan")
        out["avg_decoded_row_acc"] = float(summary.get("decoded_row_acc", float("nan")))
        out["equiv_rate"] = out["em_rate"]
        out["avg_train_steps"] = float(summary.get("train_steps", float("nan")))
        diag = summary.get("diagnostics", {})
        if isinstance(diag, dict):
            for k, v in diag.items():
                if isinstance(v, (int, float)):
                    out[f"diag_{k}"] = float(v)
        out["simpler_pred"] = out["simpler_label"] = out["simpler_tie"] = out["simpler_same"] = 0
        out["ratio_pred"] = out["ratio_label"] = out["ratio_tie"] = out["ratio_same"] = 0.0

    return out


def _safe_nested_mean(items: Any, path: List[str]) -> float:
    if not isinstance(items, list):
        return float("nan")
    vals = []
    for item in items:
        cur = item
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if isinstance(cur, (int, float)) and not math.isnan(float(cur)):
            vals.append(float(cur))
    return sum(vals) / len(vals) if vals else float("nan")


def _extract_config_meta(summary: Dict[str, Any]) -> Dict[str, Any]:
    cfg = summary.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    rel = cfg.get("relaxation", {}) if isinstance(cfg.get("relaxation", {}), dict) else {}
    scale = cfg.get("scale", {}) if isinstance(cfg.get("scale", {}), dict) else {}
    lifting = cfg.get("lifting", {}) if isinstance(cfg.get("lifting", {}), dict) else {}
    early = cfg.get("early_stop", {}) if isinstance(cfg.get("early_stop", {}), dict) else {}
    return {
        "relax_mode": rel.get("mode", "softmax"),
        "relax_hard": rel.get("hard", False),
        "gumbel_tau": rel.get("gumbel_tau", ""),
        "eval_hard": rel.get("eval_hard", False),
        "S_op": scale.get("S_op", "identity"),
        "S_k": scale.get("S_k", ""),
        "L_op": scale.get("L_op", "identity"),
        "L_k": scale.get("L_k", ""),
        "lift_use": lifting.get("use", ""),
        "early_stop_metric": early.get("metric", ""),
    }


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


def _score_tuple(run: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Ranking key for 'best' selection:
      1) higher decoded_em_rate
      2) then higher avg_decoded_row_acc
      3) then higher soft em_rate
    """
    dem = run.get("decoded_em_rate")
    dacc = run.get("avg_decoded_row_acc")
    em = run.get("em_rate")
    acc = run.get("avg_row_acc")
    dem = float(dem) if isinstance(dem, (int, float)) else float("-inf")
    dacc = float(dacc) if isinstance(dacc, (int, float)) else float("-inf")
    em = float(em) if isinstance(em, (int, float)) else float("-inf")
    acc = float(acc) if isinstance(acc, (int, float)) else float("-inf")
    return (dem, dacc, em, acc)


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
            variant, mode, eta, seed, gate_set, repel, route, lam_const16 = _infer_variant_meta(run_dir, s)
            mets = _extract_metrics(s, run_dir)
            cfg_meta = _extract_config_meta(s)

            row = {
                "folder_root": folder_root,
                "variant": variant,
                "gate_set": gate_set,
                "route": route,
                "repel": repel,
                "mode": mode,
                "eta": eta,
                "lam_const16": lam_const16,
                **cfg_meta,
                "seed": seed,
                "run_dir": str(run_dir),
                **mets,
            }
            rows.append(row)

    # ---------- per-run CSV ----------
    runs_csv = out_dir / "meta_runs.csv"
    if not rows:
        print("[warn] No runs found (no summary.json under given roots).")
        return

    fieldnames_runs = [
        "folder_root", "variant", "gate_set", "route", "repel", "mode", "eta", "lam_const16",
        "relax_mode", "relax_hard", "gumbel_tau", "eval_hard",
        "S_op", "S_k", "L_op", "L_k", "lift_use", "early_stop_metric",
        "seed", "run_dir",
        "avg_row_acc", "em_rate", "avg_decoded_row_acc", "decoded_em_rate", "equiv_rate", "n_instances",
        "avg_train_steps",
        "simpler_pred", "simpler_label", "simpler_tie", "simpler_same",
        "ratio_pred", "ratio_label", "ratio_tie", "ratio_same",
    ]
    diag_fields = sorted({k for r in rows for k in r if k.startswith("diag_")})
    fieldnames_runs.extend(diag_fields)
    with runs_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_runs)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames_runs})
    print(f"[ok] Wrote per-run: {runs_csv}")

    # ---------- grouped CSV ----------
    grouped: Dict[
        Tuple[Any, ...],
        List[Dict[str, Any]]
    ] = defaultdict(list)
    for r in rows:
        key = (
            r["folder_root"],
            r["variant"],
            r.get("gate_set"),
            r.get("route"),
            r.get("repel"),
            r.get("mode"),
            r.get("lam_const16"),
            r.get("eta"),
            r.get("relax_mode"),
            r.get("relax_hard"),
            r.get("gumbel_tau"),
            r.get("eval_hard"),
            r.get("S_op"),
            r.get("S_k"),
            r.get("L_op"),
            r.get("L_k"),
            r.get("lift_use"),
            r.get("early_stop_metric"),
        )
        grouped[key].append(r)

    group_rows: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        (
            folder_root, variant, gate_set, route, repel, mode, lam_const16, eta,
            relax_mode, relax_hard, gumbel_tau, eval_hard,
            S_op, S_k, L_op, L_k, lift_use, early_stop_metric,
        ) = key
        seeds = sorted(set(it.get("seed") for it in items if it.get("seed") is not None))
        ems = [it["em_rate"] for it in items]
        accs = [it["avg_row_acc"] for it in items]
        decoded_ems = [it["decoded_em_rate"] for it in items]
        decoded_accs = [it["avg_decoded_row_acc"] for it in items]
        eqvs = [it["equiv_rate"] for it in items]
        train_steps = [it.get("avg_train_steps", float("nan")) for it in items]

        em_mean, em_std = _mean_std(ems)
        acc_mean, acc_std = _mean_std(accs)
        decoded_em_mean, decoded_em_std = _mean_std(decoded_ems)
        decoded_acc_mean, decoded_acc_std = _mean_std(decoded_accs)
        eqv_mean, eqv_std = _mean_std(eqvs)
        train_steps_mean, _ = _mean_std(train_steps)

        # Sum counts across runs
        simpler_pred = sum(int(it.get("simpler_pred", 0)) for it in items)
        simpler_label = sum(int(it.get("simpler_label", 0)) for it in items)
        simpler_tie = sum(int(it.get("simpler_tie", 0)) for it in items)
        simpler_same = sum(int(it.get("simpler_same", 0)) for it in items)
        n_instances_total = sum(int(it.get("n_instances", 0) or 0) for it in items)

        group_rows.append({
            "folder_root": folder_root,
            "variant": variant,
            "gate_set": gate_set,
            "route": route,
            "repel": repel,
            "mode": mode,
            "lam_const16": lam_const16,
            "eta": eta,
            "relax_mode": relax_mode,
            "relax_hard": relax_hard,
            "gumbel_tau": gumbel_tau,
            "eval_hard": eval_hard,
            "S_op": S_op,
            "S_k": S_k,
            "L_op": L_op,
            "L_k": L_k,
            "lift_use": lift_use,
            "early_stop_metric": early_stop_metric,
            "num_runs": len(items),
            "num_seeds": len(seeds),
            "seeds": ",".join(map(str, seeds)) if seeds else "",
            "em_mean": em_mean,
            "em_std": em_std,
            "avg_row_acc_mean": acc_mean,
            "avg_row_acc_std": acc_std,
            "decoded_em_mean": decoded_em_mean,
            "decoded_em_std": decoded_em_std,
            "avg_decoded_row_acc_mean": decoded_acc_mean,
            "avg_decoded_row_acc_std": decoded_acc_std,
            "equiv_rate_mean": eqv_mean,
            "equiv_rate_std": eqv_std,
            "avg_train_steps_mean": train_steps_mean,
            "n_instances_total": n_instances_total,
            "simpler_pred_sum": simpler_pred,
            "simpler_label_sum": simpler_label,
            "simpler_tie_sum": simpler_tie,
            "simpler_same_sum": simpler_same,
        })
        for diag_key in diag_fields:
            m, s = _mean_std([it.get(diag_key, float("nan")) for it in items])
            group_rows[-1][f"{diag_key}_mean"] = m
            group_rows[-1][f"{diag_key}_std"] = s

    groups_csv = out_dir / "meta_groups.csv"
    with groups_csv.open("w", newline="") as f:
        fieldnames = [
            "folder_root", "variant", "gate_set", "route", "repel", "mode", "lam_const16", "eta",
            "relax_mode", "relax_hard", "gumbel_tau", "eval_hard",
            "S_op", "S_k", "L_op", "L_k", "lift_use", "early_stop_metric",
            "num_runs", "num_seeds", "seeds",
            "em_mean", "em_std",
            "avg_row_acc_mean", "avg_row_acc_std",
            "decoded_em_mean", "decoded_em_std",
            "avg_decoded_row_acc_mean", "avg_decoded_row_acc_std",
            "equiv_rate_mean", "equiv_rate_std", "avg_train_steps_mean",
            "n_instances_total",
            "simpler_pred_sum", "simpler_label_sum", "simpler_tie_sum", "simpler_same_sum",
        ]
        for diag_key in diag_fields:
            fieldnames.extend([f"{diag_key}_mean", f"{diag_key}_std"])
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(
            group_rows,
            key=lambda x: (
                x["folder_root"], x["variant"],
                str(x.get("gate_set") or ""),
                str(x.get("route") or ""),
                str(x.get("repel") or ""),
                str(x.get("mode") or ""),
                str(x.get("relax_mode") or ""),
                str(x.get("relax_hard") or ""),
                str(x.get("gumbel_tau") or ""),
                str(x.get("eval_hard") or ""),
                str(x.get("S_op") or ""),
                str(x.get("S_k") or ""),
                str(x.get("L_op") or ""),
                str(x.get("L_k") or ""),
                str(x.get("lift_use") or ""),
                str(x.get("early_stop_metric") or ""),
                x.get("lam_const16") if x.get("lam_const16") is not None else -1.0,
                x.get("eta") if x.get("eta") is not None else -1.0,
            ),
        ):
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[ok] Wrote grouped: {groups_csv}")

    # ---------- best overall ----------
    # Keep only per-run fields needed for ranking & reporting
    best_overall = max(rows, key=_score_tuple)
    best_overall_csv = out_dir / "best_overall.csv"
    with best_overall_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "relax_mode", "relax_hard", "gumbel_tau", "eval_hard",
            "S_op", "S_k", "L_op", "L_k", "lift_use", "early_stop_metric",
            "run_dir", "em_rate", "avg_row_acc",
            "decoded_em_rate", "avg_decoded_row_acc",
            "avg_train_steps", "ratio_pred",
        ])
        w.writeheader()
        w.writerow({
            "variant": best_overall.get("variant", ""),
            "relax_mode": best_overall.get("relax_mode", ""),
            "relax_hard": best_overall.get("relax_hard", ""),
            "gumbel_tau": best_overall.get("gumbel_tau", ""),
            "eval_hard": best_overall.get("eval_hard", ""),
            "S_op": best_overall.get("S_op", ""),
            "S_k": best_overall.get("S_k", ""),
            "L_op": best_overall.get("L_op", ""),
            "L_k": best_overall.get("L_k", ""),
            "lift_use": best_overall.get("lift_use", ""),
            "early_stop_metric": best_overall.get("early_stop_metric", ""),
            "run_dir": best_overall.get("run_dir", ""),
            "em_rate": best_overall.get("em_rate", ""),
            "avg_row_acc": best_overall.get("avg_row_acc", ""),
            "decoded_em_rate": best_overall.get("decoded_em_rate", ""),
            "avg_decoded_row_acc": best_overall.get("avg_decoded_row_acc", ""),
            "avg_train_steps": best_overall.get("avg_train_steps", ""),
            "ratio_pred": best_overall.get("ratio_pred", ""),
        })
    print(f"[ok] Wrote best overall: {best_overall_csv}")

    # ---------- best by setup ----------
    # Group key defining a "setup"
    setup_groups: Dict[
        Tuple[Any, ...],
        List[Dict[str, Any]]
    ] = defaultdict(list)
    for r in rows:
        setup_key = (
            r["variant"],
            str(r.get("gate_set") or ""),
            r.get("route"),
            r.get("repel"),
            r.get("mode"),
            r.get("lam_const16"),
            r.get("eta"),
            r.get("relax_mode"),
            r.get("relax_hard"),
            r.get("gumbel_tau"),
            r.get("eval_hard"),
            r.get("S_op"),
            r.get("S_k"),
            r.get("L_op"),
            r.get("L_k"),
            r.get("lift_use"),
            r.get("early_stop_metric"),
        )
        setup_groups[setup_key].append(r)

    best_by_setup_rows: List[Dict[str, Any]] = []
    for key, items in setup_groups.items():
        (
            variant, gate_set, route, repel, mode, lam_const16, eta,
            relax_mode, relax_hard, gumbel_tau, eval_hard,
            S_op, S_k, L_op, L_k, lift_use, early_stop_metric,
        ) = key
        winner = max(items, key=_score_tuple)
        best_by_setup_rows.append({
            "variant": variant,
            "gate_set": gate_set,
            "route": route,
            "repel": repel,
            "mode": mode,
            "lam_const16": lam_const16,
            "eta": eta,
            "relax_mode": relax_mode,
            "relax_hard": relax_hard,
            "gumbel_tau": gumbel_tau,
            "eval_hard": eval_hard,
            "S_op": S_op,
            "S_k": S_k,
            "L_op": L_op,
            "L_k": L_k,
            "lift_use": lift_use,
            "early_stop_metric": early_stop_metric,
            "run_dir": winner.get("run_dir", ""),
            "em_rate": winner.get("em_rate", ""),
            "avg_row_acc": winner.get("avg_row_acc", ""),
            "decoded_em_rate": winner.get("decoded_em_rate", ""),
            "avg_decoded_row_acc": winner.get("avg_decoded_row_acc", ""),
            "avg_train_steps": winner.get("avg_train_steps", ""),
            "ratio_pred": winner.get("ratio_pred", ""),
        })

    best_by_setup_csv = out_dir / "best_by_setup.csv"
    with best_by_setup_csv.open("w", newline="") as f:
        fieldnames = [
            "variant", "gate_set", "route", "repel", "mode", "lam_const16", "eta",
            "relax_mode", "relax_hard", "gumbel_tau", "eval_hard",
            "S_op", "S_k", "L_op", "L_k", "lift_use", "early_stop_metric",
            "run_dir", "em_rate", "avg_row_acc",
            "decoded_em_rate", "avg_decoded_row_acc",
            "avg_train_steps", "ratio_pred",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(
            best_by_setup_rows,
            key=lambda x: (
                x["variant"], str(x.get("gate_set") or ""),
                str(x.get("route") or ""), str(x.get("repel") or ""),
                str(x.get("mode") or ""),
                str(x.get("relax_mode") or ""),
                str(x.get("relax_hard") or ""),
                str(x.get("gumbel_tau") or ""),
                str(x.get("eval_hard") or ""),
                str(x.get("S_op") or ""),
                str(x.get("S_k") or ""),
                str(x.get("L_op") or ""),
                str(x.get("L_k") or ""),
                str(x.get("lift_use") or ""),
                str(x.get("early_stop_metric") or ""),
                x.get("lam_const16") if x.get("lam_const16") is not None else -1.0,
                x.get("eta") if x.get("eta") is not None else -1.0,
            ),
        ):
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[ok] Wrote best-by-setup: {best_by_setup_csv}")

    # ---------- optional: EM pivot for pair sweep (eta rows x mode cols) ----------
    has_pair = any((r.get("mode") is not None and r.get("eta") is not None) for r in rows)
    if has_pair:
        em_by = defaultdict(list)  # (mode, eta) -> [EMs]
        for r in rows:
            m = r.get("mode")
            e = r.get("eta")
            em = r.get("em_rate")
            if m is not None and e is not None and isinstance(em, (int, float)) and not math.isnan(em):
                em_by[(m, e)].append(em)
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
