#!/usr/bin/env python3
"""
analyze_ubc_results_gate_usage.py

Extends the existing analyzer with gate-usage aggregation.

Outputs (in --out):
  - gate_usage_runs_path.csv         : per-run path gate counts + fractions
  - gate_usage_runs_all_units.csv    : per-run all-unit argmax counts + fractions
  - gate_usage_groups_path.csv       : grouped-by-setup path gate counts + fractions
  - gate_usage_groups_all_units.csv  : grouped-by-setup all-unit counts + fractions
  - gate_usage_by_gate_set_path.csv  : overall path usage aggregated per gate_set
  - gate_usage_by_gate_set_all.csv   : overall all-unit usage aggregated per gate_set

Backward compatible:
  - If a run has no gate_usage, it is skipped for gate stats (but can still be analyzed by your old script).
Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Could not read {p}: {e}")
        return None


def _infer_setup_fields(run_dir: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    cfg = summary.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}

    parts = run_dir.parts
    variant = "unknown"
    for key in (
        "ablations", "ablations_opt", "bench_pair_sweep", "bench_default",
        "bench_default_g16", "mi_ablation", "repel_const_ablation",
        "mi_ablation_cpu", "repel_const_ablation_cpu", "full_grid",
    ):
        if key in parts:
            variant = key
            break

    seed = None
    if "seed" in cfg:
        try:
            seed = int(cfg["seed"])
        except Exception:
            seed = None
    if seed is None:
        m = re.search(r"seed(\d+)", run_dir.name)
        if m:
            try:
                seed = int(m.group(1))
            except Exception:
                seed = None

    gate_set = None
    if "gate_set" in cfg:
        gate_set = str(cfg.get("gate_set"))

    pair = cfg.get("pair", {})
    if not isinstance(pair, dict):
        pair = {}

    route = pair.get("route", None)
    repel = pair.get("repel", None)
    mode  = pair.get("mode", None)
    eta   = pair.get("eta", None)

    try:
        repel = bool(repel) if repel is not None else None
    except Exception:
        repel = None
    try:
        eta = float(eta) if eta is not None else None
    except Exception:
        eta = None

    regs = cfg.get("regs", {})
    if not isinstance(regs, dict):
        regs = {}
    lam_const16 = regs.get("lam_const16", None)
    try:
        lam_const16 = float(lam_const16) if lam_const16 is not None else None
    except Exception:
        lam_const16 = None

    return {
        "variant": variant,
        "seed": seed,
        "gate_set": gate_set,
        "route": route,
        "repel": repel,
        "mode": mode,
        "eta": eta,
        "lam_const16": lam_const16,
    }


def _extract_gate_usage(summary: Dict[str, Any]) -> Tuple[Optional[Counter], Optional[Counter]]:
    """
    Returns (path_counts, all_unit_counts) for the run, or (None, None) if missing.
    Supports:
      - summary["gate_usage"]["path_counts"] ...
      - OR summing per-instance summary["results"][i]["gate_usage"][...]
    """
    gu = summary.get("gate_usage", None)
    if isinstance(gu, dict):
        pc = gu.get("path_counts", None)
        ac = gu.get("all_unit_counts", None)
        if isinstance(pc, dict) or isinstance(ac, dict):
            path = Counter({k: int(v) for k, v in (pc or {}).items()})
            allu = Counter({k: int(v) for k, v in (ac or {}).items()})
            return path, allu

    # fallback: per-instance
    results = summary.get("results", None)
    if isinstance(results, list):
        path = Counter()
        allu = Counter()
        any_found = False
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
                any_found = True
            if isinstance(ac, dict):
                allu.update({k: int(v) for k, v in ac.items()})
                any_found = True
        if any_found:
            return path, allu

    return None, None


def _counter_to_frac_row(c: Counter) -> Dict[str, float]:
    tot = sum(c.values())
    if tot <= 0:
        return {}
    return {k: (v / tot) for k, v in c.items()}


def _sorted_gate_names(rows: List[Dict[str, Any]], key: str) -> List[str]:
    names = set()
    for r in rows:
        c = r.get(key)
        if isinstance(c, dict):
            names.update(c.keys())
    # stable-ish: by name
    return sorted(names)


def _write_gate_csv(path: Path, rows: List[Dict[str, Any]], gate_names: List[str], prefix: str) -> None:
    """
    rows entries should contain:
      - metadata fields (variant, gate_set, route, ...)
      - counts dict at rows[i][prefix+"_counts"]
      - fracs  dict at rows[i][prefix+"_fracs"]
    """
    meta_fields = [
        "folder_root", "run_dir",
        "variant", "gate_set", "route", "repel", "mode", "eta", "lam_const16",
        "seed",
    ]
    count_fields = [f"{prefix}_count__{g}" for g in gate_names]
    frac_fields  = [f"{prefix}_frac__{g}" for g in gate_names]

    out_fields = meta_fields + [f"{prefix}_total"] + count_fields + frac_fields

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            counts = r.get(f"{prefix}_counts", {}) or {}
            fracs  = r.get(f"{prefix}_fracs", {}) or {}
            out = {k: r.get(k, "") for k in meta_fields}
            out[f"{prefix}_total"] = sum(int(v) for v in counts.values()) if isinstance(counts, dict) else ""
            for g in gate_names:
                out[f"{prefix}_count__{g}"] = counts.get(g, 0)
                out[f"{prefix}_frac__{g}"]  = fracs.get(g, 0.0)
            w.writerow(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="Root directories to scan (e.g., ~/scratch/UBC-Results/full_grid)")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: first root)")
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in args.roots]
    out_dir = Path(args.out).expanduser() if args.out else roots[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows: List[Dict[str, Any]] = []
    for root in roots:
        for p in root.rglob("summary.json"):
            s = _read_json(p)
            if not s:
                continue
            run_dir = p.parent
            setup = _infer_setup_fields(run_dir, s)
            path_counts, all_counts = _extract_gate_usage(s)

            # Skip runs that don't have gate usage yet
            if path_counts is None and all_counts is None:
                continue

            row = {
                "folder_root": str(root),
                "run_dir": str(run_dir),
                **setup,
                "path_counts": dict(path_counts or Counter()),
                "path_fracs": _counter_to_frac_row(path_counts or Counter()),
                "all_counts": dict(all_counts or Counter()),
                "all_fracs": _counter_to_frac_row(all_counts or Counter()),
            }
            per_run_rows.append(row)

    if not per_run_rows:
        print("[warn] No runs with gate_usage found. (Did you patch train.py to save gate_usage?)")
        return

    # ---------- per-run outputs ----------
    gate_names_path = _sorted_gate_names(per_run_rows, "path_counts")
    gate_names_all  = _sorted_gate_names(per_run_rows, "all_counts")

    _write_gate_csv(out_dir / "gate_usage_runs_path.csv", per_run_rows, gate_names_path, prefix="path")
    print(f"[ok] Wrote: {out_dir / 'gate_usage_runs_path.csv'}")

    _write_gate_csv(out_dir / "gate_usage_runs_all_units.csv", per_run_rows, gate_names_all, prefix="all")
    print(f"[ok] Wrote: {out_dir / 'gate_usage_runs_all_units.csv'}")

    # ---------- grouped by setup ----------
    # Setup key mirrors your earlier grouping (add/remove fields as you like)
    group_key_fields = ("variant", "gate_set", "route", "repel", "mode", "lam_const16", "eta")
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in per_run_rows:
        key = tuple(r.get(k) for k in group_key_fields)
        groups[key].append(r)

    group_rows_path: List[Dict[str, Any]] = []
    group_rows_all: List[Dict[str, Any]] = []

    for key, items in groups.items():
        meta = {k: v for k, v in zip(group_key_fields, key)}
        seeds = sorted({it.get("seed") for it in items if it.get("seed") is not None})

        # sum counts
        pc = Counter()
        ac = Counter()
        for it in items:
            pc.update({k: int(v) for k, v in (it.get("path_counts") or {}).items()})
            ac.update({k: int(v) for k, v in (it.get("all_counts") or {}).items()})

        group_rows_path.append({
            "folder_root": "", "run_dir": "",
            **meta,
            "seed": ",".join(map(str, seeds)) if seeds else "",
            "path_counts": dict(pc),
            "path_fracs": _counter_to_frac_row(pc),
        })
        group_rows_all.append({
            "folder_root": "", "run_dir": "",
            **meta,
            "seed": ",".join(map(str, seeds)) if seeds else "",
            "all_counts": dict(ac),
            "all_fracs": _counter_to_frac_row(ac),
        })

    gate_names_gpath = _sorted_gate_names(group_rows_path, "path_counts")
    gate_names_gall  = _sorted_gate_names(group_rows_all, "all_counts")

    _write_gate_csv(out_dir / "gate_usage_groups_path.csv", group_rows_path, gate_names_gpath, prefix="path")
    print(f"[ok] Wrote: {out_dir / 'gate_usage_groups_path.csv'}")

    _write_gate_csv(out_dir / "gate_usage_groups_all_units.csv", group_rows_all, gate_names_gall, prefix="all")
    print(f"[ok] Wrote: {out_dir / 'gate_usage_groups_all_units.csv'}")

    # ---------- overall by gate_set (g6 overall, g16 overall) ----------
    by_gs = defaultdict(list)
    for r in per_run_rows:
        by_gs[r.get("gate_set")].append(r)

    gs_rows_path = []
    gs_rows_all  = []
    for gs, items in by_gs.items():
        pc = Counter()
        ac = Counter()
        for it in items:
            pc.update({k: int(v) for k, v in (it.get("path_counts") or {}).items()})
            ac.update({k: int(v) for k, v in (it.get("all_counts") or {}).items()})
        gs_rows_path.append({
            "folder_root": "", "run_dir": "",
            "variant": "ALL",
            "gate_set": gs,
            "route": "", "repel": "", "mode": "", "eta": "", "lam_const16": "",
            "seed": "",
            "path_counts": dict(pc),
            "path_fracs": _counter_to_frac_row(pc),
        })
        gs_rows_all.append({
            "folder_root": "", "run_dir": "",
            "variant": "ALL",
            "gate_set": gs,
            "route": "", "repel": "", "mode": "", "eta": "", "lam_const16": "",
            "seed": "",
            "all_counts": dict(ac),
            "all_fracs": _counter_to_frac_row(ac),
        })

    _write_gate_csv(out_dir / "gate_usage_by_gate_set_path.csv", gs_rows_path, _sorted_gate_names(gs_rows_path, "path_counts"), prefix="path")
    print(f"[ok] Wrote: {out_dir / 'gate_usage_by_gate_set_path.csv'}")

    _write_gate_csv(out_dir / "gate_usage_by_gate_set_all.csv", gs_rows_all, _sorted_gate_names(gs_rows_all, "all_counts"), prefix="all")
    print(f"[ok] Wrote: {out_dir / 'gate_usage_by_gate_set_all.csv'}")


if __name__ == "__main__":
    main()
