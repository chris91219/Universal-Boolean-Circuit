#!/usr/bin/env python3
"""Merge old 1200-row joint runs with a new extension shard.

This is intended for the bench_default + B=14,16 extension workflow:
  - old root: existing 1200-row joint runs
  - ext root: new 400-row B=14,16 runs under the same seed/match/config family

The script pairs runs by `(mlp_match, seed)`, computes weighted merged means by
instance count, and writes:
  - merged_runs.csv
  - merged_groups.csv

It does not assume the two roots contain identical timestamps or job ids.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def parse_seed_from_path(run_dir: Path) -> int | None:
    s = str(run_dir)
    if "_seed" not in s:
        return None
    tail = s.split("_seed")[-1]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None


def parse_match_from_path(run_dir: Path) -> str:
    for part in run_dir.parts:
        if part.startswith("mlp_"):
            return part[len("mlp_") :]
    return "unknown"


def find_run_dirs(root: Path) -> Iterable[Path]:
    for p in root.rglob("summary.json"):
        run_dir = p.parent
        if (run_dir / "results.jsonl").exists():
            yield run_dir


def numeric_mean_fields(summary: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (summary.get("means", {}) or {}).items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def collect_runs(root: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    table: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for run_dir in find_run_dirs(root):
        summary = json.loads((run_dir / "summary.json").read_text())
        match = str(summary.get("mlp_match", parse_match_from_path(run_dir)))
        seed = parse_seed_from_path(run_dir)
        if seed is None:
            continue
        key = (match, seed)
        table[key] = {
            "run_dir": str(run_dir),
            "summary": summary,
            "n_instances": int(summary.get("n_instances", 0)),
            "means": numeric_mean_fields(summary),
        }
    return table


def weighted_merge(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, float]:
    n_old = int(old["n_instances"])
    n_new = int(new["n_instances"])
    total = n_old + n_new
    if total <= 0:
        return {}
    keys = set(old["means"]) | set(new["means"])
    out: Dict[str, float] = {}
    for key in keys:
        vo = old["means"].get(key)
        vn = new["means"].get(key)
        num = 0.0
        den = 0
        if vo is not None:
            num += float(vo) * n_old
            den += n_old
        if vn is not None:
            num += float(vn) * n_new
            den += n_new
        if den > 0:
            out[key] = num / den
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root", required=True, type=Path)
    ap.add_argument("--ext-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    old_root = args.old_root.expanduser().resolve()
    ext_root = args.ext_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    old_runs = collect_runs(old_root)
    ext_runs = collect_runs(ext_root)

    common_keys = sorted(set(old_runs) & set(ext_runs))
    if not common_keys:
        raise SystemExit("No matching (mlp_match, seed) runs found between roots.")

    run_rows = []
    group_rows: Dict[str, list[Dict[str, Any]]] = {}

    for match, seed in common_keys:
        old = old_runs[(match, seed)]
        new = ext_runs[(match, seed)]
        merged = weighted_merge(old, new)
        row: Dict[str, Any] = {
            "mlp_match": match,
            "seed": seed,
            "old_run_dir": old["run_dir"],
            "ext_run_dir": new["run_dir"],
            "old_n_instances": old["n_instances"],
            "ext_n_instances": new["n_instances"],
            "merged_n_instances": old["n_instances"] + new["n_instances"],
        }
        for key, val in merged.items():
            row[key] = val
        run_rows.append(row)
        group_rows.setdefault(match, []).append(row)

    run_fields = sorted({k for row in run_rows for k in row.keys()})
    with (out_dir / "merged_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows)

    summary_rows = []
    for match, rows in sorted(group_rows.items()):
        out: Dict[str, Any] = {
            "mlp_match": match,
            "n_runs": len(rows),
            "seeds": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])),
        }
        numeric_keys = sorted({
            k for row in rows for k, v in row.items()
            if isinstance(v, (int, float)) and k not in {"seed"}
        })
        for key in numeric_keys:
            vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                out[f"{key}_mean"] = sum(vals) / len(vals)
        summary_rows.append(out)

    group_fields = sorted({k for row in summary_rows for k in row.keys()})
    with (out_dir / "merged_groups.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[ok] wrote {out_dir / 'merged_runs.csv'}")
    print(f"[ok] wrote {out_dir / 'merged_groups.csv'}")


if __name__ == "__main__":
    main()
