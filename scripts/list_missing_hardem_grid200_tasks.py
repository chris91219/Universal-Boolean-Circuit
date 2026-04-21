#!/usr/bin/env python3
"""Print hard-EM grid200 Slurm array IDs missing enough existing results."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple


def parse_grid(script: Path) -> Tuple[List[str], List[int]]:
    text = script.read_text()
    seeds_match = re.search(r"SEEDS=\(([^)]*)\)", text)
    seeds = [int(x) for x in seeds_match.group(1).split()] if seeds_match else [1, 2, 3]

    block_match = re.search(r"CONFIGS=\(\n(.*?)\n\)", text, re.S)
    if not block_match:
        raise ValueError(f"Could not parse CONFIGS from {script}")
    names: List[str] = []
    for line in block_match.group(1).splitlines():
        line = line.strip()
        if not line.startswith('"'):
            continue
        names.append(line.strip('"').split()[0])
    if not names:
        raise ValueError(f"No CONFIGS parsed from {script}")
    return names, seeds


def summary_done(path: Path, sample_n: int) -> bool:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    results = data.get("results")
    return isinstance(results, list) and len(results) >= sample_n


def jsonl_done(path: Path, sample_n: int) -> bool:
    try:
        with path.open("r") as f:
            return sum(1 for line in f if line.strip()) >= sample_n
    except Exception:
        return False


def is_done(roots: List[Path], cfg_name: str, seed: int, sample_n: int) -> bool:
    for root in roots:
        run_base = root / cfg_name / f"seed{seed}"
        if not run_base.exists():
            continue
        if any(summary_done(p, sample_n) for p in run_base.rglob("summary.json")):
            return True
        if any(jsonl_done(p, sample_n) for p in run_base.rglob("results.jsonl")):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--script",
        type=Path,
        default=Path("scripts/submit_bench_default_main_hardem_grid_cc.sh"),
        help="Grid script to parse for CONFIGS/SEEDS.",
    )
    ap.add_argument("--roots", nargs="+", required=True, type=Path, help="Roots to count as already done.")
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--min-rows", type=int, default=None, help="Rows required to count as done; defaults to --sample-n.")
    ap.add_argument("--format", choices=["csv", "lines", "table"], default="csv")
    args = ap.parse_args()

    cfg_names, seeds = parse_grid(args.script)
    roots = [p.expanduser() for p in args.roots]
    min_rows = int(args.min_rows if args.min_rows is not None else args.sample_n)

    missing = []
    for seed_idx, seed in enumerate(seeds):
        for cfg_idx, cfg_name in enumerate(cfg_names):
            task_id = seed_idx * len(cfg_names) + cfg_idx
            if not is_done(roots, cfg_name, seed, min_rows):
                missing.append((task_id, cfg_idx, cfg_name, seed))

    if args.format == "csv":
        print(",".join(str(row[0]) for row in missing))
    elif args.format == "lines":
        for row in missing:
            print(row[0])
    else:
        print("task_id\tconfig_idx\tconfig\tseed")
        for task_id, cfg_idx, cfg_name, seed in missing:
            print(f"{task_id}\t{cfg_idx}\t{cfg_name}\t{seed}")
        print(f"# missing={len(missing)} total={len(cfg_names) * len(seeds)}", file=sys.stderr)


if __name__ == "__main__":
    main()
