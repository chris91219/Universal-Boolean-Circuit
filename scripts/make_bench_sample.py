#!/usr/bin/env python3
"""Create a deterministic JSONL pilot sample from a benchmark file."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_idx"] = idx
            rows.append(row)
    return rows


def _allocations(groups: Dict[str, List[Dict[str, Any]]], n: int) -> Dict[str, int]:
    total = sum(len(v) for v in groups.values())
    if n > total:
        raise ValueError(f"Requested n={n}, but input has only {total} rows.")

    raw = {k: n * len(v) / total for k, v in groups.items()}
    alloc = {k: int(raw[k]) for k in groups}
    remaining = n - sum(alloc.values())
    for k in sorted(groups, key=lambda kk: (raw[kk] - alloc[kk], len(groups[kk]), kk), reverse=True):
        if remaining <= 0:
            break
        if alloc[k] < len(groups[k]):
            alloc[k] += 1
            remaining -= 1
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260420)
    ap.add_argument("--stratify", default="B", help="JSON key for stratified sampling; use '' for unstratified.")
    ap.add_argument("--meta", type=Path, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = _read_jsonl(args.input)
    if args.stratify:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(args.stratify, ""))].append(row)
        alloc = _allocations(groups, args.n)
        selected: List[Dict[str, Any]] = []
        for key, items in sorted(groups.items()):
            k = alloc.get(key, 0)
            selected.extend(rng.sample(items, k))
    else:
        alloc = {"all": args.n}
        selected = rng.sample(rows, args.n)

    selected.sort(key=lambda r: int(r["_source_idx"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in selected:
            clean = dict(row)
            clean.pop("_source_idx", None)
            f.write(json.dumps(clean) + "\n")

    meta = {
        "input": str(args.input),
        "output": str(args.output),
        "n_requested": args.n,
        "n_written": len(selected),
        "seed": args.seed,
        "stratify": args.stratify,
        "allocations": alloc,
        "source_indices": [int(r["_source_idx"]) for r in selected],
    }
    if args.meta:
        args.meta.parent.mkdir(parents=True, exist_ok=True)
        args.meta.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
