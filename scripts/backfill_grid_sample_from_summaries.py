#!/usr/bin/env python3
"""Backfill a sampled grid from completed full-benchmark summaries.

This avoids rerunning pilot grids when we already have full 1200-row runs.
The script takes each completed summary.json under --full-root, selects the
same deterministic sample indices used by make_bench_sample.py, recomputes the
aggregate metrics on that subset, and writes a synthetic sampled run under
--out-root with summary.json and results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _read_dataset_rows(path: Path) -> List[Dict[str, Any]]:
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
        raise ValueError(f"Requested n={n}, but dataset has only {total} rows.")

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


def sample_indices(dataset: Path, n: int, seed: int, stratify: str) -> List[int]:
    rng = random.Random(seed)
    rows = _read_dataset_rows(dataset)
    if stratify:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(stratify, ""))].append(row)
        alloc = _allocations(groups, n)
        selected: List[Dict[str, Any]] = []
        for key, items in sorted(groups.items()):
            selected.extend(rng.sample(items, alloc.get(key, 0)))
    else:
        selected = rng.sample(rows, n)
    return sorted(int(row["_source_idx"]) for row in selected)


def _mean(results: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    if not results:
        return float("nan")
    return sum(float(r.get(key, default)) for r in results) / len(results)


def _avg_diagnostics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    diag_keys = sorted({
        k
        for r in results
        for k, v in (r.get("diagnostics") or {}).items()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    })
    return {
        k: sum(float(r.get("diagnostics", {}).get(k, 0.0)) for r in results if k in r.get("diagnostics", {}))
        / max(1, sum(1 for r in results if k in r.get("diagnostics", {})))
        for k in diag_keys
    }


def synthesize_sample_summary(
    source_summary_path: Path,
    full_root: Path,
    out_root: Path,
    indices: List[int],
    sample_meta: Dict[str, Any],
) -> bool:
    summary = _read_json(source_summary_path)
    results = summary.get("results")
    if not isinstance(results, list):
        print(f"[skip] no results list: {source_summary_path}")
        return False
    if not indices or max(indices) >= len(results):
        print(f"[skip] results length {len(results)} too short for sample: {source_summary_path}")
        return False

    selected: List[Dict[str, Any]] = []
    for sample_pos, source_idx in enumerate(indices):
        row = dict(results[source_idx])
        row["idx"] = source_idx
        row["source_idx"] = source_idx
        row["sample_pos"] = sample_pos
        selected.append(row)

    new_summary = dict(summary)
    new_summary["results"] = selected
    new_summary["avg_row_acc"] = _mean(selected, "row_acc")
    new_summary["em_rate"] = _mean(selected, "em")
    new_summary["avg_decoded_row_acc"] = _mean(selected, "decoded_row_acc")
    new_summary["decoded_em_rate"] = _mean(selected, "decoded_em")
    new_summary["avg_diagnostics"] = _avg_diagnostics(selected)
    new_summary["partial"] = False
    new_summary["completed"] = True
    new_summary["reused_from_full"] = True
    new_summary["source_summary"] = str(source_summary_path)
    new_summary["source_run_dir"] = str(source_summary_path.parent)
    new_summary["sample"] = sample_meta

    rel = source_summary_path.parent.relative_to(full_root)
    dest_dir = out_root / rel
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "summary.json").write_text(json.dumps(new_summary, indent=2))
    with (dest_dir / "results.jsonl").open("w") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")
    if (source_summary_path.parent / "config.used.yaml").exists():
        (dest_dir / "config.used.yaml").write_text((source_summary_path.parent / "config.used.yaml").read_text())
    if (source_summary_path.parent / "config.json").exists():
        (dest_dir / "config.json").write_text((source_summary_path.parent / "config.json").read_text())
    print(f"[ok] {source_summary_path.parent} -> {dest_dir}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--full-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260420)
    ap.add_argument("--stratify", default="B")
    args = ap.parse_args()

    full_root = args.full_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    indices = sample_indices(args.dataset, args.n, args.seed, args.stratify)
    sample_meta = {
        "dataset": str(args.dataset),
        "n": args.n,
        "seed": args.seed,
        "stratify": args.stratify,
        "source_indices": indices,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sample.meta.json").write_text(json.dumps(sample_meta, indent=2))

    n_ok = 0
    n_seen = 0
    for summary_path in sorted(full_root.rglob("summary.json")):
        n_seen += 1
        if synthesize_sample_summary(summary_path, full_root, out_root, indices, sample_meta):
            n_ok += 1

    print(f"[done] wrote {n_ok}/{n_seen} sampled summaries to {out_root}")


if __name__ == "__main__":
    main()
