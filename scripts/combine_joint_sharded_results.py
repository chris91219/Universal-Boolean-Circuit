#!/usr/bin/env python3
"""Combine sharded joint SBC-vs-MLP runs back into normal run directories.

This is designed for the high-B extension workflow where each `(mlp_match, seed)`
job is split across contiguous dataset shards.  The combined output looks like a
regular joint run root again:

  out_root/
    mlp_neuron/combined_seed1/summary.json
    mlp_neuron/combined_seed1/results.jsonl
    mlp_neuron/combined_seed1/expr_table.csv
    ...

That lets downstream scripts such as `merge_joint_extension_results.py` consume
the recombined extension results without caring that they were sharded.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SEED_RE = re.compile(r"_seed(\d+)")
SHARD_RE = re.compile(r"_shard(\d+)of(\d+)")


def parse_seed_from_path(run_dir: Path) -> int | None:
    m = SEED_RE.search(str(run_dir))
    return int(m.group(1)) if m else None


def parse_shard_from_path(run_dir: Path) -> Tuple[int | None, int | None]:
    m = SHARD_RE.search(str(run_dir))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_match_from_path(run_dir: Path) -> str:
    for part in run_dir.parts:
        if part.startswith("mlp_"):
            return part[len("mlp_") :]
    return "unknown"


def find_run_dirs(root: Path) -> Iterable[Path]:
    for p in root.rglob("summary.json"):
        run_dir = p.parent
        if (run_dir / "results.jsonl").exists() and (run_dir / "expr_table.csv").exists():
            yield run_dir


def numeric_mean_fields(summary: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in (summary.get("means", {}) or {}).items():
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def choose_config_path(run_dir: Path) -> Path | None:
    cfg = run_dir / "config.used.yaml"
    return cfg if cfg.exists() else None


def collect_shards(root: Path) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for run_dir in find_run_dirs(root):
        summary = json.loads((run_dir / "summary.json").read_text())
        match = str(summary.get("mlp_match", parse_match_from_path(run_dir)))
        seed = parse_seed_from_path(run_dir)
        shard_idx, n_shards = parse_shard_from_path(run_dir)
        if seed is None or shard_idx is None or n_shards is None:
            continue
        groups.setdefault((match, seed), []).append(
            {
                "run_dir": run_dir,
                "summary": summary,
                "means": numeric_mean_fields(summary),
                "n_instances": int(summary.get("n_instances", 0)),
                "shard_idx": shard_idx,
                "n_shards": n_shards,
                "config_path": choose_config_path(run_dir),
            }
        )
    return groups


def weighted_means(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    total = sum(int(e["n_instances"]) for e in entries)
    if total <= 0:
        return {}
    keys = sorted({k for e in entries for k in e["means"].keys()})
    out: Dict[str, float] = {}
    for key in keys:
        num = 0.0
        den = 0
        for entry in entries:
            value = entry["means"].get(key)
            n_i = int(entry["n_instances"])
            if value is None or n_i <= 0:
                continue
            num += float(value) * n_i
            den += n_i
        if den > 0:
            out[key] = num / den
    return out


def copy_results_jsonl(entries: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    global_idx = 0
    with out_path.open("w") as out_f:
        for entry in entries:
            src = Path(entry["run_dir"]) / "results.jsonl"
            with src.open("r") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["idx"] = global_idx
                    row["combined_shard_idx"] = int(entry["shard_idx"])
                    out_f.write(json.dumps(row) + "\n")
                    global_idx += 1


def copy_expr_csv(entries: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] | None = None
    global_idx = 0
    for entry in entries:
        src = Path(entry["run_dir"]) / "expr_table.csv"
        with src.open("r", newline="") as in_f:
            reader = csv.DictReader(in_f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                row["idx"] = str(global_idx)
                rows.append(row)
                global_idx += 1
    if fieldnames is None:
        raise RuntimeError(f"No expr_table.csv rows found for {out_path}")
    with out_path.open("w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(entries: List[Dict[str, Any]], out_dir: Path, match: str, seed: int) -> None:
    first_summary = dict(entries[0]["summary"])
    means = weighted_means(entries)
    summary = {
        "config": first_summary.get("config"),
        "mlp_match": match,
        "scale": first_summary.get("scale"),
        "means": means,
        "n_instances": sum(int(e["n_instances"]) for e in entries),
        "results_jsonl": str(out_dir / "results.jsonl"),
        "expr_table_csv": str(out_dir / "expr_table.csv"),
        "combined_from_shards": [
            {
                "run_dir": str(entry["run_dir"]),
                "shard_idx": int(entry["shard_idx"]),
                "n_shards": int(entry["n_shards"]),
                "n_instances": int(entry["n_instances"]),
            }
            for entry in entries
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def write_group_csvs(rows: List[Dict[str, Any]], out_root: Path) -> None:
    run_fields = sorted({k for row in rows for k in row.keys()})
    with (out_root / "combined_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["mlp_match"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for match, match_rows in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "mlp_match": match,
            "n_runs": len(match_rows),
            "seeds": ",".join(str(r["seed"]) for r in sorted(match_rows, key=lambda r: r["seed"])),
        }
        numeric_keys = sorted(
            {
                key
                for row in match_rows
                for key, value in row.items()
                if isinstance(value, (int, float)) and key != "seed"
            }
        )
        for key in numeric_keys:
            vals = [float(r[key]) for r in match_rows if isinstance(r.get(key), (int, float))]
            if vals:
                out[f"{key}_mean"] = sum(vals) / len(vals)
        summary_rows.append(out)

    group_fields = sorted({k for row in summary_rows for k in row.keys()})
    with (out_root / "combined_groups.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharded-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--expected-shards", type=int, default=None)
    args = ap.parse_args()

    sharded_root = args.sharded_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    groups = collect_shards(sharded_root)
    if not groups:
        raise SystemExit(f"No shard runs found under {sharded_root}")

    run_rows: List[Dict[str, Any]] = []
    for (match, seed), entries in sorted(groups.items()):
        entries = sorted(entries, key=lambda e: int(e["shard_idx"]))
        n_shards_values = {int(e["n_shards"]) for e in entries}
        if len(n_shards_values) != 1:
            raise SystemExit(f"Inconsistent n_shards for {(match, seed)}: {sorted(n_shards_values)}")
        n_shards = next(iter(n_shards_values))
        if args.expected_shards is not None and n_shards != args.expected_shards:
            raise SystemExit(
                f"Expected {args.expected_shards} shards for {(match, seed)}, found metadata {n_shards}"
            )
        observed = [int(e["shard_idx"]) for e in entries]
        expected = list(range(n_shards))
        if observed != expected:
            raise SystemExit(
                f"Missing shard(s) for {(match, seed)}: observed {observed}, expected {expected}"
            )

        out_dir = out_root / f"mlp_{match}" / f"combined_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = entries[0].get("config_path")
        if cfg_path is not None:
            shutil.copy2(cfg_path, out_dir / "config.used.yaml")

        copy_results_jsonl(entries, out_dir / "results.jsonl")
        copy_expr_csv(entries, out_dir / "expr_table.csv")
        write_summary(entries, out_dir, match=match, seed=seed)

        summary = json.loads((out_dir / "summary.json").read_text())
        row: Dict[str, Any] = {
            "mlp_match": match,
            "seed": seed,
            "n_shards": n_shards,
            "n_instances": int(summary.get("n_instances", 0)),
            "run_dir": str(out_dir),
        }
        for key, value in (summary.get("means", {}) or {}).items():
            if isinstance(value, (int, float)):
                row[key] = float(value)
        run_rows.append(row)

    write_group_csvs(run_rows, out_root)
    print(f"[ok] wrote recombined runs under {out_root}")


if __name__ == "__main__":
    main()
