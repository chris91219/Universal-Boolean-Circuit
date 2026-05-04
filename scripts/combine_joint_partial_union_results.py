#!/usr/bin/env python3
"""Salvage the largest available union from joint + rowjoint extension runs.

The B=18,20 extension may be running simultaneously in two layouts:

1. `joint`    : 20 shards x 20 rows
2. `rowjoint` : 400 shards x 1 row

This script scans both raw roots, unions whatever finished for each
`(mlp_match, seed, append_dataset_idx_orig)`, chooses one candidate when the
same row exists in both sources, and writes a normal joint-run tree again:

  out_root/
    mlp_neuron/union_seed1/summary.json
    mlp_neuron/union_seed1/results.jsonl
    mlp_neuron/union_seed1/expr_table.csv
    ...

It also writes:
  - union_runs.csv
  - union_groups.csv
  - union_missing_rows.csv

By default, duplicate rows prefer the 20-row `joint` source over `rowjoint`.
That preference can be changed with `--prefer row`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SEED_RE = re.compile(r"_seed(\d+)")


def parse_seed_from_path(run_dir: Path) -> int | None:
    m = SEED_RE.search(str(run_dir))
    return int(m.group(1)) if m else None


def parse_match_from_path(run_dir: Path) -> str:
    for part in run_dir.parts:
        if part.startswith("mlp_"):
            return part[len("mlp_") :]
    return "unknown"


def find_run_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    dirs: List[Path] = []
    for p in root.rglob("summary.json"):
        run_dir = p.parent
        if (run_dir / "results.jsonl").exists() and (run_dir / "expr_table.csv").exists():
            dirs.append(run_dir)
    return dirs


def read_expr_rows(path: Path) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(row["idx"])] = row
    return out


def choose_config_path(run_dir: Path) -> Path | None:
    cfg = run_dir / "config.used.yaml"
    return cfg if cfg.exists() else None


def collect_candidates(root: Path, source: str) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for run_dir in find_run_dirs(root):
        summary = json.loads((run_dir / "summary.json").read_text())
        match = str(summary.get("mlp_match", parse_match_from_path(run_dir)))
        seed = parse_seed_from_path(run_dir)
        if seed is None:
            continue

        expr_by_local_idx = read_expr_rows(run_dir / "expr_table.csv")
        with (run_dir / "results.jsonl").open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                local_idx = int(row["idx"])
                append_idx = row.get("append_dataset_idx_orig")
                if append_idx is None:
                    continue
                expr_row = expr_by_local_idx.get(local_idx)
                if expr_row is None:
                    continue
                grouped[(match, seed)].append(
                    {
                        "source": source,
                        "run_dir": run_dir,
                        "config_path": choose_config_path(run_dir),
                        "append_idx": int(append_idx),
                        "local_idx": local_idx,
                        "result_row": row,
                        "expr_row": dict(expr_row),
                    }
                )
    return grouped


def preference_rank(source: str, prefer: str) -> int:
    if source == prefer:
        return 0
    return 1


def choose_candidate(candidates: List[Dict[str, Any]], prefer: str) -> Dict[str, Any]:
    best_rank = min(preference_rank(str(c["source"]), prefer) for c in candidates)
    best = [c for c in candidates if preference_rank(str(c["source"]), prefer) == best_rank]
    return sorted(best, key=lambda c: str(c["run_dir"]))[-1]


def mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def build_summary(rows: List[Dict[str, Any]], config: Dict[str, Any] | None, match: str, out_dir: Path) -> Dict[str, Any]:
    n = len(rows)
    ubc_em = [float(r["result_row"]["ubc"]["em"]) for r in rows]
    ubc_dec_em = [float(r["result_row"]["ubc"]["decoded_em"]) for r in rows]
    mlp_em = [float(r["result_row"]["mlp"]["em"]) for r in rows]
    ubc_row_acc = [float(r["result_row"]["ubc"]["row_acc"]) for r in rows]
    ubc_dec_row_acc = [float(r["result_row"]["ubc"]["decoded_row_acc"]) for r in rows]
    mlp_row_acc = [float(r["result_row"]["mlp"]["row_acc"]) for r in rows]
    ubc_train_steps = [float(r["result_row"]["ubc"]["train_steps"]) for r in rows]
    mlp_params = [float(r["result_row"]["mlp"]["n_params"]) for r in rows]
    ubc_soft_params = [float(r["result_row"]["ubc"]["n_soft_params"]) for r in rows]
    ubc_total_params = [float(r["result_row"]["ubc"]["n_total_params"]) for r in rows]
    mlp_expr_fit_acc = [float(r["result_row"]["mlp"]["mlp_expr_fit_acc"]) for r in rows]

    bnr_exact_l1 = []
    bnr_eps_l1 = []
    prim_hit_l1 = []
    prim_best_acc_l1 = []
    diag_lists: Dict[str, List[float]] = defaultdict(list)
    for entry in rows:
        mlp = entry["result_row"]["mlp"]
        ubc = entry["result_row"]["ubc"]
        bnr_exact = list(mlp.get("bnr_exact_per_layer", []) or [])
        bnr_eps = list(mlp.get("bnr_eps_per_layer", []) or [])
        bnr_exact_l1.append(float(bnr_exact[0]) if bnr_exact else 0.0)
        bnr_eps_l1.append(float(bnr_eps[0]) if bnr_eps else 0.0)
        prim_hit_l1.append(float(mlp.get("primitive_hit_rate_L1", 0.0)))
        prim_best_acc_l1.append(float(mlp.get("mean_best_primitive_acc_L1", 0.0)))
        for key, value in (ubc.get("diagnostics", {}) or {}).items():
            if isinstance(value, (int, float)):
                diag_lists[f"diag_{key}"].append(float(value))

    means = {
        "ubc_em_rate": mean(ubc_em),
        "ubc_decoded_em_rate": mean(ubc_dec_em),
        "mlp_em_rate": mean(mlp_em),
        "ubc_row_acc": mean(ubc_row_acc),
        "ubc_decoded_row_acc": mean(ubc_dec_row_acc),
        "mlp_row_acc": mean(mlp_row_acc),
        "ubc_train_steps": mean(ubc_train_steps),
        "mlp_params": mean(mlp_params),
        "ubc_soft_params": mean(ubc_soft_params),
        "ubc_total_params": mean(ubc_total_params),
        "mlp_expr_fit_acc": mean(mlp_expr_fit_acc),
        "bnr_exact_L1": mean(bnr_exact_l1),
        "bnr_eps_L1": mean(bnr_eps_l1),
        "prim_hit_L1": mean(prim_hit_l1),
        "prim_best_acc_L1": mean(prim_best_acc_l1),
    }
    for key, vals in sorted(diag_lists.items()):
        means[key] = mean(vals)

    return {
        "config": config,
        "mlp_match": match,
        "means": means,
        "n_instances": n,
        "results_jsonl": str(out_dir / "results.jsonl"),
        "expr_table_csv": str(out_dir / "expr_table.csv"),
    }


def write_results_jsonl(entries: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for new_idx, entry in enumerate(entries):
            row = dict(entry["result_row"])
            row["idx"] = new_idx
            row["union_append_dataset_idx_orig"] = int(entry["append_idx"])
            row["union_source"] = str(entry["source"])
            row["union_run_dir"] = str(entry["run_dir"])
            f.write(json.dumps(row) + "\n")


def write_expr_csv(entries: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = list(entries[0]["expr_row"].keys()) + [
        "union_append_dataset_idx_orig",
        "union_source",
        "union_run_dir",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for new_idx, entry in enumerate(entries):
            row = dict(entry["expr_row"])
            row["idx"] = str(new_idx)
            row["union_append_dataset_idx_orig"] = str(entry["append_idx"])
            row["union_source"] = str(entry["source"])
            row["union_run_dir"] = str(entry["run_dir"])
            writer.writerow(row)


def write_group_csvs(run_rows: List[Dict[str, Any]], missing_rows: List[Dict[str, Any]], out_root: Path) -> None:
    run_fields = sorted({k for row in run_rows for k in row.keys()})
    with (out_root / "union_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows)

    missing_fields = sorted({k for row in missing_rows for k in row.keys()})
    with (out_root / "union_missing_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=missing_fields)
        writer.writeheader()
        writer.writerows(missing_rows)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["mlp_match"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for match, rows in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "mlp_match": match,
            "n_runs": len(rows),
            "seeds": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])),
        }
        numeric_keys = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if isinstance(value, (int, float)) and key != "seed"
            }
        )
        for key in numeric_keys:
            vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                out[f"{key}_mean"] = mean(vals)
        summary_rows.append(out)

    group_fields = sorted({k for row in summary_rows for k in row.keys()})
    with (out_root / "union_groups.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-root", required=True, type=Path)
    ap.add_argument("--row-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--prefer", choices=["joint", "row"], default="joint")
    ap.add_argument("--expected-total", type=int, default=400)
    args = ap.parse_args()

    joint_root = args.joint_root.expanduser().resolve()
    row_root = args.row_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    joint = collect_candidates(joint_root, source="joint")
    row = collect_candidates(row_root, source="row")
    keys = sorted(set(joint) | set(row))
    if not keys:
        raise SystemExit("No partial runs found under either root.")

    run_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    for match, seed in keys:
        candidates = list(joint.get((match, seed), [])) + list(row.get((match, seed), []))
        by_append: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for cand in candidates:
            by_append[int(cand["append_idx"])].append(cand)

        chosen: List[Dict[str, Any]] = []
        n_from_joint = 0
        n_from_row = 0
        for append_idx in sorted(by_append):
            picked = choose_candidate(by_append[append_idx], prefer=args.prefer)
            chosen.append(picked)
            if picked["source"] == "joint":
                n_from_joint += 1
            else:
                n_from_row += 1

        chosen = sorted(chosen, key=lambda e: int(e["append_idx"]))
        out_dir = out_root / f"mlp_{match}" / f"union_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = None
        for entry in chosen:
            if entry.get("config_path") is not None:
                cfg_path = entry["config_path"]
                break
        if cfg_path is not None:
            shutil.copy2(cfg_path, out_dir / "config.used.yaml")

        if chosen:
            write_results_jsonl(chosen, out_dir / "results.jsonl")
            write_expr_csv(chosen, out_dir / "expr_table.csv")
            config = json.loads((chosen[0]["run_dir"] / "summary.json").read_text()).get("config")
            summary = build_summary(chosen, config=config, match=match, out_dir=out_dir)
            coverage = len(chosen) / max(1, int(args.expected_total))
            summary["coverage"] = {
                "n_instances": len(chosen),
                "expected_total": int(args.expected_total),
                "fraction": coverage,
                "n_from_joint": n_from_joint,
                "n_from_row": n_from_row,
            }
            summary["union_preference"] = args.prefer
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

            row_out: Dict[str, Any] = {
                "mlp_match": match,
                "seed": seed,
                "n_instances": len(chosen),
                "expected_total": int(args.expected_total),
                "coverage_fraction": coverage,
                "n_from_joint": n_from_joint,
                "n_from_row": n_from_row,
                "run_dir": str(out_dir),
            }
            for key, value in summary.get("means", {}).items():
                if isinstance(value, (int, float)):
                    row_out[key] = float(value)
            run_rows.append(row_out)

        missing = sorted(set(range(int(args.expected_total))) - set(by_append))
        missing_rows.append(
            {
                "mlp_match": match,
                "seed": seed,
                "n_missing": len(missing),
                "missing_append_indices": ",".join(str(x) for x in missing),
            }
        )

    write_group_csvs(run_rows, missing_rows, out_root)
    print(f"[ok] wrote union results under {out_root}")


if __name__ == "__main__":
    main()
