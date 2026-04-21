#!/usr/bin/env python3
"""Recover sampled-row hard-EM grid results from Slurm .out logs.

Only rows whose original full-benchmark indices belong to the deterministic
stratified sample are recovered. Timed-out logs therefore produce partial
same-slice results, never "any 200 rows".
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


ROW_RE = re.compile(
    r"^\[(?P<idx>\d+)/(?P<total>\d+)\]\s+"
    r"B=(?P<B>\d+)\s+"
    r"S:(?P<S_base>\d+)->\s*(?P<S_used>\d+)\s+"
    r"L:(?P<L_base>\d+)->\s*(?P<L_used>\d+)\s+"
    r"acc=(?P<row_acc>[0-9.]+)\s+EM=(?P<em>[01])\s+"
    r"decoded_acc=(?P<decoded_row_acc>[0-9.]+)\s+decoded_EM=(?P<decoded_em>[01])"
)
ES_RE = re.compile(r"\[early-stop\]\s+step=(?P<step>\d+)")
OUT_RE = re.compile(r"^\[info\]\s+out=(?P<out>.+)$")


def parse_task_ids(raw: str) -> List[int]:
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = [int(x) for x in part.split("-", 1)]
            ids.extend(range(lo, hi + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def parse_grid(script: Path) -> Tuple[List[List[str]], List[int]]:
    text = script.read_text()
    seeds_match = re.search(r"SEEDS=\(([^)]*)\)", text)
    seeds = [int(x) for x in seeds_match.group(1).split()] if seeds_match else [1, 2, 3]

    block_match = re.search(r"CONFIGS=\(\n(.*?)\n\)", text, re.S)
    if not block_match:
        raise ValueError(f"Could not parse CONFIGS from {script}")
    configs: List[List[str]] = []
    for line in block_match.group(1).splitlines():
        line = line.strip()
        if line.startswith('"'):
            configs.append(line.strip('"').split())
    if not configs:
        raise ValueError(f"No CONFIGS parsed from {script}")
    return configs, seeds


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


def sample_positions(dataset: Path, n: int, seed: int, stratify: str) -> Dict[int, int]:
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
    source_indices = sorted(int(row["_source_idx"]) for row in selected)
    return {source_idx: sample_pos for sample_pos, source_idx in enumerate(source_indices)}


def config_for_task(task_id: int, configs: List[List[str]], seeds: List[int], dataset: str) -> Tuple[str, int, Dict[str, Any]]:
    nconfig = len(configs)
    config_idx = task_id % nconfig
    seed_idx = task_id // nconfig
    if seed_idx >= len(seeds):
        raise ValueError(f"task_id={task_id} outside grid size {nconfig * len(seeds)}")
    seed = seeds[seed_idx]
    fields = configs[config_idx]
    (
        name, gate, opt, lr, route, repel, repel_mode, eta, prior, mi_disjoint,
        lift, relax, hard, eval_hard, gumbel_tau, s_add, l_add, t0, tmin,
        sched, phase, lam_ent, lam_const16, steps,
    ) = fields

    cfg: Dict[str, Any] = {
        "seed": seed,
        "gate_set": gate,
        "steps": int(steps),
        "optimizer": opt,
        "lr": float(lr),
        "dataset": dataset,
        "use_row_L": True,
        "use_max_L": False,
        "scale": {
            "use_row_S": True,
            "S_op": "add",
            "S_k": int(s_add),
            "S_min": 1,
            "S_max": 128,
            "L_op": "add",
            "L_k": int(l_add),
            "L_min": 2,
            "L_max": 16,
        },
        "lifting": {"use": lift == "true", "factor": 2.0},
        "relaxation": {
            "mode": relax,
            "hard": hard == "true",
            "gumbel_tau": float(gumbel_tau),
            "eval_hard": eval_hard == "true",
        },
        "anneal": {
            "T0": float(t0),
            "Tmin": float(tmin),
            "direction": "top_down",
            "schedule": sched,
            "phase_scale": float(phase),
            "start_frac": 0.0,
        },
        "sigma16": {
            "s_start": 0.25,
            "s_end": 0.10,
            "mode": "rbf",
            "radius": 0.75,
        },
        "regs": {
            "lam_entropy": float(lam_ent),
            "lam_div_units": 5.0e-4,
            "lam_div_rows": 5.0e-4,
            "lam_const16": float(lam_const16),
        },
        "pair": {
            "route": route,
            "prior_strength": float(prior),
            "mi_disjoint": mi_disjoint == "true",
            "repel": repel == "true",
            "mode": repel_mode,
            "eta": float(eta),
        },
        "early_stop": {
            "use": True,
            "metric": "decoded_em",
            "target": 1.0,
            "min_steps": 100,
            "check_every": 10,
            "patience_checks": 3,
        },
    }
    return name, seed, cfg


def parse_log(
    path: Path,
    steps: int,
    sample_pos_by_source_idx: Dict[int, int],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    rows: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    pending_steps: Optional[int] = None
    out_dir: Optional[str] = None
    for line in path.read_text(errors="replace").splitlines():
        out_m = OUT_RE.search(line)
        if out_m:
            out_dir = out_m.group("out").strip()

        es_m = ES_RE.search(line)
        if es_m:
            pending_steps = int(es_m.group("step"))
            continue

        m = ROW_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        source_idx = int(d["idx"]) - 1
        if source_idx not in sample_pos_by_source_idx or source_idx in seen:
            pending_steps = None
            continue
        seen.add(source_idx)
        row = {
            "idx": source_idx,
            "source_idx": source_idx,
            "sample_pos": sample_pos_by_source_idx[source_idx],
            "B": int(d["B"]),
            "S_base": int(d["S_base"]),
            "S_used": int(d["S_used"]),
            "L_base": int(d["L_base"]),
            "L_used": int(d["L_used"]),
            "row_acc": float(d["row_acc"]),
            "em": int(d["em"]),
            "decoded_row_acc": float(d["decoded_row_acc"]),
            "decoded_em": int(d["decoded_em"]),
            "train_steps": int(pending_steps or steps),
            "log_sample_row": True,
        }
        rows.append(row)
        pending_steps = None
        if len(rows) >= len(sample_pos_by_source_idx):
            break
    rows.sort(key=lambda r: int(r["sample_pos"]))
    return rows, out_dir


def mean(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(float(r.get(key, 0.0)) for r in rows) / len(rows)


def find_log(log_dir: Path, job_id: str, task_id: int) -> Optional[Path]:
    names = [
        f"ubc_main_hardem_grid-{job_id}_{task_id}.out",
        f"ubc_main_hardem_grid200_missing-{job_id}_{task_id}.out",
        f"ubc_main_hardem_grid200-{job_id}_{task_id}.out",
    ]
    for name in names:
        p = log_dir / name
        if p.exists():
            return p
    matches = sorted(log_dir.glob(f"*-{job_id}_{task_id}.out"))
    return matches[0] if matches else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--task-ids", required=True, help="Comma/range list, e.g. 4,6,8-10")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--grid-script", type=Path, default=Path("scripts/submit_bench_default_main_hardem_grid_cc.sh"))
    ap.add_argument("--dataset", default="data/bench_default.jsonl")
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--sample-seed", type=int, default=20260420)
    ap.add_argument("--sample-stratify", default="B")
    ap.add_argument("--min-rows", type=int, default=1)
    args = ap.parse_args()

    configs, seeds = parse_grid(args.grid_script)
    args.out_root.mkdir(parents=True, exist_ok=True)
    task_ids = parse_task_ids(args.task_ids)
    sample_pos_by_source_idx = sample_positions(
        Path(args.dataset),
        n=args.sample_n,
        seed=args.sample_seed,
        stratify=args.sample_stratify,
    )

    wrote = 0
    for task_id in task_ids:
        log_path = find_log(args.log_dir.expanduser(), args.job_id, task_id)
        if log_path is None:
            print(f"[skip] no log for task {task_id}")
            continue
        cfg_name, seed, cfg = config_for_task(task_id, configs, seeds, args.dataset)
        rows, source_out = parse_log(log_path, int(cfg["steps"]), sample_pos_by_source_idx)
        if len(rows) < args.min_rows:
            print(f"[skip] task {task_id} has only {len(rows)} rows (< {args.min_rows}): {log_path}")
            continue

        summary = {
            "config": cfg,
            "partial": len(rows) < args.sample_n,
            "completed": len(rows) >= args.sample_n,
            "reused_from_slurm_log": True,
            "sample": {
                "kind": "deterministic_stratified_sample_from_log",
                "target_rows": args.sample_n,
                "n_rows": len(rows),
                "seed": args.sample_seed,
                "stratify": args.sample_stratify,
                "source_indices": [int(r["source_idx"]) for r in rows],
                "missing_source_indices": [
                    int(source_idx)
                    for source_idx, _sample_pos in sorted(sample_pos_by_source_idx.items(), key=lambda kv: kv[1])
                    if source_idx not in {int(r["source_idx"]) for r in rows}
                ],
            },
            "source_log": str(log_path),
            "source_run_dir": source_out,
            "avg_row_acc": mean(rows, "row_acc"),
            "em_rate": mean(rows, "em"),
            "avg_decoded_row_acc": mean(rows, "decoded_row_acc"),
            "decoded_em_rate": mean(rows, "decoded_em"),
            "avg_diagnostics": {},
            "results": rows,
        }
        dest = args.out_root / cfg_name / f"seed{seed}" / f"logsample_job{args.job_id}_task{task_id}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "summary.json").write_text(json.dumps(summary, indent=2))
        (dest / "config.json").write_text(json.dumps(cfg, indent=2))
        with (dest / "results.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"[ok] task {task_id}: recovered {len(rows)} rows -> {dest}")
        wrote += 1

    print(f"[done] wrote {wrote}/{len(task_ids)} sampled-log summaries to {args.out_root}")


if __name__ == "__main__":
    main()
