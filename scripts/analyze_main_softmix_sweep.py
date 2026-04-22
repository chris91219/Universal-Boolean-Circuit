#!/usr/bin/env python3
"""Collate main soft-mixture sweep summaries.

This intentionally uses only the Python stdlib so it works on Compute Canada
even before pandas is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def find_summaries(root: Path) -> Iterable[Path]:
    yield from root.rglob("summary.json")


def infer_from_path(path: Path) -> Dict[str, Any]:
    parts = path.parts
    out: Dict[str, Any] = {}
    for part in parts:
        if part.startswith("basis_"):
            out["basis"] = part.removeprefix("basis_")
        elif part.startswith("Sadd") and "_Ladd" in part:
            m = re.match(r"Sadd(-?\d+)_Ladd(-?\d+)", part)
            if m:
                out["S_add"] = int(m.group(1))
                out["L_add"] = int(m.group(2))
        elif part in {"sbc", "mlp_neuron", "mlp_param_soft", "mlp_param_total"}:
            out["model_mode"] = part
    name = path.parent.name
    m = re.search(r"_seed(\d+)", name)
    if m:
        out["seed"] = int(m.group(1))
    m = re.search(r"_task(\d+)", name)
    if m:
        out["task_id"] = int(m.group(1))
    return out


def finite_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def flatten_summary(path: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    row = infer_from_path(path)
    means = summary.get("means", {}) or {}
    cfg = summary.get("config", {}) or {}
    scale = summary.get("scale", {}) or cfg.get("scale", {}) or {}
    row.update(
        {
            "summary_path": str(path),
            "run_dir": str(path.parent),
            "n_instances": summary.get("n_instances", len(summary.get("results", []) or [])),
            "mlp_match": summary.get("mlp_match", row.get("model_mode", "")),
            "W_base_field": summary.get("W_base_field", scale.get("W_base_field", "")),
            "D_base_field": summary.get("D_base_field", scale.get("D_base_field", "")),
            "em_rate": means.get("em_rate", summary.get("em_rate")),
            "row_acc": means.get("row_acc", summary.get("avg_row_acc")),
            "decoded_em_rate": means.get("decoded_em_rate", summary.get("decoded_em_rate")),
            "decoded_row_acc": means.get("decoded_row_acc", summary.get("avg_decoded_row_acc")),
            "train_steps": means.get("train_steps"),
            "mlp_params": means.get("mlp_params"),
            "reference_sbc_soft_params": means.get("reference_sbc_soft_params"),
            "reference_sbc_total_params": means.get("reference_sbc_total_params"),
            "mlp_bnr_exact_L1": means.get("mlp_bnr_exact_L1"),
            "mlp_bnr_eps_L1": means.get("mlp_bnr_eps_L1"),
            "mlp_bnr_exact_avg": means.get("mlp_bnr_exact_avg"),
            "mlp_bnr_eps_avg": means.get("mlp_bnr_eps_avg"),
            "mlp_unique_norm_avg": means.get("mlp_unique_norm_avg"),
        }
    )
    avg_diag = summary.get("avg_diagnostics", {}) or {}
    for key in [
        "gate_entropy_mean",
        "gate_max_prob_mean",
        "gate_margin_mean",
        "row_entropy_mean",
        "row_max_prob_mean",
        "pair_entropy_mean",
        "pair_max_prob_mean",
        "output_margin_mean",
        "bce",
    ]:
        row[key] = avg_diag.get(key)
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = sorted({k for r in rows for k in r})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_std(vals: List[float]) -> Tuple[float, float]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def group_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    group_keys = ["basis", "S_add", "L_add", "model_mode", "mlp_match"]
    for r in rows:
        groups[tuple(r.get(k) for k in group_keys)].append(r)

    metric_keys = [
        "em_rate",
        "row_acc",
        "decoded_em_rate",
        "decoded_row_acc",
        "train_steps",
        "mlp_params",
        "mlp_bnr_exact_L1",
        "mlp_bnr_eps_L1",
        "mlp_bnr_exact_avg",
        "mlp_bnr_eps_avg",
        "mlp_unique_norm_avg",
        "gate_entropy_mean",
        "gate_max_prob_mean",
        "gate_margin_mean",
        "row_entropy_mean",
        "pair_entropy_mean",
        "output_margin_mean",
    ]
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        g = {k: v for k, v in zip(group_keys, key)}
        g["n_runs"] = len(rs)
        g["seeds"] = ",".join(str(r.get("seed", "")) for r in sorted(rs, key=lambda x: str(x.get("seed", ""))))
        for metric in metric_keys:
            vals = [v for r in rs if (v := finite_float(r.get(metric))) is not None]
            mean, std = mean_std(vals)
            g[f"{metric}_mean"] = mean
            g[f"{metric}_std"] = std
        out.append(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for path in find_summaries(args.root):
        try:
            rows.append(flatten_summary(path, read_json(path)))
        except Exception as exc:  # keep sweep collation robust to partial runs
            rows.append({"summary_path": str(path), "error": repr(exc)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "meta_runs.csv", rows)
    write_csv(args.out_dir / "meta_groups.csv", group_rows([r for r in rows if "error" not in r]))
    print(f"[ok] runs={len(rows)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
