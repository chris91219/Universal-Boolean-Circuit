"""MLP-only truth-table runner with reference SBC parameter matching.

This runner is intentionally separate from the joint UBC-vs-MLP runner.  It
trains only the MLP, while computing the reference SBC shape/parameter counts
from the requested W/D fields.  That lets us cheaply screen MLP robustness
before spending cluster time on SBC training.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from . import tasks as T
from .modules import DepthStack
from .readout import normalize_expr
from .train import (
    _device,
    load_config,
    per_instance_metrics,
    safe_bce,
    seed_all,
    stable_seed,
)


class TruthTableMLP(nn.Module):
    def __init__(self, in_bits: int, hidden_dim: int, depth: int):
        super().__init__()
        self.in_bits = int(in_bits)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        layers: List[nn.Module] = []
        last = self.in_bits
        for _ in range(self.depth):
            layers.append(nn.Linear(last, self.hidden_dim))
            layers.append(nn.ReLU())
            last = self.hidden_dim
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_acts: bool = False):
        if not return_acts:
            return torch.sigmoid(self.net(x))

        acts: List[torch.Tensor] = []
        h = x
        for layer in self.net[:-1]:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                acts.append(h)
        y = torch.sigmoid(self.net[-1](h))
        return y, acts


def configure_torch_threads() -> None:
    raw = os.environ.get("UBC_TORCH_THREADS") or os.environ.get("SLURM_CPUS_PER_TASK")
    if not raw:
        return
    try:
        n = max(1, int(raw))
    except ValueError:
        return
    torch.set_num_threads(n)
    # Inter-op parallelism usually hurts these many-small-model workloads.
    try:
        torch.set_num_interop_threads(max(1, int(os.environ.get("UBC_TORCH_INTEROP_THREADS", "1"))))
    except RuntimeError:
        pass


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ubc_param_counts(
    B: int,
    W_model: int,
    D_model: int,
    gate_set: str,
    pair_cfg: Dict[str, Any],
    tau0: float,
    use_lifting: bool,
    lift_factor: float,
) -> Tuple[int, int]:
    pair_cfg = dict(pair_cfg or {})
    pair_cfg["route"] = "learned"
    model = DepthStack(
        B=B,
        L=D_model,
        S=W_model,
        tau=tau0,
        pair=pair_cfg,
        gate_set=gate_set,
        use_lifting=use_lifting,
        lift_factor=lift_factor,
    )
    n_soft = count_trainable_params(model)
    K = 16 if gate_set == "16" else 6
    n_total = n_soft + (D_model * W_model * K)
    return int(n_soft), int(n_total)


def build_mlp_param_matched(
    B: int,
    depth: int,
    target_params: int,
    min_hidden: int = 1,
    max_hidden: int = 4096,
) -> Tuple[TruthTableMLP, int]:
    best_model: TruthTableMLP | None = None
    best_n = -1
    for hidden in range(min_hidden, max_hidden + 1):
        model = TruthTableMLP(in_bits=B, hidden_dim=hidden, depth=depth)
        n_params = count_trainable_params(model)
        if n_params <= target_params:
            best_model = model
            best_n = n_params
        elif best_model is not None:
            break
        else:
            best_model = model
            best_n = n_params
            break
    assert best_model is not None
    return best_model, int(best_n)


def row_int(inst: Dict[str, Any], field: str, fallbacks: Tuple[str, ...], default: int) -> int:
    for key in (field, *fallbacks):
        if key in inst:
            return int(inst[key])
    return int(default)


def apply_scale(val: int, op: str, k: int, vmin: int, vmax: int) -> int:
    op = str(op).lower()
    if op in {"none", "identity"}:
        out = val
    elif op == "add":
        out = val + int(k)
    elif op == "mul":
        out = val * int(k)
    else:
        raise ValueError("op must be one of none|identity|add|mul")
    return int(max(vmin, min(vmax, out)))


def eval_model(model: nn.Module, X: torch.Tensor, y_true: torch.Tensor, batch_size: int = 0) -> Tuple[float, int]:
    if batch_size <= 0 or X.shape[0] <= batch_size:
        y_pred = model(X)
    else:
        preds = []
        for start in range(0, X.shape[0], batch_size):
            preds.append(model(X[start:start + batch_size]))
        y_pred = torch.cat(preds, dim=0)
    return per_instance_metrics(y_true, y_pred)


def _round_tensor(x: torch.Tensor, decimals: int) -> torch.Tensor:
    if decimals <= 0:
        return torch.round(x)
    scale = 10.0 ** decimals
    return torch.round(x * scale) / scale


def bnr_exact_fraction(layer_act: torch.Tensor, decimals: int = 6) -> float:
    """Fraction of units whose full-truth-table activations have <=2 levels."""
    A = _round_tensor(layer_act.detach().cpu(), decimals=decimals)
    if A.numel() == 0:
        return 0.0
    _, H = A.shape
    ok = 0
    for j in range(H):
        if torch.unique(A[:, j]).numel() <= 2:
            ok += 1
    return float(ok / max(1, H))


def bnr_eps_fraction(layer_act: torch.Tensor, eps: float = 1e-3) -> float:
    """Two-cluster Boolean-neuron-rate proxy with tolerance eps."""
    A = layer_act.detach().cpu().float()
    if A.numel() == 0:
        return 0.0
    _, H = A.shape
    ok = 0
    for j in range(H):
        v = A[:, j]
        med = torch.median(v)
        lo = v[v <= med]
        hi = v[v > med]
        if lo.numel() == 0 or hi.numel() == 0:
            ok += 1
            continue
        c0 = torch.median(lo)
        c1 = torch.median(hi)
        dist = torch.minimum((v - c0).abs(), (v - c1).abs()).max().item()
        if dist <= eps:
            ok += 1
    return float(ok / max(1, H))


def activation_level_stats(layer_act: torch.Tensor, decimals: int = 6) -> Dict[str, float]:
    """Cheap summary of how many distinct activation levels each hidden unit uses."""
    A = _round_tensor(layer_act.detach().cpu(), decimals=decimals)
    if A.numel() == 0:
        return {"unique_mean": 0.0, "unique_median": 0.0, "unique_norm_mean": 0.0}
    N, H = A.shape
    counts = torch.tensor([torch.unique(A[:, j]).numel() for j in range(H)], dtype=torch.float32)
    return {
        "unique_mean": float(counts.mean().item()),
        "unique_median": float(counts.median().item()),
        "unique_norm_mean": float((counts / max(1, N)).mean().item()),
    }


def mlp_activation_diagnostics(
    model: nn.Module,
    X: torch.Tensor,
    *,
    max_B: int,
    decimals: int = 6,
    eps: float = 1e-3,
) -> Dict[str, Any]:
    """BNR diagnostics for asking whether an MLP learned Boolean-like hidden units."""
    B = int(X.shape[1])
    if B > int(max_B):
        return {
            "mlp_diag_skipped": 1,
            "mlp_diag_skip_reason": f"B={B} exceeds diagnostics_max_B={max_B}",
        }

    with torch.no_grad():
        _y_pred, acts = model(X, return_acts=True)

    bnr_exact = [bnr_exact_fraction(a, decimals=decimals) for a in acts]
    bnr_eps = [bnr_eps_fraction(a, eps=eps) for a in acts]
    level_stats = [activation_level_stats(a, decimals=decimals) for a in acts]
    unique_norm = [float(s["unique_norm_mean"]) for s in level_stats]

    def avg(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    return {
        "mlp_diag_skipped": 0,
        "mlp_bnr_exact_per_layer": bnr_exact,
        "mlp_bnr_eps_per_layer": bnr_eps,
        "mlp_activation_level_stats": level_stats,
        "mlp_bnr_exact_L1": float(bnr_exact[0]) if bnr_exact else 0.0,
        "mlp_bnr_eps_L1": float(bnr_eps[0]) if bnr_eps else 0.0,
        "mlp_bnr_exact_avg": avg(bnr_exact),
        "mlp_bnr_eps_avg": avg(bnr_eps),
        "mlp_unique_norm_avg": avg(unique_norm),
    }


def train_one(
    device: torch.device,
    cfg: Dict[str, Any],
    inst: Dict[str, Any],
    idx: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    B = int(inst["B"])
    formula = str(inst["formula"])

    W_base = row_int(inst, args.W_base_field, ("S", "W_base"), int(cfg.get("S", 2)))
    D_base = row_int(inst, args.D_base_field, ("L", "D_base"), int(cfg.get("L", 2)))
    W_model = apply_scale(W_base, args.S_op, args.S_k, args.S_min, args.S_max)
    D_model = apply_scale(D_base, args.L_op, args.L_k, args.L_min, args.L_max)

    gate_set = str(cfg.get("gate_set", "16"))
    pair_cfg = dict(cfg.get("pair", {}) or {})
    anneal = cfg.get("anneal", {})
    tau0 = float(anneal.get("T0", 0.60))
    lifting = cfg.get("lifting", {})
    use_lifting = bool(lifting.get("use", True))
    lift_factor = float(lifting.get("factor", 2.0))

    n_soft_ubc, n_total_ubc = ubc_param_counts(
        B=B,
        W_model=W_model,
        D_model=D_model,
        gate_set=gate_set,
        pair_cfg=pair_cfg,
        tau0=tau0,
        use_lifting=use_lifting,
        lift_factor=lift_factor,
    )

    mode = str(args.mlp_match)
    if mode == "soft":
        mode = "param_soft"
    elif mode == "total":
        mode = "param_total"

    if mode == "neuron":
        model = TruthTableMLP(in_bits=B, hidden_dim=W_model, depth=D_model)
        n_params = count_trainable_params(model)
        target_params = None
    elif mode == "param_soft":
        target_params = n_soft_ubc
        model, n_params = build_mlp_param_matched(B=B, depth=D_model, target_params=target_params)
    elif mode == "param_total":
        target_params = n_total_ubc
        model, n_params = build_mlp_param_matched(B=B, depth=D_model, target_params=target_params)
    else:
        raise ValueError("mlp_match must be neuron|param_soft|param_total")

    seed_all(stable_seed(int(cfg.get("seed", 0)), "mlp-only", mode, idx))
    model = model.to(device)
    X, y_true = T.truth_table_from_formula(B, formula)
    X = X.to(device)
    y_true = y_true.to(device)

    opt_name = str(cfg.get("optimizer", "rmsprop")).lower()
    lr = float(cfg.get("lr", 0.001))
    opt = (
        optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-8)
        if opt_name == "rmsprop"
        else optim.Adam(model.parameters(), lr=lr)
    )

    steps = int(cfg.get("steps", 3000))
    batch_size = int(args.batch_size)
    es = cfg.get("early_stop", {}) or {}
    use_es = bool(es.get("use", True))
    min_steps = int(es.get("min_steps", 100))
    check_every = int(es.get("check_every", 10))
    patience_checks = int(es.get("patience_checks", 3))
    metric = str(es.get("metric", "em")).lower()
    target = float(es.get("target", 1.0))
    ok_streak = 0
    train_steps = 0

    n_rows = int(X.shape[0])
    for step in range(steps):
        train_steps = step + 1
        opt.zero_grad()
        if batch_size > 0 and n_rows > batch_size:
            batch_idx = torch.randint(0, n_rows, (batch_size,), device=X.device)
            xb = X.index_select(0, batch_idx)
            yb = y_true.index_select(0, batch_idx)
        else:
            xb = X
            yb = y_true
        y_pred = model(xb)
        loss = safe_bce(y_pred, yb)
        loss.backward()
        opt.step()

        if use_es and train_steps >= min_steps and train_steps % check_every == 0:
            with torch.no_grad():
                row_acc, em = eval_model(model, X, y_true, batch_size=int(args.eval_batch_size))
            cur = float(em) if metric in {"em", "decoded_em"} else float(row_acc)
            ok_streak = ok_streak + 1 if cur >= target else 0
            if ok_streak >= patience_checks:
                print(f"  [mlp early-stop] step={train_steps}, metric={metric}={cur:.3f}")
                break

    with torch.no_grad():
        row_acc, em = eval_model(model, X, y_true, batch_size=int(args.eval_batch_size))

    diagnostics: Dict[str, Any] = {}
    if bool(args.diagnostics):
        diagnostics = mlp_activation_diagnostics(
            model,
            X,
            max_B=int(args.diagnostics_max_B),
            decimals=int(args.bnr_decimals),
            eps=float(args.bnr_eps),
        )

    return {
        "idx": idx,
        "B": B,
        "B_true": inst.get("B_true"),
        "W_base": W_base,
        "D_base": D_base,
        "W_model": W_model,
        "D_model": D_model,
        "S_base": W_base,
        "L_base": D_base,
        "S_used": W_model,
        "L_used": D_model,
        "W_base_field": args.W_base_field,
        "D_base_field": args.D_base_field,
        "mlp_match": mode,
        "mlp_hidden_dim": int(getattr(model, "hidden_dim", W_model)),
        "mlp_depth": int(getattr(model, "depth", D_model)),
        "mlp_params": int(n_params),
        "target_params": target_params,
        "reference_sbc_soft_params": n_soft_ubc,
        "reference_sbc_total_params": n_total_ubc,
        "row_acc": float(row_acc),
        "em": int(em),
        "train_steps": int(train_steps),
        "diagnostics": diagnostics,
        "mlp_bnr_exact_L1": diagnostics.get("mlp_bnr_exact_L1"),
        "mlp_bnr_eps_L1": diagnostics.get("mlp_bnr_eps_L1"),
        "mlp_bnr_exact_avg": diagnostics.get("mlp_bnr_exact_avg"),
        "mlp_bnr_eps_avg": diagnostics.get("mlp_bnr_eps_avg"),
        "mlp_unique_norm_avg": diagnostics.get("mlp_unique_norm_avg"),
        "formula": formula,
        "label_expr": normalize_expr(formula),
    }


def run(cfg: Dict[str, Any], out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    configure_torch_threads()
    device = _device(cfg)
    seed_all(int(cfg.get("seed", 0)))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    insts = T.load_instances_jsonl(cfg["dataset"])
    if args.max_B is not None:
        insts = [row for row in insts if int(row["B"]) <= int(args.max_B)]

    results: List[Dict[str, Any]] = []
    results_path = out_dir / "results.jsonl"
    with results_path.open("w", buffering=1) as f:
        for idx, inst in enumerate(insts):
            res = train_one(device=device, cfg=cfg, inst=inst, idx=idx, args=args)
            results.append(res)
            f.write(json.dumps(res) + "\n")
            print(
                f"[{idx+1}/{len(insts)}] B={res['B']} "
                f"W:{res['W_base']}->{res['W_model']} "
                f"D:{res['D_base']}->{res['D_model']} "
                f"MLP({res['mlp_match']}) acc={res['row_acc']:.3f} EM={res['em']} "
                f"params={res['mlp_params']}"
            )

    n = max(1, len(results))
    by_B = Counter(int(r["B"]) for r in results)
    summary = {
        "config": cfg,
        "n_instances": len(results),
        "mlp_match": args.mlp_match,
        "W_base_field": args.W_base_field,
        "D_base_field": args.D_base_field,
        "scale": {
            "S_op": args.S_op,
            "S_k": args.S_k,
            "S_min": args.S_min,
            "S_max": args.S_max,
            "L_op": args.L_op,
            "L_k": args.L_k,
            "L_min": args.L_min,
            "L_max": args.L_max,
        },
        "max_B": args.max_B,
        "hist": {"B": dict(sorted(by_B.items()))},
        "means": {
            "row_acc": float(sum(r["row_acc"] for r in results) / n),
            "em_rate": float(sum(r["em"] for r in results) / n),
            "train_steps": float(sum(r["train_steps"] for r in results) / n),
            "mlp_params": float(sum(r["mlp_params"] for r in results) / n),
            "reference_sbc_soft_params": float(sum(r["reference_sbc_soft_params"] for r in results) / n),
            "reference_sbc_total_params": float(sum(r["reference_sbc_total_params"] for r in results) / n),
        },
        "results_jsonl": str(results_path),
    }
    diag_mean_keys = [
        "mlp_bnr_exact_L1",
        "mlp_bnr_eps_L1",
        "mlp_bnr_exact_avg",
        "mlp_bnr_eps_avg",
        "mlp_unique_norm_avg",
    ]
    for key in diag_mean_keys:
        vals = [float(r[key]) for r in results if isinstance(r.get(key), (int, float))]
        if vals:
            summary["means"][key] = float(sum(vals) / len(vals))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ok] wrote {out_dir / 'summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--mlp_match", type=str, default="neuron",
                    choices=["neuron", "soft", "total", "param_soft", "param_total"])
    ap.add_argument("--W_base_field", type=str, default="W_base")
    ap.add_argument("--D_base_field", type=str, default="D_base")
    ap.add_argument("--S_op", type=str, default="none", choices=["none", "identity", "add", "mul"])
    ap.add_argument("--S_k", type=int, default=0)
    ap.add_argument("--S_min", type=int, default=1)
    ap.add_argument("--S_max", type=int, default=128)
    ap.add_argument("--L_op", type=str, default="none", choices=["none", "identity", "add", "mul"])
    ap.add_argument("--L_k", type=int, default=0)
    ap.add_argument("--L_min", type=int, default=2)
    ap.add_argument("--L_max", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=0,
                    help="0 means full-batch training; positive uses random mini-batches.")
    ap.add_argument("--eval_batch_size", type=int, default=0,
                    help="0 means full truth-table eval in one pass; positive chunks eval.")
    ap.add_argument("--max_B", type=int, default=None)
    ap.add_argument("--diagnostics", action="store_true",
                    help="Record MLP activation BNR diagnostics after training.")
    ap.add_argument("--diagnostics_max_B", type=int, default=12,
                    help="Skip activation diagnostics above this B to avoid huge truth-table passes.")
    ap.add_argument("--bnr_decimals", type=int, default=6)
    ap.add_argument("--bnr_eps", type=float, default=1.0e-3)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["dataset"] = args.dataset
    run(cfg, Path(args.out_dir), args)


if __name__ == "__main__":
    main()
