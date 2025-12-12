# src/ubcircuit/baselines.py
# Baseline models (MLP / Transformer) matched by parameter count
# to the UBC circuits (DepthStack) on Boolean truth-table tasks.

from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .modules import DepthStack
from . import tasks as T
from .utils import seed_all, safe_bce
from .train import (
    per_instance_metrics,
    load_config,
    _device,
    normalize_expr,   # for nicer logging
)


# ----------------------------------------------------------------------
#  Helpers: parameter counting for UBC circuits
# ----------------------------------------------------------------------

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ubc_param_counts(
    B: int,
    S: int,
    L_used: int,
    gate_set: str,
    pair_cfg: Dict[str, Any],
    tau0: float,
) -> Tuple[int, int]:
    """
    Build a DepthStack just to measure its parameter counts.

    Returns:
        n_soft  : # of trainable parameters (softmax weights etc.)
        n_total : n_soft + (#frozen primitive gates), where we treat
                  each BooleanUnit as contributing K fixed primitive
                  gates (K=6 or 16).
    """
    # For parameter counting, we don't care about MI priors; route="learned"
    pair_cfg = dict(pair_cfg or {})
    pair_cfg["route"] = "learned"

    model = DepthStack(
        B=B,
        L=L_used,
        S=S,
        tau=tau0,
        pair=pair_cfg,
        gate_set=gate_set,
    )
    n_soft = count_trainable_params(model)

    # Fixed primitives: each unit has K basis gates
    K = 16 if gate_set == "16" else 6
    num_units = L_used * S
    n_fixed = num_units * K

    n_total = n_soft + n_fixed
    return n_soft, n_total


# ----------------------------------------------------------------------
#  Baseline 1: Truth-table MLP
# ----------------------------------------------------------------------

class TruthTableMLP(nn.Module):
    """
    Simple MLP baseline: f: R^B -> [0,1] with L_hidden hidden layers.

    We use ReLU between layers and sigmoid at the end.
    """

    def __init__(self, in_bits: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []

        last_dim = in_bits
        for _ in range(depth):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, B)
        logits = self.net(x)  # (N,1)
        return torch.sigmoid(logits)


def build_mlp_matching_params(
    B: int,
    L_used: int,
    target_params: int,
    min_hidden: int = 1,
    max_hidden: int = 1024,
) -> Tuple[TruthTableMLP, int]:
    """
    For given input dimension B and depth L_used, find a hidden_dim such that
      #params(MLP) <= target_params
    and as close as possible from below. If even hidden_dim=1 exceeds target,
    we fall back to hidden_dim=1.

    Returns (model, n_params_actual).
    """
    best_model: Optional[TruthTableMLP] = None
    best_n_params = -1

    for h in range(min_hidden, max_hidden + 1):
        model = TruthTableMLP(in_bits=B, hidden_dim=h, depth=L_used)
        n_params = count_trainable_params(model)

        if n_params <= target_params:
            if n_params > best_n_params:
                best_n_params = n_params
                best_model = model
        else:
            # once we pass target and we already have a valid candidate, break
            if best_model is not None:
                break
            else:
                # no candidate yet, but even h=1 is too big: we'll still use h=1
                best_model = model
                best_n_params = n_params
                break

    assert best_model is not None
    return best_model, best_n_params


# ----------------------------------------------------------------------
#  Baseline 2: Tiny Transformer over bits
# ----------------------------------------------------------------------

class TinySelfAttnBlock(nn.Module):
    """
    Single transformer-style self-attention block:
      - 1-head MultiheadAttention (batch_first=True)
      - 2-layer FFN with ReLU
      - Pre-LN residual structure
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h)
        x = x + attn_out
        h2 = self.norm2(x)
        ff_out = self.ff(h2)
        x = x + ff_out
        return x


class TruthTableTransformer(nn.Module):
    """
    Transformer baseline over bits.
      - Input: (N,B) bits
      - Treat as sequence of length B, embed each bit with shared linear layer
      - Add learned positional embeddings
      - Stack L blocks of TinySelfAttnBlock
      - Mean-pool over positions, then linear -> sigmoid
    """

    def __init__(self, B: int, d_model: int, depth: int, d_ff_scale: int = 4):
        super().__init__()
        self.B = int(B)
        self.d_model = int(d_model)

        self.val_emb = nn.Linear(1, d_model)             # shared over positions
        self.pos_emb = nn.Parameter(torch.randn(B, d_model) * 0.02)

        d_ff = d_ff_scale * d_model
        self.blocks = nn.ModuleList([
            TinySelfAttnBlock(d_model=d_model, d_ff=d_ff)
            for _ in range(depth)
        ])
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,B) in [0,1]
        N, B = x.shape
        assert B == self.B, f"Expected B={self.B}, got {B}"

        # value embedding: apply linear to each bit
        x_flat = x.view(N * B, 1)
        v = self.val_emb(x_flat).view(N, B, self.d_model)  # (N,B,d_model)

        # add positional embeddings
        pe = self.pos_emb.unsqueeze(0)                     # (1,B,d_model)
        h = v + pe

        # transformer blocks
        for blk in self.blocks:
            h = blk(h)                                     # (N,B,d_model)

        # mean-pool over bits
        h_mean = h.mean(dim=1)                             # (N,d_model)
        logits = self.out(h_mean)                          # (N,1)
        return torch.sigmoid(logits)


def build_transformer_matching_params(
    B: int,
    L_used: int,
    target_params: int,
    min_dim: int = 4,
    max_dim: int = 512,
) -> Tuple[TruthTableTransformer, int]:
    """
    Search over d_model to match parameter budget for Transformer baseline.
    We use depth=L_used (# blocks) and d_ff = 4*d_model.

    Returns (model, n_params_actual).
    """
    best_model: Optional[TruthTableTransformer] = None
    best_n_params = -1

    for d in range(min_dim, max_dim + 1, 2):  # step by 2 to keep search small
        model = TruthTableTransformer(B=B, d_model=d, depth=L_used, d_ff_scale=4)
        n_params = count_trainable_params(model)

        if n_params <= target_params:
            if n_params > best_n_params:
                best_n_params = n_params
                best_model = model
        else:
            if best_model is not None:
                break
            else:
                best_model = model
                best_n_params = n_params
                break

    assert best_model is not None
    return best_model, best_n_params


# ----------------------------------------------------------------------
#  Training loop for a single instance
# ----------------------------------------------------------------------

def _build_baseline_for_instance(
    baseline: str,
    match_mode: str,
    inst: Dict[str, Any],
    cfg: Dict[str, Any],
    L_used: int,
) -> Tuple[nn.Module, int, int, int]:
    """
    Build the param-matched baseline model for a given instance.

    Args:
        baseline : "mlp" or "transformer"
        match_mode: "soft" or "total"
        inst     : {"B": int, "S": int, ...}
        cfg      : full config dict
        L_used   : depth used for UBC circuit & baseline

    Returns:
        model          : nn.Module baseline
        n_params_model : # trainable params of baseline
        n_soft_ubc     : UBC trainable params for this instance
        n_total_ubc    : UBC total params (soft + fixed primitives)
    """
    B = int(inst["B"])
    S = int(inst["S"])
    gate_set = str(cfg.get("gate_set", "6"))
    pair_cfg = dict(cfg.get("pair", {}))
    T0 = float(cfg["anneal"]["T0"])

    n_soft_ubc, n_total_ubc = ubc_param_counts(
        B=B,
        S=S,
        L_used=L_used,
        gate_set=gate_set,
        pair_cfg=pair_cfg,
        tau0=T0,
    )

    if match_mode == "soft":
        target = n_soft_ubc
    elif match_mode == "total":
        target = n_total_ubc
    else:
        raise ValueError("match_mode must be 'soft' or 'total'")

    if baseline == "mlp":
        model, n_params_model = build_mlp_matching_params(
            B=B,
            L_used=L_used,
            target_params=target,
        )
    elif baseline == "transformer":
        model, n_params_model = build_transformer_matching_params(
            B=B,
            L_used=L_used,
            target_params=target,
        )
    else:
        raise ValueError("baseline must be 'mlp' or 'transformer'")

    return model, n_params_model, n_soft_ubc, n_total_ubc


def train_baseline_single_instance(
    device: torch.device,
    cfg: Dict[str, Any],
    inst: Dict[str, Any],
    L_override: Optional[int],
    baseline: str,
    match_mode: str,
    out_dir: Path,
    idx: int,
) -> Dict[str, Any]:
    """
    Train a param-matched baseline on a single Boolean formula instance.

    Saves model checkpoint to out_dir and returns a result dict for summary.
    """
    B = int(inst["B"])
    S = int(inst["S"])
    formula = str(inst["formula"])

    X, y_true = T.truth_table_from_formula(B, formula)
    X = X.to(device)
    y_true = y_true.to(device)

    # ---- resolve depth L_used as in train_single_instance ----
    if L_override is not None:
        L_used = int(L_override)
    elif bool(cfg.get("use_row_L", True)) and ("L" in inst):
        L_used = int(inst["L"])
    else:
        L_used = int(cfg["L"])

    # ---- build param-matched baseline ----
    model, n_params_model, n_soft_ubc, n_total_ubc = _build_baseline_for_instance(
        baseline=baseline,
        match_mode=match_mode,
        inst=inst,
        cfg=cfg,
        L_used=L_used,
    )

    model = model.to(device)

    # ---- optimizer (match train.py choices) ----
    opt_name = str(cfg["optimizer"]).lower()
    if opt_name == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-8)
    else:
        opt = optim.Adam(model.parameters(), lr=cfg["lr"])

    steps = int(cfg["steps"])

    # ---- early stopping (reuse config) ----
    es_cfg = cfg.get("early_stop", {})
    use_es = bool(es_cfg.get("use", False))
    min_steps = int(es_cfg.get("min_steps", 0))
    check_every = int(es_cfg.get("check_every", 10))
    patience_checks = int(es_cfg.get("patience_checks", 3))
    metric = str(es_cfg.get("metric", "em")).lower()  # "em" or "row_acc"
    target = float(es_cfg.get("target", 1.0))
    ok_streak = 0

    last_y_pred: Optional[torch.Tensor] = None

    for step in range(steps):
        opt.zero_grad()
        y_pred = model(X)
        last_y_pred = y_pred

        assert torch.all((y_true >= 0) & (y_true <= 1)), "Targets must be in [0,1]"
        loss = safe_bce(y_pred, y_true)
        loss.backward()
        opt.step()

        if use_es and (step + 1) >= min_steps and ((step + 1) % check_every == 0):
            with torch.no_grad():
                row_acc, em = per_instance_metrics(y_true, y_pred)
                cur_val = float(em) if metric == "em" else float(row_acc)
                if cur_val >= target:
                    ok_streak += 1
                else:
                    ok_streak = 0
                if ok_streak >= patience_checks:
                    print(f"  [baseline early-stop] step={step+1}, metric={metric}={cur_val:.3f}")
                    break

    with torch.no_grad():
        if last_y_pred is None:
            y_pred = model(X)
        else:
            y_pred = last_y_pred

        row_acc, em = per_instance_metrics(y_true, y_pred)

        label_expr = normalize_expr(formula)

    # ---- save model for post-hoc analysis ----
    ckpt_path = out_dir / f"inst{idx:04d}_B{B}_S{S}_L{L_used}_{baseline}_{match_mode}.pt"
    ckpt = {
        "baseline": baseline,
        "match_mode": match_mode,
        "instance": inst,
        "config": cfg,
        "state_dict": model.state_dict(),
        "n_params_model": n_params_model,
        "n_soft_ubc": n_soft_ubc,
        "n_total_ubc": n_total_ubc,
    }
    torch.save(ckpt, ckpt_path)
    print(f"   [saved] {ckpt_path}")

    return {
        "B": B,
        "S": S,
        "L_used": L_used,
        "formula": label_expr,
        "row_acc": row_acc,
        "em": em,
        "n_params_model": n_params_model,
        "n_soft_ubc": n_soft_ubc,
        "n_total_ubc": n_total_ubc,
    }


# ----------------------------------------------------------------------
#  Dataset runner (analogous to run_dataset)
# ----------------------------------------------------------------------

def run_baseline_dataset(
    cfg: Dict[str, Any],
    out_dir: Path,
    baseline: str,
    match_mode: str,
) -> Dict[str, Any]:
    """
    Run a chosen baseline over all instances in cfg["dataset"].

    Writes summary.json under out_dir and individual model checkpoints per instance.
    """
    device = _device(cfg)
    seed_all(cfg["seed"])

    if not cfg.get("dataset"):
        raise ValueError("Baseline runner requires cfg['dataset'] to be set.")

    insts = T.load_instances_jsonl(cfg["dataset"])

    # L strategy (row / max / fixed) as in train.run_dataset
    global_L: Optional[int] = None
    l_strategy = "row" if cfg.get("use_row_L", True) else (
        "max" if cfg.get("use_max_L", False) else "fixed"
    )
    if l_strategy == "max":
        global_L = max(int(inst.get("L", cfg["L"])) for inst in insts)
        print(f"[baseline info] Using max L across dataset: L_max = {global_L}")

    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, inst in enumerate(insts):
        if l_strategy == "row" and ("L" in inst):
            L_override = int(inst["L"])
        elif l_strategy == "max":
            L_override = int(global_L)
        else:
            L_override = None

        print(f"[{idx+1}/{len(insts)}] Baseline={baseline}, match={match_mode}, "
              f"B={inst['B']} S={inst['S']}")

        res = train_baseline_single_instance(
            device=device,
            cfg=cfg,
            inst=inst,
            L_override=L_override,
            baseline=baseline,
            match_mode=match_mode,
            out_dir=out_dir,
            idx=idx,
        )
        results.append(res)
        print(f"   acc={res['row_acc']:.3f}  EM={res['em']} "
              f"(params model={res['n_params_model']}, "
              f"UBC soft={res['n_soft_ubc']}, total={res['n_total_ubc']})")
        print(f"   label: {res['formula']}")

    # Aggregates
    n = max(1, len(results))
    avg_row_acc = sum(r["row_acc"] for r in results) / n
    em_rate = sum(r["em"] for r in results) / n

    L_used_hist = Counter(r["L_used"] for r in results)
    S_hist = Counter(r["S"] for r in results)
    B_hist = Counter(r["B"] for r in results)

    # Param stats (simple averages)
    avg_params_model = sum(r["n_params_model"] for r in results) / n
    avg_soft_ubc = sum(r["n_soft_ubc"] for r in results) / n
    avg_total_ubc = sum(r["n_total_ubc"] for r in results) / n

    # Dataset-aware config snapshot
    cfg_eff = dict(cfg)
    if cfg_eff.get("dataset"):
        if l_strategy == "row":
            cfg_eff["L"] = "per-row"
        elif l_strategy == "max":
            cfg_eff["L"] = "global_max"
            cfg_eff["L_max"] = global_L
        cfg_eff["S"] = "per-row"
        cfg_eff.pop("task", None)

    summary = {
        "config": cfg_eff,
        "baseline": baseline,
        "match_mode": match_mode,
        "l_strategy": l_strategy,
        "avg_row_acc": avg_row_acc,
        "em_rate": em_rate,
        "hist": {
            "L_used": dict(L_used_hist),
            "S": dict(S_hist),
            "B": dict(B_hist),
        },
        "params": {
            "avg_model": avg_params_model,
            "avg_ubc_soft": avg_soft_ubc,
            "avg_ubc_total": avg_total_ubc,
        },
        "results": results,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[baseline] Saved summary to: {out_dir / 'summary.json'}")
    return summary


# ----------------------------------------------------------------------
#  Single-task fallback (if you want B=2 toy baselines)
# ----------------------------------------------------------------------

def run_baseline_single(
    cfg: Dict[str, Any],
    out_dir: Path,
    baseline: str,
    match_mode: str,
) -> Dict[str, Any]:
    """
    Optional: run a baseline on the single 2-bit task specified by cfg["task"].
    Uses cfg["B"]=2, cfg["L"], cfg["S"] to build matching UBC counts.
    """
    device = _device(cfg)
    seed_all(cfg["seed"])

    B = 2
    S = int(cfg["S"])
    L_used = int(cfg["L"])
    gate_set = str(cfg.get("gate_set", "6"))
    pair_cfg = dict(cfg.get("pair", {}))
    T0 = float(cfg["anneal"]["T0"])

    X, y_true = T.make_truth_table(cfg["task"])
    X = X.to(device)
    y_true = y_true.to(device)

    n_soft_ubc, n_total_ubc = ubc_param_counts(
        B=B,
        S=S,
        L_used=L_used,
        gate_set=gate_set,
        pair_cfg=pair_cfg,
        tau0=T0,
    )

    if match_mode == "soft":
        target = n_soft_ubc
    elif match_mode == "total":
        target = n_total_ubc
    else:
        raise ValueError("match_mode must be 'soft' or 'total'")

    if baseline == "mlp":
        model, n_params_model = build_mlp_matching_params(
            B=B,
            L_used=L_used,
            target_params=target,
        )
    elif baseline == "transformer":
        model, n_params_model = build_transformer_matching_params(
            B=B,
            L_used=L_used,
            target_params=target,
        )
    else:
        raise ValueError("baseline must be 'mlp' or 'transformer'")

    model = model.to(device)

    opt_name = str(cfg["optimizer"]).lower()
    if opt_name == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-8)
    else:
        opt = optim.Adam(model.parameters(), lr=cfg["lr"])

    steps = int(cfg["steps"])
    for _ in range(steps):
        opt.zero_grad()
        y_pred = model(X)
        loss = safe_bce(y_pred, y_true)
        loss.backward()
        opt.step()

    with torch.no_grad():
        y_pred = model(X)
        row_acc, em = per_instance_metrics(y_true, y_pred)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": cfg,
        "baseline": baseline,
        "match_mode": match_mode,
        "row_acc": row_acc,
        "em": em,
        "n_params_model": n_params_model,
        "n_soft_ubc": n_soft_ubc,
        "n_total_ubc": n_total_ubc,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[baseline single] Saved summary to: {out_dir / 'summary.json'}")
    return summary


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="experiments/baselines/run")
    ap.add_argument("--dataset", type=str, default=None,
                    help="JSONL with B,S,L,formula per line (overrides config.dataset)")
    ap.add_argument("--use_row_L", action="store_true",
                    help="Use per-row L when present (default if not overridden in config)")
    ap.add_argument("--use_max_L", action="store_true",
                    help="Use max L across dataset for all rows (ignored if --use_row_L)")
    ap.add_argument("--no_row_L", action="store_true",
                    help="Disable per-row L even if present")
    ap.add_argument("--baseline", type=str, default="mlp",
                    choices=["mlp", "transformer"],
                    help="Which baseline architecture to train")
    ap.add_argument("--match_mode", type=str, default="soft",
                    choices=["soft", "total"],
                    help="Parameter matching regime: 'soft' or 'total'")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset is not None:
        cfg["dataset"] = args.dataset

    # Flags override config defaults
    if args.no_row_L:
        cfg["use_row_L"] = False
    if args.use_row_L:
        cfg["use_row_L"] = True
    if args.use_max_L:
        cfg["use_max_L"] = True

    out_dir = Path(args.out_dir)

    if cfg.get("dataset"):
        run_baseline_dataset(cfg, out_dir, baseline=args.baseline, match_mode=args.match_mode)
    else:
        run_baseline_single(cfg, out_dir, baseline=args.baseline, match_mode=args.match_mode)


if __name__ == "__main__":
    main()
