#!/usr/bin/env python3
"""
Train UBC + MLP together per-instance, with:
  - MLP match modes: param_soft | param_total | neuron
  - BNR diagnostics on trained MLP hidden units
  - Primitive-gate interpretation for MLP first hidden layer
  - UBC hard-decode gate usage + expression, and compare vs label

Usage example:
  python -m ubcircuit.joint_mlp_ubc_bnr \
    --config /path/to/cfg.yaml \
    --dataset /path/to/data.jsonl \
    --out_dir /path/to/out \
    --mlp_match neuron \
    --S_op add --S_k 10 --S_min 2 --S_max 128 \
    --save_ckpts
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .modules import DepthStack
from . import tasks as T
from .utils import seed_all, safe_bce, make_async_taus, regularizers_bundle
from .train import (
    load_config,
    _device,
    per_instance_metrics,
    normalize_expr,
    compose_readout_full,
    extract_gate_usage_from_dbg,
    resolve_pair_cfg,
    compute_B_effective,
)

# -------------------------
# Param counting utilities
# -------------------------

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ubc_param_counts(
    B: int,
    S: int,
    L_used: int,
    gate_set: str,
    pair_cfg: Dict[str, Any],
    tau0: float,
    use_lifting: bool,
    lift_factor: float,
) -> Tuple[int, int]:
    """
    Build a DepthStack just to measure param counts.

    Returns:
      n_soft  : trainable params
      n_total : n_soft + (#fixed primitives), where each BooleanUnit contributes K primitives.
    """
    # For counting: ignore MI priors, route learned
    pair_cfg = dict(pair_cfg or {})
    pair_cfg["route"] = "learned"

    model = DepthStack(
        B=B,
        L=L_used,
        S=S,
        tau=tau0,
        pair=pair_cfg,
        gate_set=gate_set,
        use_lifting=use_lifting,
        lift_factor=lift_factor,
    )
    n_soft = count_trainable_params(model)

    K = 16 if gate_set == "16" else 6
    num_units = L_used * S
    n_fixed = num_units * K
    return n_soft, n_soft + n_fixed


# -------------------------
# MLP (with activations)
# -------------------------

class TruthTableMLPActs(nn.Module):
    """
    MLP that returns hidden activations after each ReLU for BNR analysis.
    depth = # hidden layers.
    """
    def __init__(self, in_bits: int, hidden_dim: int, depth: int):
        super().__init__()
        self.in_bits = int(in_bits)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        self.linears = nn.ModuleList()
        last = self.in_bits
        for _ in range(self.depth):
            self.linears.append(nn.Linear(last, self.hidden_dim))
            last = self.hidden_dim
        self.out = nn.Linear(last, 1)

    def forward(self, x: torch.Tensor, return_acts: bool = False):
        acts: List[torch.Tensor] = []
        h = x
        for lin in self.linears:
            h = torch.relu(lin(h))
            acts.append(h)
        y = torch.sigmoid(self.out(h))
        if return_acts:
            return y, acts
        return y


def build_mlp_param_matched(
    B: int,
    depth: int,
    target_params: int,
    min_hidden: int = 1,
    max_hidden: int = 2048,
) -> Tuple[TruthTableMLPActs, int]:
    """
    Find largest hidden_dim such that params <= target_params (from below).
    """
    best_model: Optional[TruthTableMLPActs] = None
    best_n = -1

    for h in range(min_hidden, max_hidden + 1):
        m = TruthTableMLPActs(in_bits=B, hidden_dim=h, depth=depth)
        n = count_trainable_params(m)
        if n <= target_params:
            if n > best_n:
                best_n = n
                best_model = m
        else:
            if best_model is not None:
                break
            # even h=1 too big -> still take h=1
            best_model = m
            best_n = n
            break

    assert best_model is not None
    return best_model, best_n


# -------------------------
# BNR diagnostics for MLP
# -------------------------

def _round_tensor(x: torch.Tensor, decimals: int) -> torch.Tensor:
    if decimals <= 0:
        return torch.round(x)
    scale = 10.0 ** decimals
    return torch.round(x * scale) / scale

def exact_bnr_fraction(layer_act: torch.Tensor, decimals: int = 6) -> float:
    """
    Exact BNR-style: per unit, count distinct values across truth table.
    We round to avoid "float noise" making everything unique.
    """
    A = _round_tensor(layer_act.detach().cpu(), decimals=decimals)  # (N,H)
    N, H = A.shape
    ok = 0
    for j in range(H):
        uj = A[:, j]
        # torch.unique on CPU is fine for N <= 2^14
        if torch.unique(uj).numel() <= 2:
            ok += 1
    return ok / max(1, H)

def eps_bnr_fraction(layer_act: torch.Tensor, eps: float = 1e-3) -> float:
    """
    Simple ε-BNR proxy: does each unit's values cluster near 2 levels?
    We use robust two-centers from medians, and check max deviation to nearest center <= eps.
    """
    A = layer_act.detach().cpu()  # (N,H)
    N, H = A.shape
    ok = 0
    for j in range(H):
        v = A[:, j].float()
        med = v.median().values
        lo = v[v <= med]
        hi = v[v >  med]
        if lo.numel() == 0 or hi.numel() == 0:
            # essentially constant -> counts as 1-bit (<=2 levels)
            ok += 1
            continue
        c0 = lo.median().values
        c1 = hi.median().values
        d = torch.minimum((v - c0).abs(), (v - c1).abs()).max().item()
        if d <= eps:
            ok += 1
    return ok / max(1, H)


# -------------------------
# Primitive-gate interpretation for MLP layer-1
# -------------------------

_PRIM16_NAMES = [
    "FALSE","AND","A&~B","A","~A&B","B","XOR","OR",
    "NOR","XNOR","~B","A|~B","~A","~A|B","NAND","TRUE"
]

def prim16_outputs(A: torch.Tensor, B: torch.Tensor) -> List[torch.Tensor]:
    """
    A,B are boolean tensors (N,)
    Return list of 16 boolean outputs, in the same order as your Table (g1..g16).
    """
    F = torch.zeros_like(A, dtype=torch.bool)
    T = torch.ones_like(A, dtype=torch.bool)
    notA = ~A
    notB = ~B
    AND = A & B
    OR  = A | B
    XOR = A ^ B
    XNOR = ~XOR
    NAND = ~AND
    NOR  = ~OR
    out = [
        F,            # FALSE
        AND,          # AND
        A & notB,     # A&~B
        A,            # A
        notA & B,     # ~A&B
        B,            # B
        XOR,          # XOR
        OR,           # OR
        NOR,          # NOR
        XNOR,         # XNOR
        notB,         # ~B
        A | notB,     # A|~B
        notA,         # ~A
        notA | B,     # ~A|B
        NAND,         # NAND
        T,            # TRUE
    ]
    return out

def interpret_mlp_first_layer_primitives(
    X: torch.Tensor,
    act1: torch.Tensor,
    eps_for_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Binarize each hidden unit using threshold at u(x=0...0),
    then search best exact match among:
      - literals +/- (single-bit)
      - all 16 2-input gates over all ordered pairs (i,j)

    Returns stats + a histogram of gate types.
    """
    Xb = (X.detach().cpu() >= 0.5)  # (N,B) bool
    A1 = act1.detach().cpu()        # (N,H)
    N, B = Xb.shape
    _, H = A1.shape

    # find the all-zero input row (robust)
    zero_row = (Xb.sum(dim=1) == 0).nonzero(as_tuple=False)
    zero_idx = int(zero_row[0].item()) if zero_row.numel() > 0 else 0

    gate_hist = Counter()
    exact_hits = 0
    best_records = []

    # Precompute literal templates
    literals = []
    lit_names = []
    for i in range(B):
        literals.append(Xb[:, i])
        lit_names.append(f"a{i}")
        literals.append(~Xb[:, i])
        lit_names.append(f"~a{i}")

    for j in range(H):
        u = A1[:, j]
        t = float(u[zero_idx].item())
        yb = (u >= (t + eps_for_threshold))  # bool vector

        best = {"kind": None, "name": None, "i": None, "j": None, "acc": -1.0}

        # try literals
        for k, tmpl in enumerate(literals):
            acc = (yb == tmpl).float().mean().item()
            if acc > best["acc"]:
                best = {"kind": "lit", "name": lit_names[k], "i": None, "j": None, "acc": acc}

        # try all ordered pairs for 16 gates
        for i in range(B):
            Ai = Xb[:, i]
            for k in range(B):
                if k == i:
                    continue
                Bk = Xb[:, k]
                outs = prim16_outputs(Ai, Bk)
                # compute accuracies cheaply
                for gi, gout in enumerate(outs):
                    acc = (yb == gout).float().mean().item()
                    if acc > best["acc"]:
                        best = {"kind": "gate16", "name": _PRIM16_NAMES[gi], "i": i, "j": k, "acc": acc}

        best_records.append(best)
        if best["acc"] == 1.0:
            exact_hits += 1
            gate_hist[best["name"]] += 1

    return {
        "H": H,
        "exact_primitive_hits": exact_hits,
        "exact_primitive_hit_rate": exact_hits / max(1, H),
        "gate_hist": dict(gate_hist),
        "best_per_unit": best_records[: min(50, len(best_records))],  # keep small in JSON
    }


# -------------------------
# Train UBC (returns model + dbg)
# -------------------------

@dataclass
class UBCTrainOut:
    model: DepthStack
    y_pred: torch.Tensor
    dbg: List[tuple]
    taus: List[float]
    row_acc: float
    em: int
    pred_expr: str
    gate_usage: Dict[str, Any]

def train_ubc_instance(
    device: torch.device,
    cfg: Dict[str, Any],
    inst: Dict[str, Any],
    *,
    L_used: int,
    S_used: int,
) -> UBCTrainOut:
    B = int(inst["B"])
    formula = str(inst["formula"])
    X, y_true = T.truth_table_from_formula(B, formula)
    X = X.to(device)
    y_true = y_true.to(device)

    gate_set = str(cfg.get("gate_set", "6"))
    anneal_cfg = cfg["anneal"]
    T0 = float(anneal_cfg["T0"])

    # lifting
    lifting_cfg = cfg.get("lifting", {})
    use_lifting = bool(lifting_cfg.get("use", True))
    lift_factor = float(lifting_cfg.get("factor", 2.0))
    B_eff = compute_B_effective(B, use_lifting, lift_factor)

    # Pair config (MI routing etc)
    pair_cfg = dict(cfg.get("pair", {}) or {})
    pair_cfg = resolve_pair_cfg(pair_cfg, B=B, B_eff=B_eff, S=S_used, X=X, y_true=y_true)

    model = DepthStack(
        B=B,
        L=L_used,
        S=S_used,
        tau=T0,
        pair=pair_cfg,
        gate_set=gate_set,
        use_lifting=use_lifting,
        lift_factor=lift_factor,
    ).to(device)

    opt_name = str(cfg["optimizer"]).lower()
    opt = (
        optim.RMSprop(model.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-8)
        if opt_name == "rmsprop"
        else optim.Adam(model.parameters(), lr=cfg["lr"])
    )

    steps = int(cfg["steps"])
    regs_cfg = cfg["regs"]

    # early stop
    es_cfg = cfg.get("early_stop", {})
    use_es = bool(es_cfg.get("use", False))
    min_steps = int(es_cfg.get("min_steps", 0))
    check_every = int(es_cfg.get("check_every", 10))
    patience_checks = int(es_cfg.get("patience_checks", 3))
    metric = str(es_cfg.get("metric", "em")).lower()
    target = float(es_cfg.get("target", 1.0))
    ok_streak = 0

    last = None
    for step in range(steps):
        taus = make_async_taus(
            L=len(model.layers), step=step, total=steps,
            T0=anneal_cfg["T0"], Tmin=anneal_cfg["Tmin"],
            direction=anneal_cfg["direction"],
            schedule=anneal_cfg.get("schedule", "linear"),
            phase_scale=anneal_cfg.get("phase_scale", 0.4),
            start_frac=anneal_cfg.get("start_frac", 0.0),
        )

        # sigma16 bandwidth anneal if present
        if hasattr(model, "set_layer_taus_and_bandwidths") and gate_set == "16":
            s_cfg = cfg.get("sigma16", {})
            s_start = float(s_cfg.get("s_start", 0.25))
            s_end   = float(s_cfg.get("s_end",   0.10))
            model.set_layer_taus_and_bandwidths(taus, s_start=s_start, s_end=s_end)
        else:
            model.set_layer_taus(taus)

        opt.zero_grad()
        y_pred, dbg = model(X)
        last = (y_pred, dbg, taus)

        loss = safe_bce(y_pred, y_true)
        dbg_slim = [(d[0], d[1], d[2]) for d in dbg]
        reg = regularizers_bundle(
            dbg=dbg_slim, taus=taus,
            lam_entropy=regs_cfg["lam_entropy"],
            lam_div_units=regs_cfg["lam_div_units"],
            lam_div_rows=regs_cfg["lam_div_rows"],
            lam_const16=regs_cfg.get("lam_const16", 1.0e-3),
        )
        (loss + reg).backward()
        opt.step()

        if use_es and (step + 1) >= min_steps and ((step + 1) % check_every == 0):
            with torch.no_grad():
                row_acc, em = per_instance_metrics(y_true, y_pred)
                cur = float(em) if metric == "em" else float(row_acc)
                ok_streak = (ok_streak + 1) if (cur >= target) else 0
                if ok_streak >= patience_checks:
                    break

    assert last is not None
    y_pred, dbg, final_taus = last
    with torch.no_grad():
        row_acc, em = per_instance_metrics(y_true, y_pred)

        # compose hard readout expression & gate usage
        if gate_set == "16":
            from .boolean_prims16 import PRIMS16 as PRIMS_LIST
        else:
            from .boolean_prims import PRIMS as PRIMS_LIST

        pred_expr_raw = compose_readout_full(B, dbg, final_taus, PRIMS_LIST)
        pred_expr = normalize_expr(pred_expr_raw)
        gate_usage = extract_gate_usage_from_dbg(dbg, final_taus, PRIMS_LIST)

    return UBCTrainOut(
        model=model,
        y_pred=y_pred.detach(),
        dbg=dbg,
        taus=final_taus,
        row_acc=row_acc,
        em=em,
        pred_expr=pred_expr,
        gate_usage=gate_usage,
    )


# -------------------------
# Train MLP (matched)
# -------------------------

@dataclass
class MLPTrainOut:
    model: TruthTableMLPActs
    y_pred: torch.Tensor
    acts: List[torch.Tensor]
    row_acc: float
    em: int
    n_params: int
    bnr_exact: List[float]
    bnr_eps: List[float]
    prim_interp: Dict[str, Any]

def train_mlp_instance(
    device: torch.device,
    cfg: Dict[str, Any],
    inst: Dict[str, Any],
    *,
    L_used: int,
    S_used: int,
    match_mode: str,
    ubc_soft_params: int,
    ubc_total_params: int,
) -> MLPTrainOut:
    B = int(inst["B"])
    formula = str(inst["formula"])
    X, y_true = T.truth_table_from_formula(B, formula)
    X = X.to(device)
    y_true = y_true.to(device)

    # Build MLP
    if match_mode == "neuron":
        model = TruthTableMLPActs(in_bits=B, hidden_dim=S_used, depth=L_used)
        n_params = count_trainable_params(model)
    elif match_mode == "param_soft":
        model, n_params = build_mlp_param_matched(B=B, depth=L_used, target_params=ubc_soft_params)
    elif match_mode == "param_total":
        model, n_params = build_mlp_param_matched(B=B, depth=L_used, target_params=ubc_total_params)
    else:
        raise ValueError("mlp_match must be one of: neuron|param_soft|param_total")

    model = model.to(device)

    opt_name = str(cfg["optimizer"]).lower()
    opt = (
        optim.RMSprop(model.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-8)
        if opt_name == "rmsprop"
        else optim.Adam(model.parameters(), lr=cfg["lr"])
    )

    steps = int(cfg["steps"])

    # Early stop (reuse cfg)
    es_cfg = cfg.get("early_stop", {})
    use_es = bool(es_cfg.get("use", False))
    min_steps = int(es_cfg.get("min_steps", 0))
    check_every = int(es_cfg.get("check_every", 10))
    patience_checks = int(es_cfg.get("patience_checks", 3))
    metric = str(es_cfg.get("metric", "em")).lower()
    target = float(es_cfg.get("target", 1.0))
    ok_streak = 0

    last = None
    for step in range(steps):
        opt.zero_grad()
        y_pred = model(X)
        loss = safe_bce(y_pred, y_true)
        loss.backward()
        opt.step()
        last = y_pred

        if use_es and (step + 1) >= min_steps and ((step + 1) % check_every == 0):
            with torch.no_grad():
                row_acc, em = per_instance_metrics(y_true, y_pred)
                cur = float(em) if metric == "em" else float(row_acc)
                ok_streak = (ok_streak + 1) if (cur >= target) else 0
                if ok_streak >= patience_checks:
                    break

    with torch.no_grad():
        y_pred, acts = model(X, return_acts=True)
        row_acc, em = per_instance_metrics(y_true, y_pred)

        # BNR stats per layer
        bnr_exact = [exact_bnr_fraction(a, decimals=6) for a in acts]
        bnr_eps   = [eps_bnr_fraction(a, eps=1e-3) for a in acts]

        # primitive interpretation on layer 1
        prim_interp = interpret_mlp_first_layer_primitives(X, acts[0])

    return MLPTrainOut(
        model=model,
        y_pred=y_pred.detach(),
        acts=[a.detach() for a in acts],
        row_acc=row_acc,
        em=em,
        n_params=n_params,
        bnr_exact=bnr_exact,
        bnr_eps=bnr_eps,
        prim_interp=prim_interp,
    )


# -------------------------
# S/L scaling helper
# -------------------------

def apply_scale(val: int, op: str, k: int, vmin: int, vmax: int) -> int:
    if op == "none":
        out = val
    elif op == "add":
        out = val + int(k)
    elif op == "mul":
        out = val * int(k)
    else:
        raise ValueError("scale op must be one of: none|add|mul")
    return int(max(vmin, min(vmax, out)))


# -------------------------
# Main runner
# -------------------------

def run(cfg: Dict[str, Any], out_dir: Path, args) -> Dict[str, Any]:
    device = _device(cfg)
    seed_all(cfg["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "results.jsonl"

    insts = T.load_instances_jsonl(cfg["dataset"])

    # L strategy (row/max/fixed)
    global_L = None
    l_strategy = "row" if cfg.get("use_row_L", True) else ("max" if cfg.get("use_max_L", False) else "fixed")
    if l_strategy == "max":
        global_L = max(int(inst.get("L", cfg["L"])) for inst in insts)

    # Aggregates
    results = []
    agg = defaultdict(list)

    for idx, inst in enumerate(insts):
        B = int(inst["B"])
        baseS = int(inst["S"])
        baseL = int(inst.get("L", cfg["L"])) if (l_strategy == "row" and "L" in inst) else (
            int(global_L) if (l_strategy == "max") else int(cfg["L"])
        )

        # apply user scaling to S and L
        S_used = apply_scale(baseS, args.S_op, args.S_k, args.S_min, args.S_max)
        L_used = apply_scale(baseL, args.L_op, args.L_k, args.L_min, args.L_max)

        # compute UBC param targets for matching
        gate_set = str(cfg.get("gate_set", "6"))
        pair_cfg = dict(cfg.get("pair", {}))
        tau0 = float(cfg["anneal"]["T0"])
        lifting_cfg = cfg.get("lifting", {})
        use_lifting = bool(lifting_cfg.get("use", True))
        lift_factor = float(lifting_cfg.get("factor", 2.0))

        ubc_soft, ubc_total = ubc_param_counts(
            B=B, S=S_used, L_used=L_used, gate_set=gate_set, pair_cfg=pair_cfg, tau0=tau0,
            use_lifting=use_lifting, lift_factor=lift_factor
        )

        # ----- Train UBC -----
        ubc_out = train_ubc_instance(device, cfg, inst, L_used=L_used, S_used=S_used)

        # ----- Train MLP -----
        mlp_out = train_mlp_instance(
            device, cfg, inst,
            L_used=L_used, S_used=S_used,
            match_mode=args.mlp_match,
            ubc_soft_params=ubc_soft,
            ubc_total_params=ubc_total,
        )

        label_expr = normalize_expr(str(inst["formula"]))

        row = {
            "idx": idx,
            "B": B,
            "S_base": baseS,
            "S_used": S_used,
            "L_base": baseL,
            "L_used": L_used,
            "label_expr": label_expr,

            "ubc": {
                "row_acc": ubc_out.row_acc,
                "em": ubc_out.em,
                "pred_expr": ubc_out.pred_expr,
                "gate_usage": ubc_out.gate_usage,
                "n_params_soft": ubc_soft,
                "n_params_total": ubc_total,
            },
            "mlp": {
                "match_mode": args.mlp_match,
                "row_acc": mlp_out.row_acc,
                "em": mlp_out.em,
                "n_params": mlp_out.n_params,
                "bnr_exact_per_layer": mlp_out.bnr_exact,
                "bnr_eps_per_layer": mlp_out.bnr_eps,
                "prim_interp_first_layer": mlp_out.prim_interp,
            },
        }

        # save checkpoints optionally
        if args.save_ckpts:
            ckpt = {
                "instance": inst,
                "config": cfg,
                "S_used": S_used,
                "L_used": L_used,
                "ubc_state_dict": ubc_out.model.state_dict(),
                "mlp_state_dict": mlp_out.model.state_dict(),
                "mlp_match": args.mlp_match,
            }
            torch.save(ckpt, out_dir / f"ckpt_inst{idx:04d}_B{B}_S{S_used}_L{L_used}_{args.mlp_match}.pt")

        with out_jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")

        results.append(row)

        # aggregates
        agg["ubc_row_acc"].append(ubc_out.row_acc)
        agg["ubc_em"].append(ubc_out.em)
        agg["mlp_row_acc"].append(mlp_out.row_acc)
        agg["mlp_em"].append(mlp_out.em)
        agg["mlp_params"].append(mlp_out.n_params)
        agg["ubc_soft_params"].append(ubc_soft)
        agg["ubc_total_params"].append(ubc_total)
        agg["mlp_bnr_exact_L1"].append(mlp_out.bnr_exact[0] if mlp_out.bnr_exact else 0.0)
        agg["mlp_bnr_eps_L1"].append(mlp_out.bnr_eps[0] if mlp_out.bnr_eps else 0.0)
        agg["mlp_prim_hit_rate_L1"].append(float(mlp_out.prim_interp.get("exact_primitive_hit_rate", 0.0)))

        print(
            f"[{idx+1}/{len(insts)}] B={B} S={S_used} L={L_used} | "
            f"UBC EM={ubc_out.em} acc={ubc_out.row_acc:.3f} | "
            f"MLP({args.mlp_match}) EM={mlp_out.em} acc={mlp_out.row_acc:.3f} "
            f"BNR(L1)={agg['mlp_bnr_exact_L1'][-1]:.3f} primHit(L1)={agg['mlp_prim_hit_rate_L1'][-1]:.3f}"
        )

    summary = {
        "config": cfg,
        "mlp_match": args.mlp_match,
        "scale": {
            "S_op": args.S_op, "S_k": args.S_k, "S_min": args.S_min, "S_max": args.S_max,
            "L_op": args.L_op, "L_k": args.L_k, "L_min": args.L_min, "L_max": args.L_max,
        },
        "means": {
            "ubc_row_acc": float(sum(agg["ubc_row_acc"]) / max(1, len(agg["ubc_row_acc"]))),
            "ubc_em_rate": float(sum(agg["ubc_em"]) / max(1, len(agg["ubc_em"]))),
            "mlp_row_acc": float(sum(agg["mlp_row_acc"]) / max(1, len(agg["mlp_row_acc"]))),
            "mlp_em_rate": float(sum(agg["mlp_em"]) / max(1, len(agg["mlp_em"]))),
            "mlp_params": float(sum(agg["mlp_params"]) / max(1, len(agg["mlp_params"]))),
            "ubc_soft_params": float(sum(agg["ubc_soft_params"]) / max(1, len(agg["ubc_soft_params"]))),
            "ubc_total_params": float(sum(agg["ubc_total_params"]) / max(1, len(agg["ubc_total_params"]))),
            "mlp_bnr_exact_L1": float(sum(agg["mlp_bnr_exact_L1"]) / max(1, len(agg["mlp_bnr_exact_L1"]))),
            "mlp_bnr_eps_L1": float(sum(agg["mlp_bnr_eps_L1"]) / max(1, len(agg["mlp_bnr_eps_L1"]))),
            "mlp_prim_hit_rate_L1": float(sum(agg["mlp_prim_hit_rate_L1"]) / max(1, len(agg["mlp_prim_hit_rate_L1"]))),
        },
        "n_instances": len(results),
        "results_jsonl": str(out_jsonl),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ok] wrote: {out_dir / 'summary.json'} and {out_jsonl}")
    return summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--mlp_match", type=str, default="neuron",
                    choices=["neuron", "param_soft", "param_total"])

    # scaling knobs to mirror your sweep idea (S_add=10 etc)
    ap.add_argument("--S_op", type=str, default="none", choices=["none", "add", "mul"])
    ap.add_argument("--S_k", type=int, default=0)
    ap.add_argument("--S_min", type=int, default=2)
    ap.add_argument("--S_max", type=int, default=128)

    ap.add_argument("--L_op", type=str, default="none", choices=["none", "add", "mul"])
    ap.add_argument("--L_k", type=int, default=0)
    ap.add_argument("--L_min", type=int, default=2)
    ap.add_argument("--L_max", type=int, default=16)

    ap.add_argument("--save_ckpts", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg["dataset"] = args.dataset
    run(cfg, Path(args.out_dir), args)


if __name__ == "__main__":
    main()
