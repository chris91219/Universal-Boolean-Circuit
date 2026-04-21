#!/usr/bin/env python3
"""
Joint train UBC + MLP per instance, with:
- MLP match: neuron | param_soft | param_total
- UBC decode expr + gate usage
- MLP: BNR exact/eps per layer, primitive recoverability per layer, gate histograms
- Output: results.jsonl, expr_table.csv, summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    resolve_pair_cfg,
    compute_B_effective,
    extract_gate_usage_from_dbg,
)

# -----------------------------
# Expr normalize + complexity
# -----------------------------
_NOT_A_BARE        = re.compile(r"\(\s*1\s*-\s*a(\d+)\s*\)")
_NOT_PARENS_ANY    = re.compile(r"\(\s*1\s*-\s*\(\s*(.+?)\s*\)\s*\)")
_TILDE_LIT_PARENS  = re.compile(r"\(~\(\s*a(\d+)\s*\)\)")

def _to_tilde_not(expr: str) -> str:
    s = expr
    s = _NOT_A_BARE.sub(r"(~a\1)", s)
    for _ in range(8):
        s2 = _NOT_PARENS_ANY.sub(r"(~(\1))", s)
        if s2 == s:
            break
        s = s2
    s = _TILDE_LIT_PARENS.sub(r"(~a\1)", s)
    return s

def _balance_parens(expr: str) -> str:
    out = []
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            out.append(ch)
        elif ch == ")":
            if depth > 0:
                depth -= 1
                out.append(ch)
        else:
            out.append(ch)
    if depth > 0:
        tmp = []
        for ch in reversed(out):
            if ch == "(" and depth > 0:
                depth -= 1
                continue
            tmp.append(ch)
        out = list(reversed(tmp))
    return "".join(out)

def _strip_outer_parens(expr: str) -> str:
    s = expr.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return s
    return s[1:-1].strip()

def _canonical_spaces(expr: str) -> str:
    s = re.sub(r"\s+", " ", expr).strip()
    s = re.sub(r"\s*&\s*", " & ", s)
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"~\s*a", "~a", s)
    return s

def normalize_expr(expr: str) -> str:
    if not expr:
        return ""
    s = _to_tilde_not(expr)
    s = _balance_parens(s)
    s = _strip_outer_parens(s)
    s = _canonical_spaces(s)
    return s

def expr_complexity(expr: str) -> Tuple[int, int]:
    s = (expr or "").replace(" ", "")
    char_len = len(s)
    num_vars = len(re.findall(r"a\d+", s))
    num_ops  = s.count("&") + s.count("|") + s.count("~")
    return char_len, num_vars + num_ops


# -----------------------------
# Param counting + MLP
# -----------------------------
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ubc_param_counts(B: int, S: int, L_used: int, gate_set: str,
                     pair_cfg: Dict[str, Any], tau0: float,
                     use_lifting: bool, lift_factor: float) -> Tuple[int, int]:
    pair_cfg = dict(pair_cfg or {})
    pair_cfg["route"] = "learned"
    model = DepthStack(B=B, L=L_used, S=S, tau=tau0,
                       pair=pair_cfg, gate_set=gate_set,
                       use_lifting=use_lifting, lift_factor=lift_factor)
    n_soft = count_trainable_params(model)
    K = 16 if gate_set == "16" else 6
    n_fixed = (L_used * S) * K
    return n_soft, n_soft + n_fixed

class TruthTableMLPActs(nn.Module):
    def __init__(self, in_bits: int, hidden_dim: int, depth: int):
        super().__init__()
        self.linears = nn.ModuleList()
        last = int(in_bits)
        hidden_dim = int(hidden_dim)
        depth = int(depth)
        for _ in range(depth):
            self.linears.append(nn.Linear(last, hidden_dim))
            last = hidden_dim
        self.out = nn.Linear(last, 1)

    def forward(self, x: torch.Tensor, return_acts: bool = False):
        acts = []
        h = x
        for lin in self.linears:
            h = torch.relu(lin(h))
            acts.append(h)
        y = torch.sigmoid(self.out(h))
        return (y, acts) if return_acts else y

def build_mlp_param_matched(B: int, depth: int, target_params: int,
                            min_hidden: int = 1, max_hidden: int = 2048) -> Tuple[TruthTableMLPActs, int]:
    best_m = None
    best_n = -1
    for h in range(min_hidden, max_hidden + 1):
        m = TruthTableMLPActs(in_bits=B, hidden_dim=h, depth=depth)
        n = count_trainable_params(m)
        if n <= target_params:
            if n > best_n:
                best_n, best_m = n, m
        else:
            if best_m is not None:
                break
            best_m, best_n = m, n
            break
    assert best_m is not None
    return best_m, best_n


# -----------------------------
# Scaling
# -----------------------------
def apply_scale(val: int, op: str, k: int, vmin: int, vmax: int) -> int:
    op = str(op).lower()
    if op == "none":
        out = val
    elif op == "add":
        out = val + int(k)
    elif op == "mul":
        out = val * int(k)
    else:
        raise ValueError("op must be one of none|add|mul")
    return int(max(vmin, min(vmax, out)))


def row_int(inst: Dict[str, Any], field: str, fallbacks: Tuple[str, ...], default: int) -> int:
    for key in (field, *fallbacks):
        if key in inst:
            return int(inst[key])
    return int(default)


# -----------------------------
# UBC decode
# -----------------------------
def _argmax_unit_primitive(unitW: torch.Tensor, tau: float, PRIMS: List[str]) -> str:
    p = torch.softmax(unitW / max(tau, 1e-8), dim=0)
    return PRIMS[int(p.argmax().item())]

def _argmax_row_pick(L_row: torch.Tensor) -> int:
    return int(L_row.argmax().item())

def _not_expr(e: str) -> str:
    return f"(~({e}))"

def _apply_prim_to_syms(prim: str, a_sym: str, b_sym: str) -> str:
    if prim.startswith("AND"): return f"({a_sym} & {b_sym})"
    if prim.startswith("OR"):  return f"({a_sym} | {b_sym})"
    if prim.startswith("NOT(a)"): return _not_expr(a_sym)
    if prim.startswith("NOT(b)"): return _not_expr(b_sym)
    if prim.startswith("a (skip)"): return f"{a_sym}"
    if prim.startswith("b (skip)"): return f"{b_sym}"

    if prim == "FALSE": return "0"
    if prim == "TRUE":  return "1"
    if prim == "A":     return f"{a_sym}"
    if prim == "B":     return f"{b_sym}"
    if prim == "~A":    return _not_expr(a_sym)
    if prim == "~B":    return _not_expr(b_sym)
    if prim == "AND":   return f"({a_sym} & {b_sym})"
    if prim == "OR":    return f"({a_sym} | {b_sym})"
    if prim == "A&~B":  return f"({a_sym} & {_not_expr(b_sym)})"
    if prim == "~A&B":  return f"({_not_expr(a_sym)} & {b_sym})"
    if prim == "A|~B":  return f"({a_sym} | {_not_expr(b_sym)})"
    if prim == "~A|B":  return f"({_not_expr(a_sym)} | {b_sym})"
    if prim == "NAND":  return _not_expr(f"({a_sym} & {b_sym})")
    if prim == "NOR":   return _not_expr(f"({a_sym} | {b_sym})")
    if prim == "XOR":
        return f"(({a_sym} & {_not_expr(b_sym)}) | ({_not_expr(a_sym)} & {b_sym}))"
    if prim == "XNOR":
        return _not_expr(f"(({a_sym} & {_not_expr(b_sym)}) | ({_not_expr(a_sym)} & {b_sym}))")

    return f"({a_sym} | {b_sym})"

def compose_readout_ubc(B: int, dbg: List[tuple], final_taus: List[float], PRIMS: List[str]) -> str:
    base_syms = [f"a{i}" for i in range(B)]
    outs0, Lrows0, unitWs0, PL0, PR0 = dbg[0]
    tau0 = float(final_taus[0])

    if (PL0 is None) or (PR0 is None) or (not isinstance(PL0, torch.Tensor)) or (PL0.size(-1) != B):
        pair_syms = [("a0", "a1") for _ in range(Lrows0.shape[1])]
    else:
        left_idx = PL0.argmax(dim=1).tolist()
        right_idx = PR0.argmax(dim=1).tolist()
        pair_syms = [(base_syms[i], base_syms[j]) for i, j in zip(left_idx, right_idx)]

    unit_exprs = []
    for s, W in enumerate(unitWs0):
        prim = _argmax_unit_primitive(W, tau0, PRIMS)
        a_sym, b_sym = pair_syms[s]
        unit_exprs.append(_apply_prim_to_syms(prim, a_sym, b_sym))

    wires = []
    for k in range(Lrows0.shape[0]):
        u_idx = _argmax_row_pick(Lrows0[k])
        wires.append(unit_exprs[u_idx])

    for li in range(1, len(dbg) - 1):
        _outs, Lrows, unitWs, _PL, _PR = dbg[li]
        tau = float(final_taus[li])
        unit_exprs = []
        for W in unitWs:
            prim = _argmax_unit_primitive(W, tau, PRIMS)
            unit_exprs.append(_apply_prim_to_syms(prim, wires[0], wires[1]))
        new_wires = []
        for k in range(Lrows.shape[0]):
            u_idx = _argmax_row_pick(Lrows[k])
            new_wires.append(unit_exprs[u_idx])
        wires = new_wires

    _outsF, LrowsF, unitWsF, _PLF, _PRF = dbg[-1]
    tauF = float(final_taus[-1])
    final_unit_exprs = []
    for W in unitWsF:
        prim = _argmax_unit_primitive(W, tauF, PRIMS)
        final_unit_exprs.append(_apply_prim_to_syms(prim, wires[0], wires[1]))
    u_final = _argmax_row_pick(LrowsF[0])
    return final_unit_exprs[u_final]


# -----------------------------
# BNR metrics
# -----------------------------
def _round_tensor(x: torch.Tensor, decimals: int) -> torch.Tensor:
    if decimals <= 0:
        return torch.round(x)
    scale = 10.0 ** decimals
    return torch.round(x * scale) / scale

def bnr_exact_fraction(layer_act: torch.Tensor, decimals: int = 6) -> float:
    A = _round_tensor(layer_act.detach().cpu(), decimals=decimals)
    _, H = A.shape
    ok = 0
    for j in range(H):
        if torch.unique(A[:, j]).numel() <= 2:
            ok += 1
    return ok / max(1, H)

def bnr_eps_fraction(layer_act: torch.Tensor, eps: float = 1e-3) -> float:
    A = layer_act.detach().cpu().float()
    _, H = A.shape
    ok = 0
    for j in range(H):
        v = A[:, j]
        med = torch.median(v)
        lo = v[v <= med]
        hi = v[v >  med]
        if lo.numel() == 0 or hi.numel() == 0:
            ok += 1
            continue
        c0 = torch.median(lo)
        c1 = torch.median(hi)
        d = torch.minimum((v - c0).abs(), (v - c1).abs()).max().item()
        if d <= eps:
            ok += 1
    return ok / max(1, H)


# -----------------------------
# Primitive library
# -----------------------------
_PRIM16_NAMES = [
    "FALSE","AND","A&~B","A","~A&B","B","XOR","OR",
    "NOR","XNOR","~B","A|~B","~A","~A|B","NAND","TRUE"
]

def prim16_outputs(A: torch.Tensor, B: torch.Tensor) -> List[torch.Tensor]:
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
    return [
        F, AND, A & notB, A, notA & B, B, XOR, OR,
        NOR, XNOR, notB, A | notB, notA, notA | B, NAND, T
    ]

def interpret_mlp_first_layer_primitives(X: torch.Tensor, act1: torch.Tensor) -> Dict[str, Any]:
    Xb = (X.detach().cpu() >= 0.5)
    A1 = act1.detach().cpu().float()
    _, B = Xb.shape
    _, H = A1.shape

    zero_row = (Xb.sum(dim=1) == 0).nonzero(as_tuple=False)
    zero_idx = int(zero_row[0].item()) if zero_row.numel() > 0 else 0

    literals = []
    lit_names = []
    for i in range(B):
        literals.append(Xb[:, i]);  lit_names.append(f"a{i}")
        literals.append(~Xb[:, i]); lit_names.append(f"~a{i}")

    gate_hist_exact = Counter()
    best_accs = []
    exact_hits = 0

    for j in range(H):
        u = A1[:, j]
        t = float(u[zero_idx].item())
        yb = (u >= t)

        best_acc = -1.0
        best_kind = "lit"
        best_name = ""
        # literals
        for tmpl, nm in zip(literals, lit_names):
            acc = (yb == tmpl).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_kind = "lit"
                best_name = nm

        # gates
        for i in range(B):
            Ai = Xb[:, i]
            for k in range(B):
                if k == i:
                    continue
                Bk = Xb[:, k]
                outs = prim16_outputs(Ai, Bk)
                for gi, gout in enumerate(outs):
                    acc = (yb == gout).float().mean().item()
                    if acc > best_acc:
                        best_acc = acc
                        best_kind = "gate16"
                        best_name = _PRIM16_NAMES[gi]

        best_accs.append(best_acc)
        if best_acc == 1.0:
            exact_hits += 1
            if best_kind == "gate16":
                gate_hist_exact[best_name] += 1

    return {
        "primitive_hit_rate_L1": float(exact_hits / max(1, H)),
        "mean_best_primitive_acc_L1": float(sum(best_accs) / max(1, len(best_accs))),
        "gate_hist_exact": dict(gate_hist_exact),
    }


# -----------------------------
# MLP: layerwise primitive recoverability + all-layer gate hist
# -----------------------------
def extract_boolean_expr_from_mlp(X: torch.Tensor, y_pred: torch.Tensor, acts: List[torch.Tensor]) -> Dict[str, Any]:
    Xb = (X.detach().cpu() >= 0.5)
    zero_row = (Xb.sum(dim=1) == 0).nonzero(as_tuple=False)
    zero_idx = int(zero_row[0].item()) if zero_row.numel() > 0 else 0

    prev_bool = Xb
    prev_exprs = [f"a{i}" for i in range(Xb.shape[1])]

    layer_stats = []
    gate_hist_exact_all = Counter()

    for li, A in enumerate(acts):
        A = A.detach().cpu().float()
        H = A.shape[1]
        thresh = A[zero_idx, :]
        cur_bool = (A >= thresh.unsqueeze(0))

        exact_count = 0
        best_accs = []

        prev_literals = []
        for k in range(prev_bool.shape[1]):
            prev_literals.append(prev_bool[:, k])
            prev_literals.append(~prev_bool[:, k])

        P = prev_bool.shape[1]
        cur_exprs = []

        for j in range(H):
            yj = cur_bool[:, j]

            best_acc = -1.0
            best_is_gate16 = False
            best_gate_name = None
            best_expr = ""

            # literals / neg-literals of prev
            for kk in range(P):
                for t, e in [(prev_bool[:, kk], prev_exprs[kk]), (~prev_bool[:, kk], _not_expr(prev_exprs[kk]))]:
                    acc = (yj == t).float().mean().item()
                    if acc > best_acc:
                        best_acc = acc
                        best_is_gate16 = False
                        best_gate_name = None
                        best_expr = e

            # gate16(prev_p, prev_q)
            for a_idx in range(P):
                Acol = prev_bool[:, a_idx]; Aexpr = prev_exprs[a_idx]
                for b_idx in range(P):
                    if b_idx == a_idx:
                        continue
                    Bcol = prev_bool[:, b_idx]; Bexpr = prev_exprs[b_idx]
                    outs = prim16_outputs(Acol, Bcol)
                    for gi, gout in enumerate(outs):
                        acc = (yj == gout).float().mean().item()
                        if acc > best_acc:
                            name = _PRIM16_NAMES[gi]
                            best_acc = acc
                            best_is_gate16 = True
                            best_gate_name = name
                            best_expr = _apply_prim_to_syms(name if name in {"A","B","~A","~B","AND","OR","A&~B","~A&B","A|~B","~A|B","NAND","NOR","XOR","XNOR","FALSE","TRUE"} else "OR", Aexpr, Bexpr)
                            # Use the existing symbolic builder for robustness
                            best_expr = best_expr

            cur_exprs.append(best_expr)
            best_accs.append(best_acc)

            if best_acc == 1.0:
                exact_count += 1
                if best_is_gate16 and best_gate_name is not None:
                    gate_hist_exact_all[best_gate_name] += 1

        layer_stats.append({
            "layer": li,
            "H": H,
            "exact_fit_rate": float(exact_count / max(1, H)),
            "mean_best_fit_acc": float(sum(best_accs) / max(1, len(best_accs))),
        })

        prev_bool = cur_bool
        prev_exprs = cur_exprs

    # final output gate (path)
    yb = (y_pred.detach().cpu().view(-1) >= 0.5)
    P = prev_bool.shape[1]

    best_out_acc = -1.0
    best_out_is_gate16 = False
    best_out_gate = None

    for k in range(P):
        for t in [prev_bool[:, k], ~prev_bool[:, k]]:
            acc = (yb == t).float().mean().item()
            if acc > best_out_acc:
                best_out_acc = acc
                best_out_is_gate16 = False
                best_out_gate = None

    for a_idx in range(P):
        Acol = prev_bool[:, a_idx]
        for b_idx in range(P):
            if b_idx == a_idx:
                continue
            Bcol = prev_bool[:, b_idx]
            outs = prim16_outputs(Acol, Bcol)
            for gi, gout in enumerate(outs):
                acc = (yb == gout).float().mean().item()
                if acc > best_out_acc:
                    best_out_acc = acc
                    best_out_is_gate16 = True
                    best_out_gate = _PRIM16_NAMES[gi]

    gate_hist_exact_path = Counter()
    if best_out_acc == 1.0 and best_out_is_gate16 and best_out_gate is not None:
        gate_hist_exact_path[best_out_gate] += 1

    # build per-layer primitive vectors
    prim_exact_per_layer = [float(ls["exact_fit_rate"]) for ls in layer_stats]
    prim_best_acc_per_layer = [float(ls["mean_best_fit_acc"]) for ls in layer_stats]

    return {
        "layer_stats": layer_stats,
        "prim_exact_per_layer": prim_exact_per_layer,
        "prim_best_acc_per_layer": prim_best_acc_per_layer,
        "prim_exact_avg": float(sum(prim_exact_per_layer) / max(1, len(prim_exact_per_layer))) if prim_exact_per_layer else 0.0,
        "prim_best_acc_avg": float(sum(prim_best_acc_per_layer) / max(1, len(prim_best_acc_per_layer))) if prim_best_acc_per_layer else 0.0,
        "gate_hist_exact_all": dict(gate_hist_exact_all),
        "gate_hist_exact_path": dict(gate_hist_exact_path),
    }


# -----------------------------
# Train loops
# -----------------------------
@dataclass
class UBCOut:
    row_acc: float
    em: int
    dbg: List[tuple]
    taus: List[float]
    pred_expr: str
    gate_usage: Dict[str, Any]

def train_ubc_instance(device: torch.device, cfg: Dict[str, Any], inst: Dict[str, Any], *, S_used: int, L_used: int) -> UBCOut:
    B = int(inst["B"])
    formula = str(inst["formula"])
    X, y_true = T.truth_table_from_formula(B, formula)
    X = X.to(device)
    y_true = y_true.to(device)

    gate_set = str(cfg.get("gate_set", "6"))
    anneal_cfg = cfg["anneal"]
    T0 = float(anneal_cfg["T0"])

    lifting_cfg = cfg.get("lifting", {})
    use_lifting = bool(lifting_cfg.get("use", True))
    lift_factor = float(lifting_cfg.get("factor", 2.0))
    B_eff = compute_B_effective(B, use_lifting, lift_factor)

    pair_cfg = dict(cfg.get("pair", {}) or {})
    pair_cfg = resolve_pair_cfg(pair_cfg, B=B, B_eff=B_eff, S=S_used, X=X, y_true=y_true)

    model = DepthStack(B=B, L=L_used, S=S_used, tau=T0,
                       pair=pair_cfg, gate_set=gate_set,
                       use_lifting=use_lifting, lift_factor=lift_factor).to(device)

    opt_name = str(cfg["optimizer"]).lower()
    opt = (optim.RMSprop(model.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-8)
           if opt_name == "rmsprop" else
           optim.Adam(model.parameters(), lr=cfg["lr"]))

    steps = int(cfg["steps"])
    regs_cfg = cfg["regs"]

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
        if hasattr(model, "set_layer_taus_and_bandwidths") and gate_set == "16":
            s_cfg = cfg.get("sigma16", {})
            model.set_layer_taus_and_bandwidths(
                taus,
                s_start=float(s_cfg.get("s_start", 0.25)),
                s_end=float(s_cfg.get("s_end", 0.10)),
            )
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

    y_pred, dbg, final_taus = last
    with torch.no_grad():
        row_acc, em = per_instance_metrics(y_true, y_pred)
        if gate_set == "16":
            from .boolean_prims16 import PRIMS16 as PRIMS
        else:
            from .boolean_prims import PRIMS as PRIMS
        pred_expr = normalize_expr(compose_readout_ubc(B, dbg, final_taus, PRIMS))
        gate_usage = extract_gate_usage_from_dbg(dbg, final_taus, PRIMS)

    return UBCOut(row_acc=row_acc, em=em, dbg=dbg, taus=final_taus, pred_expr=pred_expr, gate_usage=gate_usage)


@dataclass
class MLPOut:
    row_acc: float
    em: int
    n_params: int

    bnr_exact_per_layer: List[float]
    bnr_eps_per_layer: List[float]

    primitive_hit_rate_L1: float
    mean_best_primitive_acc_L1: float
    gate_hist_exact_L1: Dict[str, int]

    prim_exact_per_layer: List[float]
    prim_best_acc_per_layer: List[float]
    prim_exact_avg: float
    prim_best_acc_avg: float

    gate_hist_exact_all: Dict[str, int]
    gate_hist_exact_path: Dict[str, int]


def train_mlp_instance(device: torch.device, cfg: Dict[str, Any], inst: Dict[str, Any],
                       *, S_used: int, L_used: int,
                       match_mode: str,
                       ubc_soft_params: int,
                       ubc_total_params: int) -> MLPOut:
    B = int(inst["B"])
    formula = str(inst["formula"])
    X, y_true = T.truth_table_from_formula(B, formula)
    X = X.to(device)
    y_true = y_true.to(device)

    if match_mode == "neuron":
        mlp = TruthTableMLPActs(in_bits=B, hidden_dim=S_used, depth=L_used)
        n_params = count_trainable_params(mlp)
    elif match_mode == "param_soft":
        mlp, n_params = build_mlp_param_matched(B=B, depth=L_used, target_params=ubc_soft_params)
    elif match_mode == "param_total":
        mlp, n_params = build_mlp_param_matched(B=B, depth=L_used, target_params=ubc_total_params)
    else:
        raise ValueError("mlp_match must be one of neuron|param_soft|param_total")
    mlp = mlp.to(device)

    opt_name = str(cfg["optimizer"]).lower()
    opt = (optim.RMSprop(mlp.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-8)
           if opt_name == "rmsprop" else
           optim.Adam(mlp.parameters(), lr=cfg["lr"]))

    steps = int(cfg["steps"])
    es_cfg = cfg.get("early_stop", {})
    use_es = bool(es_cfg.get("use", False))
    min_steps = int(es_cfg.get("min_steps", 0))
    check_every = int(es_cfg.get("check_every", 10))
    patience_checks = int(es_cfg.get("patience_checks", 3))
    metric = str(es_cfg.get("metric", "em")).lower()
    target = float(es_cfg.get("target", 1.0))
    ok_streak = 0

    for step in range(steps):
        opt.zero_grad()
        y_pred = mlp(X)
        loss = safe_bce(y_pred, y_true)
        loss.backward()
        opt.step()

        if use_es and (step + 1) >= min_steps and ((step + 1) % check_every == 0):
            with torch.no_grad():
                row_acc, em = per_instance_metrics(y_true, y_pred)
                cur = float(em) if metric == "em" else float(row_acc)
                ok_streak = (ok_streak + 1) if (cur >= target) else 0
                if ok_streak >= patience_checks:
                    break

    with torch.no_grad():
        y_pred, acts = mlp(X, return_acts=True)
        row_acc, em = per_instance_metrics(y_true, y_pred)

        bnr_exact = [bnr_exact_fraction(a, decimals=6) for a in acts]
        bnr_eps   = [bnr_eps_fraction(a, eps=1e-3) for a in acts]

        primL1 = interpret_mlp_first_layer_primitives(X, acts[0]) if len(acts) > 0 else {
            "primitive_hit_rate_L1": 0.0,
            "mean_best_primitive_acc_L1": 0.0,
            "gate_hist_exact": {},
        }

        expl = extract_boolean_expr_from_mlp(X, y_pred, acts)

    return MLPOut(
        row_acc=row_acc, em=em, n_params=n_params,
        bnr_exact_per_layer=bnr_exact,
        bnr_eps_per_layer=bnr_eps,
        primitive_hit_rate_L1=float(primL1["primitive_hit_rate_L1"]),
        mean_best_primitive_acc_L1=float(primL1["mean_best_primitive_acc_L1"]),
        gate_hist_exact_L1=dict(primL1.get("gate_hist_exact", {})),
        prim_exact_per_layer=list(expl.get("prim_exact_per_layer", [])),
        prim_best_acc_per_layer=list(expl.get("prim_best_acc_per_layer", [])),
        prim_exact_avg=float(expl.get("prim_exact_avg", 0.0)),
        prim_best_acc_avg=float(expl.get("prim_best_acc_avg", 0.0)),
        gate_hist_exact_all=dict(expl.get("gate_hist_exact_all", {})),
        gate_hist_exact_path=dict(expl.get("gate_hist_exact_path", {})),
    )


# -----------------------------
# Main runner
# -----------------------------
def run(cfg: Dict[str, Any], out_dir: Path, args) -> Dict[str, Any]:
    device = _device(cfg)
    seed_all(cfg["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = out_dir / "results.jsonl"
    expr_csv = out_dir / "expr_table.csv"

    insts = T.load_instances_jsonl(cfg["dataset"])
    if results_jsonl.exists():
        results_jsonl.unlink()

    expr_rows = []
    agg = defaultdict(list)

    for idx, inst in enumerate(insts):
        B = int(inst["B"])
        S_base = row_int(inst, args.W_base_field, ("S", "W_base"), int(cfg.get("S", 2)))
        L_base = row_int(inst, args.D_base_field, ("L", "D_base"), int(cfg.get("L", 2)))

        S_used = apply_scale(S_base, args.S_op, args.S_k, args.S_min, args.S_max)
        L_used = apply_scale(L_base, args.L_op, args.L_k, args.L_min, args.L_max)

        label_expr = normalize_expr(str(inst["formula"]))

        gate_set = str(cfg.get("gate_set", "6"))
        pair_cfg = dict(cfg.get("pair", {}))
        tau0 = float(cfg["anneal"]["T0"])
        lifting_cfg = cfg.get("lifting", {})
        use_lifting = bool(lifting_cfg.get("use", True))
        lift_factor = float(lifting_cfg.get("factor", 2.0))

        ubc_soft, ubc_total = ubc_param_counts(
            B=B, S=S_used, L_used=L_used,
            gate_set=gate_set, pair_cfg=pair_cfg,
            tau0=tau0, use_lifting=use_lifting, lift_factor=lift_factor,
        )

        ubc = train_ubc_instance(device, cfg, inst, S_used=S_used, L_used=L_used)
        mlp = train_mlp_instance(
            device, cfg, inst, S_used=S_used, L_used=L_used,
            match_mode=args.mlp_match,
            ubc_soft_params=ubc_soft, ubc_total_params=ubc_total,
        )

        lab_char, lab_tok = expr_complexity(label_expr)
        ubc_char, ubc_tok = expr_complexity(ubc.pred_expr)
        mlp_char, mlp_tok = expr_complexity("")  # no expr string here
        shortest = min({"label": (lab_tok, lab_char), "ubc": (ubc_tok, ubc_char), "mlp": (mlp_tok, mlp_char)}, key=lambda k: {"label": (lab_tok, lab_char), "ubc": (ubc_tok, ubc_char), "mlp": (mlp_tok, mlp_char)}[k])

        bnr_exact_L1 = mlp.bnr_exact_per_layer[0] if mlp.bnr_exact_per_layer else 0.0
        bnr_eps_L1   = mlp.bnr_eps_per_layer[0] if mlp.bnr_eps_per_layer else 0.0
        bnr_exact_avg = float(sum(mlp.bnr_exact_per_layer) / max(1, len(mlp.bnr_exact_per_layer))) if mlp.bnr_exact_per_layer else 0.0
        bnr_eps_avg   = float(sum(mlp.bnr_eps_per_layer) / max(1, len(mlp.bnr_eps_per_layer))) if mlp.bnr_eps_per_layer else 0.0

        prim_exact_L1 = mlp.prim_exact_per_layer[0] if mlp.prim_exact_per_layer else 0.0
        prim_best_L1  = mlp.prim_best_acc_per_layer[0] if mlp.prim_best_acc_per_layer else 0.0

        row = {
            "idx": idx,
            "B": B,
            "S_base": S_base,
            "S_used": S_used,
            "L_base": L_base,
            "L_used": L_used,
            "W_base": S_base,
            "W_model": S_used,
            "D_base": L_base,
            "D_model": L_used,
            "label_expr": label_expr,
            "ubc": {
                "em": ubc.em,
                "row_acc": ubc.row_acc,
                "pred_expr": ubc.pred_expr,
                "gate_usage": ubc.gate_usage,
                "n_soft_params": ubc_soft,
                "n_total_params": ubc_total,
            },
            "mlp": {
                "match_mode": args.mlp_match,
                "em": mlp.em,
                "row_acc": mlp.row_acc,
                "n_params": mlp.n_params,
                "bnr_exact_per_layer": mlp.bnr_exact_per_layer,
                "bnr_eps_per_layer": mlp.bnr_eps_per_layer,
                "primitive_hit_rate_L1": mlp.primitive_hit_rate_L1,
                "mean_best_primitive_acc_L1": mlp.mean_best_primitive_acc_L1,
                "gate_hist_exact_L1": mlp.gate_hist_exact_L1,
                "prim_exact_per_layer": mlp.prim_exact_per_layer,
                "prim_best_acc_per_layer": mlp.prim_best_acc_per_layer,
                "prim_exact_avg": mlp.prim_exact_avg,
                "prim_best_acc_avg": mlp.prim_best_acc_avg,
                "gate_hist_exact_all": mlp.gate_hist_exact_all,
                "gate_hist_exact_path": mlp.gate_hist_exact_path,
            },
            "expr_lengths": {
                "label": {"char": lab_char, "tok": lab_tok},
                "ubc":   {"char": ubc_char, "tok": ubc_tok},
                "mlp":   {"char": mlp_char, "tok": mlp_tok},
                "shortest": shortest,
            }
        }

        with results_jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"[{idx+1}/{len(insts)}] B={B} S={S_used} L={L_used} | "
            f"UBC EM={ubc.em} acc={ubc.row_acc:.3f} | "
            f"MLP({args.mlp_match}) EM={mlp.em} acc={mlp.row_acc:.3f} | "
            f"BNR(L1)={bnr_exact_L1:.3f}/{bnr_eps_L1:.3f} BNR(avg)={bnr_exact_avg:.3f}/{bnr_eps_avg:.3f} | "
            f"prim(L1)={prim_exact_L1:.3f}/{prim_best_L1:.3f} prim(avg)={mlp.prim_exact_avg:.3f}/{mlp.prim_best_acc_avg:.3f}"
        )
        print(f"   bnr_exact_per_layer: {[round(x,3) for x in mlp.bnr_exact_per_layer]}")
        print(f"   bnr_eps_per_layer  : {[round(x,3) for x in mlp.bnr_eps_per_layer]}")
        print(f"   prim_exact_per_layer: {[round(x,3) for x in mlp.prim_exact_per_layer]}")
        print(f"   prim_best_acc_per_layer: {[round(x,3) for x in mlp.prim_best_acc_per_layer]}")

        # aggregates
        agg["ubc_em"].append(float(ubc.em))
        agg["mlp_em"].append(float(mlp.em))

        agg["bnr_exact_L1"].append(float(bnr_exact_L1))
        agg["bnr_eps_L1"].append(float(bnr_eps_L1))
        agg["bnr_exact_avg"].append(float(bnr_exact_avg))
        agg["bnr_eps_avg"].append(float(bnr_eps_avg))

        agg["prim_exact_L1"].append(float(prim_exact_L1))
        agg["prim_best_L1"].append(float(prim_best_L1))
        agg["prim_exact_avg"].append(float(mlp.prim_exact_avg))
        agg["prim_best_avg"].append(float(mlp.prim_best_acc_avg))

    # minimal summary
    n = max(1, len(insts))
    summary = {
        "config": cfg,
        "mlp_match": args.mlp_match,
        "scale": {
            "W_base_field": args.W_base_field,
            "D_base_field": args.D_base_field,
            "S_op": args.S_op, "S_k": args.S_k, "S_min": args.S_min, "S_max": args.S_max,
            "L_op": args.L_op, "L_k": args.L_k, "L_min": args.L_min, "L_max": args.L_max,
        },
        "means": {
            "ubc_em_rate": float(sum(agg["ubc_em"]) / n),
            "mlp_em_rate": float(sum(agg["mlp_em"]) / n),
            "bnr_exact_L1": float(sum(agg["bnr_exact_L1"]) / n),
            "bnr_eps_L1": float(sum(agg["bnr_eps_L1"]) / n),
            "bnr_exact_avg": float(sum(agg["bnr_exact_avg"]) / n),
            "bnr_eps_avg": float(sum(agg["bnr_eps_avg"]) / n),
            "prim_exact_L1": float(sum(agg["prim_exact_L1"]) / n),
            "prim_best_L1": float(sum(agg["prim_best_L1"]) / n),
            "prim_exact_avg": float(sum(agg["prim_exact_avg"]) / n),
            "prim_best_avg": float(sum(agg["prim_best_avg"]) / n),
        },
        "n_instances": len(insts),
        "results_jsonl": str(results_jsonl),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ok] wrote summary.json + results.jsonl in {out_dir}")
    return summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--mlp_match", type=str, default="neuron",
                    choices=["neuron", "param_soft", "param_total"])

    ap.add_argument("--W_base_field", type=str, default="S",
                    help="Dataset field used as model width base, e.g. S, W_base, or W_true.")
    ap.add_argument("--S_op", type=str, default="none", choices=["none","add","mul"])
    ap.add_argument("--S_k", type=int, default=0)
    ap.add_argument("--S_min", type=int, default=2)
    ap.add_argument("--S_max", type=int, default=128)

    ap.add_argument("--D_base_field", type=str, default="L",
                    help="Dataset field used as model depth base, e.g. L, D_base, or D_true.")
    ap.add_argument("--L_op", type=str, default="none", choices=["none","add","mul"])
    ap.add_argument("--L_k", type=int, default=0)
    ap.add_argument("--L_min", type=int, default=2)
    ap.add_argument("--L_max", type=int, default=16)

    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg["dataset"] = args.dataset
    run(cfg, Path(args.out_dir), args)


if __name__ == "__main__":
    main()
