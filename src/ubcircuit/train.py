# src/ubcircuit/train.py
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from .modules import DepthStack
from .utils import seed_all, safe_bce, make_async_taus, regularizers_bundle
from . import tasks as T
from .boolean_prims16 import (
    set_sigma16_bandwidth,
    set_sigma16_mode,
    set_sigma16_radius,
)
from .mi import top_mi_pairs, priors_from_pairs


DEFAULT_CFG: Dict[str, Any] = {
    "task": "(a&b)|(~a)",   # single-task fallback
    "L": 2,
    "S": 2,
    "gate_set": "6",   # "6" (legacy) or "16" (compact)
    "sigma16": {                   # NEW: bandwidth anneal (used only if gate_set == "16")
        "s_start": 0.25,
        "s_end": 0.10,
        "mode":    "rbf",          # "rbf" | "bump" | "lagrange"
        "radius":  0.75,           # used only if mode='bump'
    },
    "steps": 1200,
    "lr": 0.05,
    "optimizer": "rmsprop",
    "seed": 2024,
    "device": "auto",
    # L selection strategy when using a dataset:
    "use_row_L": True,       # DEFAULT: use the per-row "L" field when present
    "use_max_L": False,      # else, optionally use the max L across the dataset
    "anneal": {
        "T0": 0.35, "Tmin": 0.12,
        "direction": "top_down",   # "bottom_up" | "top_down"
        "schedule": "linear",      # "linear" | "cosine"
        "phase_scale": 0.4,
        "start_frac": 0.0
    },
    "regs": {"lam_entropy": 1.0e-3, "lam_div_units": 5.0e-4, "lam_div_rows": 5.0e-4,"lam_const16": 5.0e-3},
    "pair": {
                "repel": True,                 
                "eta": 1.0, 
                "mode": "hard-log",   # right-pick repulsion 
                "route": "learned",   # NEW: "learned" | "mi_soft" | "mi_hard"
            },  
    "aux": {"use": False, "lam": 1.0e-2, "wire_targets": ["AND", "NOTA"]},
    "dataset": None,  # path to JSONL with rows like {"B":int,"S":int,"L":int,"formula":str}
    "early_stop": {
        "use": True,
        "metric": "em",          # "em" or "row_acc"
        "target": 1.0,           # 1.0 for EM; e.g., 0.999 for row_acc
        "min_steps": 100,        # warm-up
        "check_every": 10,
        "patience_checks": 3
    },
}


def load_config(path: str | None) -> Dict[str, Any]:
    cfg = DEFAULT_CFG.copy()
    if path:
        user = yaml.safe_load(open(path)) or {}
        for k, v in user.items():
            if isinstance(v, dict) and k in {"anneal", "regs", "aux", "early_stop", "pair", "sigma16"}:
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    return cfg



def _device(cfg) -> torch.device:
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


# ---------- Metrics ----------
def per_instance_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, int]:
    yb = (y_pred >= 0.5).float()
    row_acc = (yb.eq(y_true)).float().mean().item()
    em = int(torch.all(yb.eq(y_true)).item())
    return row_acc, em


# ---------- Expression normalization (align style with labels, NO SIMPLIFY) ----------
# Convert arithmetic NOT to tilde form without logical simplification.
# Examples:
#   (1 - a0)            -> (~a0)
#   (1 - (~a0))         -> (~(~a0))
#   (1 - (1 - (~a0)))   -> (~(~(~a0)))
# We intentionally DO NOT collapse ~~X -> X, etc.

_NOT_A_BARE        = re.compile(r"\(\s*1\s*-\s*a(\d+)\s*\)")
_NOT_PARENS_ANY    = re.compile(r"\(\s*1\s*-\s*\(\s*(.+?)\s*\)\s*\)")
_TILDE_LIT_PARENS  = re.compile(r"\(~\(\s*a(\d+)\s*\)\)")  # (~(aK)) -> (~aK)

def _to_tilde_not(expr: str) -> str:
    s = expr
    # (1 - aK) -> (~aK)
    s = _NOT_A_BARE.sub(r"(~a\1)", s)
    # (1 - (X)) -> (~(X))  (repeat a few times to catch nesting)
    for _ in range(8):
        s2 = _NOT_PARENS_ANY.sub(r"(~(\1))", s)
        if s2 == s: break
        s = s2
    # (~(aK)) -> (~aK) (cosmetic)
    s = _TILDE_LIT_PARENS.sub(r"(~a\1)", s)
    return s

def _balance_parens(expr: str) -> str:
    out = []
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1; out.append(ch)
        elif ch == ")":
            if depth > 0:
                depth -= 1; out.append(ch)
        else:
            out.append(ch)
    if depth > 0:
        tmp = []
        for ch in reversed(out):
            if ch == "(" and depth > 0:
                depth -= 1; continue
            tmp.append(ch)
        out = list(reversed(tmp))
    return "".join(out)

def _strip_outer_parens(expr: str) -> str:
    s = expr.strip()
    if not (s.startswith("(") and s.endswith(")")): return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(": depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(s)-1: return s
    return s[1:-1].strip()

def _force_balance_counts(expr: str) -> str:
    s = expr
    opens = s.count("("); closes = s.count(")")
    while closes > opens:
        j = s.rfind(")")
        if j < 0: break
        s = s[:j] + s[j+1:]; closes -= 1
    while opens > closes:
        i = s.find("(")
        if i < 0: break
        s = s[:i] + s[i+1:]; opens -= 1
    return s

def _canonical_spaces(expr: str) -> str:
    s = re.sub(r"\s+", " ", expr).strip()
    s = re.sub(r"\s*&\s*", " & ", s)
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"~\s*a", "~a", s)
    return s

def normalize_expr(expr: str) -> str:
    if not expr: return expr
    s = _to_tilde_not(expr)   # NO simplification beyond formatting to tilde form
    s = _balance_parens(s)
    s = _force_balance_counts(s)
    s = _strip_outer_parens(s)
    s = _canonical_spaces(s)
    return s

def expr_complexity(expr: str) -> Tuple[int, int]:
    s = expr.replace(" ", "")
    char_len = len(s)
    num_vars = len(re.findall(r"a\d+", s))
    num_ops  = s.count("&") + s.count("|") + s.count("~")
    token_score = num_vars + num_ops
    return char_len, token_score


# ---------- Layer-by-layer composed symbolic readout ----------

# def _apply_prim_to_syms(prim: str, a_sym: str, b_sym: str) -> str:
#     # No extra parens around single literals inside &, |; group the whole binary op.
#     if prim.startswith("AND"):       # "AND(a,b)=min"
#         return f"({a_sym} & {b_sym})"
#     if prim.startswith("OR"):        # "OR(a,b)=max"
#         return f"({a_sym} | {b_sym})"
#     if prim.startswith("NOT(a)"):
#         return f"(1 - {a_sym})"
#     if prim.startswith("NOT(b)"):
#         return f"(1 - {b_sym})"
#     if prim.startswith("a (skip)"):
#         return f"{a_sym}"
#     if prim.startswith("b (skip)"):
#         return f"{b_sym}"
#     return f"({a_sym} | {b_sym})"  # fallback

def _apply_prim_to_syms(prim: str, a_sym: str, b_sym: str) -> str:
    # ---- legacy 6 features
    if prim.startswith("AND"):        return f"({a_sym} & {b_sym})"
    if prim.startswith("OR("):        return f"({a_sym} | {b_sym})"
    if prim.startswith("NOT(a)"):     return f"(1 - {a_sym})"
    if prim.startswith("NOT(b)"):     return f"(1 - {b_sym})"
    if prim.startswith("a (skip)"):   return f"{a_sym}"
    if prim.startswith("b (skip)"):   return f"{b_sym}"

    # ---- 16-gate names
    if prim == "FALSE":    return "0"
    if prim == "TRUE":     return "1"
    if prim == "A":        return f"{a_sym}"
    if prim == "B":        return f"{b_sym}"
    if prim == "~A":       return f"(1 - {a_sym})"
    if prim == "~B":       return f"(1 - {b_sym})"
    if prim == "AND":      return f"({a_sym} & {b_sym})"
    if prim == "OR":       return f"({a_sym} | {b_sym})"
    if prim == "XOR":      return f"(({a_sym} & (1 - {b_sym})) | ((1 - {a_sym}) & {b_sym}))"
    if prim == "XNOR":     return f"(1 - (({a_sym} & (1 - {b_sym})) | ((1 - {a_sym}) & {b_sym})))"
    if prim == "NAND":     return f"(1 - ({a_sym} & {b_sym}))"
    if prim == "NOR":      return f"(1 - ({a_sym} | {b_sym}))"
    if prim == "A&~B":     return f"({a_sym} & (1 - {b_sym}))"
    if prim == "~A&B":     return f"((1 - {a_sym}) & {b_sym})"
    if prim == "A|~B":     return f"({a_sym} | (1 - {b_sym}))"
    if prim == "~A|B":     return f"((1 - {a_sym}) | {b_sym})"

    # Fallback
    return f"({a_sym} | {b_sym})"


def _argmax_unit_primitive(unitW: torch.Tensor, tau: float, PRIMS: List[str]) -> str:
    p = torch.softmax(unitW / max(tau, 1e-8), dim=0)
    return PRIMS[int(p.argmax().item())]

def _argmax_row_pick(L_row: torch.Tensor) -> int:
    return int(L_row.argmax().item())

def compose_readout_full(
    B: int,
    dbg: List[tuple],
    final_taus: List[float],
    PRIMS: List[str],
) -> str:
    base_syms = [f"a{i}" for i in range(B)]

    # Layer 0 (GeneralLayer: B -> 2)
    outs0, Lrows0, unitWs0, PL0, PR0 = dbg[0]
    tau0 = final_taus[0]

    if PL0 is None or PR0 is None:
        pair_syms = [("a0", "a1") for _ in range(Lrows0.shape[1])]
    else:
        PLp = torch.softmax(PL0 / max(tau0, 1e-8), dim=-1).detach()
        PRp = torch.softmax(PR0 / max(tau0, 1e-8), dim=-1).detach()
        left_idx  = PLp.argmax(dim=1).tolist()
        right_idx = PRp.argmax(dim=1).tolist()
        pair_syms = [(base_syms[i], base_syms[j]) for i, j in zip(left_idx, right_idx)]

    unit_exprs = []
    for s, W in enumerate(unitWs0):
        prim = _argmax_unit_primitive(W, tau0, PRIMS)
        a_sym, b_sym = pair_syms[s]
        unit_exprs.append(_apply_prim_to_syms(prim, a_sym, b_sym))

    wires = []
    for k in range(Lrows0.shape[0]):  # out_bits = 2
        u_idx = _argmax_row_pick(Lrows0[k])
        wires.append(unit_exprs[u_idx])

    # Middle layers (ReasoningLayer: 2 -> 2)
    for li in range(1, len(dbg) - 1):
        outs, Lrows, unitWs, _PL, _PR = dbg[li]
        tau = final_taus[li]
        unit_exprs = []
        for W in unitWs:
            prim = _argmax_unit_primitive(W, tau, PRIMS)
            unit_exprs.append(_apply_prim_to_syms(prim, wires[0], wires[1]))
        new_wires = []
        for k in range(Lrows.shape[0]):  # out_bits = 2
            u_idx = _argmax_row_pick(Lrows[k])
            new_wires.append(unit_exprs[u_idx])
        wires = new_wires

    # Final layer (ReasoningLayer: 2 -> 1)
    outsF, LrowsF, unitWsF, _PLF, _PRF = dbg[-1]
    tauF = final_taus[-1]
    final_unit_exprs = []
    for W in unitWsF:
        prim = _argmax_unit_primitive(W, tauF, PRIMS)
        final_unit_exprs.append(_apply_prim_to_syms(prim, wires[0], wires[1]))
    u_final = _argmax_row_pick(LrowsF[0])
    return final_unit_exprs[u_final]


# ---------- Training (single instance) ----------

def train_single_instance(
    device,
    cfg: Dict[str, Any],
    inst: Dict[str, Any],
    L_override: Optional[int] = None,
) -> Dict[str, Any]:
    #from .boolean_prims import PRIMS as PRIMS_LIST

    B = int(inst["B"]); S = int(inst["S"]); formula = str(inst["formula"])
    X, y_true = T.truth_table_from_formula(B, formula)    
    X = X.to(device); y_true = y_true.to(device)

    # ---- Resolve L, temps, gate set ----
    if L_override is not None:
        L_used = int(L_override)
    elif bool(cfg.get("use_row_L", True)) and ("L" in inst):
        L_used = int(inst["L"])
    else:
        L_used = int(cfg["L"])

    anneal_cfg = cfg["anneal"]
    T0 = float(anneal_cfg["T0"])
    gate_set = str(cfg.get("gate_set", "6"))
    if gate_set == "16":
        from .boolean_prims16 import PRIMS16 as PRIMS_LIST
    else:
        from .boolean_prims import PRIMS as PRIMS_LIST

    # ---- Pair config with routing modes: learned | mi_soft | mi_hard ----
    pair_cfg = dict(cfg.get("pair", {}))  # copy
    route = str(pair_cfg.get("route", "learned"))
    if (B > 2) and route in {"mi_soft", "mi_hard"}:
        from .mi import top_mi_pairs, priors_from_pairs
        pairs = top_mi_pairs(X, y_true, S=S, disjoint=bool(pair_cfg.get("mi_disjoint", True)))
        if route == "mi_hard":
            pair_cfg["fixed_pairs"] = pairs                           # for FixedPairSelector in GeneralLayer
        else:  # mi_soft
            PLp, PRp = priors_from_pairs(pairs, B)
            pair_cfg["PL_prior"] = PLp.tolist()
            pair_cfg["PR_prior"] = PRp.tolist()
            # pair_cfg["prior_strength"] can be set in YAML (default e.g. 2.0)

    # ---- Build model with the (possibly) augmented pair_cfg ----
    model = DepthStack(B=B, L=L_used, S=S, tau=T0,
                    pair=pair_cfg,
                    gate_set=gate_set).to(device)


    opt_name = str(cfg["optimizer"]).lower()
    opt = (optim.RMSprop(model.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-8)
           if opt_name == "rmsprop" else
           optim.Adam(model.parameters(), lr=cfg["lr"]))
    steps = int(cfg["steps"]); regs_cfg = cfg["regs"]; anneal_cfg = cfg["anneal"]

    es_cfg = cfg.get("early_stop", {})
    use_es = bool(es_cfg.get("use", False))
    min_steps = int(es_cfg.get("min_steps", 0))
    check_every = int(es_cfg.get("check_every", 10))
    patience_checks = int(es_cfg.get("patience_checks", 3))
    metric = str(es_cfg.get("metric", "em")).lower()   # "em" or "row_acc"
    target = float(es_cfg.get("target", 1.0))
    ok_streak = 0

    last_dbg = None
    for step in range(steps):
        taus = make_async_taus(
            L=len(model.layers), step=step, total=steps,
            T0=anneal_cfg["T0"], Tmin=anneal_cfg["Tmin"],
            direction=anneal_cfg["direction"],
            schedule=anneal_cfg.get("schedule", "linear"),
            phase_scale=anneal_cfg.get("phase_scale", 0.4),
            start_frac=anneal_cfg.get("start_frac", 0.0),
        )
        # Use bandwidth scheduling if available (and useful for gate_set=16)
        if hasattr(model, "set_layer_taus_and_bandwidths") and str(cfg.get("gate_set","6")) == "16":
            s_cfg = cfg.get("sigma16", {})
            s_start = float(s_cfg.get("s_start", 0.25))
            s_end   = float(s_cfg.get("s_end",   0.10))
            model.set_layer_taus_and_bandwidths(taus, s_start=s_start, s_end=s_end)
        else:
            model.set_layer_taus(taus)

        opt.zero_grad()
        y_pred, dbg = model(X)
        last_dbg = (y_pred, dbg, taus)

        assert torch.all((y_true >= 0) & (y_true <= 1)), "Targets must be in [0,1]"
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

        # Early stopping
        if use_es and (step + 1) >= min_steps and ((step + 1) % check_every == 0):
            with torch.no_grad():
                row_acc, em = per_instance_metrics(y_true, y_pred)
                cur_val = float(em) if metric == "em" else float(row_acc)
                if cur_val >= target:
                    ok_streak += 1
                else:
                    ok_streak = 0
                if ok_streak >= patience_checks:
                    print(f"  [early-stop] step={step+1}, metric={metric}={cur_val:.3f} (target={target})")
                    break

    # Final eval & report
    with torch.no_grad():
        if last_dbg is None:
            y_pred, dbg = model(X)
            final_taus = make_async_taus(
                L=len(model.layers), step=steps, total=steps,
                T0=anneal_cfg["T0"], Tmin=anneal_cfg["Tmin"],
                direction=anneal_cfg["direction"],
                schedule=anneal_cfg.get("schedule", "linear"),
                phase_scale=anneal_cfg.get("phase_scale", 0.4),
                start_frac=anneal_cfg.get("start_frac", 0.0),
            )
        else:
            y_pred, dbg, final_taus = last_dbg

        row_acc, em = per_instance_metrics(y_true, y_pred)

        # composed symbolic readout -> normalize
        try:
            pred_expr_raw = compose_readout_full(B, dbg, final_taus, PRIMS_LIST)
        except Exception:
            pred_expr_raw = ""
        pred_expr = normalize_expr(pred_expr_raw)
        label_expr = normalize_expr(formula)

        # equivalence == EM via truth table
        equiv = em

        # simplicity / equality
        if pred_expr == label_expr:
            simpler = "same"
        else:
            p_char, p_tok = expr_complexity(pred_expr)
            l_char, l_tok = expr_complexity(label_expr)
            if (p_tok, p_char) < (l_tok, l_char):
                simpler = "pred"
            elif (l_tok, l_char) < (p_tok, p_char):
                simpler = "label"
            else:
                simpler = "tie"

    return {
        "B": B, "S": S, "L_used": L_used, "formula": label_expr,
        "row_acc": row_acc, "em": em, "equiv": equiv,
        "pred_expr": pred_expr, "label_expr": label_expr, "simpler": simpler,
        "truth_table": {
            "X": X.cpu().tolist(),
            "y_true": y_true.cpu().tolist(),
            "y_pred": y_pred.cpu().tolist()
        }
    }


# ---------- Dataset runner ----------

def run_dataset(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    device = _device(cfg); seed_all(cfg["seed"])
    # --- NEW: set global sigma16 behavior if using gate_set=16
    if str(cfg.get("gate_set", "6")) == "16":
        s16_cfg = cfg.get("sigma16", {})
        set_sigma16_mode(str(s16_cfg.get("mode", "rbf")))
        set_sigma16_radius(float(s16_cfg.get("radius", 0.75)))
        # bandwidth only matters in mode='rbf'; we still keep the per-layer schedule below

    insts = T.load_instances_jsonl(cfg["dataset"])

    # L strategy (row / max / fixed)
    global_L: Optional[int] = None
    l_strategy = "row" if cfg.get("use_row_L", True) else (
        "max" if cfg.get("use_max_L", False) else "fixed"
    )
    if l_strategy == "max":
        global_L = max(int(inst.get("L", cfg["L"])) for inst in insts)
        print(f"[info] Using max L across dataset: L_max = {global_L}")

    results = []
    for idx, inst in enumerate(insts):
        L_override = None
        if l_strategy == "row" and ("L" in inst):
            L_override = int(inst["L"])
        elif l_strategy == "max":
            L_override = int(global_L)

        res = train_single_instance(device, cfg, inst, L_override=L_override)
        results.append(res)
        print(f"[{idx+1}/{len(insts)}] B={res['B']} S={res['S']} L={res['L_used']}  "
              f"acc={res['row_acc']:.3f}  EM={res['em']}  simpler={res['simpler']}")
        print(f"   label: {res['label_expr']}")
        print(f"   pred : {res['pred_expr']}")

    # Aggregates (macro averages across instances)
    n = max(1, len(results))
    avg_row_acc = sum(r["row_acc"] for r in results) / n
    em_rate     = sum(r["em"]      for r in results) / n
    equiv_rate  = sum(r["equiv"]   for r in results) / n

    # Histograms / breakdowns
    from collections import Counter
    L_used_hist = Counter(r["L_used"] for r in results)
    S_hist      = Counter(r["S"]      for r in results)
    B_hist      = Counter(r["B"]      for r in results)
    simpler_hist = Counter(r["simpler"] for r in results)
    # Ensure all keys are present
    for k in ["pred", "label", "tie", "same"]:
        simpler_hist.setdefault(k, 0)
    simpler_ratios = {k: simpler_hist[k] / n for k in ["pred", "label", "tie", "same"]}

    print(f"\nSimpler ratios: "
          f"pred={simpler_ratios['pred']:.3f}, "
          f"label={simpler_ratios['label']:.3f}, "
          f"tie={simpler_ratios['tie']:.3f}, "
          f"same={simpler_ratios['same']:.3f}")

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

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": cfg_eff,
        "l_strategy": l_strategy,
        "avg_row_acc": avg_row_acc,
        "em_rate": em_rate,
        "equiv_rate": equiv_rate,
        "hist": {
            "L_used": dict(L_used_hist),
            "S": dict(S_hist),
            "B": dict(B_hist),
        },
        "simpler": {
            "counts": {k: simpler_hist[k] for k in ["pred", "label", "tie", "same"]},
            "ratios": simpler_ratios,
        },
        "results": results
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {out_dir / 'summary.json'}")
    return summary



# ---------- Single-task fallback (B=2) ----------

def run_single(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    device = _device(cfg); seed_all(cfg["seed"])
    
    # --- NEW: set global sigma16 behavior if using gate_set=16
    if str(cfg.get("gate_set", "6")) == "16":
        s16_cfg = cfg.get("sigma16", {})
        set_sigma16_mode(str(s16_cfg.get("mode", "rbf")))
        set_sigma16_radius(float(s16_cfg.get("radius", 0.75)))

    X, y_true = T.make_truth_table(cfg["task"])
    X = X.to(device); y_true = y_true.to(device)

    L, S = int(cfg["L"]), int(cfg["S"])    
    #model = DepthStack(B=2, L=L, S=S, tau=cfg["anneal"]["T0"], pair=cfg.get("pair", None)).to(device)
    model = DepthStack(B=2, L=L, S=S, tau=cfg["anneal"]["T0"],
                   pair=cfg.get("pair", None),
                   gate_set=str(cfg.get("gate_set","6"))).to(device)

    opt = (optim.RMSprop(model.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-8)
           if cfg["optimizer"].lower() == "rmsprop" else
           optim.Adam(model.parameters(), lr=cfg["lr"]))
    steps = int(cfg["steps"]); regs_cfg = cfg["regs"]; anneal_cfg = cfg["anneal"]

    for step in range(steps):
        taus = make_async_taus(
            L=len(model.layers), step=step, total=steps,
            T0=anneal_cfg["T0"], Tmin=anneal_cfg["Tmin"],
            direction=anneal_cfg["direction"],
            schedule=anneal_cfg.get("schedule", "linear"),
            phase_scale=anneal_cfg.get("phase_scale", 0.4),
            start_frac=anneal_cfg.get("start_frac", 0.0),
        )
        if hasattr(model, "set_layer_taus_and_bandwidths") and str(cfg.get("gate_set","6")) == "16":
            s_cfg = cfg.get("sigma16", {})
            s_start = float(s_cfg.get("s_start", 0.25))
            s_end   = float(s_cfg.get("s_end",   0.10))
            model.set_layer_taus_and_bandwidths(taus, s_start=s_start, s_end=s_end)
        else:
            model.set_layer_taus(taus)

        opt.zero_grad()
        y_pred, dbg = model(X)
        assert torch.all((y_true >= 0) & (y_true <= 1)), "Targets must be in [0,1]"
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

    with torch.no_grad():
        y_pred, _ = model(X)
        row_acc, em = per_instance_metrics(y_true, y_pred)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"config": cfg, "row_acc": row_acc, "em": em}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {out_dir / 'summary.json'}")
    return summary


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="experiments/results/run")
    ap.add_argument("--dataset", type=str, default=None, help="JSONL with B,S,L,formula per line (overrides config.dataset)")
    ap.add_argument("--use_row_L", action="store_true", help="Use per-row L when present (default if not overridden in config)")
    ap.add_argument("--use_max_L", action="store_true", help="Use max L across dataset for all rows (ignored if --use_row_L)")
    ap.add_argument("--no_row_L", action="store_true", help="Disable per-row L even if present")
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
        run_dataset(cfg, out_dir)
    else:
        run_single(cfg, out_dir)


if __name__ == "__main__":
    main()
