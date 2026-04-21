# src/ubcircuit/train.py
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import yaml

from .modules import DepthStack
from .readout import compose_readout_expr, decoded_readout_metrics, normalize_expr
from .utils import seed_all, stable_seed, safe_bce, make_async_taus, regularizers_bundle
from . import tasks as T
from .boolean_prims16 import set_sigma16_mode, set_sigma16_radius
from .mi import top_mi_pairs, priors_from_pairs


DEFAULT_CFG: Dict[str, Any] = {
    "task": "(a&b)|(~a)",
    "L": 2,
    "S": 2,
    "gate_set": "6",   # "6" or "16"
    "sigma16": {
        "s_start": 0.25,
        "s_end": 0.10,
        "mode": "rbf",
        "radius": 0.75,
    },
    "lifting": {"use": True, "factor": 2.0},
    "relaxation": {
        "mode": "softmax",       # "softmax" | "gumbel" | "argmax_ste"
        "hard": False,           # straight-through one-hot for gumbel/argmax_ste training
        "gumbel_tau": 1.0,
        "eval_hard": False,      # deterministic one-hot during no-grad/eval forwards
    },
    "steps": 1200,
    "lr": 0.05,
    "optimizer": "rmsprop",
    "seed": 2024,
    "device": "auto",

    # L selection strategy (existing)
    "use_row_L": True,
    "use_max_L": False,

    # NEW: dataset-driven scaling for S and L (with parameter k)
    # S_used = { identity(S_base), S_base + k, S_base * k }
    # L_used = { identity(L_base), L_base + k, L_base * k }
    "scale": {
        "use_row_S": True,     # if dataset row has "S", use it as base; else cfg["S"]
        "S_op": "identity",    # "identity" | "add" | "mul"
        "S_k": 2,              # used only if op != identity
        "S_min": 2,
        "S_max": 64,

        "L_op": "identity",    # "identity" | "add" | "mul"
        "L_k": 1,              # used only if op != identity
        "L_min": 2,
        "L_max": 16,
    },

    "anneal": {
        "T0": 0.35, "Tmin": 0.12,
        "direction": "top_down",
        "schedule": "linear",
        "phase_scale": 0.4,
        "start_frac": 0.0
    },
    "regs": {
        "lam_entropy": 1.0e-3,
        "lam_div_units": 5.0e-4,
        "lam_div_rows": 5.0e-4,
        "lam_const16": 5.0e-3,
    },
    "pair": {
        "repel": True,
        "eta": 1.0,
        "mode": "hard-log",
        "route": "learned",   # "learned" | "mi_soft" | "mi_hard"
    },
    "aux": {"use": False, "lam": 1.0e-2, "wire_targets": ["AND", "NOTA"]},
    "dataset": None,
    "early_stop": {
        "use": True,
        "metric": "decoded_em",
        "target": 1.0,
        "min_steps": 100,
        "check_every": 10,
        "patience_checks": 3
    },
}


def load_config(path: str | None) -> Dict[str, Any]:
    cfg = DEFAULT_CFG.copy()
    if path:
        user = yaml.safe_load(open(path)) or {}
        for k, v in user.items():
            if isinstance(v, dict) and k in {"anneal", "regs", "aux", "early_stop", "pair", "sigma16", "lifting", "scale", "relaxation"}:
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    return cfg


def _device(cfg) -> torch.device:
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def configure_torch_threads_from_env() -> None:
    """Let CC Slurms pin PyTorch threads for tiny per-instance trainings."""
    raw = (
        os.environ.get("UBC_TORCH_THREADS")
        or os.environ.get("UBC_NUM_THREADS")
        or os.environ.get("OMP_NUM_THREADS")
    )
    if raw:
        try:
            n = max(1, int(raw))
            torch.set_num_threads(n)
        except Exception:
            pass
    print(f"[info] torch_num_threads={torch.get_num_threads()}")


# ---------- Metrics ----------
def per_instance_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, int]:
    yb = (y_pred >= 0.5).float()
    row_acc = (yb.eq(y_true)).float().mean().item()
    em = int(torch.all(yb.eq(y_true)).item())
    return row_acc, em


def configure_model_relaxation(model: torch.nn.Module, cfg: Dict[str, Any]) -> None:
    rel = cfg.get("relaxation", {}) or {}
    if hasattr(model, "set_relaxation"):
        model.set_relaxation(
            mode=str(rel.get("mode", "softmax")),
            hard=bool(rel.get("hard", False)),
            gumbel_tau=float(rel.get("gumbel_tau", 1.0)),
            eval_hard=bool(rel.get("eval_hard", False)),
        )


def set_model_schedule(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    gate_set: str,
    step: int,
    total: int,
) -> List[float]:
    anneal_cfg = cfg["anneal"]
    taus = make_async_taus(
        L=len(model.layers), step=step, total=total,
        T0=anneal_cfg["T0"], Tmin=anneal_cfg["Tmin"],
        direction=anneal_cfg["direction"],
        schedule=anneal_cfg.get("schedule", "linear"),
        phase_scale=anneal_cfg.get("phase_scale", 0.4),
        start_frac=anneal_cfg.get("start_frac", 0.0),
    )
    if hasattr(model, "set_layer_taus_and_bandwidths") and gate_set == "16":
        s_cfg = cfg.get("sigma16", {})
        s_start = float(s_cfg.get("s_start", 0.25))
        s_end = float(s_cfg.get("s_end", 0.10))
        model.set_layer_taus_and_bandwidths(taus, s_start=s_start, s_end=s_end)
    else:
        model.set_layer_taus(taus)
    return taus


def decode_from_dbg(
    B: int,
    y_true: torch.Tensor,
    dbg: List[tuple],
    taus: List[float],
    PRIMS: List[str],
    lift_W: torch.Tensor | None = None,
) -> Tuple[str, float, int]:
    pred_expr_raw = compose_readout_expr(B, dbg, taus, PRIMS, lift_W=lift_W)
    pred_expr = normalize_expr(pred_expr_raw)
    decoded_row_acc, decoded_em = decoded_readout_metrics(B, pred_expr_raw, y_true)
    return pred_expr, decoded_row_acc, decoded_em


def early_stop_metric_value(
    metric: str,
    *,
    row_acc: float,
    em: int,
    decoded_row_acc: float,
    decoded_em: int,
) -> float:
    metric = str(metric or "decoded_em").lower()
    if metric in {"em", "soft_em"}:
        return float(em)
    if metric in {"row_acc", "acc", "soft_row_acc"}:
        return float(row_acc)
    if metric in {"decoded_em", "hard_em", "readout_em"}:
        return float(decoded_em)
    if metric in {"decoded_row_acc", "decoded_acc", "hard_row_acc", "readout_row_acc"}:
        return float(decoded_row_acc)
    if metric in {"both_em", "soft_and_decoded_em"}:
        return float(min(int(em), int(decoded_em)))
    raise ValueError(
        f"Unknown early_stop.metric={metric!r}. "
        "Use em|row_acc|decoded_em|decoded_row_acc|both_em."
    )


def _dist_stats(prefix: str, probs: List[torch.Tensor]) -> Dict[str, float]:
    rows = []
    for p in probs:
        if not isinstance(p, torch.Tensor) or p.numel() == 0:
            continue
        q = p.detach().float()
        if q.dim() == 1:
            q = q.unsqueeze(0)
        rows.append(q.reshape(-1, q.shape[-1]))
    if not rows:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_entropy_mean": float("nan"),
            f"{prefix}_entropy_max": float("nan"),
            f"{prefix}_max_prob_mean": float("nan"),
            f"{prefix}_max_prob_min": float("nan"),
            f"{prefix}_margin_mean": float("nan"),
            f"{prefix}_margin_min": float("nan"),
        }
    q = torch.cat(rows, dim=0).clamp_min(1e-12)
    ent = -(q * q.log()).sum(dim=-1)
    top = torch.topk(q, k=min(2, q.shape[-1]), dim=-1).values
    max_prob = top[:, 0]
    margin = top[:, 0] - (top[:, 1] if top.shape[-1] > 1 else 0.0)
    return {
        f"{prefix}_count": int(q.shape[0]),
        f"{prefix}_entropy_mean": float(ent.mean().item()),
        f"{prefix}_entropy_max": float(ent.max().item()),
        f"{prefix}_max_prob_mean": float(max_prob.mean().item()),
        f"{prefix}_max_prob_min": float(max_prob.min().item()),
        f"{prefix}_margin_mean": float(margin.mean().item()),
        f"{prefix}_margin_min": float(margin.min().item()),
    }


def ubc_diagnostics_from_dbg(
    dbg: List[tuple],
    taus: List[float],
    *,
    y_pred: torch.Tensor | None = None,
    y_true: torch.Tensor | None = None,
) -> Dict[str, Any]:
    gate_probs: List[torch.Tensor] = []
    row_probs: List[torch.Tensor] = []
    pair_probs: List[torch.Tensor] = []
    for li, (_outs, Lrows, unitWs, PL, PR) in enumerate(dbg):
        tau = float(taus[li])
        row_probs.append(Lrows)
        for W in unitWs:
            gate_probs.append(torch.softmax(W.detach() / max(tau, 1e-8), dim=0))
        if isinstance(PL, torch.Tensor):
            pair_probs.append(PL)
        if isinstance(PR, torch.Tensor):
            pair_probs.append(PR)

    out: Dict[str, Any] = {
        "tau_min": float(min(taus)) if taus else float("nan"),
        "tau_max": float(max(taus)) if taus else float("nan"),
    }
    out.update(_dist_stats("gate", gate_probs))
    out.update(_dist_stats("row", row_probs))
    out.update(_dist_stats("pair", pair_probs))

    if y_pred is not None:
        margin = (y_pred.detach().float() - 0.5).abs()
        out["output_margin_mean"] = float(margin.mean().item())
        out["output_margin_min"] = float(margin.min().item())
    if y_pred is not None and y_true is not None:
        out["bce"] = float(safe_bce(y_pred.detach().float(), y_true.detach().float()).item())
    return out


def evaluate_ubc_model(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    *,
    B: int,
    X: torch.Tensor,
    y_true: torch.Tensor,
    gate_set: str,
    PRIMS: List[str],
    step: int,
    total: int,
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        taus = set_model_schedule(model, cfg, gate_set, step=step, total=total)
        y_pred, dbg = model(X)
        row_acc, em = per_instance_metrics(y_true, y_pred)
        lift_W = model.lift.W.detach() if getattr(model, "lift", None) is not None else None
        pred_expr, decoded_row_acc, decoded_em = decode_from_dbg(B, y_true, dbg, taus, PRIMS, lift_W=lift_W)
        diagnostics = ubc_diagnostics_from_dbg(dbg, taus, y_pred=y_pred, y_true=y_true)
    if was_training:
        model.train()
    return {
        "y_pred": y_pred,
        "dbg": dbg,
        "taus": taus,
        "row_acc": row_acc,
        "em": em,
        "pred_expr": pred_expr,
        "decoded_row_acc": decoded_row_acc,
        "decoded_em": decoded_em,
        "diagnostics": diagnostics,
    }


# ---------- Lifting: effective input width (must match modules.py) ----------
def compute_B_effective(B: int, use_lifting: bool, lift_factor: float) -> int:
    B = int(B)
    if not use_lifting:
        return B
    return max(B, int(math.ceil(float(lift_factor) * B)))


# ---------- NEW: scaling ops with parameter k ----------
def _apply_op(x: int, op: str, k: int) -> int:
    op = str(op).strip().lower()
    x = int(x)
    k = int(k)
    if op == "identity":
        return x
    if op == "add":
        return x + k
    if op == "mul":
        return x * k
    raise ValueError(f"Unknown op '{op}'. Use identity|add|mul.")

def _clamp_int(x: int, lo: int | None, hi: int | None) -> int:
    y = int(x)
    if lo is not None:
        y = max(int(lo), y)
    if hi is not None:
        y = min(int(hi), y)
    return y


def resolve_S_L_used(cfg: Dict[str, Any], inst: Dict[str, Any], L_override: Optional[int]) -> Tuple[int, int, int, int]:
    """
    Returns: (S_base, L_base, S_used, L_used)
    where base comes from dataset row (if enabled) and used comes after scaling + clamp.
    """
    scale = cfg.get("scale", {}) or {}

    # ----- Base L -----
    if L_override is not None:
        L_base = int(L_override)
    elif bool(cfg.get("use_row_L", True)) and ("L" in inst):
        L_base = int(inst["L"])
    else:
        L_base = int(cfg["L"])

    # ----- Base S -----
    if bool(scale.get("use_row_S", True)) and ("S" in inst):
        S_base = int(inst["S"])
    else:
        S_base = int(cfg["S"])

    # ----- Apply scaling -----
    S_used = _apply_op(S_base, scale.get("S_op", "identity"), scale.get("S_k", 1))
    L_used = _apply_op(L_base, scale.get("L_op", "identity"), scale.get("L_k", 1))

    # ----- Clamp -----
    S_used = _clamp_int(S_used, scale.get("S_min", 2), scale.get("S_max", 64))
    L_used = _clamp_int(L_used, scale.get("L_min", 2), scale.get("L_max", 16))

    # safety
    S_used = max(1, int(S_used))
    L_used = max(1, int(L_used))

    return S_base, L_base, S_used, L_used


# ---------- Robust MI routing ----------
def _tile_pairs(pairs: List[Tuple[int, int]], S: int) -> List[Tuple[int, int]]:
    if not pairs:
        return []
    reps = (S + len(pairs) - 1) // len(pairs)
    return (pairs * reps)[:S]


def _tile_rows(M: torch.Tensor, S: int) -> torch.Tensor:
    if M.size(0) == S:
        return M
    reps = (S + M.size(0) - 1) // M.size(0)
    return M.repeat(reps, 1)[:S]


def _expand_prior_to_Beff(P: torch.Tensor, B_eff: int, eps: float = 1e-6) -> torch.Tensor:
    if P.dim() != 2:
        raise ValueError(f"Expected 2D prior, got {tuple(P.shape)}")
    S, B_base = P.shape
    if B_eff == B_base:
        return P
    if B_eff < B_base:
        return P[:, :B_eff]
    out = torch.full((S, B_eff), float(eps), device=P.device, dtype=P.dtype)
    out[:, :B_base] += P
    out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return out


def resolve_pair_cfg(
    pair_cfg_in: Dict[str, Any],
    *,
    B: int,
    B_eff: int,
    S: int,
    X: torch.Tensor,
    y_true: torch.Tensor,
) -> Dict[str, Any]:
    pair_cfg = dict(pair_cfg_in or {})
    route = str(pair_cfg.get("route", "learned"))

    if route not in {"mi_soft", "mi_hard"}:
        return pair_cfg

    if B < 2:
        pair_cfg["route"] = "learned"
        pair_cfg.pop("PL_prior", None)
        pair_cfg.pop("PR_prior", None)
        pair_cfg.pop("fixed_pairs", None)
        return pair_cfg

    disjoint = bool(pair_cfg.get("mi_disjoint", True))
    pairs = top_mi_pairs(X, y_true, S=S, disjoint=disjoint)

    if not pairs:
        pair_cfg["route"] = "learned"
        pair_cfg.pop("PL_prior", None)
        pair_cfg.pop("PR_prior", None)
        pair_cfg.pop("fixed_pairs", None)
        return pair_cfg

    if route == "mi_hard":
        pair_cfg["fixed_pairs"] = _tile_pairs(pairs, S)
        pair_cfg.pop("PL_prior", None)
        pair_cfg.pop("PR_prior", None)
        return pair_cfg

    # mi_soft
    PLp, PRp = priors_from_pairs(pairs, B)  # (len(pairs), B)
    if PLp.size(0) != S:
        PLp = _tile_rows(PLp, S)
        PRp = _tile_rows(PRp, S)

    if int(B_eff) != int(B):
        PLp = _expand_prior_to_Beff(PLp, int(B_eff))
        PRp = _expand_prior_to_Beff(PRp, int(B_eff))

    pair_cfg["PL_prior"] = PLp.tolist()
    pair_cfg["PR_prior"] = PRp.tolist()
    pair_cfg.pop("fixed_pairs", None)
    return pair_cfg


# ---------- Gate usage extraction ----------
def extract_gate_usage_from_dbg(
    dbg: List[tuple],
    taus: List[float],
    PRIMS: List[str],
) -> Dict[str, Any]:
    path = Counter()
    allu = Counter()
    path_list: List[str] = []

    for li, layer_dbg in enumerate(dbg):
        _outs, Lrows, unitWs, _PL, _PR = layer_dbg
        tau = float(taus[li])

        for W in unitWs:
            p = torch.softmax(W / max(tau, 1e-8), dim=0)
            prim = PRIMS[int(p.argmax().item())]
            allu[prim] += 1

        if isinstance(Lrows, torch.Tensor) and Lrows.dim() == 2:
            out_bits = int(Lrows.shape[0])
            for k in range(out_bits):
                u_idx = int(Lrows[k].argmax().item())
                W = unitWs[u_idx]
                p = torch.softmax(W / max(tau, 1e-8), dim=0)
                prim = PRIMS[int(p.argmax().item())]
                path[prim] += 1
                path_list.append(prim)

    return {"path_counts": dict(path), "all_unit_counts": dict(allu), "path_list": path_list}


# ---------- Training (single instance) ----------
def train_single_instance(
    device,
    cfg: Dict[str, Any],
    inst: Dict[str, Any],
    L_override: Optional[int] = None,
) -> Dict[str, Any]:
    B = int(inst["B"])
    formula = str(inst["formula"])

    X, y_true = T.truth_table_from_formula(B, formula)
    X = X.to(device)
    y_true = y_true.to(device)

    # base + scaled S/L
    S_base, L_base, S_used, L_used = resolve_S_L_used(cfg, inst, L_override=L_override)

    anneal_cfg = cfg["anneal"]
    T0 = float(anneal_cfg["T0"])
    gate_set = str(cfg.get("gate_set", "6"))
    if gate_set == "16":
        from .boolean_prims16 import PRIMS16 as PRIMS_LIST
    else:
        from .boolean_prims import PRIMS as PRIMS_LIST

    # lifting
    lifting_cfg = cfg.get("lifting", {})
    use_lifting = bool(lifting_cfg.get("use", True))
    lift_factor = float(lifting_cfg.get("factor", 2.0))
    B_eff = compute_B_effective(B, use_lifting, lift_factor)

    # MI routing uses S_used
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
    configure_model_relaxation(model, cfg)

    opt_name = str(cfg["optimizer"]).lower()
    opt = (
        optim.RMSprop(model.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-8)
        if opt_name == "rmsprop"
        else optim.Adam(model.parameters(), lr=cfg["lr"])
    )

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

    last_step = 0
    for step in range(steps):
        last_step = step
        taus = set_model_schedule(model, cfg, gate_set, step=step, total=steps)

        opt.zero_grad()
        y_pred, dbg = model(X)

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
            eval_now = evaluate_ubc_model(
                model, cfg, B=B, X=X, y_true=y_true, gate_set=gate_set,
                PRIMS=PRIMS_LIST, step=step, total=steps,
            )
            cur_val = early_stop_metric_value(
                metric,
                row_acc=eval_now["row_acc"],
                em=eval_now["em"],
                decoded_row_acc=eval_now["decoded_row_acc"],
                decoded_em=eval_now["decoded_em"],
            )
            ok_streak = (ok_streak + 1) if (cur_val >= target) else 0
            if ok_streak >= patience_checks:
                print(f"  [early-stop] step={step+1}, metric={metric}={cur_val:.3f} (target={target})")
                break

    final_eval = evaluate_ubc_model(
        model, cfg, B=B, X=X, y_true=y_true, gate_set=gate_set,
        PRIMS=PRIMS_LIST, step=last_step, total=steps,
    )
    row_acc = final_eval["row_acc"]
    em = final_eval["em"]
    decoded_row_acc = final_eval["decoded_row_acc"]
    decoded_em = final_eval["decoded_em"]
    pred_expr = final_eval["pred_expr"]
    diagnostics = final_eval["diagnostics"]
    try:
        gate_usage = extract_gate_usage_from_dbg(final_eval["dbg"], final_eval["taus"], PRIMS_LIST)
    except Exception:
        gate_usage = {}

    return {
        "B": B,
        "S_base": S_base, "L_base": L_base,
        "S_used": S_used, "L_used": L_used,
        "row_acc": row_acc, "em": em,
        "decoded_row_acc": decoded_row_acc, "decoded_em": decoded_em,
        "formula": formula,
        "label_expr": normalize_expr(formula),
        "pred_expr": pred_expr,
        "gate_usage": gate_usage,
        "diagnostics": diagnostics,
        "train_steps": int(last_step + 1),
    }


# ---------- Dataset runner ----------
def run_dataset(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    device = _device(cfg)
    seed_all(cfg["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    if str(cfg.get("gate_set", "6")) == "16":
        s16_cfg = cfg.get("sigma16", {})
        set_sigma16_mode(str(s16_cfg.get("mode", "rbf")))
        set_sigma16_radius(float(s16_cfg.get("radius", 0.75)))

    insts = T.load_instances_jsonl(cfg["dataset"])

    global_L: Optional[int] = None
    l_strategy = "row" if cfg.get("use_row_L", True) else (
        "max" if cfg.get("use_max_L", False) else "fixed"
    )
    if l_strategy == "max":
        global_L = max(int(inst.get("L", cfg["L"])) for inst in insts)
        print(f"[info] Using max L across dataset: L_max = {global_L}")

    results = []
    results_path = out_dir / "results.jsonl"
    with results_path.open("w", buffering=1) as results_f:
        for idx, inst in enumerate(insts):
            seed_all(stable_seed(int(cfg["seed"]), "ubc", idx))
            L_override = None
            if l_strategy == "row" and ("L" in inst):
                L_override = int(inst["L"])
            elif l_strategy == "max":
                L_override = int(global_L)

            res = train_single_instance(device, cfg, inst, L_override=L_override)
            res_with_idx = {"idx": idx, **res}
            results.append(res_with_idx)
            results_f.write(json.dumps(res_with_idx) + "\n")
            results_f.flush()
            print(f"[{idx+1}/{len(insts)}] "
                  f"B={res['B']}  "
                  f"S:{res['S_base']}-> {res['S_used']}  "
                  f"L:{res['L_base']}-> {res['L_used']}  "
                  f"acc={res['row_acc']:.3f}  EM={res['em']}  "
                  f"decoded_acc={res['decoded_row_acc']:.3f}  decoded_EM={res['decoded_em']}")

    n = max(1, len(results))
    avg_row_acc = sum(r["row_acc"] for r in results) / n
    em_rate     = sum(r["em"]      for r in results) / n
    avg_decoded_row_acc = sum(r["decoded_row_acc"] for r in results) / n
    decoded_em_rate     = sum(r["decoded_em"]      for r in results) / n
    diag_keys = sorted({
        k
        for r in results
        for k, v in (r.get("diagnostics") or {}).items()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    })
    avg_diagnostics = {
        k: sum(float(r.get("diagnostics", {}).get(k, 0.0)) for r in results if k in r.get("diagnostics", {}))
        / max(1, sum(1 for r in results if k in r.get("diagnostics", {})))
        for k in diag_keys
    }

    summary = {
        "config": cfg,
        "l_strategy": l_strategy,
        "avg_row_acc": avg_row_acc,
        "em_rate": em_rate,
        "avg_decoded_row_acc": avg_decoded_row_acc,
        "decoded_em_rate": decoded_em_rate,
        "avg_diagnostics": avg_diagnostics,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {out_dir / 'summary.json'}")
    return summary


# ---------- Single-task fallback ----------
def run_single(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    device = _device(cfg)
    seed_all(cfg["seed"])

    if str(cfg.get("gate_set", "6")) == "16":
        s16_cfg = cfg.get("sigma16", {})
        set_sigma16_mode(str(s16_cfg.get("mode", "rbf")))
        set_sigma16_radius(float(s16_cfg.get("radius", 0.75)))

    X, y_true = T.make_truth_table(cfg["task"])
    X = X.to(device)
    y_true = y_true.to(device)

    # no dataset row -> base from cfg
    inst = {"B": 2, "formula": cfg["task"], "S": int(cfg["S"]), "L": int(cfg["L"])}
    S_base, L_base, S_used, L_used = resolve_S_L_used(cfg, inst, L_override=int(cfg["L"]))

    lifting_cfg = cfg.get("lifting", {})
    use_lifting = bool(lifting_cfg.get("use", True))
    lift_factor = float(lifting_cfg.get("factor", 2.0))
    B_eff = compute_B_effective(2, use_lifting, lift_factor)

    pair_cfg = dict(cfg.get("pair", {}) or {})
    pair_cfg = resolve_pair_cfg(pair_cfg, B=2, B_eff=B_eff, S=S_used, X=X, y_true=y_true)

    model = DepthStack(
        B=2, L=L_used, S=S_used, tau=cfg["anneal"]["T0"],
        pair=pair_cfg,
        gate_set=str(cfg.get("gate_set", "6")),
        use_lifting=use_lifting,
        lift_factor=lift_factor,
    ).to(device)
    gate_set = str(cfg.get("gate_set", "6"))
    if gate_set == "16":
        from .boolean_prims16 import PRIMS16 as PRIMS_LIST
    else:
        from .boolean_prims import PRIMS as PRIMS_LIST
    configure_model_relaxation(model, cfg)

    opt = (optim.RMSprop(model.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-8)
           if cfg["optimizer"].lower() == "rmsprop" else
           optim.Adam(model.parameters(), lr=cfg["lr"]))

    steps = int(cfg["steps"])
    regs_cfg = cfg["regs"]
    es_cfg = cfg.get("early_stop", {})
    use_es = bool(es_cfg.get("use", False))
    min_steps = int(es_cfg.get("min_steps", 0))
    check_every = int(es_cfg.get("check_every", 10))
    patience_checks = int(es_cfg.get("patience_checks", 3))
    metric = str(es_cfg.get("metric", "decoded_em")).lower()
    target = float(es_cfg.get("target", 1.0))
    ok_streak = 0

    last_step = 0
    for step in range(steps):
        last_step = step
        taus = set_model_schedule(model, cfg, gate_set, step=step, total=steps)

        opt.zero_grad()
        y_pred, dbg = model(X)

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
            eval_now = evaluate_ubc_model(
                model, cfg, B=2, X=X, y_true=y_true, gate_set=gate_set,
                PRIMS=PRIMS_LIST, step=step, total=steps,
            )
            cur_val = early_stop_metric_value(
                metric,
                row_acc=eval_now["row_acc"],
                em=eval_now["em"],
                decoded_row_acc=eval_now["decoded_row_acc"],
                decoded_em=eval_now["decoded_em"],
            )
            ok_streak = (ok_streak + 1) if (cur_val >= target) else 0
            if ok_streak >= patience_checks:
                print(f"  [early-stop] step={step+1}, metric={metric}={cur_val:.3f} (target={target})")
                break

    final_eval = evaluate_ubc_model(
        model, cfg, B=2, X=X, y_true=y_true, gate_set=gate_set,
        PRIMS=PRIMS_LIST, step=last_step, total=steps,
    )
    row_acc = final_eval["row_acc"]
    em = final_eval["em"]
    decoded_row_acc = final_eval["decoded_row_acc"]
    decoded_em = final_eval["decoded_em"]
    pred_expr = final_eval["pred_expr"]
    diagnostics = final_eval["diagnostics"]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": cfg,
        "row_acc": row_acc,
        "em": em,
        "decoded_row_acc": decoded_row_acc,
        "decoded_em": decoded_em,
        "label_expr": normalize_expr(cfg["task"]),
        "pred_expr": pred_expr,
        "S_used": S_used,
        "L_used": L_used,
        "diagnostics": diagnostics,
        "train_steps": int(last_step + 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {out_dir / 'summary.json'}")
    return summary


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="experiments/results/run")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--use_row_L", action="store_true")
    ap.add_argument("--use_max_L", action="store_true")
    ap.add_argument("--no_row_L", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    configure_torch_threads_from_env()
    cfg = load_config(args.config)
    if args.dataset is not None:
        cfg["dataset"] = args.dataset

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
