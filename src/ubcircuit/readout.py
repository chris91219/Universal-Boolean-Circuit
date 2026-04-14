from __future__ import annotations

import re
from typing import List, Tuple

import torch

from . import tasks as T


_NOT_A_BARE = re.compile(r"\(\s*1\s*-\s*a(\d+)\s*\)")
_NOT_PARENS_ANY = re.compile(r"\(\s*1\s*-\s*\(\s*(.+?)\s*\)\s*\)")
_TILDE_LIT_PARENS = re.compile(r"\(~\(\s*a(\d+)\s*\)\)")


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
    num_ops = s.count("&") + s.count("|") + s.count("~")
    return char_len, num_vars + num_ops


def _argmax_unit_primitive(unitW: torch.Tensor, tau: float, PRIMS: List[str]) -> str:
    p = torch.softmax(unitW / max(tau, 1e-8), dim=0)
    return PRIMS[int(p.argmax().item())]


def _argmax_row_pick(L_row: torch.Tensor) -> int:
    return int(L_row.argmax().item())


def _not_expr(expr: str) -> str:
    return f"(1 - ({expr}))"


def _lifted_base_symbols(B: int, lift_W: torch.Tensor) -> List[str]:
    syms: List[str] = []
    for row in lift_W:
        idx = int(row.argmax().item())
        if idx < B:
            syms.append(f"a{idx}")
        else:
            syms.append(_not_expr(f"a{idx - B}"))
    return syms


def _apply_prim_to_syms(prim: str, a_sym: str, b_sym: str) -> str:
    if prim.startswith("AND"):
        return f"({a_sym} & {b_sym})"
    if prim.startswith("OR"):
        return f"({a_sym} | {b_sym})"
    if prim.startswith("NOT(a)"):
        return _not_expr(a_sym)
    if prim.startswith("NOT(b)"):
        return _not_expr(b_sym)
    if prim.startswith("a (skip)"):
        return f"{a_sym}"
    if prim.startswith("b (skip)"):
        return f"{b_sym}"

    if prim == "FALSE":
        return "0"
    if prim == "TRUE":
        return "1"
    if prim == "A":
        return f"{a_sym}"
    if prim == "B":
        return f"{b_sym}"
    if prim == "~A":
        return _not_expr(a_sym)
    if prim == "~B":
        return _not_expr(b_sym)
    if prim == "AND":
        return f"({a_sym} & {b_sym})"
    if prim == "OR":
        return f"({a_sym} | {b_sym})"
    if prim == "A&~B":
        return f"({a_sym} & {_not_expr(b_sym)})"
    if prim == "~A&B":
        return f"({_not_expr(a_sym)} & {b_sym})"
    if prim == "A|~B":
        return f"({a_sym} | {_not_expr(b_sym)})"
    if prim == "~A|B":
        return f"({_not_expr(a_sym)} | {b_sym})"
    if prim == "NAND":
        return _not_expr(f"({a_sym} & {b_sym})")
    if prim == "NOR":
        return _not_expr(f"({a_sym} | {b_sym})")
    if prim == "XOR":
        return f"(({a_sym} & {_not_expr(b_sym)}) | ({_not_expr(a_sym)} & {b_sym}))"
    if prim == "XNOR":
        return _not_expr(f"(({a_sym} & {_not_expr(b_sym)}) | ({_not_expr(a_sym)} & {b_sym}))")

    return f"({a_sym} | {b_sym})"


def compose_readout_expr(
    B: int,
    dbg: List[tuple],
    final_taus: List[float],
    PRIMS: List[str],
    lift_W: torch.Tensor | None = None,
) -> str:
    base_syms = _lifted_base_symbols(B, lift_W) if lift_W is not None else [f"a{i}" for i in range(B)]
    base_width = len(base_syms)

    outs0, Lrows0, unitWs0, PL0, PR0 = dbg[0]
    tau0 = float(final_taus[0])

    if (PL0 is None) or (PR0 is None) or (not isinstance(PL0, torch.Tensor)) or (PL0.size(-1) != base_width):
        if base_width >= 2:
            fallback_pair = (base_syms[0], base_syms[1])
        elif base_width == 1:
            fallback_pair = (base_syms[0], base_syms[0])
        else:
            fallback_pair = ("a0", "a1")
        pair_syms = [fallback_pair for _ in range(Lrows0.shape[1])]
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


def decoded_readout_metrics(B: int, pred_expr_raw: str, y_true: torch.Tensor) -> Tuple[float, int]:
    if not pred_expr_raw:
        return 0.0, 0
    _, y_readout = T.truth_table_from_formula(B, pred_expr_raw)
    y_readout = y_readout.to(device=y_true.device, dtype=y_true.dtype)
    matches = y_readout.eq(y_true)
    row_acc = matches.float().mean().item()
    em = int(torch.all(matches).item())
    return row_acc, em
