# src/ubcircuit/tasks.py
# Dataset helpers for arbitrary B-bit formulas.
# Supports:
#   (1) SOP-style formulas over {~, &, |, (, ), 0/1, a0..a{B-1}}
#   (2) Advanced macros:
#         PARITY(x1,x2,...)
#         MAJ(x1,x2,...)          (>= ceil(n/2))
#         THR(k, x1,x2,...)       (>= k)
#       where each arg xi may be:
#         - aK
#         - 0 or 1
#         - a SOP sub-expression like (a0 & (~a1))
#
# NOTE: this module intentionally does NOT do algebraic simplification.

from __future__ import annotations

from typing import Dict, List, Tuple
import json, re
from pathlib import Path

import torch


# ----------------- Regex -----------------

VAR_RE = re.compile(r"a(\d+)")  # matches a0, a1, ...
_MACRO_PARITY = re.compile(r"^\s*PARITY\s*\((.*)\)\s*$", re.IGNORECASE)
_MACRO_MAJ    = re.compile(r"^\s*MAJ\s*\((.*)\)\s*$", re.IGNORECASE)
_MACRO_THR    = re.compile(r"^\s*THR\s*\(\s*(\d+)\s*,(.*)\)\s*$", re.IGNORECASE)


# ----------------- Token checks (SOP) -----------------

def _logical_not_rewrite(formula: str) -> str:
    """
    Replace "~a3" -> "(1 - a3)" everywhere to ensure outputs in {0,1}.
    This is only applied to SOP fragments (not to macro heads).
    """
    return re.sub(r"~\s*a(\d+)", r"(1 - a\1)", formula)


def _check_tokens_sop(formula_after_not_rewrite: str) -> None:
    """
    Allow only: whitespace, parentheses, &, |, '-', digits 0/1, and variables a\d+.
    """
    tmp = re.sub(r"\s+", "", formula_after_not_rewrite)   # remove whitespace
    tmp = re.sub(r"a\d+", "", tmp)                        # strip variables a0..aN
    tmp = re.sub(r"[()&|\-01]", "", tmp)                  # allowed single-char tokens
    if tmp != "":
        raise ValueError(f"Unsupported tokens in formula: {formula_after_not_rewrite}")


def _eval_sop_on_assign(formula: str, assign: List[int]) -> int:
    """
    Evaluate SOP formula with {~,&,|,(,)} over 0/1 variables a0..a{B-1}.
    Ensures output is in {0,1}.
    """
    rew = _logical_not_rewrite(formula)
    _check_tokens_sop(rew)
    env = {f"a{i}": int(assign[i]) for i in range(len(assign))}
    val = eval(rew, {"__builtins__": {}}, env)
    return int(1 if int(val) != 0 else 0)


# ----------------- Macro arg splitting -----------------

def _split_args(arg_str: str) -> List[str]:
    """
    Split macro argument string by commas at paren-depth 0.
    Example: "a0, (a1 & (~a2)), 1" -> ["a0", "(a1 & (~a2))", "1"]
    """
    s = arg_str.strip()
    out, buf, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1; buf.append(ch)
        elif ch == ")":
            depth -= 1; buf.append(ch)
        elif ch == "," and depth == 0:
            t = "".join(buf).strip()
            if t:
                out.append(t)
            buf = []
        else:
            buf.append(ch)
    t = "".join(buf).strip()
    if t:
        out.append(t)
    return out


def _eval_atom_or_sop(token: str, assign: List[int]) -> int:
    """
    Macro arg evaluator:
      - "aK" -> assign[K]
      - "0"/"1" -> 0/1
      - otherwise treat as SOP sub-expression
    """
    t = token.strip()
    if re.fullmatch(r"a\d+", t):
        k = int(t[1:])
        if k < 0 or k >= len(assign):
            raise ValueError(f"Variable {t} out of range for B={len(assign)}")
        return int(assign[k])
    if t == "0":
        return 0
    if t == "1":
        return 1
    # SOP sub-expression
    return _eval_sop_on_assign(t, assign)


# ----------------- Main evaluator (SOP + macros) -----------------

def _eval_formula_on_assign(formula: str, assign: List[int]) -> int:
    """
    Evaluate either:
      - macro formula (PARITY/MAJ/THR) at the top-level, or
      - SOP formula otherwise.

    Returns 0/1.
    """
    s = formula.strip()

    m = _MACRO_PARITY.match(s)
    if m:
        args = _split_args(m.group(1))
        if len(args) == 0:
            raise ValueError("PARITY() needs at least one argument.")
        vals = [_eval_atom_or_sop(a, assign) for a in args]
        return int(sum(vals) & 1)

    m = _MACRO_MAJ.match(s)
    if m:
        args = _split_args(m.group(1))
        if len(args) == 0:
            raise ValueError("MAJ() needs at least one argument.")
        vals = [_eval_atom_or_sop(a, assign) for a in args]
        thr = (len(vals) + 1) // 2  # ceil(n/2)
        return int(1 if sum(vals) >= thr else 0)

    m = _MACRO_THR.match(s)
    if m:
        k = int(m.group(1))
        args = _split_args(m.group(2))
        if len(args) == 0:
            raise ValueError("THR(k, ...) needs at least one argument after k.")
        vals = [_eval_atom_or_sop(a, assign) for a in args]
        return int(1 if sum(vals) >= k else 0)

    # default: SOP
    return _eval_sop_on_assign(s, assign)


# ----------------- Assignment grid -----------------

def all_assignments(B: int) -> torch.Tensor:
    """
    Return (2^B, B) tensor listing all {0,1} assignments in lexicographic order
    matching your old convention:
      bits = [(m >> (B - 1 - i)) & 1 for i in range(B)]
    """
    rows = []
    for m in range(1 << B):
        bits = [(m >> (B - 1 - i)) & 1 for i in range(B)]
        rows.append(bits)
    return torch.tensor(rows, dtype=torch.float32)


def truth_table_from_formula(B: int, formula: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: (2^B, B)
      y: (2^B, 1)
    """
    X = all_assignments(B)
    y = torch.tensor(
        [[_eval_formula_on_assign(formula, X[i].int().tolist())] for i in range(X.size(0))],
        dtype=torch.float32,
    )
    return X, y


# ----------------- JSONL helpers -----------------

def load_instances_jsonl(path: str) -> List[Dict]:
    """
    Each line: {"B": int, "S": int, "formula": "...", ...}
    """
    path = Path(path)
    instances = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    return instances


def batch_from_instances(insts: List[Dict]):
    """
    Converts instances to per-instance truth tables (no cross-instance batching since 2^B varies).
    Returns lists aligned by index:
      X_list[i] : (2^B_i, B_i)
      y_list[i] : (2^B_i, 1)
      B_list[i] : int
      formula_list[i] : str
      S_list[i] : int (desired width)
    """
    Xs, ys, Bs, Fs, Ss = [], [], [], [], []
    for inst in insts:
        B = int(inst["B"])
        S = int(inst["S"])
        f = str(inst["formula"])
        X, y = truth_table_from_formula(B, f)
        Xs.append(X); ys.append(y); Bs.append(B); Fs.append(f); Ss.append(S)
    return Xs, ys, Bs, Fs, Ss


# ----------------- Backward-compatible tiny 2-bit fallback -----------------

def make_truth_table(task: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if task.replace(" ", "") in {"(a&b)|(~a)", "or_na"}:
        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[1.], [1.], [0.], [1.]])
        return X, y
    raise ValueError(f"Unknown task name: {task}")


def subgoals_for_task(task: str, X: torch.Tensor) -> Dict[str, torch.Tensor]:
    # Only for 2-bit classic fallback
    if X.size(1) != 2:
        return {}
    a, b = X[:, 0:1], X[:, 1:2]
    AND = a * b
    NOTA = 1.0 - a
    OR = torch.maximum(AND, NOTA)
    return {"AND": AND, "NOTA": NOTA, "OR": OR}
