# src/ubcircuit/tasks.py
# Dataset helpers for arbitrary B-bit formulas and classic 2-bit fallback.

from __future__ import annotations
from typing import Dict, List, Tuple
import json, re
from pathlib import Path
import torch


VAR_RE = re.compile(r"a(\d+)")  # matches a0, a1, ...


def _logical_not_rewrite(formula: str) -> str:
    """
    Replace "~a3" -> "(1 - a3)" everywhere to ensure outputs in {0,1}.
    """
    return re.sub(r"~\s*a(\d+)", r"(1 - a\1)", formula)


def _check_tokens(formula: str) -> None:
    """
    Allow only: whitespace, parentheses, &, |, '-', digits 0/1, and variables a\d+ (after NOT rewrite).
    """
    tmp = re.sub(r"\s+", "", formula)      # remove whitespace
    tmp = re.sub(r"a\d+", "", tmp)         # strip variables a0..aN
    # remove allowed single-char tokens: parentheses, &, |, '-', and 0/1 digits
    tmp = re.sub(r"[()&|\-01]", "", tmp)
    if tmp != "":
        raise ValueError(f"Unsupported tokens in formula: {formula}")


def _eval_formula_on_assign(formula: str, assign: List[int]) -> int:
    """
    Evaluate formula with {~,&,|,(,)} over 0/1 variables a0..a{B-1}.
    Ensures output is in {0,1}.
    """
    rew = _logical_not_rewrite(formula)
    _check_tokens(rew)

    env = {f"a{i}": int(assign[i]) for i in range(len(assign))}
    val = eval(rew, {"__builtins__": {}}, env)  # uses &, |, (), ints in env
    return int(1 if val != 0 else 0)


def all_assignments(B: int) -> torch.Tensor:
    """Return (2^B, B) tensor listing all {0,1} assignments in lexicographic order."""
    rows = []
    for m in range(1 << B):
        bits = [(m >> (B - 1 - i)) & 1 for i in range(B)]
        rows.append(bits)
    return torch.tensor(rows, dtype=torch.float32)  # (2^B, B)


def truth_table_from_formula(B: int, formula: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns X:(2^B,B) and y:(2^B,1) for given formula string.
    """
    X = all_assignments(B)
    y = torch.tensor(
        [[_eval_formula_on_assign(formula, X[i].int().tolist())] for i in range(X.size(0))],
        dtype=torch.float32,
    )
    return X, y


def load_instances_jsonl(path: str) -> List[Dict]:
    """
    Each line: {"B": int, "S": int, "formula": "..." }
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
        B = int(inst["B"]); S = int(inst["S"]); f = str(inst["formula"])
        X, y = truth_table_from_formula(B, f)
        Xs.append(X); ys.append(y); Bs.append(B); Fs.append(f); Ss.append(S)
    return Xs, ys, Bs, Fs, Ss


# ---- Backward-compatible tiny task (2-bit) ----

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
