#!/usr/bin/env python3
# scripts/gen_dataset.py
# Generate Boolean-formula datasets over variables a0..a{B-1} using only ~, &, | and ().
# Each row includes: B, S (top-level OR arity, at least 2), L (heuristic depth), and formula.
#
# Heuristics:
#   - S := max(2, OR-arity-under-associativity(formula))   # flatten nested OR-chains
#   - L := 2 if S<=2 else 3                                # enough for 2-wire carry design
#
# Examples:
#   python scripts/gen_dataset.py --out data/bench_default.jsonl
#   python scripts/gen_dataset.py --grid hard --out data/bench_hard.jsonl --n_per_B 150
#   python scripts/gen_dataset.py --B 2,4,6,8 --out data/custom.jsonl --n_per_B 100 --seed 123

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import List

# ----------------- token & structure helpers -----------------

ALLOWED = re.compile(r'^[()\sa\d~&|]+$')  # only (, ), space, a, digits, ~, &, |
_VAR_RE = re.compile(r"a(\d+)")

def infer_B(formula: str) -> int:
    idxs = [int(m.group(1)) for m in _VAR_RE.finditer(formula)]
    return (max(idxs) + 1) if idxs else 0

def strip_outer_parens(s: str) -> str:
    """Strip fully-wrapping parentheses repeatedly: '(((x)))' -> 'x'."""
    s = s.strip()
    while s and s[0] == '(' and s[-1] == ')':
        depth = 0
        closed_at_end = True
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    closed_at_end = False
                    break
        if closed_at_end:
            s = s[1:-1].strip()
        else:
            break
    return s

def split_top_level_or(expr: str) -> List[str]:
    """Split by '|' at depth 0 (balanced parentheses), after stripping outer parens."""
    expr = strip_outer_parens(expr)
    terms, buf, depth = [], [], 0
    for ch in expr:
        if ch == '(':
            depth += 1; buf.append(ch)
        elif ch == ')':
            depth -= 1; buf.append(ch)
        elif ch == '|' and depth == 0:
            t = ''.join(buf).strip()
            if t: terms.append(t)
            buf = []
        else:
            buf.append(ch)
    if buf:
        t = ''.join(buf).strip()
        if t: terms.append(t)
    return terms

def flatten_or(expr: str) -> List[str]:
    """
    Recursively flatten OR as associative: (x | (y | z)) -> [x, y, z].
    Returns a list of OR-children that are not ORs at top-level anymore.
    """
    parts = split_top_level_or(expr)
    if len(parts) <= 1:
        return [strip_outer_parens(expr).strip()]
    out: List[str] = []
    for p in parts:
        sub = split_top_level_or(p)
        if len(sub) <= 1:
            out.append(strip_outer_parens(p).strip())
        else:
            out.extend(flatten_or(p))
    return out

def top_level_or_arity(expr: str) -> int:
    # OR-arity under associativity
    return max(1, len(flatten_or(expr)))

# ----------------- safe primitives (only literals negated) -----------------

def var(i: int) -> str:
    return f"a{i}"

def neg_lit(i: int) -> str:
    """Negated literal in label style: (~aK)"""
    return f"(~a{i})"

def maybe_neg(i: int, p: float) -> str:
    return neg_lit(i) if random.random() < p else var(i)

def wrap_if_neg(x: str) -> str:
    """Ensure (~aK) stays parenthesized; plain aK stays as-is."""
    return x if x.startswith("(~a") else x

def and2(a: str, b: str) -> str:  return f"({a} & {b})"
def or2(a: str, b: str)  -> str:  return f"({a} | {b})"

def and_chain(lits: List[str]) -> str:
    assert len(lits) >= 1
    expr = wrap_if_neg(lits[0])
    for t in lits[1:]:
        expr = and2(expr, wrap_if_neg(t))
    return expr

def or_chain(terms: List[str]) -> str:
    assert len(terms) >= 1
    expr = terms[0]
    for t in terms[1:]:
        expr = or2(expr, t)
    return expr

# ----------------- families (AND/OR/NOT only) -----------------

def make_dnf(B: int, n_terms: int, k_min: int, k_max: int, neg_p: float) -> str:
    """DNF: OR of AND-terms over (maybe negated) literals."""
    terms = []
    for _ in range(n_terms):
        k = random.randint(k_min, min(k_max, B))
        idxs = random.sample(range(B), k)
        lits = [maybe_neg(i, neg_p) for i in idxs]
        terms.append(and_chain(lits))
    return or_chain(terms)

def make_balanced_tree(B: int, depth: int, neg_p: float) -> str:
    """Full-ish binary tree mixing & and | over (maybe negated) literals."""
    leaves = [maybe_neg(random.randrange(B), neg_p) for _ in range(2**depth)]
    cur = leaves
    for _ in range(depth):
        nxt = []
        for i in range(0, len(cur), 2):
            a, b = cur[i], cur[i+1]
            if random.random() < 0.5:
                nxt.append(and2(wrap_if_neg(a), wrap_if_neg(b)))
            else:
                nxt.append(or2(a, b))
        cur = nxt
    return cur[0]

def sample_formula(B: int) -> str:
    """
    Emit only AND/OR/NOT-based formulas: small DNFs and balanced trees.
    No XOR / implication / equivalence families.
    """
    r = random.random()
    if r < 0.55:
        # small DNF (2–3 terms, term size 1..min(3,B)), ~ on literals only
        return make_dnf(B, n_terms=random.randint(2,3), k_min=1, k_max=min(3,B), neg_p=0.40)
    else:
        # balanced tree of depth 2–3 over literals with optional negation
        return make_balanced_tree(B, depth=random.choice([2,3]), neg_p=0.35)

# ----------------- grids & generation -----------------

DEFAULT_BS = [2,4,6,8,10,12]
HARD_BS    = [4,6,8,10,12,14]

def gen_instances(Bs: List[int], n_per_B: int, seed: int) -> List[dict]:
    random.seed(seed)
    rows: List[dict] = []
    for B in Bs:
        for _ in range(n_per_B):
            f = sample_formula(B)
            if not ALLOWED.match(f):
                continue

            # OR fan-in under associativity
            m = top_level_or_arity(f)
            S = max(2, m)
            L = 2 if m <= 2 else 3

            # keep B consistent with the vars actually used
            B_infer = infer_B(f)
            if B_infer > B:
                B = B_infer

            rows.append({"B": B, "S": S, "L": L, "formula": f})
    return rows

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--grid", type=str, default="default", choices=["default","hard"])
    ap.add_argument("--B", type=str, default=None, help="Comma list overrides grid (e.g., 2,4,6,8)")
    ap.add_argument("--n_per_B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.B is not None:
        Bs = [int(x) for x in args.B.split(",") if x.strip()]
    else:
        Bs = DEFAULT_BS if args.grid == "default" else HARD_BS

    rows = gen_instances(Bs, args.n_per_B, args.seed)

    with out.open("w") as f_out:
        for r in rows:
            if not ALLOWED.match(r["formula"]):  # safety
                continue
            f_out.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} instances to {out}")

if __name__ == "__main__":
    main()
