#!/usr/bin/env python3
# scripts/gen_dataset.py
# Generate Boolean-formula datasets over variables a0..a{B-1} using only ~, &, | and ().
# Output: JSONL with fields: B, S, formula
#
# Example:
#   python scripts/gen_dataset.py --out data/bench_default.jsonl
#   python scripts/gen_dataset.py --grid hero --out data/bench_hero.jsonl --seed 123

import argparse, json, random, math
from pathlib import Path
from typing import List, Tuple

# --------------- helpers ---------------

def lit(i: int, neg_p: float) -> str:
    """Return 'a{i}' or '~a{i}' with prob neg_p (no parens here)."""
    if random.random() < neg_p:
        return f"~a{i}"
    return f"a{i}"

def wrap_lit(x: str) -> str:
    """Put parentheses around negated literal to match label style."""
    return f"({x})" if x.startswith("~") else x

def and_chain(lits: List[str]) -> str:
    """Left-assoc AND with grouping and (~aK) parenthesized."""
    if not lits:
        return "a0"  # fallback, should not happen
    expr = wrap_lit(lits[0])
    for t in lits[1:]:
        expr = f"({expr} & {wrap_lit(t)})"
    return expr

def or_chain(terms: List[str]) -> str:
    """Left-assoc OR with grouping."""
    if not terms:
        return "a0"
    expr = terms[0]
    for t in terms[1:]:
        expr = f"({expr} | {t})"
    return expr

def rand_subset_indices(B: int, k: int) -> List[int]:
    return random.sample(range(B), k)

def xor2(a: str, b: str) -> str:
    # XOR(a,b) = (a & ~b) | (~a & b)
    return f"(({a} & {wrap_lit('~'+b[1:] if b.startswith('a') else '~'+b.split('a')[-1])}) | ({wrap_lit('~'+a[1:] if a.startswith('a') else '~'+a.split('a')[-1])} & {b}))"

def xor_chain(vars_syms: List[str]) -> str:
    """Chain XOR across symbols using only ~,&,| (parenthesized)."""
    assert len(vars_syms) >= 2
    expr = xor2(vars_syms[0], vars_syms[1])
    for v in vars_syms[2:]:
        expr = xor2(f"({expr})", v)
    return expr

def implies(p: str, q: str) -> str:
    # p -> q == (~p) | q
    return f"({wrap_lit('~'+p[1:] if p.startswith('a') else '~'+p.split('a')[-1])} | {q})"

def equiv(p: str, q: str) -> str:
    # p <-> q == (p & q) | (~p & ~q)
    pneg = wrap_lit('~'+p[1:] if p.startswith('a') else '~'+p.split('a')[-1])
    qneg = wrap_lit('~'+q[1:] if q.startswith('a') else '~'+q.split('a')[-1])
    return f"(({p} & {q}) | ({pneg} & {qneg}))"

def approx_choose_S(B: int) -> int:
    """Reasonable width suggestion; clip to {2,4,8,12,16,32} and ≤ 2*B."""
    candidates = [2,4,8,12,16,32]
    target = min(2*B, 32)
    # pick the smallest candidate >= ceil(B/1.5) but not exceeding target
    floor = math.ceil(B/1.5)
    for c in candidates:
        if c >= floor and c <= target:
            return c
    return min(target, candidates[-1])

# --------------- formula families ---------------

def make_dnf(B: int, n_terms: int, k_min: int, k_max: int, neg_p: float) -> str:
    terms = []
    for _ in range(n_terms):
        k = random.randint(k_min, min(k_max, B))
        idxs = rand_subset_indices(B, k)
        lits_ = [lit(i, neg_p) for i in idxs]
        terms.append(and_chain(lits_))
    return or_chain(terms)

def make_cnf(B: int, n_clauses: int, k_min: int, k_max: int, neg_p: float) -> str:
    # CNF: ∧ of OR-clauses -> convert to DNF-ish by distributing? too long.
    # Instead, represent as OR-of-ANDs by DeMorganing randomized structure:
    # We'll synthesize as DNF for now but with longer terms (acts like CNF difficulty).
    return make_dnf(B, n_terms=n_clauses, k_min=k_min, k_max=k_max, neg_p=neg_p)

def make_balanced_tree(B: int, depth: int, neg_p: float) -> str:
    """Generate a full-ish binary tree mixing & and | over leaves of (maybe negated) literals."""
    def leaf():
        return lit(random.randrange(B), neg_p)
    nodes = [leaf() for _ in range(2**depth)]
    # pairwise combine up
    cur = nodes
    for _ in range(depth):
        nxt = []
        for i in range(0, len(cur), 2):
            a, b = cur[i], cur[i+1]
            op = "&" if random.random() < 0.5 else "|"
            if op == "&":
                nxt.append(f"({wrap_lit(a)} & {wrap_lit(b)})")
            else:
                nxt.append(f"({a} | {b})")
        cur = nxt
    return cur[0]

def make_parity(B: int) -> str:
    idxs = list(range(B))
    random.shuffle(idxs)
    syms = [f"a{i}" for i in idxs[:max(2, min(B, 6))]]  # keep tractable length
    if len(syms) < 2:
        syms.append("a0")
    return xor_chain(syms)

def make_implication_chain(B: int, length: int) -> str:
    idxs = list(range(B))
    random.shuffle(idxs)
    chain = f"a{idxs[0]}"
    for j in idxs[1:1+max(1,length-1)]:
        chain = implies(chain, f"a{j}")
    return chain

def make_equiv_pair(B: int) -> str:
    i, j = random.sample(range(B), 2)
    return equiv(f"a{i}", f"a{j}")

# --------------- grids ---------------

DEFAULT_BS = [2,4,6,8,10,12]
DEFAULT_S  = [2,4,8,12,16]
DEFAULT_L  = [2,3,4]  # depth is *architecture*, not encoded; we still vary structure length in formulas

HERO_BS = [14]
HERO_S  = [8,12,16,32]

def sample_formula(B: int) -> str:
    """Mix of families, biased to small clean expressions."""
    r = random.random()
    if r < 0.30:
        # small DNF
        return make_dnf(B, n_terms=random.randint(2,3), k_min=1, k_max=min(3,B), neg_p=0.4)
    elif r < 0.55:
        # balanced tree
        d = random.choice([2,3])  # depth of tree (formula depth)
        return make_balanced_tree(B, depth=d, neg_p=0.35)
    elif r < 0.70:
        # implication chain
        return make_implication_chain(B, length=random.choice([2,3,4]))
    elif r < 0.85:
        # equivalence of a pair
        return make_equiv_pair(B)
    else:
        # parity snippet across ≤6 vars to keep length reasonable
        return make_parity(B)

def gen_instances(Bs: List[int], Sset: List[int], n_per_B: int, seed: int) -> List[dict]:
    random.seed(seed)
    rows = []
    for B in Bs:
        # choose a suggested S near complexity, but also include the provided Sset
        suggested = approx_choose_S(B)
        Ss = sorted(set(Sset + [suggested]))
        for _ in range(n_per_B):
            f = sample_formula(B)
            S = random.choice(Ss)
            rows.append({"B": B, "S": S, "formula": f})
    return rows

def gen_hero_instances(n_each: int, seed: int) -> List[dict]:
    random.seed(seed)
    rows = []
    for B in HERO_BS:
        for S in HERO_S:
            # a few canonical “hero” formulas
            rows += [
                {"B": B, "S": S, "formula": make_dnf(B, n_terms=3, k_min=2, k_max=3, neg_p=0.4)},
                {"B": B, "S": S, "formula": make_balanced_tree(B, depth=3, neg_p=0.35)},
                {"B": B, "S": S, "formula": make_implication_chain(B, length=4)},
                {"B": B, "S": S, "formula": make_equiv_pair(B)},
                {"B": B, "S": S, "formula": make_parity(B)},
            ]
            # add extra randoms
            for _ in range(max(0, n_each - 5)):
                rows.append({"B": B, "S": S, "formula": sample_formula(B)})
    return rows

# --------------- CLI ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--grid", type=str, default="default", choices=["default","hard","hero"])
    ap.add_argument("--B", type=str, default=None, help="Comma list, overrides grid (e.g., 2,4,6,8)")
    ap.add_argument("--S", type=str, default=None, help="Comma list, overrides grid (e.g., 2,4,8,12)")
    ap.add_argument("--n_per_B", type=int, default=200, help="#instances per B (default grid)")
    ap.add_argument("--hero_n_each", type=int, default=20, help="#instances per (B,S) in hero grid")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.grid == "hero":
        rows = gen_hero_instances(args.hero_n_each, args.seed)
    else:
        if args.B is not None:
            Bs = [int(x) for x in args.B.split(",") if x.strip()]
        else:
            Bs = DEFAULT_BS if args.grid == "default" else [4,6,8,10,12,14]  # "hard" includes 14
        if args.S is not None:
            Sset = [int(x) for x in args.S.split(",") if x.strip()]
        else:
            Sset = DEFAULT_S
        rows = gen_instances(Bs, Sset, args.n_per_B, args.seed)

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} instances to {out}")

if __name__ == "__main__":
    main()
