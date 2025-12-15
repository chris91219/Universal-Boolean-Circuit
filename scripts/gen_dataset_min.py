#!/usr/bin/env python3
"""
scripts/gen_dataset_min.py

Generate Boolean-function datasets where each row is:
  {"B": int, "S": int, "L": int, "formula": str, "meta": {...}}

Features
--------
- SOP-only labels (default): formula uses ONLY (~, &, |, parentheses, 0/1, aK)
  and is minimized to minimal SOP (DNF) using Quine–McCluskey + exact cover.
- Advanced macro rows (optional): PARITY/MAJ/THR emitted as macros (not SOP-minimized).
- Progress logs every --log_every attempts; verbose shows minimizer internals.
- Budgets to prevent hangs:
    --max_seconds, --max_minterms, --max_primes, --max_cand, --max_tries
- Enforces output B >= 2:
    - drops constants (B=0) and unary functions (B=1) after minimization/remap.
"""

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ------------------------ regex ------------------------

ALLOWED_SOP = re.compile(r'^[()\sa\d~&|01]+$')
_VAR_RE = re.compile(r"a(\d+)")

_MACRO_PARITY = re.compile(r"^\s*PARITY\s*\((.*)\)\s*$", re.IGNORECASE)
_MACRO_MAJ    = re.compile(r"^\s*MAJ\s*\((.*)\)\s*$", re.IGNORECASE)
_MACRO_THR    = re.compile(r"^\s*THR\s*\(\s*(\d+)\s*,(.*)\)\s*$", re.IGNORECASE)


# ------------------------ small utils ------------------------

class MinBudgetExceeded(RuntimeError):
    pass

def popcount(x: int) -> int:
    return x.bit_count()

def ceil_log2(n: int) -> int:
    if n <= 1:
        return 0
    return int(math.ceil(math.log2(n)))

def bits_of(m: int, B: int) -> List[int]:
    return [(m >> i) & 1 for i in range(B)]

def all_assignments_int(B: int) -> List[int]:
    return list(range(1 << B))

def canonical_spaces(expr: str) -> str:
    s = re.sub(r"\s+", " ", expr).strip()
    s = re.sub(r"\s*&\s*", " & ", s)
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"~\s*a", "~a", s)
    return s

def remap_vars(expr: str) -> Tuple[str, int, Dict[int, int]]:
    idxs = sorted(set(int(m.group(1)) for m in _VAR_RE.finditer(expr)))
    if not idxs:
        return expr, 0, {}
    mapping = {old: new for new, old in enumerate(idxs)}
    def repl(mm):
        old = int(mm.group(1))
        return f"a{mapping[old]}"
    new_expr = _VAR_RE.sub(repl, expr)
    return new_expr, len(idxs), mapping


# ------------------------ evaluation (SOP + macros) ------------------------

def _split_args(arg_str: str) -> List[str]:
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

def _eval_sop(expr: str, a: List[int]) -> int:
    """
    SOP evaluator:
      - replace aK -> a[K]
      - replace ~ -> 1-
      - allow only numeric/paren/&/|/[]/a/space/+- (after rewrite)
    """
    s = canonical_spaces(expr)
    s = _VAR_RE.sub(lambda mm: f"a[{int(mm.group(1))}]", s)
    s = s.replace("~", "1-")

    if not re.fullmatch(r"[0-9a\[\]\(\)\s\-\+\&\|]+", s):
        raise ValueError(f"Unsafe SOP after rewrite: {s}")

    val = eval(s, {"__builtins__": {}}, {"a": a})
    return 1 if int(val) != 0 else 0

def eval_expr_on_assignment(expr: str, a: List[int]) -> int:
    expr = expr.strip()

    m = _MACRO_PARITY.match(expr)
    if m:
        args = _split_args(m.group(1))
        if not args:
            raise ValueError("PARITY() needs at least one arg")
        vals = []
        for t in args:
            t = t.strip()
            if re.fullmatch(r"a\d+", t):
                i = int(t[1:])
                vals.append(a[i])
            elif t in {"0","1"}:
                vals.append(int(t))
            else:
                vals.append(_eval_sop(t, a))
        return sum(vals) & 1

    m = _MACRO_MAJ.match(expr)
    if m:
        args = _split_args(m.group(1))
        if not args:
            raise ValueError("MAJ() needs at least one arg")
        vals = []
        for t in args:
            t = t.strip()
            if re.fullmatch(r"a\d+", t):
                i = int(t[1:])
                vals.append(a[i])
            elif t in {"0","1"}:
                vals.append(int(t))
            else:
                vals.append(_eval_sop(t, a))
        thr = (len(vals) + 1) // 2
        return 1 if sum(vals) >= thr else 0

    m = _MACRO_THR.match(expr)
    if m:
        k = int(m.group(1))
        args = _split_args(m.group(2))
        if not args:
            raise ValueError("THR(k, ...) needs at least one arg")
        vals = []
        for t in args:
            t = t.strip()
            if re.fullmatch(r"a\d+", t):
                i = int(t[1:])
                vals.append(a[i])
            elif t in {"0","1"}:
                vals.append(int(t))
            else:
                vals.append(_eval_sop(t, a))
        return 1 if sum(vals) >= k else 0

    return _eval_sop(expr, a)

def truth_table(expr: str, B: int) -> List[int]:
    ys = []
    for m in all_assignments_int(B):
        a = bits_of(m, B)
        ys.append(eval_expr_on_assignment(expr, a))
    return ys


# ------------------------ QMC minimization ------------------------

@dataclass(frozen=True)
class Implicant:
    value: int   # bits on specified positions
    mask: int    # 1 means don't care at that bit
    B: int

    def covers(self, m: int) -> bool:
        keep = ~self.mask
        return (m & keep) == (self.value & keep)

    def literals_count(self) -> int:
        return self.B - popcount(self.mask)

def combine(i1: Implicant, i2: Implicant) -> Optional[Implicant]:
    if i1.B != i2.B or i1.mask != i2.mask:
        return None
    diff = (i1.value ^ i2.value) & (~i1.mask)
    if popcount(diff) != 1:
        return None
    new_mask = i1.mask | diff
    new_value = i1.value & (~diff)
    return Implicant(new_value, new_mask, i1.B)

def qm_prime_implicants(minterms: List[int], B: int,
                        *, t0: float,
                        max_seconds: float,
                        verbose: bool) -> List[Implicant]:
    if not minterms:
        return []
    current = [Implicant(m, 0, B) for m in sorted(set(minterms))]
    primes: Set[Implicant] = set()

    it = 0
    while True:
        it += 1
        if (time.time() - t0) > max_seconds:
            raise MinBudgetExceeded(f"QM timeout after {max_seconds:.2f}s (iter={it}, current={len(current)})")

        if verbose:
            print(f"      [qm] iter={it:02d} current={len(current)} primes={len(primes)}")

        groups: Dict[int, List[Implicant]] = {}
        for imp in current:
            k = popcount(imp.value & (~imp.mask))
            groups.setdefault(k, []).append(imp)

        next_set: Set[Implicant] = set()
        used: Set[Implicant] = set()

        keys = sorted(groups.keys())
        for k in keys:
            if k + 1 not in groups:
                continue
            for a in groups[k]:
                for b in groups[k + 1]:
                    c = combine(a, b)
                    if c is not None:
                        used.add(a); used.add(b)
                        next_set.add(c)

        for imp in current:
            if imp not in used:
                primes.add(imp)

        if not next_set:
            break
        current = sorted(next_set, key=lambda x: (x.mask, x.value))

    return sorted(primes, key=lambda x: (x.literals_count(), x.mask, x.value))

def implicant_to_term(p: Implicant) -> str:
    lits = []
    for i in range(p.B):
        bit = 1 << i
        if p.mask & bit:
            continue
        v = 1 if (p.value & bit) else 0
        lits.append(f"a{i}" if v == 1 else f"(~a{i})")
    if not lits:
        return "1"
    expr = lits[0]
    for t in lits[1:]:
        expr = f"({expr} & {t})"
    return expr

def choose_min_cover_exact(minterms: List[int],
                           primes: List[Implicant],
                           *,
                           t0: float,
                           max_seconds: float,
                           max_cand: int,
                           verbose: bool) -> List[Implicant]:
    M = sorted(set(minterms))
    if not M:
        return []

    cover = []
    for p in primes:
        cover.append(set(m for m in M if p.covers(m)))

    chart: Dict[int, List[int]] = {m: [] for m in M}
    for pi, cov in enumerate(cover):
        for m in cov:
            chart[m].append(pi)

    chosen: Set[int] = set()
    uncovered: Set[int] = set(M)

    changed = True
    while changed:
        changed = False
        if (time.time() - t0) > max_seconds:
            raise MinBudgetExceeded("cover timeout (essentials)")
        for m in list(uncovered):
            ps = chart[m]
            if len(ps) == 1:
                pidx = ps[0]
                if pidx not in chosen:
                    chosen.add(pidx)
                    uncovered -= cover[pidx]
                    changed = True

    if not uncovered:
        return [primes[i] for i in sorted(chosen)]

    cand = [i for i in range(len(primes)) if i not in chosen]
    cand.sort(key=lambda i: (-len(cover[i] & uncovered), primes[i].literals_count()))

    if verbose:
        print(f"      [cover] uncovered={len(uncovered)} primes={len(primes)} cand={len(cand)}")

    if len(cand) > max_cand:
        raise MinBudgetExceeded(f"too many cover candidates (cand={len(cand)} > {max_cand})")

    best_sel: Optional[List[int]] = None
    best_cost: Optional[Tuple[int, int]] = None

    def covers_all(sel: List[int]) -> bool:
        cov = set()
        for i in sel:
            cov |= cover[i]
        return uncovered.issubset(cov)

    max_cov = max(1, max(len(cover[i] & uncovered) for i in cand))
    lower_k = math.ceil(len(uncovered) / max_cov)

    from itertools import combinations
    for k in range(lower_k, len(cand) + 1):
        if (time.time() - t0) > max_seconds:
            raise MinBudgetExceeded("cover timeout (search)")
        for comb in combinations(cand, k):
            if (time.time() - t0) > max_seconds:
                raise MinBudgetExceeded("cover timeout (search combos)")
            sel = sorted(chosen | set(comb))
            if not covers_all(sel):
                continue
            terms = len(sel)
            lits = sum(primes[i].literals_count() for i in sel)
            cost = (terms, lits)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_sel = sel
        if best_sel is not None:
            break

    if best_sel is None:
        raise MinBudgetExceeded("exact cover failed within budget")

    return [primes[i] for i in best_sel]

def sop_from_minterms_budget(minterms: List[int], B: int,
                            *, t0: float,
                            max_seconds: float,
                            max_primes: int,
                            max_cand: int,
                            verbose: bool) -> str:
    if not minterms:
        return "0"
    if len(minterms) == (1 << B):
        return "1"

    primes = qm_prime_implicants(minterms, B, t0=t0, max_seconds=max_seconds, verbose=verbose)

    if verbose:
        print(f"      [min] primes={len(primes)}")

    if len(primes) > max_primes:
        raise MinBudgetExceeded(f"too many prime implicants ({len(primes)} > {max_primes})")

    cover = choose_min_cover_exact(minterms, primes, t0=t0, max_seconds=max_seconds, max_cand=max_cand, verbose=verbose)
    terms = [implicant_to_term(p) for p in cover]
    terms.sort(key=lambda s: (s.count("&"), len(s), s))

    if len(terms) == 1:
        return terms[0]
    expr = terms[0]
    for t in terms[1:]:
        expr = f"({expr} | {t})"
    return expr

def minimize_to_sop_budget(expr: str, B: int,
                          *, max_seconds: float,
                          max_minterms: int,
                          max_primes: int,
                          max_cand: int,
                          verbose: bool) -> Tuple[str, List[int]]:
    t0 = time.time()
    ys = truth_table(expr, B)
    minterms = [m for m, y in enumerate(ys) if y == 1]

    if verbose:
        print(f"      [min] B={B} ones={len(minterms)}/{1<<B}")

    if len(minterms) > max_minterms and len(minterms) < (1 << B) - max_minterms:
        raise MinBudgetExceeded(f"too dense: ones={len(minterms)} (max_minterms={max_minterms})")

    sop = sop_from_minterms_budget(
        minterms, B,
        t0=t0,
        max_seconds=max_seconds,
        max_primes=max_primes,
        max_cand=max_cand,
        verbose=verbose,
    )
    return sop, ys


# ------------------------ derive S,L from minimized SOP ------------------------

def top_level_or_arity_sop(expr: str) -> int:
    expr = expr.strip()
    if expr in {"0", "1"}:
        return 1
    depth = 0
    parts, buf = [], []
    for ch in expr:
        if ch == "(":
            depth += 1; buf.append(ch)
        elif ch == ")":
            depth -= 1; buf.append(ch)
        elif ch == "|" and depth == 0:
            t = "".join(buf).strip()
            if t:
                parts.append(t)
            buf = []
        else:
            buf.append(ch)
    t = "".join(buf).strip()
    if t:
        parts.append(t)
    return max(1, len(parts))

def max_term_literals(expr: str) -> int:
    if expr in {"0", "1"}:
        return 0
    depth = 0
    parts, buf = [], []
    for ch in expr:
        if ch == "(":
            depth += 1; buf.append(ch)
        elif ch == ")":
            depth -= 1; buf.append(ch)
        elif ch == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    best = 0
    for p in parts:
        best = max(best, len(_VAR_RE.findall(p)))
    return best

def compute_S_L(expr: str, B: int, family: str) -> Tuple[int, int]:
    fam = family.lower()
    if fam == "parity":
        return 2, max(2, ceil_log2(max(2, B)) + 1)
    if fam in {"maj", "thr"}:
        return 2, 2

    num_terms = top_level_or_arity_sop(expr)
    S = max(2, num_terms)
    k = max_term_literals(expr)
    and_depth = ceil_log2(max(1, k))
    or_depth = ceil_log2(max(1, num_terms))
    L = max(2, and_depth + or_depth)
    return S, L


# ------------------------ samplers ------------------------

def _lit(i: int, neg_p: float) -> str:
    return f"(~a{i})" if random.random() < neg_p else f"a{i}"

def _and2(a: str, b: str) -> str:
    return f"({a} & {b})"

def _or2(a: str, b: str) -> str:
    return f"({a} | {b})"

def sample_formula_g6(B: int) -> str:
    if random.random() < 0.65:
        n_terms = random.randint(2, 3)
        terms = []
        for _ in range(n_terms):
            k = random.randint(1, min(3, B))
            idxs = random.sample(range(B), k)
            t = _lit(idxs[0], 0.40)
            for j in idxs[1:]:
                t = _and2(t, _lit(j, 0.40))
            terms.append(t)
        expr = terms[0]
        for t in terms[1:]:
            expr = _or2(expr, t)
        return expr
    else:
        depth = random.choice([2, 3])
        leaves = [_lit(random.randrange(B), 0.35) for _ in range(2**depth)]
        cur = leaves
        for _ in range(depth):
            nxt = []
            for i in range(0, len(cur), 2):
                a, b = cur[i], cur[i+1]
                nxt.append(_and2(a, b) if random.random() < 0.5 else _or2(a, b))
            cur = nxt
        return cur[0]

def sample_formula_g16(B: int) -> str:
    def gate(a: str, b: str, g: str) -> str:
        if g == "FALSE": return "0"
        if g == "TRUE":  return "1"
        if g == "A":     return a
        if g == "B":     return b
        if g == "~A":    return f"(~{a})" if a.startswith("a") else f"(~({a}))"
        if g == "~B":    return f"(~{b})" if b.startswith("a") else f"(~({b}))"
        if g == "AND":   return _and2(a, b)
        if g == "OR":    return _or2(a, b)
        if g == "NAND":  return f"(~({_and2(a,b)}))"
        if g == "NOR":   return f"(~({_or2(a,b)}))"
        if g == "A&~B":  return _and2(a, f"(~{b})" if b.startswith("a") else f"(~({b}))")
        if g == "~A&B":  return _and2(f"(~{a})" if a.startswith("a") else f"(~({a}))", b)
        if g == "A|~B":  return _or2(a, f"(~{b})" if b.startswith("a") else f"(~({b}))")
        if g == "~A|B":  return _or2(f"(~{a})" if a.startswith("a") else f"(~({a}))", b)
        if g == "XOR":
            return _or2(_and2(a, f"(~{b})" if b.startswith("a") else f"(~({b}))"),
                        _and2(f"(~{a})" if a.startswith("a") else f"(~({a}))", b))
        if g == "XNOR":
            return f"(~({gate(a,b,'XOR')}))"
        return _or2(a, b)

    GATES = ["AND","OR","NAND","NOR","A","B","~A","~B","A&~B","~A&B","A|~B","~A|B","XOR","XNOR","TRUE","FALSE"]
    depth = random.choice([2, 3])
    leaves = [_lit(random.randrange(B), 0.35) for _ in range(2**depth)]
    cur = leaves
    for _ in range(depth):
        nxt = []
        for i in range(0, len(cur), 2):
            nxt.append(gate(cur[i], cur[i+1], random.choice(GATES)))
        cur = nxt
    return cur[0]

def sample_advanced(B: int, family: str) -> str:
    vs = [f"a{i}" for i in range(B)]
    if family == "parity":
        return "PARITY(" + ",".join(vs) + ")"
    if family == "maj":
        return "MAJ(" + ",".join(vs) + ")"
    if family == "thr":
        k = random.randint(1, B)
        return f"THR({k}," + ",".join(vs) + ")"
    raise ValueError(f"unknown family: {family}")


# ------------------------ generation loop ------------------------

DEFAULT_BS = [2,4,6,8,10,12]
HARD_BS    = [4,6,8,10,12,14]

def parse_adv_mix(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, v = chunk.split(":")
        out[k.strip().lower()] = float(v)
    return out

def gen_dataset(
    Bs: List[int],
    n_per_B: int,
    seed: int,
    gate_set: str,
    *,
    advanced: bool,
    adv_mix: Dict[str, float],
    adv_rate: float,
    verbose: bool,
    log_every: int,
    max_tries: int,
    max_seconds: float,
    max_minterms: int,
    max_primes: int,
    max_cand: int,
    min_B_out: int,
) -> List[dict]:
    random.seed(seed)
    rows: List[dict] = []
    seen: Set[str] = set()

    if advanced:
        tot = sum(adv_mix.values())
        if tot <= 0:
            raise ValueError("adv_mix total weight must be >0")
        adv_keys = list(adv_mix.keys())
        adv_probs = [adv_mix[k] / tot for k in adv_keys]
    else:
        adv_keys, adv_probs = [], []

    tries = 0
    for B0 in Bs:
        need = n_per_B
        while need > 0:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f"Exceeded max_tries={max_tries}. Generated={len(rows)} rows so far.")

            if (tries % log_every) == 0:
                print(f"[gen] tries={tries}  B_orig={B0}  remaining_for_B={need}  total_rows={len(rows)}")

            B = int(B0)

            # advanced macros
            if advanced and (random.random() < adv_rate):
                fam = random.choices(adv_keys, weights=adv_probs, k=1)[0]
                if B < min_B_out:
                    continue
                expr = sample_advanced(B, fam)
                minimized = canonical_spaces(expr)
                B_eff = B
                if B_eff < min_B_out:
                    continue
                S, L = compute_S_L(minimized, B_eff, fam)

                key = f"{fam}::{B_eff}::{minimized}"
                if key in seen:
                    continue
                seen.add(key)

                rows.append({
                    "B": int(B_eff),
                    "S": int(S),
                    "L": int(L),
                    "formula": minimized,
                    "meta": {"gate_set": gate_set, "family": fam, "B_orig": int(B0)},
                })
                need -= 1
                continue

            # SOP family
            expr = sample_formula_g16(B) if gate_set == "16" else sample_formula_g6(B)

            try:
                sop, _ys = minimize_to_sop_budget(
                    expr, B,
                    max_seconds=max_seconds,
                    max_minterms=max_minterms,
                    max_primes=max_primes,
                    max_cand=max_cand,
                    verbose=(verbose and (tries % log_every == 0)),
                )
            except MinBudgetExceeded as e:
                if verbose and (tries % log_every == 0):
                    print(f"    [skip] {e}")
                continue
            except Exception as e:
                if verbose and (tries % log_every == 0):
                    print(f"    [skip] minimizer error: {type(e).__name__}: {e}")
                continue

            sop = canonical_spaces(sop)
            sop2, B_eff, _map = remap_vars(sop)
            sop2 = canonical_spaces(sop2)

            # enforce B >= min_B_out (drops constants and unary functions)
            if B_eff < min_B_out:
                continue

            if sop2 == "" or not ALLOWED_SOP.match(sop2):
                continue

            S, L = compute_S_L(sop2, B_eff, "sop")
            key = f"sop::{B_eff}::{sop2}"
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "B": int(B_eff),
                "S": int(S),
                "L": int(L),
                "formula": sop2,
                "meta": {"gate_set": gate_set, "family": "sop", "B_orig": int(B0)},
            })
            need -= 1

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--grid", type=str, default="default", choices=["default", "hard"])
    ap.add_argument("--B", type=str, default=None, help="Override grid, e.g. 2,4,6,8")
    ap.add_argument("--n_per_B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--gate_set", type=str, default="6", choices=["6", "16"])

    # enforce output minimum B
    ap.add_argument("--min_B_out", type=int, default=2,
                    help="Drop instances whose minimized/remapped B is smaller than this (default 2).")

    # advanced macros
    ap.add_argument("--advanced", action="store_true")
    ap.add_argument("--adv_rate", type=float, default=0.35)
    ap.add_argument("--adv_mix", type=str, default="parity:1,maj:1,thr:1")

    # logging + budgets
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--max_tries", type=int, default=200000)

    ap.add_argument("--max_seconds", type=float, default=0.25)
    ap.add_argument("--max_minterms", type=int, default=256)
    ap.add_argument("--max_primes", type=int, default=800)
    ap.add_argument("--max_cand", type=int, default=40)

    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.B is not None:
        Bs = [int(x) for x in args.B.split(",") if x.strip()]
    else:
        Bs = DEFAULT_BS if args.grid == "default" else HARD_BS

    adv_mix = parse_adv_mix(args.adv_mix)

    rows = gen_dataset(
        Bs=Bs,
        n_per_B=args.n_per_B,
        seed=args.seed,
        gate_set=args.gate_set,
        advanced=bool(args.advanced),
        adv_mix=adv_mix,
        adv_rate=float(args.adv_rate),
        verbose=bool(args.verbose),
        log_every=int(args.log_every),
        max_tries=int(args.max_tries),
        max_seconds=float(args.max_seconds),
        max_minterms=int(args.max_minterms),
        max_primes=int(args.max_primes),
        max_cand=int(args.max_cand),
        min_B_out=int(args.min_B_out),
    )

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[ok] Wrote {len(rows)} instances -> {out}")

if __name__ == "__main__":
    main()
