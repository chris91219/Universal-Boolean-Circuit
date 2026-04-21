#!/usr/bin/env python3
"""Build annotated high-B and noise-variable benchmark datasets.

The original `formula` is always kept unchanged and remains the label used by
the trainer/truth-table evaluator. SymPy simplification is recorded only as
metadata (`reduced_expr`, `B_true`, `S_true`, `L_true`, ...), so variables that
cancel algebraically still act as noise variables for the truth table.

Default outputs:
  - data/bench_default_ext_b20.jsonl
      Existing bench_default rows copied first, then B=14,16,18,20 rows.
  - data/bench_default_noise_add{0,2,4,6,8}.jsonl
      Same extended 2000 formulas, with ambient B increased by k.

Small-B robustness outputs can be created with:
  python scripts/build_robustness_datasets.py --mode fresh \
      --fresh-B 2,4,6,8,10 --n-per-B 400 \
      --extended-out data/bench_small_b10_2000.jsonl \
      --noise-out-prefix data/bench_small_b10_2000_noise_add

Naming convention:
  - B is the ambient truth-table dimension.
  - W_base/D_base are generator-provided width/depth budget proxies.
  - W_true/D_true are reduced-expression fan-in-2 circuit width/depth stats.
  - S/L are kept as backward-compatible aliases for W_base/D_base because the
    existing trainers still consume those names by default.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import sympy as sp
from sympy.logic.boolalg import And, BooleanFalse, BooleanTrue, Not, Or, simplify_logic


VAR_RE = re.compile(r"a(\d+)")


def _load_gen_dataset_module():
    path = Path(__file__).resolve().parent / "gen_dataset.py"
    spec = importlib.util.spec_from_file_location("ubc_gen_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_gen_dataset_module()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def infer_vars(formula: str) -> List[int]:
    return sorted({int(m.group(1)) for m in VAR_RE.finditer(formula)})


def remap_formula_vars(formula: str, mapping: Dict[int, int]) -> str:
    def repl(match: re.Match[str]) -> str:
        old = int(match.group(1))
        return f"a{mapping[old]}"

    return VAR_RE.sub(repl, formula)


def expr_token_count(formula: str) -> int:
    return len(re.findall(r"a\d+|[~&|()01]", formula.replace(" ", "")))


def expr_tree_depth(formula: str) -> int:
    depth = max_depth = 0
    for ch in formula:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


def ceil_log2(n: int) -> int:
    if n <= 1:
        return 0
    return int(math.ceil(math.log2(n)))


def formula_to_sympy(formula: str, B: int) -> sp.Basic:
    env = {f"a{i}": sp.Symbol(f"a{i}") for i in range(B)}
    safe = re.sub(r"\b0\b", "False", formula)
    safe = re.sub(r"\b1\b", "True", safe)
    return eval(safe, {"__builtins__": {}}, env)


def sympy_to_formula(expr: sp.Basic) -> str:
    if expr == BooleanTrue() or expr == sp.true:
        return "1"
    if expr == BooleanFalse() or expr == sp.false:
        return "0"
    if isinstance(expr, sp.Symbol):
        return str(expr)
    if isinstance(expr, Not):
        return f"(~{sympy_to_formula(expr.args[0])})"
    if isinstance(expr, And):
        args = sorted(expr.args, key=str)
        cur = sympy_to_formula(args[0])
        for arg in args[1:]:
            cur = f"({cur} & {sympy_to_formula(arg)})"
        return cur
    if isinstance(expr, Or):
        args = sorted(expr.args, key=str)
        cur = sympy_to_formula(args[0])
        for arg in args[1:]:
            cur = f"({cur} | {sympy_to_formula(arg)})"
        return cur
    # This should be rare because we request DNF over &, |, ~.
    return str(expr).replace("~", "~").replace("True", "1").replace("False", "0")


def dnf_terms(expr: sp.Basic) -> List[sp.Basic]:
    if expr == BooleanFalse() or expr == sp.false:
        return []
    if expr == BooleanTrue() or expr == sp.true:
        return [expr]
    if isinstance(expr, Or):
        return list(expr.args)
    return [expr]


def term_literal_count(term: sp.Basic) -> int:
    if term == BooleanTrue() or term == sp.true:
        return 0
    if term == BooleanFalse() or term == sp.false:
        return 0
    if isinstance(term, And):
        return len(term.args)
    return 1


def dnf_layered_circuit_stats(expr: sp.Basic) -> Dict[str, Any]:
    """Natural layered fan-in-2 DNF circuit stats for a simplified expression.

    Literals and negated literals are treated as depth-0 inputs because the
    paper/model includes bit-lifting over variables and their negations.
    Multi-literal terms are balanced AND trees; terms are then combined by a
    balanced OR tree. Projection gates are allowed to carry early terms forward
    so all layers remain consecutive fan-in-2 circuit layers.
    """
    terms = dnf_terms(expr)
    n_terms = len(terms)
    term_lits = [term_literal_count(t) for t in terms]

    if n_terms == 0:
        return {
            "D_true": 0,
            "W_true": 0,
            "dnf_terms_true": 0,
            "max_term_literals_true": 0,
            "term_literals_true": [],
            "and_depth_true": 0,
            "or_depth_true": 0,
        }

    term_depths = [ceil_log2(k) for k in term_lits]
    and_depth = max(term_depths, default=0)
    or_depth = ceil_log2(n_terms)
    d_true = and_depth + or_depth

    widths: List[int] = []
    for layer in range(1, and_depth + 1):
        # Terms that finished early are carried by projection gates.
        widths.append(sum(max(1, math.ceil(k / (2 ** layer))) for k in term_lits))
    for layer in range(1, or_depth + 1):
        widths.append(math.ceil(n_terms / (2 ** layer)))

    max_term_literals = int(max(term_lits, default=0))
    # Constants need no literal/computational wire.  Non-constant literals have
    # D_true=0 but still occupy one carried wire in the natural circuit view.
    w_true = int(max(widths, default=1 if max_term_literals > 0 else 0))

    return {
        "D_true": int(d_true),
        "W_true": w_true,
        "dnf_terms_true": int(n_terms),
        "max_term_literals_true": max_term_literals,
        "term_literals_true": [int(k) for k in term_lits],
        "and_depth_true": int(and_depth),
        "or_depth_true": int(or_depth),
    }


def reduce_formula(formula: str, B: int) -> Dict[str, Any]:
    expr = formula_to_sympy(formula, B)
    reduced = simplify_logic(expr, form="dnf", force=True)
    reduced_expr = sympy_to_formula(reduced)
    true_vars = infer_vars(reduced_expr)
    s_true = int(GEN.top_level_or_arity(reduced_expr))
    l_true_heuristic = 2 if s_true <= 2 else 3
    circuit_stats = dnf_layered_circuit_stats(reduced)
    original_vars = infer_vars(formula)
    return {
        "reduced_expr": reduced_expr,
        "reduced_expr_form": "sympy_dnf",
        "B_true": len(true_vars),
        "B_true_span": (max(true_vars) + 1) if true_vars else 0,
        "true_vars": true_vars,
        "S_true": s_true,
        "L_true": circuit_stats["D_true"],
        "D_true": circuit_stats["D_true"],
        "W_true": circuit_stats["W_true"],
        "L_true_heuristic": l_true_heuristic,
        "D_true_method": "balanced_dnf_fanin2_with_projection_carries",
        **circuit_stats,
        "reduced_tree_depth": expr_tree_depth(reduced_expr),
        "formula_vars": original_vars,
        "formula_var_count": len(original_vars),
        "formula_token_count": expr_token_count(formula),
        "reduced_token_count": expr_token_count(reduced_expr),
    }


def annotate_row(row: Dict[str, Any], *, source: str, source_idx: int | None = None) -> Dict[str, Any]:
    out = dict(row)
    B = int(out["B"])
    formula = str(out["formula"])
    out["W_base"] = int(out.get("W_base", out.get("S", 2)))
    out["D_base"] = int(out.get("D_base", out.get("L", 2)))
    # Backward compatibility for existing trainers/analyzers.
    out["S"] = int(out.get("S", out["W_base"]))
    out["L"] = int(out.get("L", out["D_base"]))
    reduced = reduce_formula(formula, B)
    out.update(reduced)
    out["source"] = source
    if source_idx is not None:
        out["source_idx"] = int(source_idx)
    out["ambient_noise_vars"] = max(0, B - int(out["B_true"]))
    out["generated_expr_semantic_gap"] = int(out["formula_var_count"]) - int(out["B_true"])
    return out


def generate_rows(Bs: Sequence[int], n_per_B: int, seed: int, *, source_prefix: str = "generated_B") -> List[Dict[str, Any]]:
    random.seed(seed)
    rows: List[Dict[str, Any]] = []
    for B in Bs:
        for local_idx in range(n_per_B):
            formula = GEN.sample_formula(B)
            s_gen = max(2, int(GEN.top_level_or_arity(formula)))
            l_gen = 2 if s_gen <= 2 else 3
            row = {
                "B": int(B),
                "S": s_gen,
                "L": l_gen,
                "W_base": s_gen,
                "D_base": l_gen,
                "formula": formula,
                "source_local_idx": local_idx,
            }
            rows.append(annotate_row(row, source=f"{source_prefix}{B}", source_idx=None))
    return rows


def build_extended(base_rows: List[Dict[str, Any]], append_Bs: Sequence[int], n_per_B: int, seed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [
        annotate_row(row, source="bench_default_existing", source_idx=i)
        for i, row in enumerate(base_rows)
    ]
    out.extend(generate_rows(append_Bs, n_per_B=n_per_B, seed=seed))
    for idx, row in enumerate(out):
        row["dataset_idx"] = idx
    return out


def build_fresh(Bs: Sequence[int], n_per_B: int, seed: int) -> List[Dict[str, Any]]:
    out = generate_rows(Bs, n_per_B=n_per_B, seed=seed, source_prefix="fresh_B")
    for idx, row in enumerate(out):
        row["dataset_idx"] = idx
    return out


def make_noise_rows(base_rows: List[Dict[str, Any]], noise_add: int, noise_basis: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    noise_basis = str(noise_basis)
    for i, row in enumerate(base_rows):
        base_B = int(row["B"])
        new_row = dict(row)
        new_row["noise_base_dataset_idx"] = int(row.get("dataset_idx", i))
        new_row["noise_base_source"] = row.get("source", "unknown")
        if "source_idx" in row:
            new_row["noise_base_source_idx"] = row["source_idx"]
        formula_vars = [int(v) for v in row.get("formula_vars", infer_vars(str(row["formula"])))]
        true_vars = [int(v) for v in row.get("true_vars", [])]
        new_row["B_base"] = base_B
        new_row["noise_basis"] = noise_basis
        new_row["noise_add_requested"] = noise_add

        if noise_basis == "B_base":
            new_B = base_B + noise_add
            added_noise = list(range(base_B, new_B))
        elif noise_basis == "B_true":
            true_set = set(true_vars)
            formula_only_vars = [v for v in formula_vars if v not in true_set]
            compact_vars = true_vars + formula_only_vars
            mapping = {old: new for new, old in enumerate(compact_vars)}
            new_row["formula"] = remap_formula_vars(str(row["formula"]), mapping)
            new_row["formula_var_remap"] = {f"a{old}": f"a{new}" for old, new in mapping.items()}
            b_true = len(true_vars)
            b_min_for_formula = len(compact_vars)
            target_B = b_true + noise_add
            new_B = max(b_min_for_formula, target_B)
            added_noise = list(range(b_min_for_formula, new_B))
            new_row["B_true_compact"] = b_true
            new_row["B_min_for_formula"] = b_min_for_formula
            new_row["B_noise_target"] = target_B
            new_row["noise_add_effective_total"] = new_B - b_true
            new_row["formula_only_semantic_noise_vars"] = len(formula_only_vars)
        else:
            raise ValueError("noise_basis must be B_base or B_true")

        new_row["B"] = new_B
        new_row["noise_vars_added"] = noise_add
        new_row["added_noise_var_indices"] = added_noise
        annotated = annotate_row(new_row, source=f"noise_{noise_basis}_add{noise_add}", source_idx=i)
        annotated["dataset_idx"] = i
        out.append(annotated)
    return out


def write_meta(path: Path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extend", "fresh"], default="extend")
    ap.add_argument("--base", type=Path, default=Path("data/bench_default.jsonl"))
    ap.add_argument("--extended-out", type=Path, default=Path("data/bench_default_ext_b20.jsonl"))
    ap.add_argument("--append-B", default="14,16,18,20")
    ap.add_argument("--fresh-B", default="2,4,6,8,10")
    ap.add_argument("--n-per-B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260421)
    ap.add_argument("--noise-add", default="0,2,4,6,8")
    ap.add_argument("--noise-out-prefix", type=Path, default=Path("data/bench_default_noise_add"))
    ap.add_argument("--noise-basis", choices=["B_base", "B_true"], default="B_base")
    ap.add_argument("--skip-extended", action="store_true")
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args()

    base_rows = read_jsonl(args.base) if args.mode == "extend" else []
    append_Bs = parse_int_list(args.append_B)
    fresh_Bs = parse_int_list(args.fresh_B)
    noise_adds = parse_int_list(args.noise_add)

    extended: List[Dict[str, Any]] | None = None

    if not args.skip_extended:
        if args.mode == "fresh":
            extended = build_fresh(fresh_Bs, n_per_B=args.n_per_B, seed=args.seed)
            meta_kind = "fresh_small_B"
            meta_extra = {"fresh_B": fresh_Bs}
        else:
            extended = build_extended(base_rows, append_Bs=append_Bs, n_per_B=args.n_per_B, seed=args.seed)
            meta_kind = "extended_high_B"
            meta_extra = {
                "base": str(args.base),
                "base_rows": len(base_rows),
                "append_B": append_Bs,
            }
        write_jsonl(args.extended_out, extended)
        write_meta(
            args.extended_out.with_suffix(args.extended_out.suffix + ".meta.json"),
            {
                "kind": meta_kind,
                "extended_out": str(args.extended_out),
                "n_per_B": args.n_per_B,
                "seed": args.seed,
                "total_rows": len(extended),
                **meta_extra,
                "truth_table_formula_field": "formula",
                "reduced_expr_is_metadata_only": True,
                "ambient_dimension_field": "B",
                "junta_size_field": "B_true",
                "base_width_field": "W_base",
                "base_depth_field": "D_base",
                "legacy_base_width_alias": "S",
                "legacy_base_depth_alias": "L",
                "true_depth_field": "D_true",
                "legacy_depth_alias": "L_true",
                "legacy_heuristic_depth_field": "L_true_heuristic",
                "true_width_field": "W_true",
                "true_depth_width_method": "balanced_dnf_fanin2_with_projection_carries",
            },
        )
        print(f"[ok] wrote {len(extended)} rows -> {args.extended_out}")

    if not args.skip_noise:
        if extended is None:
            if args.extended_out.exists():
                extended = read_jsonl(args.extended_out)
            elif args.mode == "fresh":
                extended = build_fresh(fresh_Bs, n_per_B=args.n_per_B, seed=args.seed)
            else:
                extended = build_extended(base_rows, append_Bs=append_Bs, n_per_B=args.n_per_B, seed=args.seed)
        for k in noise_adds:
            path = Path(f"{args.noise_out_prefix}{k}.jsonl")
            rows = make_noise_rows(extended, noise_add=k, noise_basis=args.noise_basis)
            write_jsonl(path, rows)
            write_meta(
                path.with_suffix(path.suffix + ".meta.json"),
                {
                    "kind": "noise_variable_robustness",
                    "base": str(args.extended_out),
                    "base_kind": "fresh_small_B" if args.mode == "fresh" else "extended_high_B",
                    "out": str(path),
                    "base_rows": len(extended),
                    "noise_vars_added": k,
                    "noise_basis": args.noise_basis,
                    "total_rows": len(rows),
                    "truth_table_formula_field": "formula",
                    "reduced_expr_is_metadata_only": True,
                    "ambient_dimension_field": "B",
                    "junta_size_field": "B_true",
                    "base_width_field": "W_base",
                    "base_depth_field": "D_base",
                    "legacy_base_width_alias": "S",
                    "legacy_base_depth_alias": "L",
                    "true_depth_field": "D_true",
                    "legacy_depth_alias": "L_true",
                    "legacy_heuristic_depth_field": "L_true_heuristic",
                    "true_width_field": "W_true",
                    "true_depth_width_method": "balanced_dnf_fanin2_with_projection_carries",
                },
            )
            print(f"[ok] wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
