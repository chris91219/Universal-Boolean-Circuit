# Universal-Boolean-Circuit (UBC)

End-to-end trainable Boolean reasoning layers that **discover and execute** propositional logic with **AND/OR/NOT** primitives. The stack composes parallel probabilistic gates via row-wise softmax (routing), with a differentiable **pair selector** to scale from 2-bit inputs to **B-bit** inputs. Training is on full truth tables; readout produces a **composed symbolic formula** equivalent to the learned function.

## Highlights

* **Primitives:** `AND(a,b)=min`, `OR(a,b)=max`, `NOT`, plus skips.
* **Layered circuit:** stack of reasoning layers with soft routing.
* **Scaling:** works for B up to \~12 routinely (hero cases at B=14).
* **Readable:** layer-by-layer **composed symbolic readout** (string-normalized).
* **Metrics:** per-instance row-accuracy, EM (exact table match), equivalence (via table), and a “simpler” score; reports `simpler: same` when strings match.
* **Training aids:** async temperature annealing, entropy + diversity regs, early stopping.

## Repo Layout

```
src/ubcircuit/      # library code (models, utils, train)
scripts/            # dataset generator, SLURM helpers, etc.
configs/            # example configs
data/               # datasets (.jsonl) you generate
experiments/        # results (logs & summary.json)
```

## Install (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU Torch (or use cu121 wheel if you have CUDA):
pip install torch==2.6.* --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Dataset format

JSONL with one instance per line:

```json
{"B": 8, "S": 8, "formula": "(a0 & (~a3)) | (a4 & a7)"}
```

* Variables are `a0..a{B-1}`.
* Allowed ops: `~` (NOT), `&`, `|`, and parentheses.
* Negations on literals are printed as `(~aK)` for labels/normalization.

### Generate benchmark datasets

```bash
# default grid: B in {2,4,6,8,10,12}, S in {2,4,8,12,16}
python scripts/gen_dataset.py --out data/bench_default.jsonl

# include B=14 hero cases
python scripts/gen_dataset.py --grid hard --out data/bench_hard.jsonl

# hero set (B=14, S up to 32), a few canonical + random
python scripts/gen_dataset.py --grid hero --out data/bench_hero.jsonl
```

## Train

### Multi-instance (dataset file)

```bash
python -m ubcircuit.train \
  --dataset data/bench_default.jsonl \
  --out_dir experiments/results/bench_default
```

* Prints per-instance metrics and formulas; writes `summary.json` with all results.
* **Early stop** is on by default (`em` target 1.0, warm-up 100 steps). Configure via `early_stop` in config.

### Single toy (2-bit)

```bash
python -m ubcircuit.train \
  --config configs/bool_2bit_or_na.yaml \
  --out_dir experiments/results/toy
```

## Key knobs (via config or CLI)

* `L` (depth), `S` (width), `steps`, `optimizer`, `lr`
* `anneal`: `T0`, `Tmin`, `direction`, `schedule`, `phase_scale`
* `regs`: `lam_entropy`, `lam_div_units`, `lam_div_rows`
* `early_stop`: `use`, `metric`, `target`, `min_steps`, `check_every`, `patience_checks`

## Practical ranges

* Routine: **B ≤ 12**, **S ≤ 16**, **L ≤ 4**
* Hero: **B=14**, **S ≤ 32** (few instances)
* Avoid routine training with **B ≥ 16** (truth table explodes).

## Outputs

`experiments/results/.../summary.json` contains:

* Per-instance: `row_acc`, `em`, `equiv`, `pred_expr`, `label_expr`, `simpler`, and truth tables.
* Aggregates: `avg_row_acc`, `em_rate`, `equiv_rate`.

## License

See `LICENSE`.

