#!/bin/bash
# Extend the old 1200-row bench_default benchmark with only B=18,20 rows
# (200 each), then run the old best soft-EM joint SBC-vs-MLP sweep on just
# the added 400 rows.  The 400 rows are sharded into 20 contiguous chunks of
# 20 instances each so the workload parallelizes well on CC.
#
# This keeps the old main-text comparison apples-to-apples:
#   - runner: ubcircuit.joint_mlp_ubc_bnr_1216
#   - lifting: false
#   - early stop: soft EM
#   - S_add=10, L_add=0
#   - match modes: neuron / param_soft / param_total
#   - seeds: 1..5
#
# Submit from repo root:
#   sbatch scripts/submit_bench_default_b1820_joint_sharded_cc.sh
#
# After the array finishes, recombine shards:
#   sbatch scripts/slurm_combine_b1820_joint_shards_cc.sh

#SBATCH --job-name=ubc_b1820_joint
#SBATCH --account=def-ssanner
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-299
#SBATCH --output=/scratch/uoftwhli/UBC-Results/slurm-logs/%x-%A_%a.out

set -euo pipefail

START_ISO="$(date -Is)"
START_SEC=$SECONDS
on_exit () {
  local end_iso dur h m s
  end_iso="$(date -Is)"
  dur=$((SECONDS - START_SEC))
  h=$((dur / 3600))
  m=$(((dur % 3600) / 60))
  s=$((dur % 60))
  printf "\n[Timer] Start: %s\n[Timer] End  : %s\n[Timer] Elapsed: %02d:%02d:%02d (%d s)\n" \
         "$START_ISO" "$end_iso" "$h" "$m" "$s" "$dur"
}
trap on_exit EXIT

if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH:-$HOME/scratch}/UBC-Results}"
LOG_DIR="${RESULTS_ROOT}/slurm-logs"
mkdir -p "${RESULTS_ROOT}" "${LOG_DIR}"

module purge
module load StdEnv/2023
module load python/3.11

if [[ ! -f "${PROJECT_DIR}/ENV/bin/activate" ]]; then
  echo "[error] Missing virtualenv: ${PROJECT_DIR}/ENV/bin/activate"
  echo "[error] Submit from the repo root or pass PROJECT_DIR=/path/to/Universal-Boolean-Circuit."
  exit 1
fi
source "${PROJECT_DIR}/ENV/bin/activate"
hash -r

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

BASE_DATASET="${BASE_DATASET:-${PROJECT_DIR}/data/bench_default.jsonl}"
COMBINED_DATASET="${COMBINED_DATASET:-${PROJECT_DIR}/data/bench_default_plus_b1820_1600.jsonl}"
APPEND_DATASET="${APPEND_DATASET:-${PROJECT_DIR}/data/bench_default_plus_b1820_added400.jsonl}"
DATASET="${DATASET:-${APPEND_DATASET}}"
APPEND_BS="${APPEND_BS:-18,20}"
N_PER_B="${N_PER_B:-200}"
DATASET_SEED="${DATASET_SEED:-20260501}"
SHARD_SIZE="${SHARD_SIZE:-20}"
N_SHARDS="${N_SHARDS:-20}"

if [[ ! -f "${DATASET}" ]]; then
  if [[ "${DATASET}" != "${APPEND_DATASET}" ]]; then
    echo "[error] Missing run dataset: ${DATASET}"
    exit 1
  fi
  if [[ ! -f "${BASE_DATASET}" ]]; then
    echo "[error] Missing base dataset required to generate ${APPEND_DATASET}: ${BASE_DATASET}"
    exit 1
  fi
  echo "[info] generating benchmark extension datasets..."
  python "${PROJECT_DIR}/scripts/build_robustness_datasets.py" \
    --mode extend \
    --base "${BASE_DATASET}" \
    --extended-out "${COMBINED_DATASET}" \
    --append-only-out "${APPEND_DATASET}" \
    --append-B "${APPEND_BS}" \
    --n-per-B "${N_PER_B}" \
    --seed "${DATASET_SEED}" \
    --skip-noise
fi

if [[ ! -f "${DATASET}" ]]; then
  echo "[error] Missing run dataset after generation: ${DATASET}"
  exit 1
fi

GATE_SET="${GATE_SET:-16}"
OPT="${OPT:-rmsprop}"
ROUTE="${ROUTE:-mi_soft}"
DIR="${DIR:-top_down}"
REPEL="${REPEL:-false}"
REPEL_MODE="${REPEL_MODE:-log}"
LAM_CONST16="${LAM_CONST16:-0.0}"
STEPS="${STEPS:-3000}"
LR="${LR:-0.001}"
ETA="${ETA:-2.0}"
LAM_ENT="${LAM_ENT:-1.0e-3}"
LAM_DIV_U="${LAM_DIV_U:-5.0e-4}"
LAM_DIV_R="${LAM_DIV_R:-5.0e-4}"
PRIOR_STRENGTH="${PRIOR_STRENGTH:-2.0}"
MI_DISJOINT="${MI_DISJOINT:-true}"
T0="${T0:-0.60}"
TMIN="${TMIN:-0.06}"
SCHED="${SCHED:-cosine}"
PHASE="${PHASE:-0.5}"
S16_START="${S16_START:-0.25}"
S16_END="${S16_END:-0.10}"
S16_MODE="${S16_MODE:-rbf}"
S16_RADIUS="${S16_RADIUS:-0.75}"
LIFT_USE="${LIFT_USE:-false}"
LIFT_FACTOR="${LIFT_FACTOR:-2.0}"
S_ADD="${S_ADD:-10}"
L_ADD="${L_ADD:-0}"

SEEDS=(1 2 3 4 5)
MATCH_MODES=(neuron param_soft param_total)
NSEED=${#SEEDS[@]}
NMODE=${#MATCH_MODES[@]}
TOTAL=$((NSEED * NMODE * N_SHARDS))

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  exit 1
fi

task="${SLURM_ARRAY_TASK_ID}"
shard_idx=$((task % N_SHARDS))
mode_idx=$(((task / N_SHARDS) % NMODE))
seed_idx=$((task / (N_SHARDS * NMODE)))
SEED="${SEEDS[$seed_idx]}"
MATCH="${MATCH_MODES[$mode_idx]}"

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_b1820_joint_seed${SEED}.yaml"
SHARD_DATASET="${TMPDIR_USE}/bench_default_plus_b1820_added400_shard$(printf '%02d' "${shard_idx}")of$(printf '%02d' "${N_SHARDS}").jsonl"

python - "${DATASET}" "${SHARD_DATASET}" "${shard_idx}" "${SHARD_SIZE}" "${N_SHARDS}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
shard_idx = int(sys.argv[3])
shard_size = int(sys.argv[4])
n_shards = int(sys.argv[5])

rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
expected_total = shard_size * n_shards
if len(rows) != expected_total:
    raise SystemExit(
        f"Expected {expected_total} rows in append dataset for exact sharding, found {len(rows)} in {src}"
    )

start = shard_idx * shard_size
end = start + shard_size
subset = rows[start:end]
for local_idx, row in enumerate(subset):
    row = dict(row)
    row["append_dataset_idx_orig"] = int(row.get("dataset_idx", start + local_idx))
    row["dataset_idx"] = local_idx
    row["shard_idx"] = shard_idx
    row["n_shards"] = n_shards
    subset[local_idx] = row

dst.write_text("".join(json.dumps(row) + "\n" for row in subset))
print(f"[ok] wrote shard {shard_idx+1}/{n_shards}: {dst} ({len(subset)} rows)")
PY

cat > "${CFG}" <<EOF
seed: ${SEED}
gate_set: "${GATE_SET}"
steps: ${STEPS}
optimizer: ${OPT}
lr: ${LR}
use_row_L: true
use_max_L: false
lifting:
  use: ${LIFT_USE}
  factor: ${LIFT_FACTOR}
relaxation:
  mode: softmax
  hard: false
  gumbel_tau: 1.0
  eval_hard: false
anneal:
  T0: ${T0}
  Tmin: ${TMIN}
  direction: ${DIR}
  schedule: ${SCHED}
  phase_scale: ${PHASE}
  start_frac: 0.0
sigma16:
  s_start: ${S16_START}
  s_end:   ${S16_END}
  mode:    "${S16_MODE}"
  radius:  ${S16_RADIUS}
regs:
  lam_entropy: ${LAM_ENT}
  lam_div_units: ${LAM_DIV_U}
  lam_div_rows: ${LAM_DIV_R}
  lam_const16: ${LAM_CONST16}
pair:
  route: ${ROUTE}
  prior_strength: ${PRIOR_STRENGTH}
  mi_disjoint: ${MI_DISJOINT}
  repel: ${REPEL}
  mode: "${REPEL_MODE}"
  eta: ${ETA}
early_stop:
  use: true
  metric: em
  target: 1.0
  min_steps: 100
  check_every: 10
  patience_checks: 3
EOF

STAMP="$(date +"%Y%m%d-%H%M%S")"
RUN_DIR="${RESULTS_ROOT}/bench_default_b1820_joint_1216_sharded/Sadd${S_ADD}_Ladd${L_ADD}/g${GATE_SET}/${OPT}/${ROUTE}/${DIR}/mlp_${MATCH}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}_shard$(printf '%02d' "${shard_idx}")of$(printf '%02d' "${N_SHARDS}")"
mkdir -p "${RUN_DIR}"
cp "${CFG}" "${RUN_DIR}/config.used.yaml"

echo "[info] project=${PROJECT_DIR}"
echo "[info] dataset=${DATASET}"
echo "[info] shard_dataset=${SHARD_DATASET}"
echo "[info] combined_dataset=${COMBINED_DATASET}"
echo "[info] seed=${SEED} match=${MATCH} shard=$((shard_idx + 1))/${N_SHARDS} S_add=${S_ADD} L_add=${L_ADD}"
echo "[info] cfg=${CFG}"
echo "[info] out=${RUN_DIR}"

cd "${PROJECT_DIR}"

python - <<'PY'
import os, platform, sys
print("Python:", platform.python_version())
print("Python executable:", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
import torch
import yaml
import ubcircuit
print("torch:", torch.__version__)
print("yaml:", getattr(yaml, "__version__", "unknown"))
print("ubcircuit:", getattr(ubcircuit, "__file__", "unknown"))
PY

python -m ubcircuit.joint_mlp_ubc_bnr_1216 \
  --config "${CFG}" \
  --dataset "${SHARD_DATASET}" \
  --out_dir "${RUN_DIR}" \
  --mlp_match "${MATCH}" \
  --S_op add --S_k "${S_ADD}" --S_min 2 --S_max 128 \
  --L_op add --L_k "${L_ADD}" --L_min 2 --L_max 16

echo "[ok] done. summary: ${RUN_DIR}/summary.json"
