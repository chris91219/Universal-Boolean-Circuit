#!/bin/bash
# Main/old soft SBC grid over S_add and L_add on Compute Canada.
# This keeps the original softmax relaxation, but uses the new decoded-EM
# early stopping and decoded-readout diagnostics.
#
# Submit from repo root:
#   sbatch scripts/submit_bench_default_main_sadd_ladd_grid_cc.sh

#SBATCH --job-name=ubc_main_sl_grid
#SBATCH --account=def-ssanner
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-179
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

DATASET="${PROJECT_DIR}/data/bench_default.jsonl"

# Best soft main setup so far, with only S/L capacity varied.
GATE_SET=16
OPT=rmsprop
ROUTE=mi_soft
DIR=top_down
REPEL=false
REPEL_MODE=log
LAM_CONST16=0.0
STEPS=3000
LR=0.001
ETA=2.0
LAM_ENT=1.0e-3
LAM_DIV_U=5.0e-4
LAM_DIV_R=5.0e-4
PRIOR_STRENGTH=2.0
MI_DISJOINT=true
T0=0.60
TMIN=0.06
SCHED=cosine
PHASE=0.5
S16_START=0.25
S16_END=0.10
S16_MODE=rbf
S16_RADIUS=0.75
LIFT_USE=true
LIFT_FACTOR=2.0

# Phase-1 decoded-EM capacity grid.
# Negative S_add is intentional: narrower circuits may decode more faithfully.
S_ADD_LIST=(-2 -1 0 1 2 4 6 8 10)
L_ADD_LIST=(-1 0 1 2)
SEEDS=(1 2 3 4 5)

NS=${#S_ADD_LIST[@]}
NL=${#L_ADD_LIST[@]}
NSEED=${#SEEDS[@]}
TOTAL=$((NS * NL * NSEED))

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  exit 1
fi

seed_idx=$((SLURM_ARRAY_TASK_ID % NSEED))
cfg_idx=$((SLURM_ARRAY_TASK_ID / NSEED))
s_idx=$((cfg_idx % NS))
l_idx=$((cfg_idx / NS))

S_ADD="${S_ADD_LIST[$s_idx]}"
L_ADD="${L_ADD_LIST[$l_idx]}"
SEED="${SEEDS[$seed_idx]}"

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_main_sl_grid_Sadd${S_ADD}_Ladd${L_ADD}_seed${SEED}.yaml"

cat > "${CFG}" <<EOF
seed: ${SEED}
gate_set: "${GATE_SET}"
steps: ${STEPS}
optimizer: ${OPT}
lr: ${LR}
use_row_L: true
use_max_L: false
scale:
  use_row_S: true
  S_op: add
  S_k: ${S_ADD}
  S_min: 1
  S_max: 128
  L_op: add
  L_k: ${L_ADD}
  L_min: 2
  L_max: 16
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
  metric: decoded_em
  target: 1.0
  min_steps: 100
  check_every: 10
  patience_checks: 3
EOF

STAMP="$(date +"%Y%m%d-%H%M%S")"
RUN_DIR="${RESULTS_ROOT}/bench_default_main_sadd_ladd_grid/softmax/g${GATE_SET}/${OPT}/${ROUTE}/${DIR}/lift_${LIFT_USE}/Sadd${S_ADD}_Ladd${L_ADD}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
mkdir -p "${RUN_DIR}"
cp "${CFG}" "${RUN_DIR}/config.used.yaml"

echo "[info] project=${PROJECT_DIR}"
echo "[info] dataset=${DATASET}"
echo "[info] S_add=${S_ADD} L_add=${L_ADD} seed=${SEED}"
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

python -m ubcircuit.train \
  --dataset "${DATASET}" \
  --use_row_L \
  --config "${CFG}" \
  --out_dir "${RUN_DIR}"

echo "[ok] done. summary: ${RUN_DIR}/summary.json"
