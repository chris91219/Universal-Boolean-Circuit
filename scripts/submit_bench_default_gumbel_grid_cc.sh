#!/bin/bash
# UBC-only decoded-EM hunt on Compute Canada.
# Phase-1 grid: softmax controls + Gumbel/STE hardening configs across 5 seeds.
#
# Submit from repo root:
#   sbatch scripts/submit_bench_default_gumbel_grid_cc.sh

#SBATCH --job-name=ubc_gumbel_grid
#SBATCH --account=def-ssanner
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-79
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
SEEDS=(1 2 3 4 5)

# Fields:
# name mode hard eval_hard gumbel_tau S_add L_add Tmin lam_entropy lift_use steps
CONFIGS=(
  "soft_s0_liftT softmax false false 1.0 0 0 0.06 0.001 true 3000"
  "soft_s10_liftF softmax false false 1.0 10 0 0.06 0.001 false 3000"
  "gumbel_s0_tau1_T006_liftT gumbel true true 1.0 0 0 0.06 0.001 true 3000"
  "gumbel_s0_tau05_T003_liftT gumbel true true 0.5 0 0 0.03 0.001 true 3000"
  "gumbel_s4_tau1_T006_liftT gumbel true true 1.0 4 0 0.06 0.001 true 3000"
  "gumbel_s4_tau05_T003_liftT gumbel true true 0.5 4 0 0.03 0.001 true 3000"
  "gumbel_s10_tau1_T006_liftF gumbel true true 1.0 10 0 0.06 0.001 false 3000"
  "gumbel_s10_tau05_T003_liftF gumbel true true 0.5 10 0 0.03 0.001 false 3000"
  "gumbel_s10_tau03_T0015_liftF gumbel true true 0.3 10 0 0.015 0.001 false 3000"
  "gumbel_s10_tau05_T003_ent005_liftF gumbel true true 0.5 10 0 0.03 0.005 false 3000"
  "gumbel_s6_tau05_T003_liftF gumbel true true 0.5 6 0 0.03 0.001 false 3000"
  "gumbel_s14_tau05_T003_liftF gumbel true true 0.5 14 0 0.03 0.001 false 3000"
  "argmax_s4_T003_liftT argmax_ste true true 1.0 4 0 0.03 0.001 true 3000"
  "argmax_s10_T003_liftF argmax_ste true true 1.0 10 0 0.03 0.001 false 3000"
  "gumbel_s10_tau05_T003_liftT gumbel true true 0.5 10 0 0.03 0.001 true 3000"
  "gumbel_s4_tau05_T0015_ent005_liftT gumbel true true 0.5 4 0 0.015 0.005 true 3000"
)

NSEED=${#SEEDS[@]}
NCONFIG=${#CONFIGS[@]}
TOTAL=$((NSEED * NCONFIG))

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  exit 1
fi

config_idx=$((SLURM_ARRAY_TASK_ID % NCONFIG))
seed_idx=$((SLURM_ARRAY_TASK_ID / NCONFIG))
SEED="${SEEDS[$seed_idx]}"
read -r CFG_NAME RELAX_MODE RELAX_HARD EVAL_HARD GUMBEL_TAU S_ADD L_ADD TMIN LAM_ENT LIFT_USE STEPS <<< "${CONFIGS[$config_idx]}"

GATE_SET=16
OPT=rmsprop
ROUTE=mi_soft
DIR=top_down
REPEL=false
REPEL_MODE=log
LAM_CONST16=0.0
LR=0.001
ETA=2.0
LAM_DIV_U=5.0e-4
LAM_DIV_R=5.0e-4
PRIOR_STRENGTH=2.0
MI_DISJOINT=true
T0=0.60
SCHED=cosine
PHASE=0.5
S16_START=0.25
S16_END=0.10
S16_MODE=rbf
S16_RADIUS=0.75
LIFT_FACTOR=2.0

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_gumbel_grid_${CFG_NAME}_seed${SEED}.yaml"

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
  S_min: 2
  S_max: 128
  L_op: add
  L_k: ${L_ADD}
  L_min: 2
  L_max: 16
lifting:
  use: ${LIFT_USE}
  factor: ${LIFT_FACTOR}
relaxation:
  mode: ${RELAX_MODE}
  hard: ${RELAX_HARD}
  gumbel_tau: ${GUMBEL_TAU}
  eval_hard: ${EVAL_HARD}
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
RUN_DIR="${RESULTS_ROOT}/bench_default_gumbel_grid/${CFG_NAME}/seed${SEED}/${STAMP}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
mkdir -p "${RUN_DIR}"
cp "${CFG}" "${RUN_DIR}/config.used.yaml"

echo "[info] project=${PROJECT_DIR}"
echo "[info] dataset=${DATASET}"
echo "[info] config=${CFG_NAME} idx=${config_idx}/${NCONFIG}"
echo "[info] seed=${SEED}"
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
