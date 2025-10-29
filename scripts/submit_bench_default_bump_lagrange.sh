#!/bin/bash
#SBATCH --job-name=ubc_mode_cmp
#SBATCH --account=def-ssanner
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-35
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%A_%a.out
set -euo pipefail

PROJECT_DIR=/home/uoftwhli/projects/def-ssanner/uoftwhli/Universal-Boolean-Circuit
RESULTS_ROOT=/home/uoftwhli/scratch/UBC-Results
DATASET="${PROJECT_DIR}/data/bench_default.jsonl"

module purge; module load StdEnv/2023 python/3.11
source "${PROJECT_DIR}/ENV/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""

# axes: seed(4) × const(3) × mode(3)
TASK_ID=${SLURM_ARRAY_TASK_ID}
seed_set=(0 1 2 4)
mode_set=("rbf" "bump" "lagrange")
const_name_set=("none" "weak" "strong")
const_val_set=(0.0 1.0e-3 5.0e-3)

seed_i=$(( TASK_ID % 4 )); v=$(( TASK_ID / 4 ))
const_i=$(( v % 3 ));     v=$(( v / 3 ))
mode_i=$(( v % 3  ))

SEED=${seed_set[$seed_i]}
MODE=${mode_set[$mode_i]}
CONST_NAME=${const_name_set[$const_i]}
LAM_CONST16=${const_val_set[$const_i]}

# fixed best-ish knobs from your table
GATE_SET=16
OPT=adam
ROUTE=mi_soft
DIR=top_down
REPEL=false
REPEL_MODE=log
ETA=2.0
PRIOR_STRENGTH=2.0
MI_DISJOINT=true
STEPS=3000
LR=0.001
LAM_ENT=1.0e-3
LAM_DIV_U=5.0e-4
LAM_DIV_R=5.0e-4

# anneal (your g16 defaults)
T0=0.60; TMIN=0.06; SCHED="cosine"; PHASE=0.5
S16_START=0.25; S16_END=0.10
RADIUS=0.75   # for bump only

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"; mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_modes_seed${SEED}_${MODE}_${CONST_NAME}.yaml"
{
  echo "seed: ${SEED}"
  echo "gate_set: \"${GATE_SET}\""
  echo "steps: ${STEPS}"
  echo "optimizer: ${OPT}"
  echo "lr: ${LR}"
  echo "use_row_L: true"
  echo "anneal:"
  echo "  T0: ${T0}"
  echo "  Tmin: ${TMIN}"
  echo "  direction: ${DIR}"
  echo "  schedule: ${SCHED}"
  echo "  phase_scale: ${PHASE}"
  echo "sigma16:"
  echo "  mode: \"${MODE}\""         # <- NEW
  echo "  s_start: ${S16_START}"     # used in rbf only
  echo "  s_end:   ${S16_END}"
  echo "  radius:  ${RADIUS}"        # used in bump only
  echo "regs:"
  echo "  lam_entropy: ${LAM_ENT}"
  echo "  lam_div_units: ${LAM_DIV_U}"
  echo "  lam_div_rows: ${LAM_DIV_R}"
  echo "  lam_const16: ${LAM_CONST16}"
  echo "pair:"
  echo "  route: ${ROUTE}"
  echo "  prior_strength: ${PRIOR_STRENGTH}"
  echo "  mi_disjoint: ${MI_DISJOINT}"
  echo "  repel: ${REPEL}"
  echo "  mode: \"${REPEL_MODE}\""
  echo "  eta: ${ETA}"
  echo "early_stop:"
  echo "  use: true"
  echo "  metric: em"
  echo "  target: 1.0"
  echo "  min_steps: 100"
  echo "  check_every: 10"
  echo "  patience_checks: 3"
} > "${CFG}"

STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="${RESULTS_ROOT}/mode_cmp/g16/${OPT}/${ROUTE}/${DIR}/${MODE}/${CONST_NAME}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"

python -m ubcircuit.train --dataset "${DATASET}" --use_row_L --config "${CFG}" --out_dir "${RUN_DIR}"
echo "Done -> ${RUN_DIR}/summary.json"
