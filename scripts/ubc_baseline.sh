#!/bin/bash
#SBATCH --job-name=ubc_baseline_grid
#SBATCH --account=def-ssanner
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-79
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%A_%a.out
set -euo pipefail

PROJECT_DIR=/home/uoftwhli/projects/def-ssanner/uoftwhli/Universal-Boolean-Circuit
RESULTS_ROOT=/home/uoftwhli/scratch/UBC-Results
DATASET="${PROJECT_DIR}/data/bench_default.jsonl"

module purge; module load StdEnv/2023 python/3.11
source "${PROJECT_DIR}/ENV/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""

# -----------------------------------------
# Axes:
#   gate_set      ∈ {6, 16}             (2)
#   optimizer     ∈ {adam, rmsprop}     (2)
#   baseline      ∈ {mlp, transformer}  (2)
#   match_mode    ∈ {soft, total}       (2)
#   seed          ∈ {0,1,2,3,4}         (5)
# TASK_ID: 0..79
# -----------------------------------------

gate_set_arr=(6 16)
opt_arr=("adam" "rmsprop")
baseline_arr=("mlp" "transformer")
match_mode_arr=("soft" "total")
seed_arr=(0 1 2 3 4)

TASK_ID=${SLURM_ARRAY_TASK_ID}

seed_i=$(( TASK_ID % 5 )); v=$(( TASK_ID / 5 ))
match_i=$(( v % 2 ));       v=$(( v / 2 ))
base_i=$(( v % 2 ));        v=$(( v / 2 ))
opt_i=$(( v % 2 ));         v=$(( v / 2 ))
gate_i=$(( v % 2 ))

SEED=${seed_arr[$seed_i]}
MATCH_MODE=${match_mode_arr[$match_i]}
BASELINE=${baseline_arr[$base_i]}
OPT=${opt_arr[$opt_i]}
GATE_SET=${gate_set_arr[$gate_i]}

echo "TASK_ID=${TASK_ID} -> gate_set=${GATE_SET}, opt=${OPT}, baseline=${BASELINE}, match_mode=${MATCH_MODE}, seed=${SEED}"

# -----------------------------------------
# Hyperparameters (aligned with best g16 run style)
# -----------------------------------------
ROUTE=mi_soft
DIR=top_down
REPEL=false
REPEL_MODE=log
ETA=2.0
PRIOR_STRENGTH=2.0
MI_DISJOINT=true

STEPS=3000
LR=0.001   # same LR for both adam/rmsprop for clean comparison

LAM_ENT=1.0e-3
LAM_DIV_U=5.0e-4
LAM_DIV_R=5.0e-4
LAM_CONST16=0.0  # "none" constant penalty

# Anneal (only relevant for UBC param-count config; baselines ignore tau)
T0=0.60
TMIN=0.06
SCHED="cosine"
PHASE=0.5

S16_START=0.25
S16_END=0.10
RADIUS=0.75   # irrelevant if mode=rbf but harmless

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_baseline_g${GATE_SET}_${OPT}_${BASELINE}_${MATCH_MODE}_seed${SEED}.yaml"

# -----------------------------------------
# Write ephemeral YAML config (for UBC param-count)
# -----------------------------------------
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
  echo "  mode: \"rbf\""          # fixed; no bump/lagrange grid
  echo "  s_start: ${S16_START}"
  echo "  s_end:   ${S16_END}"
  echo "  radius:  ${RADIUS}"
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
RUN_DIR="${RESULTS_ROOT}/baselines/full_grid/g${GATE_SET}/${BASELINE}_${MATCH_MODE}/${OPT}/${ROUTE}/${DIR}/seed${SEED}/${STAMP}_job${SLURM_JOB_ID}_task${TASK_ID}"
mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"

python -m ubcircuit.baselines \
  --dataset "${DATASET}" \
  --use_row_L \
  --config "${CFG}" \
  --out_dir "${RUN_DIR}" \
  --baseline "${BASELINE}" \
  --match_mode "${MATCH_MODE}"

echo "Done -> ${RUN_DIR}/summary.json"
