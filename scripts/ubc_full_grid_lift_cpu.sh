#!/bin/bash
#SBATCH --job-name=ubc_lift_full_grid_cpu
#SBATCH --account=def-ssanner
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# 2 gate x 2 opt x 3 route x 2 dir x 3 repel x 3 const x 5 seeds = 1080
#SBATCH --array=0-1079
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs-lift/%x-%A_%a.out

set -euo pipefail

# ---- wall-clock timer ----
START_ISO="$(date -Is)"; START_SEC=$SECONDS
on_exit () {
  local end_iso dur h m s
  end_iso="$(date -Is)"; dur=$((SECONDS - START_SEC))
  h=$((dur/3600)); m=$(((dur%3600)/60)); s=$((dur%60))
  printf "\n[Timer] Start: %s\n[Timer] End  : %s\n[Timer] Elapsed: %02d:%02d:%02d (%d s)\n" \
         "$START_ISO" "$end_iso" "$h" "$m" "$s" "$dur"
}
trap on_exit EXIT

# -------- paths / env ----------
PROJECT_DIR=/home/uoftwhli/projects/def-ssanner/uoftwhli/Universal-Boolean-Circuit
RESULTS_ROOT=/home/uoftwhli/scratch/UBC-Results
LOG_DIR=${RESULTS_ROOT}/slurm-logs-lift
mkdir -p "${LOG_DIR}"

module purge
module load StdEnv/2023
module load python/3.11
source "${PROJECT_DIR}/ENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""

# -------- dataset ----------
DATASET="${PROJECT_DIR}/data/bench_default.jsonl"

# -------- grid decoding ----------
# Axes (index order from fastest to slowest):
#   seed ∈ {0..4}                          (5)
#   const ∈ {none, weak, strong}           (3)
#   repel ∈ {none, log, hard-log}          (3)
#   dir ∈ {top_down, bottom_up}            (2)
#   route ∈ {learned, mi_soft, mi_hard}    (3)
#   opt ∈ {adam, rmsprop}                  (2)
#   gate ∈ {6, 16}                         (2)
TASK_ID=${SLURM_ARRAY_TASK_ID}
seed=$(( TASK_ID % 5 ))
v=$(( TASK_ID / 5 ))

CONST_I=$(( v % 3 )); v=$(( v / 3 ))
REPEL_I=$(( v % 3 )); v=$(( v / 3 ))
DIR_I=$(( v % 2 ));   v=$(( v / 2  ))
ROUTE_I=$(( v % 3 )); v=$(( v / 3  ))
OPT_I=$(( v % 2 ));   v=$(( v / 2  ))
GATE_I=$(( v % 2 ))

# Decode gate
case "${GATE_I}" in
  0) GATE_SET=6 ;;
  1) GATE_SET=16 ;;
esac

# Decode optimizer
case "${OPT_I}" in
  0) OPT=adam ;;
  1) OPT=rmsprop ;;
esac

# Decode routing
case "${ROUTE_I}" in
  0) ROUTE="learned" ;;
  1) ROUTE="mi_soft" ;;
  2) ROUTE="mi_hard" ;;
esac

# Decode direction
case "${DIR_I}" in
  0) DIR="top_down" ;;
  1) DIR="bottom_up" ;;
esac

# Decode repulsion
case "${REPEL_I}" in
  0) REPEL=false; REPEL_MODE="log" ;;   # mode ignored when REPEL=false
  1) REPEL=true;  REPEL_MODE="log" ;;
  2) REPEL=true;  REPEL_MODE="hard-log" ;;
esac

# Decode const penalty (16-way only; ignored by g6)
case "${CONST_I}" in
  0) CONST_NAME=none;   LAM_CONST16=0.0    ;;
  1) CONST_NAME=weak;   LAM_CONST16=1.0e-3 ;;
  2) CONST_NAME=strong; LAM_CONST16=5.0e-3 ;;
esac

# -------- fixed knobs ----------
STEPS=3000
LR=0.001         # fix LR for both optimizers to isolate 'opt' effect
ETA=2.0          # repulsion strength
LAM_ENT=1.0e-3
LAM_DIV_U=5.0e-4
LAM_DIV_R=5.0e-4
# MI-soft defaults
PRIOR_STRENGTH=2.0
MI_DISJOINT=true

# Anneal per gate_set
if [[ "${GATE_SET}" == 16 ]]; then
  T0=0.60; TMIN=0.06; SCHED="cosine"; PHASE=0.5
  S16_START=0.25; S16_END=0.10
else
  T0=0.35; TMIN=0.12; SCHED="linear"; PHASE=0.4
fi

# -------- build run config ----------
TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_lift_g${GATE_SET}_${OPT}_${ROUTE}_${DIR}_${REPEL_MODE}_rep${REPEL}_${CONST_NAME}_seed${seed}.yaml"

{
  echo "seed: ${seed}"
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
  if [[ "${GATE_SET}" == 16 ]]; then
    echo "sigma16:"
    echo "  s_start: ${S16_START}"
    echo "  s_end:   ${S16_END}"
  fi
  echo "lifting:"
  echo "  use: true"
  echo "  factor: 2.0"
  echo "regs:"
  echo "  lam_entropy: ${LAM_ENT}"
  echo "  lam_div_units: ${LAM_DIV_U}"
  echo "  lam_div_rows: ${LAM_DIV_R}"
  if [[ "${GATE_SET}" == 16 ]]; then
    echo "  lam_const16: ${LAM_CONST16}"
  fi
  echo "pair:"
  # Routing selection
  if [[ "${ROUTE}" == "mi_soft" ]]; then
    echo "  route: mi_soft"
    echo "  prior_strength: ${PRIOR_STRENGTH}"
    echo "  mi_disjoint: ${MI_DISJOINT}"
  elif [[ "${ROUTE}" == "mi_hard" ]]; then
    echo "  route: mi_hard"
    echo "  mi_disjoint: ${MI_DISJOINT}"
  else
    echo "  route: learned"
  fi
  # Repulsion (learned & mi_soft still use it; mi_hard ignores at runtime)
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

# -------- output dir ----------
STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="${RESULTS_ROOT}/lift_full_grid/g${GATE_SET}/${OPT}/${ROUTE}/${DIR}/${REPEL_MODE}_rep${REPEL}/${CONST_NAME}/${STAMP}_job${SLURM_JOB_ID}_seed${seed}"
mkdir -p "${RUN_DIR}"

# -------- env info ----------
python - <<'PY'
import os, platform
print("Python:", platform.python_version())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PY

# -------- launch ----------
cd "${PROJECT_DIR}"
echo "[info] (LIFT) gate_set=${GATE_SET} opt=${OPT} route=${ROUTE} dir=${DIR} repel=${REPEL} mode=${REPEL_MODE} const=${CONST_NAME}(${LAM_CONST16}) seed=${seed}"
python -m ubcircuit.train \
  --dataset "${DATASET}" \
  --use_row_L \
  --config "${CFG}" \
  --out_dir "${RUN_DIR}"

echo "Done. Results at: ${RUN_DIR}/summary.json"
