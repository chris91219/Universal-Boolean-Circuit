#!/bin/bash
#SBATCH --job-name=ubc_mi_all_rms
#SBATCH --account=def-ssanner
#SBATCH --time=0-04:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-44               # 3 repel x 3 const x 5 seeds = 45
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%A_%a.out

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
LOG_DIR=${RESULTS_ROOT}/slurm-logs
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
# factors:
#   REPEL_MODE ∈ {none, log, hard-log}          (#=3)
#   CONST_LEVEL ∈ {none, weak, strong}          (#=3)
#   SEED ∈ {0..4}                                (#=5)
# index formula: (((repel_i)*3 + const_i)*5 + seed)
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$(( TASK_ID % 5 ))
v=$(( TASK_ID / 5 ))
CONST_I=$(( v % 3 )); v=$(( v / 3 ))
REPEL_I=$(( v % 3 ))

case "${REPEL_I}" in
  0) REPEL=false; REPEL_MODE="log" ;;  # mode ignored when repel=false
  1) REPEL=true;  REPEL_MODE="log" ;;
  2) REPEL=true;  REPEL_MODE="hard-log" ;;
esac

case "${CONST_I}" in
  0) CONST_NAME=none;   LAM_CONST16=0.0    ;;
  1) CONST_NAME=weak;   LAM_CONST16=1.0e-3 ;;
  2) CONST_NAME=strong; LAM_CONST16=5.0e-3 ;;
esac

# -------- training knobs (g16) ----------
GATE_SET=16
STEPS=2000
LR=0.001
# anneal (g16): stronger + cosine
T0=0.60; TMIN=0.06; SCHED="cosine"; PHASE=0.5
# sigma16 bandwidth anneal
S16_START=0.25; S16_END=0.10
# regs (same as your “reasonable” defaults; lam_const16 overridden per grid)
LAM_ENT=1.0e-3
LAM_DIV_U=5.0e-4
LAM_DIV_R=5.0e-4
ETA=2.0

# -------- temp config ----------
TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_g${GATE_SET}_${REPEL_MODE}_rep${REPEL}_const${CONST_NAME}_seed${SEED}.yaml"

{
  echo "seed: ${SEED}"
  echo "gate_set: \"${GATE_SET}\""
  echo "steps: ${STEPS}"
  echo "optimizer: rmsprop"
  echo "lr: ${LR}"
  echo "use_row_L: true"
  echo "anneal:"
  echo "  T0: ${T0}"
  echo "  Tmin: ${TMIN}"
  echo "  direction: top_down"
  echo "  schedule: ${SCHED}"
  echo "  phase_scale: ${PHASE}"
  echo "sigma16:"
  echo "  s_start: ${S16_START}"
  echo "  s_end:   ${S16_END}"
  echo "regs:"
  echo "  lam_entropy: ${LAM_ENT}"
  echo "  lam_div_units: ${LAM_DIV_U}"
  echo "  lam_div_rows: ${LAM_DIV_R}"
  echo "  lam_const16: ${LAM_CONST16}"          # <<< configurable const penalty
  echo "pair:"
  echo "  route: learned"                       # MI off for this grid; can add later
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

# -------- run name / output dir ----------
STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="${RESULTS_ROOT}/repel_const_ablation/g${GATE_SET}/${REPEL_MODE}_rep${REPEL}_rms/${CONST_NAME}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
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
echo "[info] gate_set=${GATE_SET} repel=${REPEL} mode=${REPEL_MODE} const=${CONST_NAME}(${LAM_CONST16}) seed=${SEED}"
python -m ubcircuit.train \
  --dataset "${DATASET}" \
  --use_row_L \
  --config "${CFG}" \
  --out_dir "${RUN_DIR}"

echo "Done. Results at: ${RUN_DIR}/summary.json"
