#!/bin/bash
#SBATCH --job-name=ubc_rerun_best_gates
#SBATCH --account=def-ssanner
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%j.out

set -euo pipefail

# -------- wall-clock timer ----------
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

# -------- best setup (from grid search) ----------
GATE_SET=16
OPT=rmsprop
ROUTE=mi_soft
DIR=top_down
REPEL=false
REPEL_MODE=log          # ignored when repel=false, but keep for logging
CONST_NAME=none
LAM_CONST16=0.0

# fixed knobs (match your grid job)
STEPS=3000
LR=0.001
ETA=2.0
LAM_ENT=1.0e-3
LAM_DIV_U=5.0e-4
LAM_DIV_R=5.0e-4
PRIOR_STRENGTH=2.0
MI_DISJOINT=true

# anneal (match your grid job for g16)
T0=0.60
TMIN=0.06
SCHED="cosine"
PHASE=0.5

# sigma16 (your grid job defaults)
S16_START=0.25
S16_END=0.10
S16_MODE="rbf"
S16_RADIUS=0.75

# IMPORTANT: rerun with the SAME seed as the recorded best run
SEED=4

# -------- config YAML (written to SLURM_TMPDIR) ----------
TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_best_g${GATE_SET}_${OPT}_${ROUTE}_${DIR}_rep${REPEL}_${CONST_NAME}_seed${SEED}.yaml"

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
  echo "  start_frac: 0.0"
  echo "sigma16:"
  echo "  s_start: ${S16_START}"
  echo "  s_end:   ${S16_END}"
  echo "  mode:    \"${S16_MODE}\""
  echo "  radius:  ${S16_RADIUS}"
  echo "regs:"
  echo "  lam_entropy: ${LAM_ENT}"
  echo "  lam_div_units: ${LAM_DIV_U}"
  echo "  lam_div_rows: ${LAM_DIV_R}"
  echo "  lam_const16: ${LAM_CONST16}"
  echo "pair:"
  echo "  route: mi_soft"
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

# -------- output dir ----------
STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="${RESULTS_ROOT}/rerun_best_gates/g${GATE_SET}/${OPT}/${ROUTE}/${DIR}/log_rep${REPEL}/${CONST_NAME}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
mkdir -p "${RUN_DIR}"

# -------- env info ----------
python - <<'PY'
import os, platform
print("Python:", platform.python_version())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PY

# -------- train ----------
cd "${PROJECT_DIR}"
echo "[info] Re-running best setup with gate_usage patch -> ${RUN_DIR}"
python -m ubcircuit.train \
  --dataset "${DATASET}" \
  --use_row_L \
  --config "${CFG}" \
  --out_dir "${RUN_DIR}"

echo "[info] Training done. summary: ${RUN_DIR}/summary.json"

# -------- gate analysis ----------
# This script expects the patched summary.json (with gate_usage)
# If you named it differently, adjust the path below.
ANALYSIS_OUT="${RUN_DIR}/analysis_gates"
mkdir -p "${ANALYSIS_OUT}"

echo "[info] Running gate usage analysis -> ${ANALYSIS_OUT}"
python "${PROJECT_DIR}/scripts/analyze_ubc_results_gate_usage.py" \
  "${RUN_DIR}" \
  --out "${ANALYSIS_OUT}"

echo "[ok] Gate analysis complete. Outputs:"
ls -lh "${ANALYSIS_OUT}"
