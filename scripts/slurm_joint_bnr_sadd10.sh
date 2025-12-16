#!/bin/bash
#SBATCH --job-name=ubc_joint_bnr_Sadd10
#SBATCH --account=def-ssanner
#SBATCH --time=0-10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-14
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%A_%a.out

set -euo pipefail

START_ISO="$(date -Is)"; START_SEC=$SECONDS
on_exit () {
  local end_iso dur h m s
  end_iso="$(date -Is)"; dur=$((SECONDS - START_SEC))
  h=$((dur/3600)); m=$(((dur%3600)/60)); s=$((dur%60))
  printf "\n[Timer] Start: %s\n[Timer] End  : %s\n[Timer] Elapsed: %02d:%02d:%02d (%d s)\n" \
         "$START_ISO" "$end_iso" "$h" "$m" "$s" "$dur"
}
trap on_exit EXIT

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

DATASET="${PROJECT_DIR}/data/bench_default.jsonl"

# ---- best UBC knobs (same as before) ----
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
SCHED="cosine"
PHASE=0.5

S16_START=0.25
S16_END=0.10
S16_MODE="rbf"
S16_RADIUS=0.75

# ---- fixed scaling for this experiment ----
S_ADD=10
L_ADD=0

# ---- sweep axes ----
SEEDS=(1 2 3 4 5)
MATCH_MODES=(neuron param_soft param_total)

NSEED=${#SEEDS[@]}
NMODE=${#MATCH_MODES[@]}
TOTAL=$((NSEED * NMODE))

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  exit 1
fi

mode_idx=$((SLURM_ARRAY_TASK_ID % NMODE))
seed_idx=$((SLURM_ARRAY_TASK_ID / NMODE))

SEED=${SEEDS[$seed_idx]}
MATCH=${MATCH_MODES[$mode_idx]}

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_joint_bnr_seed${SEED}.yaml"

# Config is shared across match modes; match mode is CLI arg
{
  echo "seed: ${SEED}"
  echo "gate_set: \"${GATE_SET}\""
  echo "steps: ${STEPS}"
  echo "optimizer: ${OPT}"
  echo "lr: ${LR}"
  echo "use_row_L: true"
  echo "use_max_L: false"

  # ---- FORCE NO LIFTING (as before) ----
  echo "lifting:"
  echo "  use: false"
  echo "  factor: 2.0"

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
RUN_DIR="${RESULTS_ROOT}/joint_bnr_Sadd${S_ADD}_Ladd${L_ADD}/g${GATE_SET}/${OPT}/${ROUTE}/${DIR}/mlp_${MATCH}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
mkdir -p "${RUN_DIR}"

echo "[info] seed=${SEED} match=${MATCH} S_add=${S_ADD} L_add=${L_ADD}"
echo "[info] cfg=${CFG}"
echo "[info] out=${RUN_DIR}"

cd "${PROJECT_DIR}"

python -m ubcircuit.joint_mlp_ubc_bnr \
  --config "${CFG}" \
  --dataset "${DATASET}" \
  --out_dir "${RUN_DIR}" \
  --mlp_match "${MATCH}" \
  --S_op add --S_k "${S_ADD}" --S_min 2 --S_max 128 \
  --L_op add --L_k "${L_ADD}" --L_min 2 --L_max 16 \
  --save_ckpts

echo "[ok] done. summary: ${RUN_DIR}/summary.json"
