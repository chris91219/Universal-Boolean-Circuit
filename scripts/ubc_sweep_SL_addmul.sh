#!/bin/bash
#SBATCH --job-name=ubc_sweep_SL_addmul
#SBATCH --account=def-ssanner
#SBATCH --time=0-09:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-41
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
# DATASET="${PROJECT_DIR}/data/bench_min_g16.jsonl"

# ---- best baseline knobs ----
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

# ---- sweep sets ----
L_ADD_LIST=(0 1)
S_ADD_LIST=(0 2 4 6 8)
S_MULT_LIST=(2 3)
SEEDS=(1 2 3)

# ---- build config list: each entry "mode,S_k,L_add" ----
CONFIGS=()
for LADD in "${L_ADD_LIST[@]}"; do
  for SADD in "${S_ADD_LIST[@]}"; do
    CONFIGS+=("add,${SADD},${LADD}")
  done
done
for LADD in "${L_ADD_LIST[@]}"; do
  for SMUL in "${S_MULT_LIST[@]}"; do
    CONFIGS+=("mul,${SMUL},${LADD}")
  done
done

NCONF=${#CONFIGS[@]}
NSEED=${#SEEDS[@]}
TOTAL=$((NCONF*NSEED))

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  exit 1
fi

seed_idx=$((SLURM_ARRAY_TASK_ID % NSEED))
cfg_idx=$((SLURM_ARRAY_TASK_ID / NSEED))

SEED=${SEEDS[$seed_idx]}
IFS=',' read -r MODE S_K L_ADD <<< "${CONFIGS[$cfg_idx]}"

# YAML scaling
# MODE=add -> S_op=add, S_k=S_add
# MODE=mul -> S_op=mul, S_k=S_mult
S_OP="${MODE}"

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_${MODE}_Sk${S_K}_Ladd${L_ADD}_seed${SEED}.yaml"

{
  echo "seed: ${SEED}"
  echo "gate_set: \"${GATE_SET}\""
  echo "steps: ${STEPS}"
  echo "optimizer: ${OPT}"
  echo "lr: ${LR}"
  echo "use_row_L: true"
  echo "use_max_L: false"

  # ---- FORCE NO LIFTING ----
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

  echo "scale:"
  echo "  use_row_S: true"
  echo "  S_op: ${S_OP}"
  echo "  S_k: ${S_K}"
  echo "  S_min: 2"
  echo "  S_max: 128"
  echo "  L_op: add"
  echo "  L_k: ${L_ADD}"
  echo "  L_min: 2"
  echo "  L_max: 16"

  echo "early_stop:"
  echo "  use: true"
  echo "  metric: em"
  echo "  target: 1.0"
  echo "  min_steps: 100"
  echo "  check_every: 10"
  echo "  patience_checks: 3"
} > "${CFG}"

STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="${RESULTS_ROOT}/sweep_SL_addmul/g${GATE_SET}/${OPT}/${ROUTE}/${DIR}/${MODE}_Sk${S_K}_Ladd${L_ADD}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
mkdir -p "${RUN_DIR}"

echo "[info] sweep: MODE=${MODE} S_k=${S_K} L_add=${L_ADD} seed=${SEED}"
echo "[info] cfg: ${CFG}"
echo "[info] out: ${RUN_DIR}"

cd "${PROJECT_DIR}"
python -m ubcircuit.train \
  --dataset "${DATASET}" \
  --use_row_L \
  --config "${CFG}" \
  --out_dir "${RUN_DIR}"

echo "[ok] done. summary: ${RUN_DIR}/summary.json"
