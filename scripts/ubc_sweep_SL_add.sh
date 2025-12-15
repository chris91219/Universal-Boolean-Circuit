#!/bin/bash
#SBATCH --job-name=ubc_sweep_SL_add
#SBATCH --account=def-ssanner
#SBATCH --time=0-02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-35
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%A_%a.out

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
# If using the minimized dataset:
# DATASET="${PROJECT_DIR}/data/bench_min_g16.jsonl"

# -------- base setup (from your best rerun) ----------
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

# -------- additive sweep grid ----------
S_ADD_LIST=(0 2 4 6)   # S_used = S_base + S_add
L_ADD_LIST=(0 1 2)     # L_used = L_base + L_add
SEEDS=(1 2 3)          # 3 seeds per setup

NS=${#S_ADD_LIST[@]}
NL=${#L_ADD_LIST[@]}
NSEED=${#SEEDS[@]}
TOTAL=$((NS*NL*NSEED))

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  exit 1
fi

seed_idx=$((SLURM_ARRAY_TASK_ID % NSEED))
cfg_idx=$((SLURM_ARRAY_TASK_ID / NSEED))
s_idx=$((cfg_idx % NS))
l_idx=$((cfg_idx / NS))

S_ADD=${S_ADD_LIST[$s_idx]}
L_ADD=${L_ADD_LIST[$l_idx]}
SEED=${SEEDS[$seed_idx]}

# -------- config YAML (written to SLURM_TMPDIR) ----------
TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_addSL_g${GATE_SET}_Sadd${S_ADD}_Ladd${L_ADD}_seed${SEED}.yaml"

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
  echo "  factor: 2.0"   # ignored when use=false, keep for completeness

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
  echo "  S_op: add"
  echo "  S_k: ${S_ADD}"
  echo "  S_min: 2"
  echo "  S_max: 64"
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

# -------- output dir ----------
STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="${RESULTS_ROOT}/sweep_SL_add/g${GATE_SET}/${OPT}/${ROUTE}/${DIR}/Sadd${S_ADD}_Ladd${L_ADD}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
mkdir -p "${RUN_DIR}"

echo "[info] sweep(add): S_add=${S_ADD} L_add=${L_ADD} seed=${SEED}"
echo "[info] cfg: ${CFG}"
echo "[info] out: ${RUN_DIR}"

# -------- train ----------
cd "${PROJECT_DIR}"
python -m ubcircuit.train \
  --dataset "${DATASET}" \
  --use_row_L \
  --config "${CFG}" \
  --out_dir "${RUN_DIR}"

echo "[ok] done. summary: ${RUN_DIR}/summary.json"
