#!/bin/bash
# Main NeurIPS soft-mixture sweep on Compute Canada.
#
# Goal:
#   Compare soft SBC against param-matched MLPs on the new 2000-row B<=10
#   benchmark, with enough seeds and S_add capacity to support the main-text
#   claim "SBC as an explicit soft mixture of Boolean circuits".
#
# Default grid:
#   dataset: data/bench_small_b10_2000_noise_add0.jsonl
#   model:   sbc, mlp_neuron, mlp_param_soft, mlp_param_total
#   S_add:   10,12,14,16,18
#   seeds:   1,2,3,4,5
#
# Submit from repo root:
#   sbatch scripts/submit_main_softmix_sweep_cc.sh
#
# Useful overrides:
#   S_ADD_LIST_STR="12 14 16 18" sbatch scripts/submit_main_softmix_sweep_cc.sh
#   DATASET=/path/to/dataset.jsonl sbatch scripts/submit_main_softmix_sweep_cc.sh

#SBATCH --job-name=ubc_main_softmix
#SBATCH --account=def-ssanner
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-99
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
  echo "[error] Submit from repo root or pass PROJECT_DIR=/path/to/Universal-Boolean-Circuit."
  exit 1
fi
source "${PROJECT_DIR}/ENV/bin/activate"
hash -r

export UBC_NUM_THREADS="${UBC_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"
export UBC_TORCH_THREADS="${UBC_TORCH_THREADS:-${UBC_NUM_THREADS}}"
export UBC_TORCH_INTEROP_THREADS="${UBC_TORCH_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${UBC_NUM_THREADS}"
export MKL_NUM_THREADS="${UBC_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${UBC_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${UBC_NUM_THREADS}"
export VECLIB_MAXIMUM_THREADS="${UBC_NUM_THREADS}"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

DATASET="${DATASET:-${PROJECT_DIR}/data/bench_small_b10_2000_noise_add0.jsonl}"
BASIS="${BASIS:-base}"
if [[ "${BASIS}" == "true" ]]; then
  W_FIELD="${W_FIELD:-W_true}"
  D_FIELD="${D_FIELD:-D_true}"
else
  W_FIELD="${W_FIELD:-W_base}"
  D_FIELD="${D_FIELD:-D_base}"
fi

read -r -a MODEL_MODES <<< "${MODEL_MODES_STR:-sbc mlp_neuron mlp_param_soft mlp_param_total}"
read -r -a S_ADD_LIST <<< "${S_ADD_LIST_STR:-10 12 14 16 18}"
read -r -a SEEDS <<< "${SEEDS_STR:-1 2 3 4 5}"

NMODEL=${#MODEL_MODES[@]}
NS=${#S_ADD_LIST[@]}
NSEED=${#SEEDS[@]}
TOTAL=$((NMODEL * NS * NSEED))

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  echo "[hint] Submit with --array=0-$((TOTAL - 1)) if you override grid sizes."
  exit 1
fi

task=${SLURM_ARRAY_TASK_ID}
model_idx=$((task % NMODEL)); task=$((task / NMODEL))
s_idx=$((task % NS)); task=$((task / NS))
seed_idx=$((task % NSEED))

MODEL_MODE="${MODEL_MODES[$model_idx]}"
S_ADD="${S_ADD_LIST[$s_idx]}"
SEED="${SEEDS[$seed_idx]}"
L_ADD="${L_ADD:-0}"

if [[ ! -f "${DATASET}" ]]; then
  echo "[error] Missing dataset: ${DATASET}"
  exit 1
fi

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"

CFG="${TMPDIR_USE}/cfg_main_softmix_${MODEL_MODE}_Sadd${S_ADD}_seed${SEED}.yaml"
cat > "${CFG}" <<EOF
seed: ${SEED}
gate_set: "16"
steps: ${STEPS:-3000}
optimizer: rmsprop
lr: ${LR:-0.001}
use_row_L: true
use_max_L: false
scale:
  use_row_S: true
  W_base_field: "${W_FIELD}"
  D_base_field: "${D_FIELD}"
  S_op: add
  S_k: ${S_ADD}
  S_min: 1
  S_max: 160
  L_op: add
  L_k: ${L_ADD}
  L_min: 2
  L_max: 16
lifting:
  use: ${LIFT_USE:-true}
  factor: ${LIFT_FACTOR:-2.0}
relaxation:
  mode: softmax
  hard: false
  gumbel_tau: 1.0
  eval_hard: false
anneal:
  T0: ${T0:-0.60}
  Tmin: ${TMIN:-0.06}
  direction: top_down
  schedule: cosine
  phase_scale: 0.5
  start_frac: 0.0
sigma16:
  s_start: 0.25
  s_end:   0.10
  mode:    "rbf"
  radius:  0.75
regs:
  lam_entropy: 1.0e-3
  lam_div_units: 5.0e-4
  lam_div_rows: 5.0e-4
  lam_const16: 0.0
pair:
  route: mi_soft
  prior_strength: 2.0
  mi_disjoint: true
  repel: false
  mode: "log"
  eta: 2.0
early_stop:
  use: true
  metric: ${EARLY_STOP_METRIC:-em}
  target: 1.0
  min_steps: 100
  check_every: 10
  patience_checks: 3
EOF

STAMP="$(date +"%Y%m%d-%H%M%S")"
RUN_DIR="${RESULTS_ROOT}/main_softmix_sweep/$(basename "${DATASET}" .jsonl)/basis_${BASIS}/Sadd${S_ADD}_Ladd${L_ADD}/${MODEL_MODE}/${STAMP}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_seed${SEED}"
mkdir -p "${RUN_DIR}"
cp "${CFG}" "${RUN_DIR}/config.used.yaml"

echo "[info] project=${PROJECT_DIR}"
echo "[info] dataset=${DATASET}"
echo "[info] basis=${BASIS} W_field=${W_FIELD} D_field=${D_FIELD}"
echo "[info] model=${MODEL_MODE} S_add=${S_ADD} L_add=${L_ADD} seed=${SEED}"
echo "[info] out=${RUN_DIR}"

cd "${PROJECT_DIR}"

python - <<'PY'
import os, platform, sys, torch, ubcircuit
print("Python:", platform.python_version())
print("Python executable:", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("UBC_TORCH_THREADS:", os.environ.get("UBC_TORCH_THREADS"))
print("torch:", torch.__version__)
print("ubcircuit:", getattr(ubcircuit, "__file__", "unknown"))
PY

if [[ "${MODEL_MODE}" == "sbc" ]]; then
  python -m ubcircuit.train \
    --dataset "${DATASET}" \
    --use_row_L \
    --config "${CFG}" \
    --out_dir "${RUN_DIR}"
else
  MATCH="${MODEL_MODE#mlp_}"
  python -m ubcircuit.mlp_only \
    --config "${CFG}" \
    --dataset "${DATASET}" \
    --out_dir "${RUN_DIR}" \
    --mlp_match "${MATCH}" \
    --W_base_field "${W_FIELD}" \
    --D_base_field "${D_FIELD}" \
    --S_op add --S_k "${S_ADD}" --S_min 1 --S_max 160 \
    --L_op add --L_k "${L_ADD}" --L_min 2 --L_max 16 \
    --batch_size "${MLP_BATCH_SIZE:-0}" \
    --eval_batch_size "${MLP_EVAL_BATCH_SIZE:-0}" \
    --diagnostics \
    --diagnostics_max_B "${DIAGNOSTICS_MAX_B:-10}"
fi

echo "[ok] done. summary: ${RUN_DIR}/summary.json"
