#!/bin/bash
# Main/old SBC hard-EM grid on Compute Canada.
# Curated configs over relaxation, annealing, routing, lifting, regularization,
# optimizer/LR, gate set, and S/L capacity. The array is config-fastest:
#   --array=0-35   -> 1 seed over all 36 configs
#   --array=0-107  -> 3 seeds over all 36 configs (default)
#
# Submit from repo root:
#   sbatch scripts/submit_bench_default_main_hardem_grid_cc.sh
#
# Optional pilot/sample controls, used by the grid200 wrapper:
#   DATASET_SAMPLE_N=200 DATASET_SAMPLE_SEED=20260420 DATASET_SAMPLE_STRATIFY=B
#   RUN_ROOT=/scratch/$USER/UBC-Results/bench_default_main_hardem_grid200
#   UBC_NUM_THREADS=1

#SBATCH --job-name=ubc_main_hardem_grid
#SBATCH --account=def-ssanner
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-107
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
UBC_NUM_THREADS="${UBC_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"
export UBC_NUM_THREADS
export UBC_TORCH_THREADS="${UBC_TORCH_THREADS:-${UBC_NUM_THREADS}}"
export OMP_NUM_THREADS="${UBC_NUM_THREADS}"
export MKL_NUM_THREADS="${UBC_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${UBC_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${UBC_NUM_THREADS}"
export VECLIB_MAXIMUM_THREADS="${UBC_NUM_THREADS}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

FULL_DATASET="${DATASET:-${PROJECT_DIR}/data/bench_default.jsonl}"
SEEDS=(1 2 3)

# Fields:
# name gate opt lr route repel repel_mode eta prior mi_disjoint lift relax hard eval_hard gumbel_tau S_add L_add T0 Tmin sched phase lam_entropy lam_const16 steps
CONFIGS=(
  "soft_base_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 0 0.60 0.06 cosine 0.5 0.001 0.0 3000"
  "soft_base_liftF 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true false softmax false false 1.0 0 0 0.60 0.06 cosine 0.5 0.001 0.0 3000"
  "soft_sharp_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "soft_sharp_liftF 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true false softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "soft_narrowS2_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 -2 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "soft_narrowS1_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 -1 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "soft_s2_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 2 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "soft_s4_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 4 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "soft_s8_liftF 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true false softmax false false 1.0 8 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "soft_s10_liftF 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true false softmax false false 1.0 10 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "soft_lessDepth_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 -1 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "soft_moreDepth_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 1 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "soft_moreDepth_s4_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 4 1 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "mihard_liftT 16 rmsprop 0.001 mi_hard false log 2.0 0.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "mihard_s4_liftF 16 rmsprop 0.001 mi_hard false log 2.0 0.0 true false softmax false false 1.0 4 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "learned_liftT 16 rmsprop 0.001 learned false log 2.0 0.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "learned_repel_liftT 16 rmsprop 0.001 learned true hard-log 2.0 0.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "misoft_repel_log_liftT 16 rmsprop 0.001 mi_soft true log 2.0 2.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "misoft_repel_hardlog_liftT 16 rmsprop 0.001 mi_soft true hard-log 2.0 2.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "misoft_prior1_liftT 16 rmsprop 0.001 mi_soft false log 2.0 1.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "misoft_prior4_liftT 16 rmsprop 0.001 mi_soft false log 2.0 4.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "adam_soft_liftT 16 adam 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "rms_lr0003_liftT 16 rmsprop 0.0003 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "rms_lr003_liftT 16 rmsprop 0.003 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.0 3000"
  "g6_base_misoft 6 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 0 0.35 0.12 linear 0.4 0.001 0.0 3000"
  "g6_sharp_misoft 6 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 0 0.35 0.06 linear 0.4 0.005 0.0 3000"
  "g6_mihard 6 rmsprop 0.001 mi_hard false log 2.0 0.0 true true softmax false false 1.0 0 0 0.35 0.06 linear 0.4 0.005 0.0 3000"
  "const001_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.001 3000"
  "const005_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.005 0.005 3000"
  "lowEnt_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true softmax false false 1.0 0 0 0.60 0.03 cosine 0.5 0.0001 0.0 3000"
  "gumbel_tau1_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true gumbel true true 1.0 0 0 0.60 0.06 cosine 0.5 0.001 0.0 3000"
  "gumbel_tau05_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true gumbel true true 0.5 0 0 0.60 0.03 cosine 0.5 0.001 0.0 3000"
  "gumbel_s4_tau05_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true gumbel true true 0.5 4 0 0.60 0.03 cosine 0.5 0.001 0.0 3000"
  "gumbel_s10_tau05_liftF 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true false gumbel true true 0.5 10 0 0.60 0.03 cosine 0.5 0.001 0.0 3000"
  "argmax_s0_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true argmax_ste true true 1.0 0 0 0.60 0.03 cosine 0.5 0.001 0.0 3000"
  "argmax_s4_liftT 16 rmsprop 0.001 mi_soft false log 2.0 2.0 true true argmax_ste true true 1.0 4 0 0.60 0.03 cosine 0.5 0.001 0.0 3000"
)

NCONFIG=${#CONFIGS[@]}
NSEED=${#SEEDS[@]}
TOTAL=$((NCONFIG * NSEED))

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  exit 1
fi

config_idx=$((SLURM_ARRAY_TASK_ID % NCONFIG))
seed_idx=$((SLURM_ARRAY_TASK_ID / NCONFIG))
SEED="${SEEDS[$seed_idx]}"
read -r CFG_NAME GATE_SET OPT LR ROUTE REPEL REPEL_MODE ETA PRIOR_STRENGTH MI_DISJOINT LIFT_USE RELAX_MODE RELAX_HARD EVAL_HARD GUMBEL_TAU S_ADD L_ADD T0 TMIN SCHED PHASE LAM_ENT LAM_CONST16 STEPS <<< "${CONFIGS[$config_idx]}"

LAM_DIV_U=5.0e-4
LAM_DIV_R=5.0e-4
S16_START=0.25
S16_END=0.10
S16_MODE=rbf
S16_RADIUS=0.75
LIFT_FACTOR=2.0

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
SAMPLE_META=""
if [[ -n "${DATASET_SAMPLE_N:-}" ]]; then
  DATASET_SAMPLE_SEED="${DATASET_SAMPLE_SEED:-20260420}"
  DATASET_SAMPLE_STRATIFY="${DATASET_SAMPLE_STRATIFY:-B}"
  DATASET="${TMPDIR_USE}/bench_default_sample${DATASET_SAMPLE_N}_seed${DATASET_SAMPLE_SEED}.jsonl"
  SAMPLE_META="${TMPDIR_USE}/bench_default_sample${DATASET_SAMPLE_N}_seed${DATASET_SAMPLE_SEED}.meta.json"
  python "${PROJECT_DIR}/scripts/make_bench_sample.py" \
    --input "${FULL_DATASET}" \
    --output "${DATASET}" \
    --n "${DATASET_SAMPLE_N}" \
    --seed "${DATASET_SAMPLE_SEED}" \
    --stratify "${DATASET_SAMPLE_STRATIFY}" \
    --meta "${SAMPLE_META}"
else
  DATASET="${FULL_DATASET}"
fi

CFG="${TMPDIR_USE}/cfg_main_hardem_${CFG_NAME}_seed${SEED}.yaml"

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
  S_min: 1
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
  direction: top_down
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
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/bench_default_main_hardem_grid}"
RUN_DIR="${RUN_ROOT}/${CFG_NAME}/seed${SEED}/${STAMP}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
mkdir -p "${RUN_DIR}"
cp "${CFG}" "${RUN_DIR}/config.used.yaml"
if [[ -n "${SAMPLE_META}" && -f "${SAMPLE_META}" ]]; then
  cp "${SAMPLE_META}" "${RUN_DIR}/dataset.sample.meta.json"
fi

echo "[info] project=${PROJECT_DIR}"
echo "[info] dataset=${DATASET}"
echo "[info] full_dataset=${FULL_DATASET}"
echo "[info] run_root=${RUN_ROOT}"
echo "[info] config=${CFG_NAME} idx=${config_idx}/${NCONFIG}"
echo "[info] seed=${SEED}"
echo "[info] cfg=${CFG}"
echo "[info] out=${RUN_DIR}"
echo "[info] UBC_NUM_THREADS=${UBC_NUM_THREADS}"

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
print("torch num threads:", torch.get_num_threads())
print("yaml:", getattr(yaml, "__version__", "unknown"))
print("ubcircuit:", getattr(ubcircuit, "__file__", "unknown"))
PY

python -m ubcircuit.train \
  --dataset "${DATASET}" \
  --use_row_L \
  --config "${CFG}" \
  --out_dir "${RUN_DIR}"

echo "[ok] done. summary: ${RUN_DIR}/summary.json"
