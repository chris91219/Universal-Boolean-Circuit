#!/bin/bash
# Best joint UBC-vs-MLP benchmark run on Compute Canada.
# This uses the newer joint runner with decoded-readout metrics.
# If your CC allocation changed, update the account line below or submit with:
#   sbatch --account=<your_allocation> scripts/submit_bench_default_best_joint_1216_cc.sh

#SBATCH --job-name=ubc_best_joint1216
#SBATCH --account=def-ssanner
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-14
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

# Slurm copies submitted scripts into a spool directory before execution, so
# BASH_SOURCE points at that copy. SLURM_SUBMIT_DIR is the repo directory when
# submitted from the project root; PROJECT_DIR can still override explicitly.
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

DATASET="${PROJECT_DIR}/data/bench_default.jsonl"

# Best joint setup so far, following scripts/slurm_joint_bnr_sadd10.sh,
# but using ubcircuit.joint_mlp_ubc_bnr_1216.
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
SCHED=cosine
PHASE=0.5
S16_START=0.25
S16_END=0.10
S16_MODE=rbf
S16_RADIUS=0.75
LIFT_USE=false
LIFT_FACTOR=2.0
S_ADD=10
L_ADD=0

SEEDS=(1 2 3 4 5)
MATCH_MODES=(neuron param_soft param_total)
NSEED=${#SEEDS[@]}
NMODE=${#MATCH_MODES[@]}
TOTAL=$((NSEED * NMODE))

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  exit 1
fi

mode_idx=$((SLURM_ARRAY_TASK_ID % NMODE))
seed_idx=$((SLURM_ARRAY_TASK_ID / NMODE))

SEED="${SEEDS[$seed_idx]}"
MATCH="${MATCH_MODES[$mode_idx]}"

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
CFG="${TMPDIR_USE}/cfg_best_joint1216_seed${SEED}.yaml"

cat > "${CFG}" <<EOF
seed: ${SEED}
gate_set: "${GATE_SET}"
steps: ${STEPS}
optimizer: ${OPT}
lr: ${LR}
use_row_L: true
use_max_L: false
lifting:
  use: ${LIFT_USE}
  factor: ${LIFT_FACTOR}
relaxation:
  mode: softmax
  hard: false
  gumbel_tau: 1.0
  eval_hard: false
anneal:
  T0: ${T0}
  Tmin: ${TMIN}
  direction: ${DIR}
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
RUN_DIR="${RESULTS_ROOT}/bench_default_best_joint_1216/Sadd${S_ADD}_Ladd${L_ADD}/g${GATE_SET}/${OPT}/${ROUTE}/${DIR}/mlp_${MATCH}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
mkdir -p "${RUN_DIR}"
cp "${CFG}" "${RUN_DIR}/config.used.yaml"

echo "[info] project=${PROJECT_DIR}"
echo "[info] dataset=${DATASET}"
echo "[info] seed=${SEED} match=${MATCH}"
echo "[info] cfg=${CFG}"
echo "[info] out=${RUN_DIR}"

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
print("yaml:", getattr(yaml, "__version__", "unknown"))
print("ubcircuit:", getattr(ubcircuit, "__file__", "unknown"))
PY

python -m ubcircuit.joint_mlp_ubc_bnr_1216 \
  --config "${CFG}" \
  --dataset "${DATASET}" \
  --out_dir "${RUN_DIR}" \
  --mlp_match "${MATCH}" \
  --S_op add --S_k "${S_ADD}" --S_min 2 --S_max 128 \
  --L_op add --L_k "${L_ADD}" --L_min 2 --L_max 16

echo "[ok] done. summary: ${RUN_DIR}/summary.json"
