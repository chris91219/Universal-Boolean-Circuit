#!/bin/bash
# SBC robustness jump/restart sweep on Compute Canada.
#
# This is orthogonal to submit_robustness_sbc_cc.sh: it trains softly first,
# then performs stochastic hard restarts from the learned soft basin using either
# Gumbel-ST or deterministic argmax-STE.
#
# Grid:
#   datasets: bench_small_b10_2000_noise_true_add{0,2,4,6,8,10}
#   shape bases: W/D_base and W/D_true
#   jump modes: gumbel and argmax_ste
#   W_add/S_add: 0,2,4,6,8,10 with D_add/L_add=0
#
# Submit from repo root:
#   sbatch scripts/submit_robustness_sbc_jump_cc.sh
#
# Useful pilot:
#   sbatch --array=0-23 scripts/submit_robustness_sbc_jump_cc.sh
#
# Optional overrides:
#   PAIR_ROUTE=mi_soft SBC_JUMP_STEPS=2400 JUMP_ATTEMPTS=2 sbatch ...

#SBATCH --job-name=ubc_robust_jump
#SBATCH --account=def-ssanner
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-143
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

DATASET_KEYS=(true_add0 true_add2 true_add4 true_add6 true_add8 true_add10)
DATASET_FILES=(
  "${PROJECT_DIR}/data/bench_small_b10_2000_noise_true_add0.jsonl"
  "${PROJECT_DIR}/data/bench_small_b10_2000_noise_true_add2.jsonl"
  "${PROJECT_DIR}/data/bench_small_b10_2000_noise_true_add4.jsonl"
  "${PROJECT_DIR}/data/bench_small_b10_2000_noise_true_add6.jsonl"
  "${PROJECT_DIR}/data/bench_small_b10_2000_noise_true_add8.jsonl"
  "${PROJECT_DIR}/data/bench_small_b10_2000_noise_true_add10.jsonl"
)
BASIS_KEYS=(base true)
W_FIELDS=(W_base W_true)
D_FIELDS=(D_base D_true)
JUMP_MODES=(gumbel argmax_ste)
S_ADD_LIST=(0 2 4 6 8 10)

NDATA=${#DATASET_KEYS[@]}
NBASIS=${#BASIS_KEYS[@]}
NJUMP=${#JUMP_MODES[@]}
NS=${#S_ADD_LIST[@]}
TOTAL=$((NDATA * NBASIS * NJUMP * NS))

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  exit 1
fi

task=${SLURM_ARRAY_TASK_ID}
s_idx=$((task % NS)); task=$((task / NS))
jump_idx=$((task % NJUMP)); task=$((task / NJUMP))
basis_idx=$((task % NBASIS)); task=$((task / NBASIS))
data_idx=$((task % NDATA))

S_ADD="${S_ADD_LIST[$s_idx]}"
JUMP_MODE="${JUMP_MODES[$jump_idx]}"
BASIS="${BASIS_KEYS[$basis_idx]}"
W_FIELD="${W_FIELDS[$basis_idx]}"
D_FIELD="${D_FIELDS[$basis_idx]}"
DATASET_KEY="${DATASET_KEYS[$data_idx]}"
DATASET_IN="${DATASET_FILES[$data_idx]}"
MAX_B="${MAX_B:-0}"
SEED="${SEED:-1}"
PAIR_ROUTE="${PAIR_ROUTE:-mi_hard}"

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
FILTERED_DATASET="${TMPDIR_USE}/${DATASET_KEY}_maxB${MAX_B}.jsonl"
export DATASET_IN FILTERED_DATASET MAX_B
python - <<'PY'
import collections, json, os
src = os.environ["DATASET_IN"]
dst = os.environ["FILTERED_DATASET"]
max_b = int(os.environ.get("MAX_B", "0"))
kept = total = 0
hist = collections.Counter()
with open(src) as f_in, open(dst, "w") as f_out:
    for line in f_in:
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        if max_b <= 0 or int(row["B"]) <= max_b:
            f_out.write(json.dumps(row) + "\n")
            kept += 1
            hist[int(row["B"])] += 1
cond = "all rows" if max_b <= 0 else f"B<= {max_b}"
print(f"[filter] {src} -> {dst}: kept {kept}/{total} rows with {cond}")
print("[filter] B hist:", dict(sorted(hist.items())))
if kept == 0:
    raise SystemExit("filtered dataset is empty")
PY

CFG="${TMPDIR_USE}/cfg_robust_jump_${DATASET_KEY}_${BASIS}_${JUMP_MODE}_Sadd${S_ADD}.yaml"

cat > "${CFG}" <<EOF
seed: ${SEED}
gate_set: "16"
steps: ${SBC_JUMP_STEPS:-3000}
optimizer: rmsprop
lr: ${SBC_LR:-0.001}
use_row_L: true
use_max_L: false
scale:
  use_row_S: true
  W_base_field: "${W_FIELD}"
  D_base_field: "${D_FIELD}"
  S_op: add
  S_k: ${S_ADD}
  S_min: 1
  S_max: 128
  L_op: add
  L_k: 0
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
jump:
  use: true
  mode: "${JUMP_MODE}"
  hard: true
  gumbel_tau: ${JUMP_GUMBEL_TAU:-0.5}
  eval_hard: true
  start_frac: ${JUMP_START_FRAC:-0.6}
  attempts: ${JUMP_ATTEMPTS:-3}
  interval: ${JUMP_INTERVAL:-0}
  strength: ${JUMP_STRENGTH:-4.0}
  noise_std: ${JUMP_NOISE_STD:-0.10}
  sample: ${JUMP_SAMPLE:-true}
  reset_optimizer: true
  restore_anchor_each_attempt: true
  keep_best: true
  keep_best_metric: decoded_row_acc
  include_gates: true
  include_rows: true
  include_pairs: true
  include_lift: true
anneal:
  T0: ${T0:-0.60}
  Tmin: ${TMIN:-0.03}
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
  lam_entropy: ${LAM_ENTROPY:-5.0e-3}
  lam_div_units: 5.0e-4
  lam_div_rows: 5.0e-4
  lam_const16: ${LAM_CONST16:-0.0}
pair:
  route: "${PAIR_ROUTE}"
  prior_strength: ${PRIOR_STRENGTH:-0.0}
  mi_disjoint: true
  repel: false
  mode: "log"
  eta: 2.0
early_stop:
  use: true
  metric: decoded_em
  target: 1.0
  min_steps: 100
  check_every: 10
  patience_checks: 3
EOF

STAMP="$(date +"%Y%m%d-%H%M%S")"
RUN_DIR="${RESULTS_ROOT}/robustness_sbc_jump/${DATASET_KEY}/basis_${BASIS}/${JUMP_MODE}/Sadd${S_ADD}_Ladd0/${STAMP}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_seed${SEED}"
mkdir -p "${RUN_DIR}"
cp "${CFG}" "${RUN_DIR}/config.used.yaml"

echo "[info] project=${PROJECT_DIR}"
echo "[info] dataset_in=${DATASET_IN}"
echo "[info] dataset_filtered=${FILTERED_DATASET}"
echo "[info] dataset_key=${DATASET_KEY} max_B=${MAX_B}"
echo "[info] basis=${BASIS} W_field=${W_FIELD} D_field=${D_FIELD}"
echo "[info] jump_mode=${JUMP_MODE} pair_route=${PAIR_ROUTE}"
echo "[info] S_add=${S_ADD} L_add=0 seed=${SEED}"
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

python -m ubcircuit.train \
  --dataset "${FILTERED_DATASET}" \
  --use_row_L \
  --config "${CFG}" \
  --out_dir "${RUN_DIR}"

echo "[ok] done. summary: ${RUN_DIR}/summary.json"
