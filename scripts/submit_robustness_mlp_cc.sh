#!/bin/bash
# MLP-only robustness screen on Compute Canada.
#
# Grid:
#   datasets: bench_small_b10_2000_noise_true_add{0,2,4,6,8,10}
#   reference shape bases: W/D_base and W/D_true
#   W_add/S_add: 2,4,6,8,10 with D_add/L_add=0
#   MLP matching: neuron, param_soft, param_total
#
# The default suite measures noise relative to B_true.  The generated formula is
# preserved up to compact variable renumbering, while ambient B is set from the
# semantic variable count plus the requested noise budget whenever possible.
#
# Submit from repo root:
#   sbatch scripts/submit_robustness_mlp_cc.sh

#SBATCH --job-name=ubc_robust_mlp
#SBATCH --account=def-ssanner
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-179
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

# Threading: MLPs use larger dense matmuls than SBC, so modest threading helps.
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
S_ADD_LIST=(2 4 6 8 10)
MATCH_MODES=(neuron param_soft param_total)

NDATA=${#DATASET_KEYS[@]}
NBASIS=${#BASIS_KEYS[@]}
NS=${#S_ADD_LIST[@]}
NMATCH=${#MATCH_MODES[@]}
TOTAL=$((NDATA * NBASIS * NS * NMATCH))

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}"
  exit 1
fi

task=${SLURM_ARRAY_TASK_ID}
match_idx=$((task % NMATCH)); task=$((task / NMATCH))
s_idx=$((task % NS)); task=$((task / NS))
basis_idx=$((task % NBASIS)); task=$((task / NBASIS))
data_idx=$((task % NDATA))

MATCH="${MATCH_MODES[$match_idx]}"
S_ADD="${S_ADD_LIST[$s_idx]}"
BASIS="${BASIS_KEYS[$basis_idx]}"
W_FIELD="${W_FIELDS[$basis_idx]}"
D_FIELD="${D_FIELDS[$basis_idx]}"
DATASET_KEY="${DATASET_KEYS[$data_idx]}"
DATASET_IN="${DATASET_FILES[$data_idx]}"
MAX_B="${MAX_B:-0}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-0}"
MLP_EVAL_BATCH_SIZE="${MLP_EVAL_BATCH_SIZE:-0}"

TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
FILTERED_DATASET="${TMPDIR_USE}/${DATASET_KEY}_maxB${MAX_B}.jsonl"
export DATASET_IN FILTERED_DATASET MAX_B
python - <<'PY'
import collections, json, os
src = os.environ["DATASET_IN"]
dst = os.environ["FILTERED_DATASET"]
max_b = int(os.environ.get("MAX_B", "20"))
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

CFG="${TMPDIR_USE}/cfg_robust_mlp_${DATASET_KEY}_${BASIS}_Sadd${S_ADD}_${MATCH}.yaml"
SEED="${SEED:-1}"

cat > "${CFG}" <<EOF
seed: ${SEED}
gate_set: "16"
steps: ${MLP_STEPS:-3000}
optimizer: rmsprop
lr: ${MLP_LR:-0.001}
use_row_L: true
use_max_L: false
lifting:
  use: ${LIFT_USE:-true}
  factor: ${LIFT_FACTOR:-2.0}
anneal:
  T0: ${T0:-0.60}
  Tmin: ${TMIN:-0.06}
  direction: top_down
  schedule: cosine
  phase_scale: 0.5
  start_frac: 0.0
pair:
  route: mi_soft
  prior_strength: 2.0
  mi_disjoint: true
  repel: false
  mode: "log"
  eta: 2.0
early_stop:
  use: true
  metric: em
  target: 1.0
  min_steps: 100
  check_every: 10
  patience_checks: 3
EOF

STAMP="$(date +"%Y%m%d-%H%M%S")"
RUN_DIR="${RESULTS_ROOT}/robustness_mlp/${DATASET_KEY}/basis_${BASIS}/Sadd${S_ADD}_Ladd0/mlp_${MATCH}/${STAMP}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
mkdir -p "${RUN_DIR}"
cp "${CFG}" "${RUN_DIR}/config.used.yaml"

echo "[info] project=${PROJECT_DIR}"
echo "[info] dataset_in=${DATASET_IN}"
echo "[info] dataset_filtered=${FILTERED_DATASET}"
echo "[info] dataset_key=${DATASET_KEY} max_B=${MAX_B}"
echo "[info] basis=${BASIS} W_field=${W_FIELD} D_field=${D_FIELD}"
echo "[info] S_add=${S_ADD} L_add=0 match=${MATCH}"
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

python -m ubcircuit.mlp_only \
  --config "${CFG}" \
  --dataset "${FILTERED_DATASET}" \
  --out_dir "${RUN_DIR}" \
  --mlp_match "${MATCH}" \
  --W_base_field "${W_FIELD}" \
  --D_base_field "${D_FIELD}" \
  --S_op add --S_k "${S_ADD}" --S_min 1 --S_max 128 \
  --L_op add --L_k 0 --L_min 2 --L_max 16 \
  --batch_size "${MLP_BATCH_SIZE}" \
  --eval_batch_size "${MLP_EVAL_BATCH_SIZE}"

echo "[ok] done. summary: ${RUN_DIR}/summary.json"
