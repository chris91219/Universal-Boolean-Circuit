#!/bin/bash
#SBATCH --job-name=ubc_pair_repel_sweep_cpu
#SBATCH --account=def-ssanner
#SBATCH --time=0-02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-39
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%A_%a.out

set -euo pipefail

# ---- wall-clock timer ----
START_ISO="$(date -Is)"
START_SEC=$SECONDS
on_exit () {
  local end_iso dur h m s
  end_iso="$(date -Is)"
  dur=$((SECONDS - START_SEC))
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

# Your repo venv
source "${PROJECT_DIR}/ENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""   # force CPU

# -------- dataset ----------
DATASET="${PROJECT_DIR}/data/bench_default.jsonl"

# -------- sweep definition ----------
# Modes and etas grid (edit here if you want more/less)
MODES=(mul log)                  # 2 modes
ETAS=(0.5 1.0 2.0 3.0)           # 4 eta values
NUM_MODES=${#MODES[@]}           # 2
NUM_ETAS=${#ETAS[@]}             # 4
NUM_SEEDS=5                      # 5 runs each

TOTAL=$((NUM_MODES * NUM_ETAS * NUM_SEEDS))  # 40
TASK_ID=${SLURM_ARRAY_TASK_ID}

if [[ ${TASK_ID} -ge ${TOTAL} ]]; then
  echo "TASK_ID ${TASK_ID} >= TOTAL ${TOTAL} — adjust --array or grid."; exit 1
fi

# -------- index mapping: task_id -> (mode_i, eta_i, seed_i) ----------
GROUP_SIZE=$((NUM_ETAS * NUM_SEEDS))        # 20
mode_i=$(( TASK_ID / GROUP_SIZE ))          # 0..1
rem=$(( TASK_ID % GROUP_SIZE ))
eta_i=$(( rem / NUM_SEEDS ))                # 0..3
seed_i=$(( rem % NUM_SEEDS ))               # 0..4

MODE=${MODES[$mode_i]}
ETA=${ETAS[$eta_i]}
SEED=${seed_i}                               # use 0..4 as seeds (or add an offset)

# -------- temp config to override pair + seed ----------
TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
TMP_CFG="${TMPDIR_USE}/ubc_pair_${MODE}_eta${ETA}_seed${SEED}.yaml"
cat > "${TMP_CFG}" <<EOF
seed: ${SEED}
pair:
  repel: true
  mode: "${MODE}"   # "mul" matches Prof's prob-space formula; "log" is log-bias variant
  eta: ${ETA}
# You can also tweak regs/anneal here if desired, e.g.:
# regs:
#   lam_entropy: 1.0e-3
#   lam_div_units: 5.0e-4
#   lam_div_rows: 5.0e-4
EOF

# -------- run name / output dir ----------
STAMP=$(date +"%Y%m%d-%H%M%S")
OUT_SUBDIR="bench_pair_sweep/${MODE}_eta${ETA}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
RUN_DIR="${RESULTS_ROOT}/${OUT_SUBDIR}"
mkdir -p "${RUN_DIR}"

# -------- CPU/Python info ----------
python - <<'PY'
import os, platform
print("Python:", platform.python_version())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PY

# -------- launch ----------
cd "${PROJECT_DIR}"
echo "[Info] MODE=${MODE}  ETA=${ETA}  SEED=${SEED}"
python -m ubcircuit.train \
  --dataset "${DATASET}" \
  --use_row_L \
  --config "${TMP_CFG}" \
  --out_dir "${RUN_DIR}"

echo "Done. Results at: ${RUN_DIR}/summary.json"
