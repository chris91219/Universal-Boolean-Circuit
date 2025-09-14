#!/bin/bash
#SBATCH --job-name=ubc_bench_default_cpu
#SBATCH --account=def-ssanner
#SBATCH --time=0-06:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%A_%a.out

set -euo pipefail

# ---- wall-clock timer (prints on normal exit or failure) ----
START_ISO="$(date -Is)"
START_SEC=$SECONDS
on_exit () {
  local end_iso dur h m s
  end_iso="$(date -Is)"
  dur=$((SECONDS - START_SEC))
  h=$((dur/3600)); m=$(((dur%3600)/60)); s=$((dur%60))
  printf "\n[Timer] Start: %s\n[Timer] End  : %s\n[Timer] Elapsed: %02d:%02d:%02d (%d seconds)\n" \
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

# -------- seed from array ----------
SEED=${SLURM_ARRAY_TASK_ID}

# temp config to override seed (and any other overrides you want)
TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"
TMP_CFG="${TMPDIR_USE}/ubc_seed_${SEED}.yaml"
cat > "${TMP_CFG}" <<EOF
seed: ${SEED}
# early_stop:
#   use: true
#   metric: em
#   target: 1.0
#   min_steps: 100
#   check_every: 10
#   patience_checks: 3
EOF

# -------- run name / output dir ----------
STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="${RESULTS_ROOT}/bench_default/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
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
python -m ubcircuit.train \
  --dataset "${DATASET}" \
  --config "${TMP_CFG}" \
  --out_dir "${RUN_DIR}"

echo "Done. Results at: ${RUN_DIR}/summary.json"
