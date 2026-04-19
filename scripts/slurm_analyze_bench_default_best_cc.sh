#!/bin/bash
# Summarize the new best bench_default runs on Compute Canada.
# It analyzes:
#   - $SCRATCH/UBC-Results/bench_default_best_main
#   - $SCRATCH/UBC-Results/bench_default_best_joint_1216
#
# If your CC allocation changed, update the account line below or submit with:
#   sbatch --account=<your_allocation> scripts/slurm_analyze_bench_default_best_cc.sh

#SBATCH --job-name=ubc_best_summary
#SBATCH --account=def-ssanner
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/uoftwhli/UBC-Results/slurm-logs/%x-%j.out

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

MAIN_ROOT="${MAIN_ROOT:-${RESULTS_ROOT}/bench_default_best_main}"
JOINT_ROOT="${JOINT_ROOT:-${RESULTS_ROOT}/bench_default_best_joint_1216}"

STAMP="$(date +"%Y%m%d-%H%M%S")"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${RESULTS_ROOT}/bench_default_best_analysis/${STAMP}_job${SLURM_JOB_ID}}"
MAIN_OUT="${ANALYSIS_ROOT}/main"
JOINT_OUT="${ANALYSIS_ROOT}/joint_1216"
mkdir -p "${MAIN_OUT}" "${JOINT_OUT}"

module purge
module load StdEnv/2023
module load python/3.11

if [[ -f "${PROJECT_DIR}/ENV/bin/activate" ]]; then
  source "${PROJECT_DIR}/ENV/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "[info] project=${PROJECT_DIR}"
echo "[info] results_root=${RESULTS_ROOT}"
echo "[info] main_root=${MAIN_ROOT}"
echo "[info] joint_root=${JOINT_ROOT}"
echo "[info] analysis_root=${ANALYSIS_ROOT}"

cd "${PROJECT_DIR}"

python - <<'PY'
import os, platform
print("Python:", platform.python_version())
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
try:
    import pandas as pd
    print("pandas:", pd.__version__)
except Exception as exc:
    print("pandas import failed:", exc)
    raise
PY

echo "[info] Analyzing main UBC runs..."
python "${PROJECT_DIR}/scripts/analyze_ubc_results.py" \
  "${MAIN_ROOT}" \
  --out "${MAIN_OUT}"

echo "[info] Analyzing joint 1216 runs..."
python "${PROJECT_DIR}/scripts/analyze_joint_bnr_runs.py" \
  "${JOINT_ROOT}" \
  --out "${JOINT_OUT}"

echo "[ok] analysis complete"
echo "[ok] main outputs:"
ls -lh "${MAIN_OUT}"
echo "[ok] joint outputs:"
ls -lh "${JOINT_OUT}"
