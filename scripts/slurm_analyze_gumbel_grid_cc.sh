#!/bin/bash
# Analyze UBC Gumbel grid runs on Compute Canada.
#
# Submit from repo root after the grid has produced summaries:
#   sbatch scripts/slurm_analyze_gumbel_grid_cc.sh

#SBATCH --job-name=ubc_gumbel_summary
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

if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH:-$HOME/scratch}/UBC-Results}"
GRID_ROOT="${RESULTS_ROOT}/bench_default_gumbel_grid"
STAMP="$(date +"%Y%m%d-%H%M%S")_job${SLURM_JOB_ID}"
OUT_DIR="${RESULTS_ROOT}/bench_default_gumbel_grid_analysis/${STAMP}"
mkdir -p "${OUT_DIR}" "${RESULTS_ROOT}/slurm-logs"

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
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "[info] project=${PROJECT_DIR}"
echo "[info] grid_root=${GRID_ROOT}"
echo "[info] out_dir=${OUT_DIR}"

cd "${PROJECT_DIR}"
python scripts/analyze_ubc_results.py "${GRID_ROOT}" --out "${OUT_DIR}"

echo "[ok] analysis complete: ${OUT_DIR}"
