#!/bin/bash
# Reuse completed full 1200-row hard-EM grid runs to create the same 200-row
# pilot summaries, then analyze them. This avoids rerunning configs that have
# already completed on the full benchmark.
#
# Submit from repo root:
#   sbatch scripts/slurm_backfill_main_hardem_grid200_from_full_cc.sh

#SBATCH --job-name=ubc_hardem200_backfill
#SBATCH --account=def-ssanner
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=1
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
FULL_ROOT="${FULL_ROOT:-${RESULTS_ROOT}/bench_default_main_hardem_grid}"
OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/bench_default_main_hardem_grid200_from_full}"
STAMP="$(date +"%Y%m%d-%H%M%S")_job${SLURM_JOB_ID}"
ANALYSIS_OUT="${ANALYSIS_OUT:-${RESULTS_ROOT}/bench_default_main_hardem_grid200_from_full_analysis/${STAMP}}"
DATASET="${DATASET:-${PROJECT_DIR}/data/bench_default.jsonl}"
SAMPLE_N="${SAMPLE_N:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-20260420}"
SAMPLE_STRATIFY="${SAMPLE_STRATIFY:-B}"

mkdir -p "${OUT_ROOT}" "${ANALYSIS_OUT}" "${RESULTS_ROOT}/slurm-logs"

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
echo "[info] full_root=${FULL_ROOT}"
echo "[info] out_root=${OUT_ROOT}"
echo "[info] analysis_out=${ANALYSIS_OUT}"
echo "[info] dataset=${DATASET}"
echo "[info] sample_n=${SAMPLE_N} sample_seed=${SAMPLE_SEED} sample_stratify=${SAMPLE_STRATIFY}"

cd "${PROJECT_DIR}"

python "${PROJECT_DIR}/scripts/backfill_grid_sample_from_summaries.py" \
  --dataset "${DATASET}" \
  --full-root "${FULL_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --n "${SAMPLE_N}" \
  --seed "${SAMPLE_SEED}" \
  --stratify "${SAMPLE_STRATIFY}"

python "${PROJECT_DIR}/scripts/analyze_ubc_results.py" \
  "${OUT_ROOT}" \
  --out "${ANALYSIS_OUT}"

echo "[ok] backfilled sampled summaries: ${OUT_ROOT}"
echo "[ok] analysis complete: ${ANALYSIS_OUT}"
ls -lh "${ANALYSIS_OUT}"
