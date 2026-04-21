#!/bin/bash
# Recover same-slice sampled pilot summaries from timed-out full hard-EM grid
# Slurm logs. This only keeps rows whose original 1200-row indices are in the
# deterministic stratified sample used by grid200.
#
# Submit after the old full grid timeout:
#   sbatch scripts/slurm_backfill_main_hardem_grid200_from_logs_cc.sh

#SBATCH --job-name=ubc_hardem200_logfill
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
LOG_DIR="${LOG_DIR:-${RESULTS_ROOT}/slurm-logs}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-36229060}"
FAILED_TASK_IDS="${FAILED_TASK_IDS:-4,6,7,8,9,11,12,14,32,33,35,40,43,44,45,47,48,50,68,69,71,77,78,79,80,81,83,84,86,88,104,105,107}"
OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/bench_default_main_hardem_grid200_from_logs}"
SAMPLE_N="${SAMPLE_N:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-20260420}"
SAMPLE_STRATIFY="${SAMPLE_STRATIFY:-B}"
MIN_ROWS="${MIN_ROWS:-1}"
DATASET="${DATASET:-${PROJECT_DIR}/data/bench_default.jsonl}"
STAMP="$(date +"%Y%m%d-%H%M%S")_job${SLURM_JOB_ID}"
ANALYSIS_OUT="${ANALYSIS_OUT:-${RESULTS_ROOT}/bench_default_main_hardem_grid200_from_logs_analysis/${STAMP}}"

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
echo "[info] log_dir=${LOG_DIR}"
echo "[info] source_job_id=${SOURCE_JOB_ID}"
echo "[info] failed_task_ids=${FAILED_TASK_IDS}"
echo "[info] out_root=${OUT_ROOT}"
echo "[info] sample_n=${SAMPLE_N} sample_seed=${SAMPLE_SEED} sample_stratify=${SAMPLE_STRATIFY} min_rows=${MIN_ROWS}"

cd "${PROJECT_DIR}"

python "${PROJECT_DIR}/scripts/backfill_hardem_grid_prefix_from_logs.py" \
  --log-dir "${LOG_DIR}" \
  --job-id "${SOURCE_JOB_ID}" \
  --task-ids "${FAILED_TASK_IDS}" \
  --out-root "${OUT_ROOT}" \
  --dataset "${DATASET}" \
  --sample-n "${SAMPLE_N}" \
  --sample-seed "${SAMPLE_SEED}" \
  --sample-stratify "${SAMPLE_STRATIFY}" \
  --min-rows "${MIN_ROWS}"

python "${PROJECT_DIR}/scripts/analyze_ubc_results.py" \
  "${OUT_ROOT}" \
  --out "${ANALYSIS_OUT}"

echo "[ok] sampled-log backfill complete: ${OUT_ROOT}"
echo "[ok] analysis complete: ${ANALYSIS_OUT}"
ls -lh "${ANALYSIS_OUT}"
