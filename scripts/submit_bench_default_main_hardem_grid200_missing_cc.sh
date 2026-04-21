#!/bin/bash
# Missing-only 200-row pilot hard-EM grid. Submit this after running the
# backfill-from-full job. It is safe to submit over 0-107: tasks with an
# existing 200-row summary/results file exit immediately; only missing configs
# train on the sampled 200 rows.
#
# Submit all and let completed tasks self-skip:
#   sbatch scripts/submit_bench_default_main_hardem_grid200_missing_cc.sh
#
# Or first print the exact missing task IDs:
#   python scripts/list_missing_hardem_grid200_tasks.py \
#     --roots ~/scratch/UBC-Results/bench_default_main_hardem_grid200_from_full \
#             ~/scratch/UBC-Results/bench_default_main_hardem_grid200_from_logs \
#             ~/scratch/UBC-Results/bench_default_main_hardem_grid200 \
#     --format csv

#SBATCH --job-name=ubc_main_hardem_grid200_missing
#SBATCH --account=def-ssanner
#SBATCH --time=0-16:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-107
#SBATCH --output=/scratch/uoftwhli/UBC-Results/slurm-logs/%x-%A_%a.out

set -euo pipefail

if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH:-$HOME/scratch}/UBC-Results}"

export DATASET_SAMPLE_N="${DATASET_SAMPLE_N:-200}"
export DATASET_SAMPLE_SEED="${DATASET_SAMPLE_SEED:-20260420}"
export DATASET_SAMPLE_STRATIFY="${DATASET_SAMPLE_STRATIFY:-B}"
export RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/bench_default_main_hardem_grid200}"

export SKIP_IF_SAMPLE_DONE=1
export SKIP_MIN_ROWS="${SKIP_MIN_ROWS:-${DATASET_SAMPLE_N}}"
export SKIP_ROOTS="${SKIP_ROOTS:-${RESULTS_ROOT}/bench_default_main_hardem_grid200_from_full:${RESULTS_ROOT}/bench_default_main_hardem_grid200_from_logs:${RESULTS_ROOT}/bench_default_main_hardem_grid200}"

# Tiny serial per-instance trainings tend to do better without many BLAS/OpenMP
# threads fighting over small matmuls.
export UBC_NUM_THREADS="${UBC_NUM_THREADS:-1}"
export UBC_TORCH_THREADS="${UBC_TORCH_THREADS:-${UBC_NUM_THREADS}}"

exec bash "${PROJECT_DIR}/scripts/submit_bench_default_main_hardem_grid_cc.sh"
