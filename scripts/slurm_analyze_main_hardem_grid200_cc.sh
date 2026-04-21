#!/bin/bash
# Analyze the 200-row pilot hard-EM grid on Compute Canada.
#
# Submit from repo root after the pilot grid finishes or times out:
#   sbatch scripts/slurm_analyze_main_hardem_grid200_cc.sh

#SBATCH --job-name=ubc_main_hardem200_summary
#SBATCH --account=def-ssanner
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/scratch/uoftwhli/UBC-Results/slurm-logs/%x-%j.out

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
STAMP="$(date +"%Y%m%d-%H%M%S")_job${SLURM_JOB_ID}"

export GRID_ROOT="${GRID_ROOT:-${RESULTS_ROOT}/bench_default_main_hardem_grid200}"
export OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/bench_default_main_hardem_grid200_analysis/${STAMP}}"
export UBC_NUM_THREADS="${UBC_NUM_THREADS:-1}"
export UBC_TORCH_THREADS="${UBC_TORCH_THREADS:-${UBC_NUM_THREADS}}"

exec bash "${PROJECT_DIR}/scripts/slurm_analyze_main_hardem_grid_cc.sh"
