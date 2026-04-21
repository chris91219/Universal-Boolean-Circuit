#!/bin/bash
# Pilot version of the main/old SBC hard-EM grid on a deterministic 200-row
# sample from bench_default.jsonl. This keeps all grid configs/seeds identical
# to submit_bench_default_main_hardem_grid_cc.sh, but makes the first pass
# cheap enough for promotion/discard decisions.
#
# Submit from repo root:
#   sbatch scripts/submit_bench_default_main_hardem_grid200_cc.sh
#
# Fast smoke over one seed:
#   sbatch --array=0-35 scripts/submit_bench_default_main_hardem_grid200_cc.sh

#SBATCH --job-name=ubc_main_hardem_grid200
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

# These trainings are many tiny serial PyTorch jobs; one thread is usually
# faster and easier on CC nodes than oversubscribing small matmuls.
export UBC_NUM_THREADS="${UBC_NUM_THREADS:-1}"
export UBC_TORCH_THREADS="${UBC_TORCH_THREADS:-${UBC_NUM_THREADS}}"

exec bash "${PROJECT_DIR}/scripts/submit_bench_default_main_hardem_grid_cc.sh"
