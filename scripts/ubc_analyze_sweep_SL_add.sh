#!/bin/bash
#SBATCH --job-name=ubc_analyze_sweep_SL_add
#SBATCH --account=def-ssanner
#SBATCH --time=0-00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%j.out

set -euo pipefail

PROJECT_DIR=/home/uoftwhli/projects/def-ssanner/uoftwhli/Universal-Boolean-Circuit
RESULTS_ROOT=/home/uoftwhli/scratch/UBC-Results

module purge
module load StdEnv/2023
module load python/3.11
source "${PROJECT_DIR}/ENV/bin/activate"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

SWEEP_ROOT="${RESULTS_ROOT}/sweep_SL_add"
OUT="${SWEEP_ROOT}/analysis"
mkdir -p "${OUT}"

cd "${PROJECT_DIR}"
python scripts/analyze_ubc_sweep_SL_add.py "${SWEEP_ROOT}" --out "${OUT}"

echo "[ok] analysis written to ${OUT}"
ls -lh "${OUT}"
