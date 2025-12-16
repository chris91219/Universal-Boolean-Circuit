#!/bin/bash
#SBATCH --job-name=ubc_joint_bnr_analyze
#SBATCH --account=def-ssanner
#SBATCH --time=0-00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%A.out
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR=/home/uoftwhli/projects/def-ssanner/uoftwhli/Universal-Boolean-Circuit
RESULTS_ROOT=/home/uoftwhli/scratch/UBC-Results
LOG_DIR=${RESULTS_ROOT}/slurm-logs
mkdir -p "${LOG_DIR}"

module purge
module load StdEnv/2023
module load python/3.11
source "${PROJECT_DIR}/ENV/bin/activate"

ROOT_RUN_DIR="${RESULTS_ROOT}/joint_bnr_Sadd10_Ladd0"
OUT_DIR="${ROOT_RUN_DIR}/analysis_joint_bnr"

cd "${PROJECT_DIR}"
python scripts/analyze_joint_bnr_runs.py "${ROOT_RUN_DIR}" --out "${OUT_DIR}"

echo "[ok] analysis written to ${OUT_DIR}"
