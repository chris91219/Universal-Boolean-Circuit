#!/bin/bash
# Recombine the row-sharded B=18,20 extension runs back into normal joint-run dirs.
#
# Run this after the array from submit_bench_default_b1820_joint_rowshard_cc.sh
# has finished successfully.
#
# Submit from repo root:
#   sbatch scripts/slurm_combine_b1820_joint_rowshard_cc.sh

#SBATCH --job-name=ubc_b1820_rowcombine
#SBATCH --account=def-ssanner
#SBATCH --time=02:00:00
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
SHARDED_ROOT="${SHARDED_ROOT:-${RESULTS_ROOT}/bench_default_b1820_joint_1216_rowshard_raw/Sadd10_Ladd0/g16/rmsprop/mi_soft/top_down}"
OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/bench_default_b1820_joint_1216_rowshard/Sadd10_Ladd0/g16/rmsprop/mi_soft/top_down}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-400}"

module purge
module load StdEnv/2023
module load python/3.11

if [[ ! -f "${PROJECT_DIR}/ENV/bin/activate" ]]; then
  echo "[error] Missing virtualenv: ${PROJECT_DIR}/ENV/bin/activate"
  exit 1
fi
source "${PROJECT_DIR}/ENV/bin/activate"
hash -r

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

mkdir -p "${OUT_ROOT}"

echo "[info] project=${PROJECT_DIR}"
echo "[info] sharded_root=${SHARDED_ROOT}"
echo "[info] out_root=${OUT_ROOT}"
echo "[info] expected_shards=${EXPECTED_SHARDS}"

cd "${PROJECT_DIR}"
python "${PROJECT_DIR}/scripts/combine_joint_sharded_results.py" \
  --sharded-root "${SHARDED_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --expected-shards "${EXPECTED_SHARDS}"

echo "[ok] combined shard runs under ${OUT_ROOT}"
