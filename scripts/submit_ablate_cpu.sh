#!/bin/bash
#SBATCH --job-name=ubc_ablate_cpu
#SBATCH --account=def-ssanner
#SBATCH --time=0-02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# 7 variants * 5 seeds = 35 tasks -> 0-34
#SBATCH --array=0-34
#SBATCH --mail-user=chriswenhao.li@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/uoftwhli/scratch/UBC-Results/slurm-logs/%x-%A_%a.out

set -euo pipefail

# ---- wall-clock timer ----
START_ISO="$(date -Is)"
START_SEC=$SECONDS
on_exit () {
  local end_iso dur h m s
  end_iso="$(date -Is)"
  dur=$((SECONDS - START_SEC))
  h=$((dur/3600)); m=$(((dur%3600)/60)); s=$((dur%60))
  printf "\n[Timer] Start: %s\n[Timer] End  : %s\n[Timer] Elapsed: %02d:%02d:%02d (%d seconds)\n" \
         "$START_ISO" "$end_iso" "$h" "$m" "$s" "$dur"
}
trap on_exit EXIT

# -------- paths / env ----------
PROJECT_DIR=/home/uoftwhli/projects/def-ssanner/uoftwhli/Universal-Boolean-Circuit
RESULTS_ROOT=/home/uoftwhli/scratch/UBC-Results
LOG_DIR=${RESULTS_ROOT}/slurm-logs
mkdir -p "${LOG_DIR}"

module purge
module load StdEnv/2023
module load python/3.11
source "${PROJECT_DIR}/ENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""   # force CPU

# -------- dataset ----------
DATASET="${PROJECT_DIR}/data/bench_default.jsonl"

# -------- sweep setup ----------
N_VARIANTS=7
N_SEEDS=5
VIDX=$(( SLURM_ARRAY_TASK_ID % N_VARIANTS ))
SEED=$(( SLURM_ARRAY_TASK_ID / N_VARIANTS ))

# -------- per-variant overrides (write a small YAML) ----------
TMPDIR_USE="${SLURM_TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_USE}"

case "${VIDX}" in
  0)
    VAR=base
    # No anneal (T0=Tmin), no entropy, no diversity
    cat > "${TMPDIR_USE}/ovr.yaml" <<'YML'
use_row_L: true
anneal:
  T0: 0.35
  Tmin: 0.35      # = T0 disables anneal
  direction: top_down
  schedule: linear
  phase_scale: 0.0
  start_frac: 0.0
regs:
  lam_entropy: 0.0
  lam_div_units: 0.0
  lam_div_rows: 0.0
YML
    ;;

  1)
    VAR=anneal_td
    cat > "${TMPDIR_USE}/ovr.yaml" <<'YML'
use_row_L: true
anneal:
  T0: 0.35
  Tmin: 0.12
  direction: top_down
  schedule: linear
  phase_scale: 0.4
  start_frac: 0.0
regs:
  lam_entropy: 1.0e-3
  lam_div_units: 0.0
  lam_div_rows: 0.0
YML
    ;;

  2)
    VAR=anneal_bu
    cat > "${TMPDIR_USE}/ovr.yaml" <<'YML'
use_row_L: true
anneal:
  T0: 0.35
  Tmin: 0.12
  direction: bottom_up
  schedule: linear
  phase_scale: 0.4
  start_frac: 0.0
regs:
  lam_entropy: 1.0e-3
  lam_div_units: 0.0
  lam_div_rows: 0.0
YML
    ;;

  3)
    VAR=td_div_units
    cat > "${TMPDIR_USE}/ovr.yaml" <<'YML'
use_row_L: true
anneal:
  T0: 0.35
  Tmin: 0.12
  direction: top_down
  schedule: linear
  phase_scale: 0.4
  start_frac: 0.0
regs:
  lam_entropy: 1.0e-3
  lam_div_units: 5.0e-4
  lam_div_rows: 0.0
YML
    ;;

  4)
    VAR=td_div_rows
    cat > "${TMPDIR_USE}/ovr.yaml" <<'YML'
use_row_L: true
anneal:
  T0: 0.35
  Tmin: 0.12
  direction: top_down
  schedule: linear
  phase_scale: 0.4
  start_frac: 0.0
regs:
  lam_entropy: 1.0e-3
  lam_div_units: 0.0
  lam_div_rows: 5.0e-4
YML
    ;;

  5)
    VAR=td_div_both
    cat > "${TMPDIR_USE}/ovr.yaml" <<'YML'
use_row_L: true
anneal:
  T0: 0.35
  Tmin: 0.12
  direction: top_down
  schedule: linear
  phase_scale: 0.4
  start_frac: 0.0
regs:
  lam_entropy: 1.0e-3
  lam_div_units: 5.0e-4
  lam_div_rows: 5.0e-4
YML
    ;;

  6)
    VAR=anneal_td_cosine
    cat > "${TMPDIR_USE}/ovr.yaml" <<'YML'
use_row_L: true
anneal:
  T0: 0.35
  Tmin: 0.12
  direction: top_down
  schedule: cosine
  phase_scale: 0.4
  start_frac: 0.0
regs:
  lam_entropy: 1.0e-3
  lam_div_units: 0.0
  lam_div_rows: 0.0
YML
    ;;
esac

# Merge seed into a temp run config
TMP_CFG="${TMPDIR_USE}/run_${VAR}_seed${SEED}.yaml"
cat > "${TMP_CFG}" <<EOF
seed: ${SEED}
steps: 1200
optimizer: rmsprop
lr: 0.05
$(cat "${TMPDIR_USE}/ovr.yaml")
EOF

# -------- run name / output dir ----------
STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="${RESULTS_ROOT}/ablations/${VAR}/${STAMP}_job${SLURM_JOB_ID}_seed${SEED}"
mkdir -p "${RUN_DIR}"

# -------- CPU/Python info ----------
python - <<'PY'
import os, platform
print("Python:", platform.python_version())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PY

# -------- launch ----------
cd "${PROJECT_DIR}"
echo "[info] Variant=${VAR} Seed=${SEED}"
python -m ubcircuit.train \
  --dataset "${DATASET}" \
  --use_row_L \
  --config "${TMP_CFG}" \
  --out_dir "${RUN_DIR}"

echo "Done. Results at: ${RUN_DIR}/summary.json"
