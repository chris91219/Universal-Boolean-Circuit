#!/usr/bin/env bash
# Pull the curated UBC result files from Google Drive into the local .results/
# folder using scripts/results_rclone_filter.txt.
#
# Default:
#   bash scripts/rclone_pull_results_to_local.sh
#
# Overrides:
#   REMOTE='GDrive:UofT_PhD/Collab_Prof_Kratsios/UBC-Results' \
#   DEST='.results/UBC-Results' \
#   bash scripts/rclone_pull_results_to_local.sh
#
# Dry run:
#   DRY_RUN=1 bash scripts/rclone_pull_results_to_local.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE="${REMOTE:-GDrive:UofT_PhD/Collab_Prof_Kratsios/UBC-Results}"
DEST="${DEST:-${PROJECT_DIR}/.results/UBC-Results}"
FILTER="${FILTER:-${PROJECT_DIR}/scripts/results_rclone_filter.txt}"

mkdir -p "${DEST}"

cmd=(rclone copy "${REMOTE}" "${DEST}" --filter-from "${FILTER}" --progress)
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-false}" == "true" ]]; then
  cmd+=(--dry-run)
fi

echo "[info] remote=${REMOTE}"
echo "[info] dest=${DEST}"
echo "[info] filter=${FILTER}"
echo "[info] command: ${cmd[*]}"

"${cmd[@]}" "$@"

echo "[ok] curated results synced to ${DEST}"
