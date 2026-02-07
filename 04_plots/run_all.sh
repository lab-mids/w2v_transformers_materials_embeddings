#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v snakemake >/dev/null 2>&1; then
  echo "Error: snakemake is not available in PATH. Activate the environment that provides Snakemake." >&2
  exit 1
fi

# How many cores/threads Snakemake should use per workflow
# Override by running: COMMON_ARGS="-j 4" ./run_all.sh
COMMON_ARGS="${COMMON_ARGS:--j 1}"

workflows=(
  "workflows/error_plot.smk"
  "workflows/error_heatmap.smk"
  "workflows/fraction_retained.smk"
  "workflows/material_system_panels.smk"
  "workflows/word_embedding_distribution.smk"
)

for i in "${!workflows[@]}"; do
  wf="${workflows[$i]}"
  echo "=== Running workflow $((i + 1))/${#workflows[@]}: $wf ==="
  snakemake -s "$wf" $COMMON_ARGS
  echo
done

echo "=== All workflows finished successfully ==="
