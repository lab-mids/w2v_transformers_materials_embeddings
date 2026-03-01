#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v snakemake >/dev/null 2>&1; then
  echo "Error: snakemake is not available in PATH." >&2
  echo "Activate the environment that provides Snakemake, e.g.:" >&2
  echo "  conda activate word_embedding_transformer" >&2
  exit 1
fi

# How many cores/threads Snakemake should use per workflow.
# Override by running: COMMON_ARGS="-j 4" ./run_all.sh
COMMON_ARGS="${COMMON_ARGS:--j 1}"

workflows=(
  "MatSciBERT"
  "MatSciBERT_Full"
  "Qwen"
  "Qwen_Full"
  "Word2Vec"
)

for i in "${!workflows[@]}"; do
  wf="${workflows[$i]}"
  echo "=== Running workflow $((i + 1))/${#workflows[@]}: $wf ==="
  (
    cd "$wf"
    snakemake -s Snakefile $COMMON_ARGS
  )
  echo
done

echo "=== All workflows finished successfully ==="
