#!/usr/bin/env bash
set -euo pipefail

# How many cores/threads Snakemake should use per workflow
COMMON_ARGS="-j 1"

echo "=== Running workflow 1 ==="
snakemake -s MatSciBERT/Snakefile $COMMON_ARGS

echo "=== Running workflow 2  ==="
snakemake -s MatSciBERT_Full/Snakefile $COMMON_ARGS

echo "=== Running workflow 3  ==="
snakemake -s Qwen/Snakefile $COMMON_ARGS

echo "=== Running workflow 4 ==="
snakemake -s Qwen_Full/Snakefile $COMMON_ARGS

echo "=== Running workflow 5 ==="
snakemake -s Word2Vec/Snakefile $COMMON_ARGS

echo "=== All workflows finished successfully ==="