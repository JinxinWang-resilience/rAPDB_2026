#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [existing-output-directory]" >&2
  exit 2
fi

DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/../outputs_n10000_runall_mu0_mu2_strict"
if [ ! -d "$DEFAULT_OUTPUT_DIR" ]; then
  DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/apdb_real_data_python/outputs_n10000_runall_mu0_mu2_strict"
fi
OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"
DATA_DIR="apdb_real_data_matlab"

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Missing original output directory: $OUTPUT_DIR" >&2
  exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "Missing original input directory: $DATA_DIR" >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/apdb_matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"

python -m apdb_real_data_python.complete_gpu --outdir "$OUTPUT_DIR" --data-dir "$DATA_DIR"
