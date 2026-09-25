#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -d apdb_real_data_python ] || { echo "Missing apdb_real_data_python directory."; exit 1; }
[ -d apdb_real_data_matlab ] || { echo "Missing apdb_real_data_matlab directory."; exit 1; }

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/apdb_matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"

python -m apdb_real_data_python.main_real_data \
  --run-all \
  --n-samples 5000 \
  --max-iter 9999 \
  --stop-criteria 1e-7 \
  --reg-param 1 \
  --strict-mosek \
  --mosek-tol 1e-9 \
  --progress on \
  --progress-every 200 \
  --mu-values 2 \
  --outdir apdb_real_data_python/outputs_n5000_runall_mu2_k1000

echo "Run completed."
echo "Outputs:"
echo "  apdb_real_data_python/outputs_n5000_runall_mu2_k1000"
