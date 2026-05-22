#!/usr/bin/env bash
set -euo pipefail

[ -d apdb_real_data_python ] || { echo "Missing apdb_real_data_python directory."; exit 1; }
[ -d apdb_python ] || { echo "Missing apdb_python directory."; exit 1; }


python -m apdb_python.main_qcqp \
  --run-all \
  --n 10 \
  --m 1000 \
  --numsim 5 \
  --seed 123 \
  --max-iter 50000 \
  --epsilon 1e-8 \
  --strict-mosek \
  --mosek-tol 1e-10 \
  --outdir apdb_python/outputs_runall_n10_m1000_numsim4_strict

echo "All runs completed."
echo "Outputs:"
echo "  apdb_python/outputs_runall_n10_m1000_numsim4_strict"
