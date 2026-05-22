# APDB Python Port

Python translation of `apdb_matlab` QCQP experiments with CPU/GPU options.

## Install

```bash
pip install -r apdb_python/requirements.txt
```

If MOSEK is used, ensure license is configured (for example, `MOSEKLM_LICENSE_FILE`).

## Run

```bash
python -m apdb_python.main_qcqp --device cpu --n 10 --m 1000 --numsim 2 --seed 123 --max-iter 50000 --epsilon 1e-8 --plot-style both
python -m apdb_python.main_qcqp --device cuda --n 10 --m 3000 --numsim 2 --seed 123 --max-iter 50000 --epsilon 1e-8 --plot-style both
python -m apdb_python.main_qcqp --run-all --n 10 --m 1000 --numsim 1 --plot-style strict
```

`--run-all` mode:
- always runs CPU;
- runs GPU only when CUDA is available;
- saves per-device outputs and a CPU/GPU comparison table.

Gamma tuning behavior:
- XY and YX families tune `gamma0` on CPU using grid `[1, 3, 6, 9]`;
- tuning uses `mode=0` only;
- tuning budget is `min(2000, max_iter)`;
- selected family-wise best `gamma0` is used by base/restart/ada-restart variants.

For strict MATLAB reference solving (no SCS fallback):

```bash
python -m apdb_python.main_qcqp --strict-mosek
```

By default, MOSEK reference solve uses high-accuracy tolerances (MATLAB `cvx_precision high` style) with
`mosek_tol=1e-9`.

You can override this behavior:

```bash
# keep strict MOSEK, but disable high-accuracy settings
python -m apdb_python.main_qcqp --strict-mosek --no-mosek-high-accuracy

# keep high-accuracy mode and set a custom tolerance
python -m apdb_python.main_qcqp --strict-mosek --mosek-tol 1e-11
```

## Outputs

Results are written to `apdb_python/outputs/`:

- `results_apdb_both_modes_*.mat`
- `results_summary_table_*.csv`
- `results_summary_table_*.mat`
- `result_apdb_*.png`
- `results_summary_table_compare_cpu_gpu.csv` (when `--run-all`)
- `results_summary_table_compare_cpu_gpu.mat` (when `--run-all`)
- `run_manifest.json` (when `--run-all`)

If `--plot-style both` is used, two figures are generated with suffixes:

- `result_apdb_*_strict.png`
- `result_apdb_*_custom.png`

Legacy names (`results_apdb_both_modes.mat`, `results_summary_table.csv/.mat`) are also generated for MATLAB-compatible downstream scripts.
