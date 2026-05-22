# APDB Real-Data Python Port

Standalone Python implementation of `apdb_real_data_matlab` for `sido0_train`.

## Run CPU then GPU

```bash
python -m apdb_real_data_python.main_real_data --run-all --outdir apdb_real_data_python/outputs
```

## Run CPU only

```bash
python -m apdb_real_data_python.main_real_data --device cpu --outdir apdb_real_data_python/outputs
```

## Run On Server With Progress

```bash
python -m apdb_real_data_python.main_real_data \
  --device cpu \
  --n-samples 10000 \
  --progress on \
  --outdir apdb_real_data_python/outputs
```

## Notes

- MOSEK reference solve uses tolerance `1e-9` by default.
- For strict MATLAB alignment, use `--alignment-mode legacy_matlab` (default).
- In `legacy_matlab` mode, the reference-solution tolerance is tightened internally
  to `1e-11` when needed, so the distance-to-`x*` stopping rule can match MATLAB behavior.
- Restart parameters are fixed to match MATLAB:
  - monotone: `rAPDB-yx K=600`, `rAPDB-yx-ada K=1000`
  - nonmonotone: `rAPDB-yx K=200`, `rAPDB-yx-ada K=200`
- APDB-yx variants run for `--mu-values 0,2` by default; result tables include a
  `Mu` column, with EGM/MOSEK reported as `Mu=-`.
- Default sample count is `--n-samples 10000`.
- Progress controls:
  - `--progress auto|on|off` (default `auto`)
  - `--progress-refresh <seconds>` (default `0.5`)
  - `--progress-every <iters>` (default `20`, larger means sparser updates)
- Plots are generated from CPU results only.

## Outputs

- `summary_table_cpu.csv/.mat`
- `summary_table_gpu.csv/.mat` (when GPU runs)
- `summary_cpu_gpu.csv/.mat`
- `experiment_outputs_cpu.mat`
- `experiment_outputs_gpu.mat` (when GPU runs)
- `comparison_plot_cpu.png/.pdf`
- `run_manifest.json`
