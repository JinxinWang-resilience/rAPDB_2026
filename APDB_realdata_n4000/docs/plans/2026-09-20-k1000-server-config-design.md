# K=1000 Server Experiment Design

## Decision

Use `K=1000` for both monotone restart solvers:

- `rAPDB-yx` monotone: `K=1000`
- `rAPDB-yx-ada` monotone checkpoint interval: `K=1000`
- both nonmonotone variants remain at `K=200`

The restart interval remains a formal runner default rather than a new CLI option. This is the smallest change needed for the next server experiment and keeps the command line aligned with the current experiment interface.

## Code and artifact changes

1. Update `RESTART_CONFIG` in `apdb_real_data_python/runner.py`. The same object is already written to `run_manifest.json`, so every new run records the actual `1000/200` configuration.
2. Change only the legacy-MAT fallback interval for monotone fixed restart in `apdb_real_data_python/plot_cpu.py` from 600 to 1000. Real restart event arrays continue to take precedence.
3. Update configuration and plotting tests before production code, then update the README.
4. Add `run_apdb_real_data_n5000_runall_mu2_k1000.sh`. Keep the older, historically named script untouched so old experiments remain reproducible.

## Server run contract

The new script runs CPU first and CUDA second when CUDA is available. It explicitly fixes `n_samples=5000`, `max_iter=9999`, stopping tolerance `1e-7`, seed 17, `reg_param=1`, `mu=2`, MATLAB-compatible data alignment, strict MOSEK with tolerance `1e-11`, and EGM scale 0.1. Results go to a new directory whose name includes `mu2` and `k1000`.

The script cannot pass `K` on the command line because `K` is intentionally controlled by `RESTART_CONFIG`; therefore the checked-out code and generated manifest are part of the experiment record.

## Verification

- Focused tests must first fail against the old `K=600` behavior and pass after implementation.
- The full Python test suite must pass.
- The new shell script must pass `bash -n`.
- A CLI help smoke test must succeed without loading the dataset or starting MOSEK.

