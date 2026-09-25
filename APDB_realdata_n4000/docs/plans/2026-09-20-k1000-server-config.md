# K=1000 Server Configuration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `K=1000` the formal monotone restart interval for both real-data restart solvers and provide a reproducible n=5000 server launcher.

**Architecture:** Keep restart intervals centralized in `runner.RESTART_CONFIG`, which already drives both solver calls and manifest metadata. Keep plot fallback metadata synchronized with that formal default, while recorded restart events remain authoritative. Add a new explicit server script without altering the historically named launcher.

**Tech Stack:** Python 3, pytest, Bash, PyTorch/CVXPY/MOSEK experiment CLI, Matplotlib.

---

### Task 1: Lock the formal restart configuration with failing tests

**Files:**
- Modify: `apdb_real_data_python/tests/test_runner_config.py`
- Modify: `apdb_real_data_python/tests/test_plot_style.py`

**Step 1: Write the failing runner tests**

Change monotone expectations for both solvers to 1000 while retaining nonmonotone expectations at 200:

```python
assert calls_r[0]["k_restart"] == 1000
assert calls_ra[0]["k_restart"] == 1000
assert runner.RESTART_CONFIG == {
    "rapdb_yx": {"mono": 1000, "nonmono": 200},
    "rapdb_yx_ada": {"mono": 1000, "nonmono": 200},
}
```

Update the manifest assertion to the same dictionary.

**Step 2: Write the failing plot fallback tests**

Assert that the monotone fixed-restart plot spec uses 1000 and use iteration samples around 1000 and 2000:

```python
assert mono_spec["restart_every"] == 1000
assert _marker_kwargs(mono_spec, [1, 1000, 1001, 2000, 2001])["markevery"] == [1, 3]
```

Leave event-priority tests semantically unchanged; recorded event values need not be multiples of K.

**Step 3: Run focused tests to verify they fail**

Run:

```bash
pytest -q apdb_real_data_python/tests/test_runner_config.py apdb_real_data_python/tests/test_plot_style.py
```

Expected: failures showing production defaults are still 600.

### Task 2: Implement K=1000 in runner and plotting fallback

**Files:**
- Modify: `apdb_real_data_python/runner.py`
- Modify: `apdb_real_data_python/plot_cpu.py`

**Step 1: Change the centralized runner configuration**

```python
RESTART_CONFIG = {
    "rapdb_yx": {"mono": 1000, "nonmono": 200},
    "rapdb_yx_ada": {"mono": 1000, "nonmono": 200},
}
```

Do not change solver restart equations or nonmonotone values.

**Step 2: Change the legacy plot fallback**

Set the monotone `rAPDB-yx` plot spec's `restart_every` to 1000. Do not add K to the legend, do not add an “at equal K” caption, and do not change line widths or colors.

**Step 3: Run the focused tests**

Run the same two-file pytest command.

Expected: all focused tests pass.

### Task 3: Document the formal defaults and add the server launcher

**Files:**
- Modify: `apdb_real_data_python/README.md`
- Create: `run_apdb_real_data_n5000_runall_mu2_k1000.sh`
- Test: `apdb_real_data_python/tests/test_server_launcher.py`

**Step 1: Add a failing launcher contract test**

Read the new script and assert it exists, uses `--run-all`, `--n-samples 5000`, `--max-iter 9999`, `--stop-criteria 1e-7`, `--seed 17`, `--reg-param 1`, `--mu-values 2`, `--alignment-mode legacy_matlab`, `--strict-mosek`, `--mosek-tol 1e-11`, `--egm-tau-scale 0.1`, and a unique `outputs_n5000_runall_mu2_k1000` directory.

**Step 2: Run the launcher test to verify it fails**

Run:

```bash
pytest -q apdb_real_data_python/tests/test_server_launcher.py
```

Expected: failure because the new launcher does not yet exist.

**Step 3: Add the launcher**

Create a strict Bash script that validates the two required data/code directories, creates a Matplotlib cache, and runs:

```bash
python -m apdb_real_data_python.main_real_data \
  --run-all \
  --n-samples 5000 \
  --max-iter 9999 \
  --stop-criteria 1e-7 \
  --seed 17 \
  --reg-param 1 \
  --alignment-mode legacy_matlab \
  --strict-mosek \
  --mosek-tol 1e-11 \
  --egm-tau-scale 0.1 \
  --progress on \
  --progress-every 200 \
  --mu-values 2 \
  --outdir apdb_real_data_python/outputs_n5000_runall_mu2_k1000
```

**Step 4: Update README**

Replace both monotone 600 values with 1000 and add a server-launch example naming the new script. State that K comes from `RESTART_CONFIG` and is saved in the manifest.

**Step 5: Run launcher tests and syntax check**

Run:

```bash
pytest -q apdb_real_data_python/tests/test_server_launcher.py
bash -n run_apdb_real_data_n5000_runall_mu2_k1000.sh
```

Expected: both commands succeed.

### Task 4: Full verification

**Files:**
- Verify: `apdb_real_data_python/tests/`

**Step 1: Run the entire test suite**

```bash
pytest -q apdb_real_data_python/tests
```

Expected: all tests pass.

**Step 2: Run the CLI help smoke test**

```bash
python -m apdb_real_data_python.main_real_data --help
```

Expected: exit status 0 and documented experiment flags.

**Step 3: Inspect final changes**

Because this workspace is not a Git checkout, inspect changed files directly with `rg`, `sed`, and shell syntax checks rather than creating commits. Confirm no saved experiment outputs were overwritten.

