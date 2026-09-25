"""Complete a CPU-only n10000 run on a CUDA host without rerunning CPU algorithms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from apdb_real_data_python.data_pipeline import RealDataProblem, load_sido_real_problem
from apdb_real_data_python.io_utils import save_curves_mat, save_table_csv, save_table_mat
from apdb_real_data_python.plot_cpu import load_cpu_curves_mat, plot_cpu_curves
from apdb_real_data_python.reference_mosek import solve_reference_mosek
from apdb_real_data_python.reporting import build_cpu_gpu_summary
from apdb_real_data_python.runner import RESTART_CONFIG, _build_solver_input, _single_device_run
from apdb_real_data_python.solvers import run_egm


def input_hashes(data_dir: str | Path) -> dict[str, str]:
    result = {}
    for name in ("sido0_train.mat", "sido0_train.targets"):
        path = Path(data_dir) / name
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def reference_check(problem: RealDataProblem, optsol: np.ndarray, saved_optval: float) -> tuple[bool, dict]:
    """Check old solution on reconstructed data; not proof of byte-identical input."""
    kd = problem.input_data
    alpha = np.asarray(optsol, dtype=np.float64).reshape(-1)
    y = np.asarray(kd.y_train, dtype=np.float64).reshape(-1)
    if alpha.size != y.size or not np.isfinite(alpha).all():
        return False, {"reason": "solution shape or finite values invalid"}
    if len(kd.g_tr_class) != len(kd.trace_kernels) or len(kd.g_tr_class) == 0:
        return False, {"reason": "kernel dimensions invalid"}
    quad = [float(alpha @ (g @ alpha) / tr) for g, tr in zip(kd.g_tr_class, kd.trace_kernels)]
    objective = float(-2.0 * alpha.sum() + problem.reg_param * (alpha @ alpha)
                      + np.sum(kd.trace_kernels) * max(quad))
    metrics = {
        "old_solution_min": float(alpha.min()),
        "old_solution_label_balance": float(y @ alpha),
        "old_solution_objective_recomputed": objective,
        "old_solution_objective_saved": float(saved_optval),
        "old_solution_objective_abs_difference": abs(objective - float(saved_optval)),
    }
    consistent = (
        np.isfinite(objective)
        and alpha.min() >= -1e-7
        and abs(y @ alpha) <= 1e-5 * max(1.0, float(np.linalg.norm(alpha)))
        and math.isclose(objective, float(saved_optval), rel_tol=1e-6, abs_tol=1e-4)
    )
    return bool(consistent), metrics


def _key_rows(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return list(zip(frame.Algorithm.astype(str), frame.Mode.astype(str), frame.Mu.astype(str)))


def _curve_key(row: tuple[str, str, str]) -> str:
    algorithm, mode, mu = row
    return f"{algorithm}|{mode}" if mu == "-" else f"{algorithm}|{mode}|mu={float(mu):g}"


def _require_run_parameters(manifest: dict) -> None:
    for key in ("n_samples", "train_size", "seed", "alignment_mode", "max_iter", "stop_criteria",
                "mu_values", "mosek_tol_effective", "restart_config"):
        if key not in manifest:
            raise ValueError(f"Original run_manifest.json is missing {key}.")
    if manifest.get("data") != "sido0_train" or manifest.get("run_all") is not True:
        raise ValueError("Expected the original all-algorithm sido0_train run.")
    if manifest["alignment_mode"] != "legacy_matlab" or int(manifest["seed"]) != 17:
        raise ValueError("Unexpected original data preprocessing parameters.")
    if int(manifest["n_samples"]) != 12500 or int(manifest["max_iter"]) != 14999:
        raise ValueError("Expected the original n10000 run with n_samples=12500 and max_iter=14999.")
    if not math.isclose(float(manifest["stop_criteria"]), 1e-6, rel_tol=0, abs_tol=1e-15):
        raise ValueError("Unexpected original stop criterion.")
    if [float(mu) for mu in manifest["mu_values"]] != [2.0]:
        raise ValueError("Unexpected original Mu values.")
    if manifest["restart_config"] != RESTART_CONFIG or manifest.get("x_old_policy") != "frozen_x0":
        raise ValueError("Original restart or anchor policy differs from this implementation.")
    if not math.isclose(float(manifest["mosek_tol_effective"]), 1e-11, rel_tol=0, abs_tol=1e-16):
        raise ValueError("Unexpected effective MOSEK tolerance.")


def complete_missing_gpu(output_dir: str | Path, data_dir: str | Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU completion requires a CUDA host.")

    out = Path(output_dir).expanduser().resolve()
    data = Path(data_dir).expanduser().resolve()
    manifest_path = out / "run_manifest.json"
    cpu_mat_path = out / "experiment_outputs_cpu.mat"
    cpu_csv_path = out / "summary_table_cpu.csv"
    for path in (manifest_path, cpu_mat_path, cpu_csv_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing original CPU result: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_run_parameters(manifest)
    if manifest.get("gpu_used"):
        if all((out / name).is_file() for name in (
            "experiment_outputs_gpu.mat", "summary_table_gpu.csv", "summary_cpu_gpu.csv",
            "comparison_plot_cpu.pdf", "comparison_plot_cpu.png")):
            print("GPU completion already present; no files changed.")
            return manifest
        raise RuntimeError("Manifest says GPU is complete, but one or more GPU artifacts are missing.")

    cpu_df = pd.read_csv(cpu_csv_path, dtype={"Algorithm": str, "Mode": str, "Mu": str})
    if cpu_df.empty or "MOSEK(CVX)" not in set(cpu_df.Algorithm):
        raise ValueError("Original CPU summary is incomplete or missing MOSEK.")
    cpu_mat = loadmat(cpu_mat_path, squeeze_me=True)
    for key in ("optsol", "optval", "time_cvx", "rel_err_egm", "egm_tau_scale_used"):
        if key not in cpu_mat:
            raise ValueError(f"Original CPU MAT is missing {key}.")
    old_alpha = np.asarray(cpu_mat["optsol"], dtype=np.float64).reshape(-1)
    if old_alpha.size != int(manifest["train_size"]):
        raise ValueError("Original MOSEK solution length differs from recorded train size.")
    old_optval = float(cpu_mat["optval"])
    old_time = float(cpu_mat["time_cvx"])
    tau_scale = float(cpu_mat["egm_tau_scale_used"])

    hashes = input_hashes(data)
    problem = load_sido_real_problem(
        data, n_samples=int(manifest["n_samples"]), reg_param=1.0,
        seed=int(manifest["seed"]), alignment_mode=manifest["alignment_mode"],
    )
    if problem.n_samples != int(manifest["n_samples"]) or len(problem.input_data.y_train) != int(manifest["train_size"]):
        raise ValueError("Reconstructed sample or train size differs from the original run.")

    reference_match, check = reference_check(problem, old_alpha, old_optval)
    if reference_match:
        first = run_egm(
            _build_solver_input(problem, old_alpha), stop_criteria=-1.0,
            max_iter=1, device="cpu", progress={"mode": "off"}, tau_scale=tau_scale,
        )
        saved_first = float(np.asarray(cpu_mat["rel_err_egm"]).reshape(-1)[0])
        computed_first = float(np.asarray(first.rel_err_x).reshape(-1)[0])
        check["old_egm_first_residual_saved"] = saved_first
        check["old_egm_first_residual_recomputed"] = computed_first
        reference_match = bool(math.isclose(saved_first, computed_first, rel_tol=1e-5, abs_tol=1e-8))

    if reference_match:
        selected_alpha, selected_optval, selected_time = old_alpha, old_optval, old_time
        print("Old MOSEK solution passes reconstructed-data checks; reusing it for GPU.")
    else:
        effective_tol = float(manifest["mosek_tol_effective"])
        selected_alpha, selected_optval, selected_time = solve_reference_mosek(
            _build_solver_input(problem, old_alpha), strict_mosek=True, mosek_tol=effective_tol,
        )
        print("Old reference is incompatible with reconstructed data; recomputed MOSEK for GPU only.")

    progress_cfg = {
        "mode": manifest.get("progress_mode", "off"),
        "refresh_sec": float(manifest.get("progress_refresh", 0.5)),
        "every_n": int(manifest.get("progress_every", 20)),
    }
    gpu_df, gpu_raw, gpu_curves = _single_device_run(
        problem=problem, optsol=np.asarray(selected_alpha, dtype=np.float64),
        optval=float(selected_optval), time_mosek=float(selected_time),
        max_iter=int(manifest["max_iter"]), stop_criteria=float(manifest["stop_criteria"]),
        device="cuda", progress_cfg=progress_cfg,
        egm_tau_scale=tau_scale,
        egm_tune=False,
        egm_tune_max_iter=int(manifest.get("egm_tune_max_iter", 1000)),
        egm_tune_grid=tuple(float(x) for x in manifest.get("egm_tune_grid", [0.05, 0.1, 0.2, 0.5])),
        mu_values=tuple(float(x) for x in manifest["mu_values"]),
    )
    expected = set(_key_rows(cpu_df[cpu_df.Algorithm != "MOSEK(CVX)"]))
    observed = set(_key_rows(gpu_df[gpu_df.Algorithm != "MOSEK(CVX)"]))
    if not expected or observed != expected or len(observed) != len(gpu_df) - 1:
        raise RuntimeError(f"GPU algorithms incomplete: expected {expected}, got {observed}.")
    expected_curves = {_curve_key(row) for row in expected}
    if set(gpu_curves) != expected_curves:
        raise RuntimeError(f"GPU curves incomplete: expected {expected_curves}, got {set(gpu_curves)}.")
    for curve in gpu_curves.values():
        arrays = [np.asarray(curve[field], dtype=np.float64).reshape(-1) for field in ("iter", "time", "rel")]
        if not arrays[0].size or len({array.size for array in arrays}) != 1 or not all(np.isfinite(array).all() for array in arrays):
            raise RuntimeError("GPU curve arrays are empty, mismatched, or nonfinite.")

    # The CPU MOSEK row is historical. Do not label it as GPU time or compute a speed ratio.
    gpu_algorithm_df = gpu_df[gpu_df.Algorithm != "MOSEK(CVX)"].copy()
    cpu_df["Mu"] = cpu_df.Mu.astype(str)
    gpu_algorithm_df["Mu"] = gpu_algorithm_df.Mu.astype(str)
    gpu_raw = dict(gpu_raw)
    gpu_raw["ResultTable"] = gpu_algorithm_df.to_records(index=False)
    merged = build_cpu_gpu_summary(cpu_df, gpu_algorithm_df)
    if len(merged) != len(cpu_df):
        raise RuntimeError("CPU/GPU summary keys did not match one-to-one.")
    merged["ReferenceMatch"] = bool(reference_match)
    merged["ReferenceNote"] = (
        "same numerical reference; timing may span nodes" if reference_match
        else "CPU uses old MOSEK reference; GPU uses recomputed reference; timing may span nodes"
    )

    selected_alpha = np.asarray(selected_alpha, dtype=np.float64).reshape(-1)
    distance = float(np.linalg.norm(selected_alpha - old_alpha))
    new_manifest = dict(manifest)
    new_manifest["gpu_used"] = True
    new_manifest["gpu_completion"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_available_at_completion": True,
        "input_sha256": hashes,
        "reference_match": bool(reference_match),
        "reference_check": check,
        "old_optval": old_optval,
        "gpu_optval": float(selected_optval),
        "old_new_reference_l2_distance": distance,
        "old_new_reference_relative_distance": distance / max(1.0, float(np.linalg.norm(old_alpha))),
        "cpu_reference": "original MOSEK solution",
        "gpu_reference": "original MOSEK solution" if reference_match else "recomputed MOSEK solution",
        "data_identity_note": "Numerical compatibility check only; original input hash was not recorded.",
        "timing_note": "CPU and GPU timings may be from different hosts; MOSEK has no GPU speed ratio.",
    }

    # Only publish complete GPU results. A killed/failed GPU run leaves original artifacts intact.
    with tempfile.TemporaryDirectory(prefix=".gpu-completion-", dir=out) as temp_name:
        temp = Path(temp_name)
        gpu_raw = dict(gpu_raw)
        gpu_raw.update({
            "optsol": selected_alpha,
            "optval": float(selected_optval),
            "time_cvx": float(selected_time),
            "reference_match": int(reference_match),
            "old_new_reference_l2_distance": distance,
        })
        save_curves_mat(temp / "experiment_outputs_gpu.mat", gpu_raw)
        save_table_csv(temp / "summary_table_gpu.csv", gpu_algorithm_df)
        save_table_mat(temp / "summary_table_gpu.mat", gpu_algorithm_df)
        save_table_csv(temp / "summary_cpu_gpu.csv", merged)
        save_table_mat(temp / "summary_cpu_gpu.mat", merged)
        plot_cpu_curves(load_cpu_curves_mat(cpu_mat_path), temp, gpu_curves=gpu_curves,
                        reference_match=bool(reference_match))
        (temp / "run_manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n", encoding="utf-8")
        for name in (
            "experiment_outputs_gpu.mat", "summary_table_gpu.csv", "summary_table_gpu.mat",
            "summary_cpu_gpu.csv", "summary_cpu_gpu.mat", "comparison_plot_cpu.png",
            "comparison_plot_cpu.pdf", "run_manifest.json",
        ):
            os.replace(temp / name, out / name)

    print(f"GPU completion saved in {out}; reference_match={reference_match}.")
    return new_manifest


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=root / "apdb_real_data_python" / "outputs_n10000_runall_mu0_mu2_strict")
    parser.add_argument("--data-dir", type=Path, default=root / "apdb_real_data_matlab")
    args = parser.parse_args(argv)
    complete_missing_gpu(args.outdir, args.data_dir)


if __name__ == "__main__":
    main()
