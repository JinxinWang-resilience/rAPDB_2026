from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from apdb_real_data_python.data_pipeline import RealDataProblem
from apdb_real_data_python.io_utils import save_curves_mat, save_table_csv, save_table_mat
from apdb_real_data_python.plot_cpu import plot_cpu_curves
from apdb_real_data_python.reference_mosek import solve_reference_mosek
from apdb_real_data_python.reporting import build_cpu_gpu_summary, build_single_device_table
from apdb_real_data_python.solvers import run_apdb_yx, run_egm, run_rapdb_yx, run_rapdb_yx_ada

DEFAULT_EGM_TAU_SCALE = 0.1
DEFAULT_EGM_TUNE_GRID = (0.05, 0.1, 0.2, 0.5)
DEFAULT_MU_VALUES = (0.0, 2.0)


def _format_mu_label(mu: float) -> str:
    text = f"{float(mu):g}"
    return text.replace("-", "m").replace(".", "p")


def _build_solver_input(problem: RealDataProblem, optsol: np.ndarray) -> dict[str, Any]:
    kd = problem.input_data
    return {
        "G_tr_class": kd.g_tr_class,
        "Kernel_tr_class": kd.kernel_tr_class,
        "Kernel_test_class": kd.kernel_test_class,
        "G_tr_max_norm": kd.g_tr_max_norm,
        "Ytrain": kd.y_train,
        "Ytest": kd.y_test,
        "trace_kernels": kd.trace_kernels,
        "reg_param": problem.reg_param,
        "x0": problem.x0,
        "y0": problem.y0,
        "optsol": optsol,
    }


def _tune_egm_tau_scale(
    *,
    inp: dict[str, Any],
    stop_criteria: float,
    max_iter: int,
    device: str,
    progress_cfg: dict[str, Any],
    tau_grid: tuple[float, ...],
) -> tuple[float, list[dict[str, float]]]:
    if max_iter <= 0:
        raise ValueError("egm_tune_max_iter must be positive.")
    if len(tau_grid) == 0:
        raise ValueError("egm_tune_grid must not be empty.")

    records: list[dict[str, float]] = []
    for scale in tau_grid:
        scale_f = float(scale)
        if scale_f <= 0.0:
            raise ValueError("All egm_tune_grid values must be positive.")
        out = run_egm(
            inp,
            stop_criteria=stop_criteria,
            max_iter=max_iter,
            device=device,
            progress=progress_cfg,
            tau_scale=scale_f,
        )
        records.append(
            {
                "tau_scale": scale_f,
                "final_residual": float(out.rel_err_x[-1]),
                "final_time": float(out.time_period[-1]),
                "final_iter": float(out.iter_epoch[-1]),
            }
        )

    best = min(records, key=lambda r: (r["final_residual"], r["final_time"], r["tau_scale"]))
    return float(best["tau_scale"]), records


def _single_device_run(
    *,
    problem: RealDataProblem,
    optsol: np.ndarray,
    optval: float,
    time_mosek: float,
    max_iter: int,
    stop_criteria: float,
    device: str,
    progress_cfg: dict[str, Any],
    egm_tau_scale: float | None = None,
    egm_tune: bool = False,
    egm_tune_max_iter: int = 1000,
    egm_tune_grid: tuple[float, ...] = DEFAULT_EGM_TUNE_GRID,
    mu_values: tuple[float, ...] = DEFAULT_MU_VALUES,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    inp = _build_solver_input(problem, optsol)
    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    raw: dict[str, Any] = {}

    rows.append(
        {
            "Algorithm": "MOSEK(CVX)",
            "Mode": "-",
            "Mu": "-",
            "Iterations": np.nan,
            "FinalResidual": np.nan,
            "TimeSeconds": float(time_mosek),
        }
    )

    if egm_tau_scale is not None and float(egm_tau_scale) <= 0.0:
        raise ValueError("egm_tau_scale must be positive.")
    if egm_tune_max_iter <= 0:
        raise ValueError("egm_tune_max_iter must be positive.")
    if len(egm_tune_grid) == 0:
        raise ValueError("egm_tune_grid must not be empty.")
    if len(mu_values) == 0:
        raise ValueError("mu_values must not be empty.")
    if any(float(v) < 0.0 for v in mu_values):
        raise ValueError("All mu_values must be nonnegative.")

    selected_tau_scale = DEFAULT_EGM_TAU_SCALE if egm_tau_scale is None else float(egm_tau_scale)
    tune_records: list[dict[str, float]] = []
    if egm_tau_scale is None and bool(egm_tune):
        selected_tau_scale, tune_records = _tune_egm_tau_scale(
            inp=inp,
            stop_criteria=stop_criteria,
            max_iter=egm_tune_max_iter,
            device=device,
            progress_cfg=progress_cfg,
            tau_grid=tuple(float(v) for v in egm_tune_grid),
        )

    out_egm = run_egm(
        inp,
        stop_criteria=stop_criteria,
        max_iter=max_iter,
        device=device,
        progress=progress_cfg,
        tau_scale=selected_tau_scale,
    )
    rows.append(
        {
            "Algorithm": "EGM",
            "Mode": "-",
            "Mu": "-",
            "Iterations": float(out_egm.iter_epoch[-1]),
            "FinalResidual": float(out_egm.rel_err_x[-1]),
            "TimeSeconds": float(out_egm.time_period[-1]),
        }
    )
    curves["EGM|-"] = {"iter": out_egm.iter_epoch, "time": out_egm.time_period, "rel": out_egm.rel_err_x}
    raw["rel_err_egm"] = out_egm.rel_err_x
    raw["time_egm"] = out_egm.time_period
    raw["iter_egm"] = out_egm.iter_epoch
    raw["egm_tau_scale_used"] = float(selected_tau_scale)
    if tune_records:
        raw["egm_tau_tuning_grid"] = np.asarray([r["tau_scale"] for r in tune_records], dtype=np.float64)
        raw["egm_tau_tuning_final_residual"] = np.asarray(
            [r["final_residual"] for r in tune_records], dtype=np.float64
        )
        raw["egm_tau_tuning_final_time"] = np.asarray([r["final_time"] for r in tune_records], dtype=np.float64)

    for mu in tuple(float(v) for v in mu_values):
        mu_label = _format_mu_label(mu)
        for mode in [0, 1]:
            mode_name = "mono" if mode == 0 else "nonmono"
            k_restart = 600 if mode == 0 else 200
            k_restart_ada = 1000 if mode == 0 else 200

            out_apdb = run_apdb_yx(
                inp,
                stop_criteria=stop_criteria,
                max_iter=max_iter,
                non_mono=mode,
                device=device,
                progress=progress_cfg,
                mu=mu,
            )
            rows.append(
                {
                    "Algorithm": "APDB-yx",
                    "Mode": mode_name,
                    "Mu": mu,
                    "Iterations": float(out_apdb.iter_epoch[-1]),
                    "FinalResidual": float(out_apdb.rel_err_x[-1]),
                    "TimeSeconds": float(out_apdb.time_period[-1]),
                }
            )
            curves[f"APDB-yx|{mode_name}|mu={mu_label}"] = {
                "iter": out_apdb.iter_epoch,
                "time": out_apdb.time_period,
                "rel": out_apdb.rel_err_x,
            }
            raw[f"rel_err_apdb_mu{mu_label}_{mode}"] = out_apdb.rel_err_x
            raw[f"time_apdb_mu{mu_label}_{mode}"] = out_apdb.time_period
            raw[f"iter_apdb_mu{mu_label}_{mode}"] = out_apdb.iter_epoch

            out_r = run_rapdb_yx(
                inp,
                stop_criteria=stop_criteria,
                max_iter=max_iter,
                k_restart=k_restart,
                non_mono=mode,
                device=device,
                progress=progress_cfg,
                mu=mu,
            )
            rows.append(
                {
                    "Algorithm": "rAPDB-yx",
                    "Mode": mode_name,
                    "Mu": mu,
                    "Iterations": float(out_r.iter_epoch[-1]),
                    "FinalResidual": float(out_r.rel_err_x[-1]),
                    "TimeSeconds": float(out_r.time_period[-1]),
                }
            )
            curves[f"rAPDB-yx|{mode_name}|mu={mu_label}"] = {
                "iter": out_r.iter_epoch,
                "time": out_r.time_period,
                "rel": out_r.rel_err_x,
            }
            raw[f"rel_err_rapdb_mu{mu_label}_{mode}"] = out_r.rel_err_x
            raw[f"time_rapdb_mu{mu_label}_{mode}"] = out_r.time_period
            raw[f"iter_rapdb_mu{mu_label}_{mode}"] = out_r.iter_epoch

            out_ra = run_rapdb_yx_ada(
                inp,
                stop_criteria=stop_criteria,
                max_iter=max_iter,
                k_restart=k_restart_ada,
                non_mono=mode,
                device=device,
                progress=progress_cfg,
                mu=mu,
            )
            rows.append(
                {
                    "Algorithm": "rAPDB-yx-ada",
                    "Mode": mode_name,
                    "Mu": mu,
                    "Iterations": float(out_ra.iter_epoch[-1]),
                    "FinalResidual": float(out_ra.rel_err_x[-1]),
                    "TimeSeconds": float(out_ra.time_period[-1]),
                }
            )
            curves[f"rAPDB-yx-ada|{mode_name}|mu={mu_label}"] = {
                "iter": out_ra.iter_epoch,
                "time": out_ra.time_period,
                "rel": out_ra.rel_err_x,
            }
            raw[f"rel_err_rapdb_ada_mu{mu_label}_{mode}"] = out_ra.rel_err_x
            raw[f"time_rapdb_ada_mu{mu_label}_{mode}"] = out_ra.time_period
            raw[f"iter_rapdb_ada_mu{mu_label}_{mode}"] = out_ra.iter_epoch

    df = build_single_device_table(rows)
    raw["ResultTable"] = df.to_records(index=False)
    raw["optval"] = float(optval)
    raw["optsol"] = np.asarray(optsol, dtype=np.float64)
    raw["time_cvx"] = float(time_mosek)
    return df, raw, curves


def run_real_data_experiment(
    *,
    problem: RealDataProblem,
    outdir: str | Path,
    run_all: bool,
    device: str,
    max_iter: int,
    stop_criteria: float,
    strict_mosek: bool,
    mosek_tol: float,
    progress_mode: str = "auto",
    progress_refresh: float = 0.5,
    progress_every: int = 20,
    egm_tau_scale: float | None = None,
    egm_tune: bool = False,
    egm_tune_max_iter: int = 1000,
    egm_tune_grid: tuple[float, ...] = DEFAULT_EGM_TUNE_GRID,
    mu_values: tuple[float, ...] = DEFAULT_MU_VALUES,
) -> dict[str, Any]:
    progress_mode = progress_mode.lower().strip()
    if progress_mode not in {"on", "off", "auto"}:
        raise ValueError(f"Unsupported progress_mode: {progress_mode}")
    if progress_refresh <= 0.0:
        raise ValueError("progress_refresh must be positive.")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive.")
    if egm_tau_scale is not None and float(egm_tau_scale) <= 0.0:
        raise ValueError("egm_tau_scale must be positive.")
    if egm_tune_max_iter <= 0:
        raise ValueError("egm_tune_max_iter must be positive.")
    if len(egm_tune_grid) == 0:
        raise ValueError("egm_tune_grid must not be empty.")
    if any(float(v) <= 0.0 for v in egm_tune_grid):
        raise ValueError("All egm_tune_grid values must be positive.")
    if len(mu_values) == 0:
        raise ValueError("mu_values must not be empty.")
    if any(float(v) < 0.0 for v in mu_values):
        raise ValueError("All mu_values must be nonnegative.")

    progress_cfg = {
        "mode": progress_mode,
        "refresh_sec": float(progress_refresh),
        "every_n": int(progress_every),
    }

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    ref_problem = {
        "G_tr_class": problem.input_data.g_tr_class,
        "Ytrain": problem.input_data.y_train,
        "trace_kernels": problem.input_data.trace_kernels,
        "reg_param": problem.reg_param,
    }
    effective_mosek_tol = float(mosek_tol)
    if problem.alignment_mode == "legacy_matlab" and effective_mosek_tol > 1e-11:
        # In this KML instance, 1e-9 objective-level tolerance can still leave an
        # optimizer mismatch large enough to block 1e-8 solution-distance stopping.
        # Tighten only the reference solve used for residual/stopping alignment.
        effective_mosek_tol = 1e-11

    optsol, optval, time_mosek = solve_reference_mosek(
        ref_problem,
        strict_mosek=strict_mosek,
        mosek_tol=effective_mosek_tol,
    )

    cpu_df, cpu_raw, cpu_curves = _single_device_run(
        problem=problem,
        optsol=optsol,
        optval=optval,
        time_mosek=time_mosek,
        max_iter=max_iter,
        stop_criteria=stop_criteria,
        device="cpu" if run_all else device,
        progress_cfg=progress_cfg,
        egm_tau_scale=egm_tau_scale,
        egm_tune=egm_tune,
        egm_tune_max_iter=egm_tune_max_iter,
        egm_tune_grid=egm_tune_grid,
        mu_values=mu_values,
    )

    save_table_csv(out / "summary_table_cpu.csv", cpu_df)
    save_table_mat(out / "summary_table_cpu.mat", cpu_df)
    save_curves_mat(out / "experiment_outputs_cpu.mat", cpu_raw)
    plot_png, plot_pdf = plot_cpu_curves(cpu_curves, out)

    gpu_df: pd.DataFrame | None = None
    gpu_raw: dict[str, Any] | None = None
    gpu_used = False

    if run_all:
        if torch.cuda.is_available():
            gpu_df, gpu_raw, _ = _single_device_run(
                problem=problem,
                optsol=optsol,
                optval=optval,
                time_mosek=time_mosek,
                max_iter=max_iter,
                stop_criteria=stop_criteria,
                device="cuda",
                progress_cfg=progress_cfg,
                egm_tau_scale=egm_tau_scale,
                egm_tune=egm_tune,
                egm_tune_max_iter=egm_tune_max_iter,
                egm_tune_grid=egm_tune_grid,
                mu_values=mu_values,
            )
            save_table_csv(out / "summary_table_gpu.csv", gpu_df)
            save_table_mat(out / "summary_table_gpu.mat", gpu_df)
            save_curves_mat(out / "experiment_outputs_gpu.mat", gpu_raw)
            gpu_used = True

    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        gpu_df = cpu_df
        gpu_raw = cpu_raw
        cpu_df = pd.DataFrame(columns=cpu_df.columns)

    compare_df = build_cpu_gpu_summary(cpu_df if len(cpu_df) else pd.DataFrame(gpu_df), gpu_df)
    save_table_csv(out / "summary_cpu_gpu.csv", compare_df)
    save_table_mat(out / "summary_cpu_gpu.mat", compare_df)
    print("\n================ CPU/GPU Summary ================")
    print(compare_df.to_string(index=False))

    manifest = {
        "data": problem.data_name,
        "n_samples": int(problem.n_samples),
        "seed": int(problem.seed),
        "train_size": int(problem.input_data.y_train.shape[0]),
        "test_size": int(problem.input_data.y_test.shape[0]),
        "run_all": bool(run_all),
        "device": device,
        "gpu_available": bool(torch.cuda.is_available()),
        "gpu_used": bool(gpu_used),
        "max_iter": int(max_iter),
        "stop_criteria": float(stop_criteria),
        "progress_mode": progress_mode,
        "progress_refresh": float(progress_refresh),
        "progress_every": int(progress_every),
        "egm_tau_scale": None if egm_tau_scale is None else float(egm_tau_scale),
        "egm_tune": bool(egm_tune),
        "egm_tune_max_iter": int(egm_tune_max_iter),
        "egm_tune_grid": [float(v) for v in egm_tune_grid],
        "mu_values": [float(v) for v in mu_values],
        "alignment_mode": problem.alignment_mode,
        "mosek_tol_requested": float(mosek_tol),
        "mosek_tol_effective": float(effective_mosek_tol),
    }
    with (out / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "cpu_df": cpu_df,
        "gpu_df": gpu_df,
        "compare_df": compare_df,
        "plot_png": plot_png,
        "plot_pdf": plot_pdf,
    }
