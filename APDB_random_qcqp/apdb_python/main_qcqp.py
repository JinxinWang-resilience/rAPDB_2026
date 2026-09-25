from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from apdb_python.data_gen import generate_random_qcqp, make_rng
from apdb_python.io_utils import (
    load_problem_from_mat,
    save_results_mat,
    save_summary_table,
    save_summary_table_mat,
)
from apdb_python.plotting import plot_results
from apdb_python.solvers import (
    run_apdb_c_lr,
    run_apdb_c_lr_ada_restart,
    run_apdb_c_lr_restart,
    run_apdb_c_xy,
    run_apdb_c_xy_ada_restart,
    run_apdb_c_xy_restart,
    solve_qcqp_reference,
    tune_gamma_xy_family,
    tune_gamma_yx_family,
    tune_tau_egm,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="APDB QCQP MATLAB-to-Python experiment runner")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="run CPU first, then GPU if CUDA is available",
    )
    parser.add_argument("--n", type=int, default=10, help="number of constraints")
    parser.add_argument("--m", type=int, default=90, help="number of primal variables")
    parser.add_argument("--numsim", type=int, default=2, help="number of simulations")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-iter", type=int, default=50000)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--outdir", type=str, default="apdb_python/outputs")
    parser.add_argument(
        "--xy-ls-debug",
        action="store_true",
        help="print tau after each line-search backtracking step for XY-family solvers",
    )
    parser.add_argument(
        "--plot-style",
        choices=["strict", "custom", "both"],
        default="both",
        help="strict: MATLAB style; custom: clearer markers; both: output both figures",
    )
    parser.add_argument(
        "--strict-mosek",
        action="store_true",
        help="if set, fail when MOSEK is unavailable instead of falling back to SCS",
    )
    parser.add_argument(
        "--no-mosek-high-accuracy",
        action="store_true",
        help="disable MOSEK high-accuracy tolerances for reference solve",
    )
    parser.add_argument(
        "--mosek-tol",
        type=float,
        default=1e-10,
        help="MOSEK high-accuracy tolerance used by the reference solve",
    )
    parser.add_argument(
        "--problem-mat",
        type=str,
        default="",
        help="optional .mat file containing A,Q,b,d,c and optional x0,y0,optval",
    )
    return parser


def _mode_cells(numsim: int) -> list[list[np.ndarray]]:
    return [[np.array([], dtype=np.float64) for _ in range(numsim)] for _ in range(2)]


def _final_mean(series: list[np.ndarray]) -> float:
    vals = [float(v[-1]) for v in series if len(v) > 0]
    return float(np.mean(vals)) if vals else float("nan")


def _print_summary(rows: list[dict], header_suffix: str = "") -> None:
    title = "================ Average over simulations ================"
    if header_suffix:
        print(f"\n{title} [{header_suffix}]")
    else:
        print(f"\n{title}")
    print(
        f"{'Group':<14} {'Algorithm':<18} {'Time':<12} {'Iter':<12} "
        f"{'SuboptRes':<18} {'InfeasRes':<18} {'GxiTime':<12}"
    )
    print("-" * 118)
    for row in rows:
        print(
            f"{row['Group']:<14} {row['Algorithm']:<18} {row['Time']:<12.2e} "
            f"{row['Iter']:<12.2e} {row['SuboptRes']:<18.2e} {row['InfeasRes']:<18.2e} {row['GxiTime']:<12.2e}"
        )


def _prepare_run_context(args: argparse.Namespace) -> dict[str, Any]:
    rng = make_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    case_data: dict[str, Any] | None = None
    n_run = args.n
    m_run = args.m
    if args.problem_mat:
        case_data = load_problem_from_mat(args.problem_mat)
        n_case = int(case_data["problem"]["c"].shape[0])
        m_case = int(case_data["problem"]["b"].shape[0])
        if n_case != n_run or m_case != m_run:
            print(f"Override n,m from problem mat: ({n_run},{m_run}) -> ({n_case},{m_case})")
        n_run = n_case
        m_run = m_case

    if case_data is None:
        x0_base = rng.random(m_run)
        y0_base = np.zeros(n_run, dtype=np.float64)
        problems = [generate_random_qcqp(n_run, m_run, rng) for _ in range(args.numsim)]
    else:
        x0_base = np.asarray(case_data.get("x0", rng.random(m_run)), dtype=np.float64).reshape(-1)
        y0_base = np.asarray(case_data.get("y0", np.zeros(n_run, dtype=np.float64)), dtype=np.float64).reshape(-1)
        problems = [case_data["problem"] for _ in range(args.numsim)]

    mosek_time = np.zeros(args.numsim, dtype=np.float64)
    optvals = np.zeros(args.numsim, dtype=np.float64)
    if case_data is not None and "optval" in case_data:
        print("using optval from problem .mat file")
        optvals[:] = float(case_data["optval"])
    else:
        for sim in range(args.numsim):
            print(f"starting cvxpy/mosek reference... (simulation {sim + 1}/{args.numsim})")
            _, optval, ref_time = solve_qcqp_reference(
                problems[sim],
                strict_mosek=args.strict_mosek,
                high_accuracy=not args.no_mosek_high_accuracy,
                mosek_tol=args.mosek_tol,
            )
            optvals[sim] = float(optval)
            mosek_time[sim] = float(ref_time)

    gamma_grid_family = np.asarray([1.0, 3.0, 6.0, 9.0], dtype=np.float64)
    tune_iter_family = int(min(2000, args.max_iter))
    best_gamma_xy_sim = np.ones(args.numsim, dtype=np.float64)
    best_gamma_yx_sim = np.ones(args.numsim, dtype=np.float64)
    tune_info_xy_sim: list[dict[str, Any]] = [{} for _ in range(args.numsim)]
    tune_info_yx_sim: list[dict[str, Any]] = [{} for _ in range(args.numsim)]
    for sim in range(args.numsim):
        best_gamma_xy, tune_info_xy = tune_gamma_xy_family(
            runner=run_apdb_c_xy,
            input_data=problems[sim],
            optval=float(optvals[sim]),
            x0=x0_base.copy(),
            y0=y0_base.copy(),
            epsilon=args.epsilon,
            max_iter=args.max_iter,
            gamma_grid=gamma_grid_family,
            tune_iter=tune_iter_family,
            device="cpu",
            sim_idx=sim,
            ls_debug=args.xy_ls_debug,
        )
        best_gamma_xy_sim[sim] = float(best_gamma_xy)
        tune_info_xy_sim[sim] = tune_info_xy
        print(
            f"selected XY gamma [sim {sim + 1}] = {best_gamma_xy:.2e}; "
            f"mode0_subopt={tune_info_xy['best_subopt_mode0']:.3e}, "
            f"score={tune_info_xy['best_score']:.3e}"
        )
        best_gamma_yx, tune_info_yx = tune_gamma_yx_family(
            runner=run_apdb_c_lr,
            input_data=problems[sim],
            optval=float(optvals[sim]),
            x0=x0_base.copy(),
            y0=y0_base.copy(),
            epsilon=args.epsilon,
            max_iter=args.max_iter,
            gamma_grid=gamma_grid_family,
            tune_iter=tune_iter_family,
            device="cpu",
            sim_idx=sim,
        )
        best_gamma_yx_sim[sim] = float(best_gamma_yx)
        tune_info_yx_sim[sim] = tune_info_yx
        print(
            f"selected YX gamma [sim {sim + 1}] = {best_gamma_yx:.2e}; "
            f"mode0_subopt={tune_info_yx['best_subopt_mode0']:.3e}, "
            f"score={tune_info_yx['best_score']:.3e}"
        )

    return {
        "n_run": n_run,
        "m_run": m_run,
        "x0_base": x0_base,
        "y0_base": y0_base,
        "problems": problems,
        "optvals": optvals,
        "mosek_time": mosek_time,
        "Total_time_Mosek": float(np.sum(mosek_time)),
        "gamma_grid_family": gamma_grid_family,
        "tune_iter_family": tune_iter_family,
        "tune_info_xy_sim": tune_info_xy_sim,
        "tune_info_yx_sim": tune_info_yx_sim,
        "best_gamma_xy_sim": best_gamma_xy_sim,
        "best_gamma_yx_sim": best_gamma_yx_sim,
    }


def _run_for_device(
    *,
    args: argparse.Namespace,
    context: dict[str, Any],
    device: str,
    outdir: Path,
    save_legacy: bool,
) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    n_run = int(context["n_run"])
    m_run = int(context["m_run"])
    x0_base = np.asarray(context["x0_base"], dtype=np.float64)
    y0_base = np.asarray(context["y0_base"], dtype=np.float64)
    problems = context["problems"]
    optvals = np.asarray(context["optvals"], dtype=np.float64)
    best_gamma_xy_sim = np.asarray(context.get("best_gamma_xy_sim", np.ones(args.numsim, dtype=np.float64)), dtype=np.float64)
    best_gamma_yx_sim = np.asarray(context.get("best_gamma_yx_sim", np.ones(args.numsim, dtype=np.float64)), dtype=np.float64)
    best_tau_egm = [0.0 for _ in range(args.numsim)]
    tune_table_egm = [None for _ in range(args.numsim)]

    rel_infeas_err_egm = [np.array([], dtype=np.float64) for _ in range(args.numsim)]
    rel_subopt_egm = [np.array([], dtype=np.float64) for _ in range(args.numsim)]
    time_period_egm = [np.array([], dtype=np.float64) for _ in range(args.numsim)]
    iter_epoch_egm = [np.array([], dtype=np.float64) for _ in range(args.numsim)]
    oracle_egm = [np.array([], dtype=np.float64) for _ in range(args.numsim)]

    rel_infeas_1 = _mode_cells(args.numsim)
    rel_subopt_1 = _mode_cells(args.numsim)
    time_period_1 = _mode_cells(args.numsim)
    iter_epoch_1 = _mode_cells(args.numsim)
    oracle_1 = _mode_cells(args.numsim)

    rel_infeas_1_ada = _mode_cells(args.numsim)
    rel_subopt_1_ada = _mode_cells(args.numsim)
    time_period_1_ada = _mode_cells(args.numsim)
    iter_epoch_1_ada = _mode_cells(args.numsim)
    oracle_1_ada = _mode_cells(args.numsim)
    gxi_time_1_ada = _mode_cells(args.numsim)
    restart_iter_epoch_1_ada = _mode_cells(args.numsim)
    restart_oracle_1_ada = _mode_cells(args.numsim)

    rel_infeas_r1 = _mode_cells(args.numsim)
    rel_subopt_r1 = _mode_cells(args.numsim)
    time_period_r1 = _mode_cells(args.numsim)
    iter_epoch_r1 = _mode_cells(args.numsim)
    oracle_r1 = _mode_cells(args.numsim)

    rel_infeas_xy_1 = _mode_cells(args.numsim)
    rel_subopt_xy_1 = _mode_cells(args.numsim)
    time_period_xy_1 = _mode_cells(args.numsim)
    iter_epoch_xy_1 = _mode_cells(args.numsim)
    oracle_xy_1 = _mode_cells(args.numsim)

    rel_infeas_xy_r1 = _mode_cells(args.numsim)
    rel_subopt_xy_r1 = _mode_cells(args.numsim)
    time_period_xy_r1 = _mode_cells(args.numsim)
    iter_epoch_xy_r1 = _mode_cells(args.numsim)
    oracle_xy_r1 = _mode_cells(args.numsim)

    rel_infeas_xy_ada1 = _mode_cells(args.numsim)
    rel_subopt_xy_ada1 = _mode_cells(args.numsim)
    time_period_xy_ada1 = _mode_cells(args.numsim)
    iter_epoch_xy_ada1 = _mode_cells(args.numsim)
    oracle_xy_ada1 = _mode_cells(args.numsim)
    gxi_time_xy_ada1 = _mode_cells(args.numsim)
    restart_iter_epoch_xy_ada1 = _mode_cells(args.numsim)
    restart_oracle_xy_ada1 = _mode_cells(args.numsim)

    for sim in range(args.numsim):
        print(f"\n=== Simulation {sim + 1}/{args.numsim} [{device}] ===")
        problem = problems[sim]
        x0 = x0_base.copy()
        y0 = y0_base.copy()
        optval = float(optvals[sim])
        gamma_xy = float(best_gamma_xy_sim[sim])
        gamma_yx = float(best_gamma_yx_sim[sim])

        opts_egm = {
            "epoch": 1,
            "verbose": 300,
            "tau_grid": np.array([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3], dtype=np.float64),
            "tune_iter": min(2000, args.max_iter),
        }

        best_tau, best_out, tune_table = tune_tau_egm(
            input_data=problem,
            optval=optval,
            x0=x0,
            y0=y0,
            epsilon=args.epsilon,
            max_iter=args.max_iter,
            opts=opts_egm,
            device=device,
        )

        best_tau_egm[sim] = best_tau
        tune_table_egm[sim] = tune_table
        rel_infeas_err_egm[sim] = best_out["rel_infeas_err"]
        rel_subopt_egm[sim] = best_out["rel_subopt_err"]
        time_period_egm[sim] = best_out["time_period"]
        iter_epoch_egm[sim] = best_out["iter_epoch"]
        oracle_egm[sim] = best_out["oracle"]

        print(f"Final selected EGM tau = {best_tau:.2e}")

        for mode_idx, whether_nonmono in enumerate([0, 1]):
            if whether_nonmono == 0:
                k = 2000
                k1 = 800
            else:
                k = 1000
                k1 = 400

            print(f"\nRunning APDB with whether_nonmono = {whether_nonmono} [{device}]")

            out = run_apdb_c_lr(
                problem, optval, x0, y0, args.epsilon, args.max_iter, whether_nonmono, device=device, gamma0=gamma_yx
            )
            rel_infeas_1[mode_idx][sim] = out["rel_infeas_err"]
            rel_subopt_1[mode_idx][sim] = out["rel_subopt_err"]
            time_period_1[mode_idx][sim] = out["time_period"]
            iter_epoch_1[mode_idx][sim] = out["iter_epoch"]
            oracle_1[mode_idx][sim] = out["oracle"]

            out = run_apdb_c_lr_ada_restart(
                problem, optval, x0, y0, args.epsilon, args.max_iter, whether_nonmono, device=device, gamma0=gamma_yx
            )
            rel_infeas_1_ada[mode_idx][sim] = out["rel_infeas_err"]
            rel_subopt_1_ada[mode_idx][sim] = out["rel_subopt_err"]
            time_period_1_ada[mode_idx][sim] = out["time_period"]
            iter_epoch_1_ada[mode_idx][sim] = out["iter_epoch"]
            oracle_1_ada[mode_idx][sim] = out["oracle"]
            gxi_time_1_ada[mode_idx][sim] = np.asarray([out.get("gxi_check_restart_time_total", np.nan)], dtype=np.float64)
            restart_iter_epoch_1_ada[mode_idx][sim] = np.asarray(out.get("restart_iter_epoch", []), dtype=np.float64)
            restart_oracle_1_ada[mode_idx][sim] = np.asarray(out.get("restart_oracle", []), dtype=np.float64)

            out = run_apdb_c_lr_restart(
                problem,
                optval,
                x0,
                y0,
                args.epsilon,
                args.max_iter,
                whether_nonmono,
                k1,
                device=device,
                gamma0=gamma_yx,
            )
            rel_infeas_r1[mode_idx][sim] = out["rel_infeas_err"]
            rel_subopt_r1[mode_idx][sim] = out["rel_subopt_err"]
            time_period_r1[mode_idx][sim] = out["time_period"]
            iter_epoch_r1[mode_idx][sim] = out["iter_epoch"]
            oracle_r1[mode_idx][sim] = out["oracle"]

            out = run_apdb_c_xy(
                problem,
                optval,
                x0,
                y0,
                args.epsilon,
                args.max_iter,
                whether_nonmono,
                device=device,
                gamma0=gamma_xy,
                ls_debug=args.xy_ls_debug,
            )
            rel_infeas_xy_1[mode_idx][sim] = out["rel_infeas_err"]
            rel_subopt_xy_1[mode_idx][sim] = out["rel_subopt_err"]
            time_period_xy_1[mode_idx][sim] = out["time_period"]
            iter_epoch_xy_1[mode_idx][sim] = out["iter_epoch"]
            oracle_xy_1[mode_idx][sim] = out["oracle"]

            out = run_apdb_c_xy_restart(
                problem,
                optval,
                x0,
                y0,
                args.epsilon,
                args.max_iter,
                whether_nonmono,
                k,
                device=device,
                gamma0=gamma_xy,
                ls_debug=args.xy_ls_debug,
            )
            rel_infeas_xy_r1[mode_idx][sim] = out["rel_infeas_err"]
            rel_subopt_xy_r1[mode_idx][sim] = out["rel_subopt_err"]
            time_period_xy_r1[mode_idx][sim] = out["time_period"]
            iter_epoch_xy_r1[mode_idx][sim] = out["iter_epoch"]
            oracle_xy_r1[mode_idx][sim] = out["oracle"]

            out = run_apdb_c_xy_ada_restart(
                problem,
                optval,
                x0,
                y0,
                args.epsilon,
                args.max_iter,
                whether_nonmono,
                device=device,
                gamma0=gamma_xy,
                ls_debug=args.xy_ls_debug,
            )
            rel_infeas_xy_ada1[mode_idx][sim] = out["rel_infeas_err"]
            rel_subopt_xy_ada1[mode_idx][sim] = out["rel_subopt_err"]
            time_period_xy_ada1[mode_idx][sim] = out["time_period"]
            iter_epoch_xy_ada1[mode_idx][sim] = out["iter_epoch"]
            oracle_xy_ada1[mode_idx][sim] = out["oracle"]
            gxi_time_xy_ada1[mode_idx][sim] = np.asarray([out.get("gxi_check_restart_time_total", np.nan)], dtype=np.float64)
            restart_iter_epoch_xy_ada1[mode_idx][sim] = np.asarray(out.get("restart_iter_epoch", []), dtype=np.float64)
            restart_oracle_xy_ada1[mode_idx][sim] = np.asarray(out.get("restart_oracle", []), dtype=np.float64)

    results = {
        "mosek_time": context["mosek_time"],
        "Total_time_Mosek": context["Total_time_Mosek"],
        "best_tau_egm": best_tau_egm,
        "rel_infeas_err_egm": rel_infeas_err_egm,
        "rel_subopt_egm": rel_subopt_egm,
        "time_period_egm": time_period_egm,
        "iter_epoch_egm": iter_epoch_egm,
        "oracle_egm": oracle_egm,
        "rel_infeas_1": rel_infeas_1,
        "rel_subopt_1": rel_subopt_1,
        "time_period_1": time_period_1,
        "iter_epoch_1": iter_epoch_1,
        "oracle_1": oracle_1,
        "rel_infeas_1_ada": rel_infeas_1_ada,
        "rel_subopt_1_ada": rel_subopt_1_ada,
        "time_period_1_ada": time_period_1_ada,
        "iter_epoch_1_ada": iter_epoch_1_ada,
        "oracle_1_ada": oracle_1_ada,
        "gxi_time_1_ada": gxi_time_1_ada,
        "restart_iter_epoch_1_ada": restart_iter_epoch_1_ada,
        "restart_oracle_1_ada": restart_oracle_1_ada,
        "rel_infeas_r1": rel_infeas_r1,
        "rel_subopt_r1": rel_subopt_r1,
        "time_period_r1": time_period_r1,
        "iter_epoch_r1": iter_epoch_r1,
        "oracle_r1": oracle_r1,
        "rel_infeas_xy_1": rel_infeas_xy_1,
        "rel_subopt_xy_1": rel_subopt_xy_1,
        "time_period_xy_1": time_period_xy_1,
        "iter_epoch_xy_1": iter_epoch_xy_1,
        "oracle_xy_1": oracle_xy_1,
        "rel_infeas_xy_r1": rel_infeas_xy_r1,
        "rel_subopt_xy_r1": rel_subopt_xy_r1,
        "time_period_xy_r1": time_period_xy_r1,
        "iter_epoch_xy_r1": iter_epoch_xy_r1,
        "oracle_xy_r1": oracle_xy_r1,
        "rel_infeas_xy_ada1": rel_infeas_xy_ada1,
        "rel_subopt_xy_ada1": rel_subopt_xy_ada1,
        "time_period_xy_ada1": time_period_xy_ada1,
        "iter_epoch_xy_ada1": iter_epoch_xy_ada1,
        "oracle_xy_ada1": oracle_xy_ada1,
        "gxi_time_xy_ada1": gxi_time_xy_ada1,
        "restart_iter_epoch_xy_ada1": restart_iter_epoch_xy_ada1,
        "restart_oracle_xy_ada1": restart_oracle_xy_ada1,
        "best_gamma_xy_sim": best_gamma_xy_sim,
        "best_gamma_yx_sim": best_gamma_yx_sim,
    }

    results_mat = outdir / f"results_apdb_both_modes_n{n_run}_m{m_run}_numsim{args.numsim}_{device}.mat"
    save_results_mat(results_mat, results)
    if save_legacy:
        save_results_mat(outdir / "results_apdb_both_modes.mat", results)

    rows = [
        {
            "Group": "Reference",
            "Algorithm": "MOSEK/CVX",
            "Time": float(np.mean(context["mosek_time"][: args.numsim])),
            "Iter": float("nan"),
            "SuboptRes": 0.0,
            "InfeasRes": 0.0,
            "GxiTime": float("nan"),
        },
        {
            "Group": "EGM",
            "Algorithm": "EGM",
            "Time": _final_mean(time_period_egm),
            "Iter": _final_mean(iter_epoch_egm),
            "SuboptRes": _final_mean(rel_subopt_egm),
            "InfeasRes": _final_mean(rel_infeas_err_egm),
            "GxiTime": float("nan"),
        },
    ]

    labels = [
        ("APDB-yx", time_period_1, iter_epoch_1, rel_subopt_1, rel_infeas_1, None),
        ("rAPDB-yx-ada", time_period_1_ada, iter_epoch_1_ada, rel_subopt_1_ada, rel_infeas_1_ada, gxi_time_1_ada),
        ("rAPDB-yx", time_period_r1, iter_epoch_r1, rel_subopt_r1, rel_infeas_r1, None),
        ("APDB-xy", time_period_xy_1, iter_epoch_xy_1, rel_subopt_xy_1, rel_infeas_xy_1, None),
        ("rAPDB-xy-ada", time_period_xy_ada1, iter_epoch_xy_ada1, rel_subopt_xy_ada1, rel_infeas_xy_ada1, gxi_time_xy_ada1),
        ("rAPDB-xy", time_period_xy_r1, iter_epoch_xy_r1, rel_subopt_xy_r1, rel_infeas_xy_r1, None),
    ]

    for mode_idx, group in [(0, "Monotone"), (1, "Nonmonotone")]:
        for name, t_s, i_s, sub_s, inf_s, gxi_s in labels:
            rows.append(
                {
                    "Group": group,
                    "Algorithm": name,
                    "Time": _final_mean(t_s[mode_idx]),
                    "Iter": _final_mean(i_s[mode_idx]),
                    "SuboptRes": _final_mean(sub_s[mode_idx]),
                    "InfeasRes": _final_mean(inf_s[mode_idx]),
                    "GxiTime": _final_mean(gxi_s[mode_idx]) if gxi_s is not None else float("nan"),
                }
            )

    _print_summary(rows, header_suffix=device)

    csv_path = outdir / f"results_summary_table_n{n_run}_m{m_run}_numsim{args.numsim}_{device}.csv"
    mat_path = outdir / f"results_summary_table_n{n_run}_m{m_run}_numsim{args.numsim}_{device}.mat"
    df = save_summary_table(csv_path, rows)
    save_summary_table_mat(mat_path, df)

    if save_legacy:
        save_summary_table(outdir / "results_summary_table.csv", rows)
        save_summary_table_mat(outdir / "results_summary_table.mat", df)

    fig_paths = plot_results(results, args.numsim, n_run, m_run, device, outdir, plot_style=args.plot_style)
    print(f"Saved MAT results: {results_mat}")
    print(f"Saved summary CSV: {csv_path}")
    for fig_path in fig_paths:
        print(f"Saved plot: {fig_path}")

    return {
        "device": device,
        "results": results,
        "summary": df,
        "plot_paths": fig_paths,
        "results_mat_path": results_mat,
        "summary_csv_path": csv_path,
        "summary_mat_path": mat_path,
    }


def _build_device_compare_table(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame | None) -> pd.DataFrame:
    key_cols = ["Group", "Algorithm"]
    metrics = ["Time", "Iter", "SuboptRes", "InfeasRes"]

    cpu = cpu_df[key_cols + metrics].copy().rename(columns={m: f"{m}_cpu" for m in metrics})
    if gpu_df is None:
        merged = cpu.copy()
        for m in metrics:
            merged[f"{m}_gpu"] = np.nan
            merged[f"{m}_abs_diff"] = np.nan
            merged[f"{m}_rel_diff"] = np.nan
        return merged

    gpu = gpu_df[key_cols + metrics].copy().rename(columns={m: f"{m}_gpu" for m in metrics})
    merged = cpu.merge(gpu, on=key_cols, how="outer")
    for m in metrics:
        a = merged[f"{m}_cpu"].to_numpy(dtype=np.float64)
        b = merged[f"{m}_gpu"].to_numpy(dtype=np.float64)
        abs_diff = np.abs(a - b)
        den = np.abs(a) + 1e-16
        rel_diff = abs_diff / den
        rel_diff[np.isnan(a) | np.isnan(b)] = np.nan
        merged[f"{m}_abs_diff"] = abs_diff
        merged[f"{m}_rel_diff"] = rel_diff
    return merged


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    context = _prepare_run_context(args)

    if args.run_all:
        print("\n=== Run mode: CPU mandatory, GPU optional ===")
        cpu_art = _run_for_device(args=args, context=context, device="cpu", outdir=outdir, save_legacy=True)

        gpu_art: dict[str, Any] | None = None
        gpu_skipped_reason: str | None = None
        if torch.cuda.is_available():
            gpu_art = _run_for_device(args=args, context=context, device="cuda", outdir=outdir, save_legacy=False)
        else:
            gpu_skipped_reason = "CUDA not available"
            print(f"Skip GPU run: {gpu_skipped_reason}")

        compare_df = _build_device_compare_table(cpu_art["summary"], None if gpu_art is None else gpu_art["summary"])

        compare_csv = outdir / "results_summary_table_compare_cpu_gpu.csv"
        compare_mat = outdir / "results_summary_table_compare_cpu_gpu.mat"
        compare_df.to_csv(compare_csv, index=False)
        save_summary_table_mat(compare_mat, compare_df)

        manifest = {
            "run_all": True,
            "gpu_available": bool(torch.cuda.is_available()),
            "devices_ran": ["cpu"] + (["cuda"] if gpu_art is not None else []),
            "gpu_skipped_reason": gpu_skipped_reason,
            "numsim": int(args.numsim),
            "n": int(context["n_run"]),
            "m": int(context["m_run"]),
            "seed": int(args.seed),
            "max_iter": int(args.max_iter),
            "epsilon": float(args.epsilon),
            "problem_mat": args.problem_mat,
            "gamma_tune": {
                "grid": _to_jsonable(context.get("gamma_grid_family", np.asarray([1.0, 3.0, 6.0, 9.0], dtype=np.float64))),
                "tune_iter": int(context.get("tune_iter_family", min(2000, args.max_iter))),
                "best_gamma_xy_sim": _to_jsonable(context.get("best_gamma_xy_sim", [])),
                "best_gamma_yx_sim": _to_jsonable(context.get("best_gamma_yx_sim", [])),
                "xy_details": _to_jsonable(context.get("tune_info_xy_sim", [])),
                "yx_details": _to_jsonable(context.get("tune_info_yx_sim", [])),
            },
        }
        manifest_path = outdir / "run_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Saved CPU/GPU compare CSV: {compare_csv}")
        print(f"Saved CPU/GPU compare MAT: {compare_mat}")
        print(f"Saved run manifest: {manifest_path}")

        return {
            "cpu": cpu_art,
            "gpu": gpu_art,
            "overview": {
                "compare_table": compare_df,
                "compare_csv_path": compare_csv,
                "compare_mat_path": compare_mat,
                "manifest_path": manifest_path,
            },
        }

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")

    single = _run_for_device(args=args, context=context, device=args.device, outdir=outdir, save_legacy=True)
    return {
        "results": single["results"],
        "summary": single["summary"],
        "plot_paths": single["plot_paths"],
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
