from __future__ import annotations

import numpy as np
import pandas as pd

from apdb_python.solvers.egm import run_egm


def tune_tau_egm(
    input_data: dict,
    optval: float,
    x0: np.ndarray,
    y0: np.ndarray,
    epsilon: float,
    max_iter: int,
    opts: dict,
    device: str = "cpu",
) -> tuple[float, dict, pd.DataFrame]:
    tau_grid = np.asarray(opts.get("tau_grid", np.logspace(-5, -2, 10)), dtype=np.float64)
    tune_iter = int(opts.get("tune_iter", min(1000, max_iter)))

    score_list = np.full(tau_grid.shape, np.inf, dtype=np.float64)
    final_infeas = np.full(tau_grid.shape, np.inf, dtype=np.float64)
    final_subopt = np.full(tau_grid.shape, np.inf, dtype=np.float64)

    print("\n========== Tuning EGM tau ==========")
    for idx, tau in enumerate(tau_grid):
        opts_tmp = dict(opts)
        opts_tmp["tau"] = float(tau)
        try:
            out = run_egm(
                input_data=input_data,
                optval=optval,
                x0=x0,
                y0=y0,
                stop_criteria=epsilon,
                max_iter=tune_iter,
                opts=opts_tmp,
                device=device,
            )
            if out["rel_infeas_err"].size and out["rel_subopt_err"].size:
                final_infeas[idx] = out["rel_infeas_err"][-1]
                final_subopt[idx] = out["rel_subopt_err"][-1]
                score_list[idx] = max(final_infeas[idx], final_subopt[idx])
        except Exception as exc:  # noqa: BLE001
            print(f"tau = {tau:.2e} failed: {exc}")

        print(
            f"tau = {tau:.2e}, score = {score_list[idx]:.3e}, "
            f"infeas = {final_infeas[idx]:.3e}, subopt = {final_subopt[idx]:.3e}"
        )

    best_idx = int(np.argmin(score_list))
    best_tau = float(tau_grid[best_idx])
    print(f"Best tau selected: {best_tau:.2e}")
    print("====================================\n")

    table = pd.DataFrame(
        {
            "tau": tau_grid,
            "score": score_list,
            "final_infeas": final_infeas,
            "final_subopt": final_subopt,
        }
    )

    best_opts = dict(opts)
    best_opts["tau"] = best_tau
    best_out = run_egm(
        input_data=input_data,
        optval=optval,
        x0=x0,
        y0=y0,
        stop_criteria=epsilon,
        max_iter=max_iter,
        opts=best_opts,
        device=device,
    )
    best_out["opts"] = best_opts

    return best_tau, best_out, table
