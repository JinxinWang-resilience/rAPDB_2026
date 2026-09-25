from __future__ import annotations

import time
from typing import Any

import cvxpy as cp
import numpy as np


def mosek_high_accuracy_params(tol: float = 1e-9) -> dict[str, float]:
    return {
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": float(tol),
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": float(tol),
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": float(tol),
    }


def solve_reference_mosek(
    problem: dict[str, Any],
    *,
    strict_mosek: bool = True,
    mosek_tol: float = 1e-9,
) -> tuple[np.ndarray, float, float]:
    g_tr_class = [np.asarray(g, dtype=np.float64) for g in problem["G_tr_class"]]
    y_train = np.asarray(problem["Ytrain"], dtype=np.float64).reshape(-1)
    trace_kernels = np.asarray(problem["trace_kernels"], dtype=np.float64).reshape(-1)
    reg_param = float(problem["reg_param"])

    m = y_train.shape[0]
    num_kern = trace_kernels.shape[0]
    c = float(np.sum(trace_kernels))

    alpha = cp.Variable(m)
    t = cp.Variable()

    quad_vals = [cp.quad_form(alpha, cp.psd_wrap(g)) / trace_kernels[i] for i, g in enumerate(g_tr_class)]
    constraints = [
        t * np.ones(num_kern) >= cp.hstack(quad_vals),
        y_train @ alpha == 0,
        alpha >= 0,
    ]

    objective = cp.Minimize(-2.0 * cp.sum(alpha) + reg_param * cp.sum_squares(alpha) + c * t)
    cvx_prob = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    try:
        cvx_prob.solve(
            solver=cp.MOSEK,
            verbose=False,
            mosek_params=mosek_high_accuracy_params(mosek_tol),
        )
    except Exception as exc:
        if strict_mosek:
            raise RuntimeError("MOSEK solve failed and strict_mosek=True.") from exc
        cvx_prob.solve(solver=cp.SCS, verbose=False, eps=1e-6)

    elapsed = time.perf_counter() - t0

    if alpha.value is None or cvx_prob.value is None:
        raise RuntimeError("Reference solve failed to produce a solution.")

    return np.asarray(alpha.value, dtype=np.float64).reshape(-1), float(cvx_prob.value), float(elapsed)
