from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from apdb_python.core import to_numpy_problem


def solve_qcqp_reference(
    input_data: dict,
    box: float = 10.0,
    strict_mosek: bool = False,
    high_accuracy: bool = True,
    mosek_tol: float = 1e-10,
) -> tuple[np.ndarray, float, float]:
    prob = to_numpy_problem(input_data)
    m = prob.b.shape[0]

    x = cp.Variable(m)
    obj = 0.5 * cp.quad_form(x, cp.psd_wrap(prob.A)) + prob.b @ x
    constraints = []
    for j, q_j in enumerate(prob.Q):
        constraints.append(
            0.5 * cp.quad_form(x, cp.psd_wrap(q_j)) + prob.d[j] @ x - prob.c[j] <= 0
        )
    constraints.extend([x <= box, x >= -box])

    problem = cp.Problem(cp.Minimize(obj), constraints)
    t0 = time.perf_counter()
    try:
        mosek_kwargs: dict = {"solver": cp.MOSEK, "verbose": False}
        if high_accuracy:
            mosek_kwargs["mosek_params"] = {
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS": float(mosek_tol),
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": float(mosek_tol),
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": float(mosek_tol),
            }
        problem.solve(**mosek_kwargs)
    except Exception as exc:
        if strict_mosek:
            raise RuntimeError("MOSEK solve failed and strict_mosek=True.") from exc
        # Fallback keeps pipeline usable when MOSEK is unavailable.
        problem.solve(solver=cp.SCS, verbose=False, eps=1e-6)
    elapsed = time.perf_counter() - t0

    if x.value is None or problem.value is None:
        raise RuntimeError("Reference solver failed to produce a solution.")

    return np.asarray(x.value).reshape(-1), float(problem.value), float(elapsed)
