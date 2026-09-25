from __future__ import annotations

import time

import torch

from apdb_real_data_python.operators import inner_prod_matvec_torch, proj_pos_plane_torch, projsplx_torch
from apdb_real_data_python.solvers.common import (
    append_metrics,
    finish_result,
    init_shared_params,
    make_progress_reporter,
    to_torch_input,
)

DEFAULT_EGM_TAU_SCALE = 0.1


def run_egm(
    input_data: dict,
    *,
    stop_criteria: float,
    max_iter: int,
    device: str = "cpu",
    progress: dict | None = None,
    tau_scale: float | None = None,
):
    inp = to_torch_input(input_data, device=device)
    params = init_shared_params(inp)

    effective_scale = DEFAULT_EGM_TAU_SCALE if tau_scale is None else float(tau_scale)
    if effective_scale <= 0.0:
        raise ValueError("tau_scale must be positive.")
    tau = params.tau * effective_scale
    c_over_trace = params.c / inp.trace_kernels

    x = inp.x0.clone()
    y = inp.y0.clone()

    rel_err: list[float] = []
    time_curve: list[float] = []
    iter_curve: list[float] = []
    elapsed = 0.0
    norm_optsol = 1.0 + float(torch.norm(inp.optsol).item())
    progress_reporter = make_progress_reporter(name="EGM", max_iter=max_iter, progress=progress)

    iter_idx = 0
    while iter_idx < max_iter and float(torch.norm(x - inp.optsol).item()) / norm_optsol > stop_criteria:
        iter_idx += 1
        t0 = time.perf_counter()

        grad2, grad1 = inner_prod_matvec_torch(inp.g_tr_class, x, return_matvec=True)
        grad_x = -2.0 * torch.ones_like(x) + 2.0 * (grad1 @ (c_over_trace * y)) + 2.0 * inp.reg_param * x
        x_mid = proj_pos_plane_torch(x - tau * grad_x, inp.y_train)

        grad_y = c_over_trace * grad2
        y_mid = projsplx_torch(y + tau * grad_y)

        grad2_mid, grad1_mid = inner_prod_matvec_torch(inp.g_tr_class, x_mid, return_matvec=True)
        grad_x2 = -2.0 * torch.ones_like(x) + 2.0 * (grad1_mid @ (c_over_trace * y_mid)) + 2.0 * inp.reg_param * x_mid
        x_new = proj_pos_plane_torch(x - tau * grad_x2, inp.y_train)

        grad_y2 = c_over_trace * grad2_mid
        y_new = projsplx_torch(y + tau * grad_y2)

        x = x_new
        y = y_new
        elapsed += time.perf_counter() - t0
        append_metrics(rel_err, time_curve, iter_curve, x=x, optsol=inp.optsol, elapsed=elapsed, iter_idx=iter_idx)
        progress_reporter.update(iter_idx=iter_idx, rel_err=rel_err[-1], elapsed=elapsed)

    append_metrics(rel_err, time_curve, iter_curve, x=x, optsol=inp.optsol, elapsed=elapsed, iter_idx=iter_idx)
    progress_reporter.update(iter_idx=iter_idx, rel_err=rel_err[-1], elapsed=elapsed)
    progress_reporter.close()
    return finish_result(rel_err, time_curve, iter_curve)
