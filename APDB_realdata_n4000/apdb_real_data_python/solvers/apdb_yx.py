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
    validate_apdb_strong_convex_settings,
)


def run_apdb_yx(
    input_data: dict,
    *,
    stop_criteria: float,
    max_iter: int,
    non_mono: int,
    device: str = "cpu",
    progress: dict | None = None,
    mu: float = 2.0,
):
    inp = to_torch_input(input_data, device=device)
    mu_f = float(mu)
    validate_apdb_strong_convex_settings(inp, mu=mu_f)
    params = init_shared_params(inp, strong_convexity=mu_f)

    x = inp.x0.clone()
    y = inp.y0.clone()
    # Intentional fixed-anchor momentum: the dual reference remains at x0.
    x_anchor = inp.x0.clone()

    rel_err: list[float] = []
    time_curve: list[float] = []
    iter_curve: list[float] = []

    tau = params.tau
    gamma = params.gamma
    sigma_old = params.sigma_old
    tau_old = params.tau_old
    alpha_old = params.alpha_old
    elapsed = 0.0

    norm_optsol = 1.0 + float(torch.norm(inp.optsol).item())
    c_over_trace = params.c / inp.trace_kernels
    grad3_anchor, _ = inner_prod_matvec_torch(inp.g_tr_class, x_anchor, return_matvec=False)

    iter_idx = 0
    mode_name = "mono" if int(non_mono) == 0 else "nonmono"
    progress_reporter = make_progress_reporter(
        name=f"APDB-yx({mode_name}, mu={mu_f:g})",
        max_iter=max_iter,
        progress=progress,
    )
    while iter_idx < max_iter and float(torch.norm(x - inp.optsol).item()) / norm_optsol > stop_criteria:
        iter_idx += 1
        t0 = time.perf_counter()

        while True:
            sigma = gamma * tau
            theta = sigma_old / sigma
            alpha = params.c_alpha / sigma

            grad2, grad1 = inner_prod_matvec_torch(inp.g_tr_class, x, return_matvec=True)

            grad_y = (1.0 + theta) * (c_over_trace * grad2) - theta * (c_over_trace * grad3_anchor)
            y_tilde = projsplx_torch(y + sigma * grad_y)

            grad_x_linearized = -2.0 * torch.ones_like(x) + 2.0 * (grad1 @ (c_over_trace * y_tilde))
            x_tilde = proj_pos_plane_torch(
                (x - tau * grad_x_linearized) / (1.0 + mu_f * tau),
                inp.y_train,
            )

            x_diff = x_tilde - x
            y_diff = y_tilde - y

            grad2_diff, _ = inner_prod_matvec_torch(inp.g_tr_class, x_diff, return_matvec=True)
            e_val = torch.sum((c_over_trace * y_tilde) * grad2_diff)

            grad2_new, _ = inner_prod_matvec_torch(inp.g_tr_class, x_tilde, return_matvec=True)
            aux = c_over_trace * (grad2_new - grad2)

            e_val = (
                e_val
                - (1.0 - params.delta) * torch.dot(x_diff, x_diff) / (2.0 * tau)
                + torch.dot(aux, aux) / (2.0 * alpha)
                - ((1.0 - params.delta) / sigma - theta * alpha_old) * torch.dot(y_diff, y_diff) / 2.0
            )
            if float(e_val.item()) <= 0.0:
                break
            tau = tau * params.eta

        gamma_old = gamma
        gamma = gamma * (1.0 + mu_f * tau)
        if int(non_mono) == 0:
            tau_new = tau * ((gamma_old / gamma) ** 0.5)
        else:
            tau_new = min(tau * ((gamma_old / gamma * (1.0 + tau / tau_old)) ** 0.5), 10.0)

        sigma_old = sigma
        tau_old = tau
        tau = tau_new
        alpha_old = alpha

        x = x_tilde
        y = y_tilde

        elapsed += time.perf_counter() - t0
        append_metrics(rel_err, time_curve, iter_curve, x=x, optsol=inp.optsol, elapsed=elapsed, iter_idx=iter_idx)
        progress_reporter.update(iter_idx=iter_idx, rel_err=rel_err[-1], elapsed=elapsed)

    append_metrics(rel_err, time_curve, iter_curve, x=x, optsol=inp.optsol, elapsed=elapsed, iter_idx=iter_idx)
    progress_reporter.update(iter_idx=iter_idx, rel_err=rel_err[-1], elapsed=elapsed)
    progress_reporter.close()
    return finish_result(rel_err, time_curve, iter_curve)
