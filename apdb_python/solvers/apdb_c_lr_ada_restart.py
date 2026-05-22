from __future__ import annotations

import time

import numpy as np
import torch

from apdb_python.core import project_box, smoothed_duality_gap, to_numpy, to_numpy_problem, to_torch_problem
from apdb_python.solvers.common import compute_g_x, compute_g_y, evaluate_metrics, finalize_history


def run_apdb_c_lr_ada_restart(
    input_data: dict,
    optval: float,
    x0: np.ndarray,
    y0: np.ndarray,
    stop_criteria: float,
    max_iter: int,
    non_mono: int,
    device: str = "cpu",
    gamma0: float = 1.0,
) -> dict:
    prob = to_torch_problem(input_data, device=device)
    np_prob = to_numpy_problem(input_data)
    n_con, m_vars = prob.d.shape

    eta = 0.7
    tau = 3e-1
    gamma = float(gamma0)
    c_alpha = 0.4
    delta = 0.5
    sigma_old = gamma * tau
    tau_old = tau
    alpha_old = c_alpha / sigma_old
    mu = 0.0
    xi = 0.04
    gxi_check_period = 500 if int(non_mono) == 0 else 200

    x = torch.as_tensor(x0, dtype=torch.float64, device=device).reshape(-1)
    y = torch.as_tensor(y0, dtype=torch.float64, device=device).reshape(-1)

    x_cum = torch.zeros(m_vars, dtype=torch.float64, device=device)
    y_cum = torch.zeros(n_con, dtype=torch.float64, device=device)
    sigma_0 = sigma_old
    tk = 0.0

    iter_idx = 0
    orc = 0
    elapsed = 0.0
    gxi_elapsed = 0.0
    inner = np.zeros(max_iter, dtype=np.float64)
    g_y = compute_g_y(x, prob)
    optval_rel = abs(float(optval)) + 1.0
    g_xi_old = np.inf

    history = {k: [] for k in ["rel_infeas_err", "rel_subopt_err", "time_period", "iter_epoch", "oracle"]}

    while iter_idx < max_iter:
        t0 = time.perf_counter()
        iter_idx += 1

        g_y_old = g_y
        g_y = compute_g_y(x, prob)

        while True:
            sigma = gamma * tau
            theta = sigma_old / sigma
            alpha = c_alpha / sigma

            y_tild = torch.clamp(y + sigma * ((1.0 + theta) * g_y - theta * g_y_old), min=0.0)
            g_x = compute_g_x(x, y_tild, prob)
            orc += 1
            x_tild = project_box(x - tau * g_x)

            xdiff = x_tild - x
            ydiff = y_tild - y

            e_val = torch.tensor(0.0, dtype=torch.float64, device=device)
            aux = torch.tensor(0.0, dtype=torch.float64, device=device)
            for j, q_j in enumerate(prob.Q):
                e_val = e_val + (xdiff @ (q_j @ xdiff)) * y_tild[j]
                term = 0.5 * (x_tild @ (q_j @ x_tild)) - 0.5 * (x @ (q_j @ x)) + prob.d[j] @ xdiff
                aux = aux + term.pow(2)

            e_val = (
                0.5 * e_val
                + 0.5 * (xdiff @ (prob.A @ xdiff))
                - (1.0 - delta) * xdiff.norm().pow(2) / (2.0 * tau)
                + aux / (2.0 * alpha)
                - ((1.0 - delta) / sigma - theta * alpha_old) * ydiff.norm().pow(2) / 2.0
            )

            if float(e_val) <= 0.0:
                break
            inner[iter_idx - 1] += 1
            tau = tau * eta

        if iter_idx == 50:
            tg0 = time.perf_counter()
            x_curr = to_numpy(x)
            y_curr = to_numpy(y)
            g_xi_old = smoothed_duality_gap(np_prob, x_curr, y_curr, xi)
            gxi_elapsed += time.perf_counter() - tg0

        gamma_old = gamma
        gamma = gamma * (1.0 + mu * tau)
        if non_mono == 0:
            tau_new = tau * np.sqrt(gamma_old / gamma)
        else:
            tau_new = min(tau * np.sqrt(gamma_old / gamma * (1.0 + tau / tau_old)), 1.0)

        sigma_old = sigma
        tau_old = tau
        tau = tau_new
        alpha_old = alpha

        x = x_tild
        y = y_tild

        w = sigma / sigma_0
        x_cum = x_cum + w * x
        y_cum = y_cum + w * y
        tk += w

        if iter_idx % gxi_check_period == 0 and np.isfinite(g_xi_old):
            tg0 = time.perf_counter()
            x_curr = to_numpy(x)
            y_curr = to_numpy(y)
            g_xi = smoothed_duality_gap(np_prob, x_curr, y_curr, xi)
            if g_xi < g_xi_old * 0.5:
                x_cum = torch.zeros_like(x_cum)
                y_cum = torch.zeros_like(y_cum)
                tk = 0.0
                x = torch.as_tensor(x_curr, dtype=torch.float64, device=device)
                y = torch.as_tensor(y_curr, dtype=torch.float64, device=device)
                g_xi_old = g_xi

                tau = 3e-1
                sigma_old = gamma * tau
                alpha_old = c_alpha / sigma_old
                tau_old = tau
                _ = compute_g_x(x, y, prob)
            gxi_elapsed += time.perf_counter() - tg0

        elapsed += time.perf_counter() - t0

        rel_sub, infeas = evaluate_metrics(x, prob, optval, optval_rel)
        history["rel_subopt_err"].append(rel_sub)
        history["rel_infeas_err"].append(infeas)
        history["time_period"].append(elapsed)
        history["oracle"].append(float(orc))
        history["iter_epoch"].append(float(iter_idx))

        if max(rel_sub, infeas) < stop_criteria:
            break

    out = finalize_history(history)
    out["x_final"] = to_numpy(x)
    out["y_final"] = to_numpy(y)
    out["inner_ls"] = inner[:iter_idx]
    out["gxi_check_restart_time_total"] = float(gxi_elapsed)
    return out
