from __future__ import annotations

import time

import numpy as np
import torch

from apdb_python.core import project_box, to_numpy, to_torch_problem
from apdb_python.solvers.common import compute_g_x, compute_g_y, evaluate_metrics, finalize_history


def run_apdb_c_xy_restart(
    input_data: dict,
    optval: float,
    x0: np.ndarray,
    y0: np.ndarray,
    stop_criteria: float,
    max_iter: int,
    non_mono: int,
    k_restart: int,
    device: str = "cpu",
    gamma0: float = 1.0,
    gamma_0: float = 1.0,
    ls_debug: bool = False,
) -> dict:
    prob = to_torch_problem(input_data, device=device)
    n_con, m_vars = prob.d.shape

    eta = 0.7
    tau = 3e-1
    gamma = float(gamma0)
    gamma_0 = float(gamma_0)
    c_alpha = 0.25
    c_beta = 0.3
    delta = 0.4
    sigma_old = gamma * tau
    alpha_old = c_alpha / tau
    beta_old = gamma_0 * c_beta / sigma_old
    tau_old = tau
    mu = 0.0

    x = torch.as_tensor(x0, dtype=torch.float64, device=device).reshape(-1)
    y = torch.as_tensor(y0, dtype=torch.float64, device=device).reshape(-1)

    x_cum = torch.zeros(m_vars, dtype=torch.float64, device=device)
    y_cum = torch.zeros(n_con, dtype=torch.float64, device=device)
    sigma_0 = sigma_old
    tk = 0.0

    g_x = compute_g_x(x, y, prob)
    orc = 1

    iter_idx = 0
    elapsed = 0.0
    inner = np.zeros(max_iter, dtype=np.float64)
    optval_rel = abs(float(optval)) + 1.0

    history = {k: [] for k in ["rel_infeas_err", "rel_subopt_err", "time_period", "iter_epoch", "oracle"]}

    while iter_idx < max_iter:
        t0 = time.perf_counter()
        iter_idx += 1
        g_x_old = g_x

        while True:
            sigma = gamma * tau
            theta = sigma_old / sigma
            alpha = c_alpha / tau
            beta = gamma_0 * c_beta / sigma

            g_x = compute_g_x(x, y, prob)
            x_tild = project_box(x - tau * ((1.0 + theta) * g_x - theta * g_x_old))
            xdiff = x_tild - x

            g_y = compute_g_y(x_tild, prob)
            orc += 1
            y_tild = torch.clamp(y + sigma * g_y, min=0.0)
            ydiff = y_tild - y

            aux1 = torch.zeros(m_vars, dtype=torch.float64, device=device)
            aux2 = prob.A @ xdiff
            for j, q_j in enumerate(prob.Q):
                aux1 = aux1 + (q_j @ x_tild + prob.d[j]) * ydiff[j]
                aux2 = aux2 + (q_j @ xdiff) * y[j]

            e_val = (
                0.5 * aux1.norm().pow(2) / alpha
                + 0.5 * aux2.norm().pow(2) / beta
                + (delta - 1.0) * ydiff.norm().pow(2) / (2.0 * sigma)
                - ((1.0 - delta) / tau - theta * (alpha_old + beta_old)) * xdiff.norm().pow(2) / 2.0
            )

            if float(e_val) <= 0.0:
                break
            inner[iter_idx - 1] += 1
            tau = tau * eta
            if ls_debug:
                print(
                    f"[rAPDB-xy][mode={int(non_mono)}][iter={iter_idx}] "
                    f"ls_step={int(inner[iter_idx - 1])} tau={tau:.6e}"
                )

        gamma_old = gamma
        gamma = gamma * (1.0 + mu * tau)
        if non_mono == 0:
            tau_new = tau * np.sqrt(gamma_old / gamma)
        else:
            tau_new = min(tau * np.sqrt(gamma_old / gamma * (1.0 + tau / tau_old)), 1.0)

        sigma_old = sigma
        tau_old = tau
        alpha_old = alpha
        beta_old = beta
        tau = tau_new

        x = x_tild
        y = y_tild

        w = sigma / sigma_0
        x_cum = x_cum + w * x
        y_cum = y_cum + w * y
        tk += w

        if iter_idx % k_restart == 0:
            x = x
            y = y
            x_cum = torch.zeros_like(x_cum)
            y_cum = torch.zeros_like(y_cum)
            tk = 0.0

            tau = 3e-1
            sigma_old = gamma * tau
            alpha_old = c_alpha / tau
            beta_old = gamma_0 * c_beta / sigma_old
            tau_old = tau
            g_x = compute_g_x(x, y, prob)

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
    return out
