from __future__ import annotations

import time

import numpy as np
import torch

from apdb_python.core import (
    infeasibility_mean,
    primal_obj,
    project_box,
    saddle_operator,
    scalar,
    to_numpy,
    to_torch_problem,
)


def run_egm(
    input_data: dict,
    optval: float,
    x0: np.ndarray,
    y0: np.ndarray,
    stop_criteria: float,
    max_iter: int,
    opts: dict,
    device: str = "cpu",
) -> dict:
    prob = to_torch_problem(input_data, device=device)

    x = torch.as_tensor(x0, dtype=torch.float64, device=device).reshape(-1)
    y = torch.as_tensor(y0, dtype=torch.float64, device=device).reshape(-1)
    x = project_box(x)
    y = torch.clamp(y, min=0.0)

    tau = float(opts.get("tau", 1e-4))
    epoch = int(opts.get("epoch", 1))

    rel_subopt_err: list[float] = []
    rel_infeas_err: list[float] = []
    time_period: list[float] = []
    iter_epoch: list[float] = []
    oracle: list[float] = []

    iter_idx = 0
    orc = 0
    elapsed = 0.0
    optval_rel = abs(float(optval)) + 1.0

    while iter_idx < max_iter:
        t0 = time.perf_counter()
        iter_idx += 1

        grad_x, g_val = saddle_operator(x, y, prob)
        orc += 1

        x_bar = project_box(x - tau * grad_x)
        y_bar = torch.clamp(y + tau * g_val, min=0.0)

        grad_x_bar, g_val_bar = saddle_operator(x_bar, y_bar, prob)
        orc += 1

        x = project_box(x - tau * grad_x_bar)
        y = torch.clamp(y + tau * g_val_bar, min=0.0)

        elapsed += time.perf_counter() - t0

        if iter_idx % epoch == 0:
            subopt = scalar(primal_obj(x, prob))
            infeas = scalar(infeasibility_mean(x, prob))

            rel_sub = abs(subopt - optval) / optval_rel
            rel_subopt_err.append(rel_sub)
            rel_infeas_err.append(infeas)
            time_period.append(elapsed)
            oracle.append(float(orc))
            iter_epoch.append(float(iter_idx))

            if max(rel_sub, infeas) < stop_criteria:
                break

    return {
        "rel_infeas_err": np.asarray(rel_infeas_err, dtype=np.float64),
        "rel_subopt_err": np.asarray(rel_subopt_err, dtype=np.float64),
        "time_period": np.asarray(time_period, dtype=np.float64),
        "iter_epoch": np.asarray(iter_epoch, dtype=np.float64),
        "oracle": np.asarray(oracle, dtype=np.float64),
        "x_final": to_numpy(x),
        "y_final": to_numpy(y),
    }
