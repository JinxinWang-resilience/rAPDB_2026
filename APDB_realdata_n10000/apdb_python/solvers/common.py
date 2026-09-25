from __future__ import annotations

import numpy as np
import torch

from apdb_python.core import TorchProblem, constraint_values, primal_obj, scalar


def compute_g_y(x: torch.Tensor, prob: TorchProblem) -> torch.Tensor:
    vals = []
    for j, q_j in enumerate(prob.Q):
        vals.append(0.5 * torch.dot(x, q_j @ x) + torch.dot(prob.d[j], x) - prob.c[j])
    return torch.stack(vals)


def compute_g_x(x: torch.Tensor, y: torch.Tensor, prob: TorchProblem) -> torch.Tensor:
    g_x = prob.A @ x + prob.b
    for j, q_j in enumerate(prob.Q):
        g_x = g_x + (q_j @ x + prob.d[j]) * y[j]
    return g_x


def evaluate_metrics(x: torch.Tensor, prob: TorchProblem, optval: float, optval_rel: float) -> tuple[float, float]:
    subopt = scalar(primal_obj(x, prob))
    infeas = scalar(torch.clamp(constraint_values(x, prob), min=0.0).mean())
    rel_sub = abs(subopt - optval) / abs(optval_rel)
    return rel_sub, infeas


def finalize_history(history: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float64) for k, v in history.items()}
