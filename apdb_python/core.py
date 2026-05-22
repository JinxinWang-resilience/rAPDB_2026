from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class TorchProblem:
    A: torch.Tensor
    Q: list[torch.Tensor]
    b: torch.Tensor
    d: torch.Tensor
    c: torch.Tensor
    sc: float


@dataclass
class NumpyProblem:
    A: np.ndarray
    Q: list[np.ndarray]
    b: np.ndarray
    d: np.ndarray
    c: np.ndarray
    sc: float


def to_numpy_problem(input_data: dict[str, Any]) -> NumpyProblem:
    return NumpyProblem(
        A=np.asarray(input_data["A"], dtype=np.float64),
        Q=[np.asarray(q, dtype=np.float64) for q in input_data["Q"]],
        b=np.asarray(input_data["b"], dtype=np.float64).reshape(-1),
        d=np.asarray(input_data["d"], dtype=np.float64),
        c=np.asarray(input_data["c"], dtype=np.float64).reshape(-1),
        sc=float(input_data.get("sc", 0.0)),
    )


def to_torch_problem(input_data: dict[str, Any], device: str = "cpu") -> TorchProblem:
    nprob = to_numpy_problem(input_data)
    dtype = torch.float64
    return TorchProblem(
        A=torch.as_tensor(nprob.A, dtype=dtype, device=device),
        Q=[torch.as_tensor(q, dtype=dtype, device=device) for q in nprob.Q],
        b=torch.as_tensor(nprob.b, dtype=dtype, device=device),
        d=torch.as_tensor(nprob.d, dtype=dtype, device=device),
        c=torch.as_tensor(nprob.c, dtype=dtype, device=device),
        sc=nprob.sc,
    )


def project_box(x: torch.Tensor, lb: float = -10.0, ub: float = 10.0) -> torch.Tensor:
    return torch.clamp(x, min=lb, max=ub)


def constraint_values(x: torch.Tensor, prob: TorchProblem) -> torch.Tensor:
    vals = []
    for j, q_j in enumerate(prob.Q):
        vals.append(0.5 * torch.dot(x, q_j @ x) + torch.dot(prob.d[j], x) - prob.c[j])
    return torch.stack(vals)


def primal_obj(x: torch.Tensor, prob: TorchProblem) -> torch.Tensor:
    return 0.5 * torch.dot(x, prob.A @ x) + torch.dot(prob.b, x)


def infeasibility_mean(x: torch.Tensor, prob: TorchProblem) -> torch.Tensor:
    g = constraint_values(x, prob)
    return torch.clamp(g, min=0).mean()


def saddle_operator(x: torch.Tensor, y: torch.Tensor, prob: TorchProblem) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = prob.A @ x + prob.b
    g_val = []
    for j, q_j in enumerate(prob.Q):
        qx = q_j @ x
        g_j = 0.5 * torch.dot(x, qx) + torch.dot(prob.d[j], x) - prob.c[j]
        grad_x = grad_x + y[j] * (qx + prob.d[j])
        g_val.append(g_j)
    return grad_x, torch.stack(g_val)


def smooth_gap_qp_solve(
    np_prob: NumpyProblem,
    x_curr: np.ndarray,
    y_curr: np.ndarray,
    xi: float,
    lb: float = -10.0,
    ub: float = 10.0,
) -> tuple[np.ndarray, float]:
    try:
        import osqp
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("OSQP is required for smoothed-gap QP solve") from exc

    m = x_curr.shape[0]
    qp_Q = np_prob.A + 2.0 * xi * np.eye(m)
    for j, q_j in enumerate(np_prob.Q):
        qp_Q = qp_Q + y_curr[j] * q_j
    qp_Q = 0.5 * (qp_Q + qp_Q.T)

    qp_q = np_prob.b - 2.0 * xi * x_curr
    for j in range(len(np_prob.c)):
        qp_q = qp_q + y_curr[j] * np_prob.d[j]

    P = sp.csc_matrix(qp_Q)
    q = np.asarray(qp_q, dtype=np.float64).reshape(-1)
    A = sp.eye(m, dtype=np.float64, format="csc")
    l = np.full(m, float(lb), dtype=np.float64)
    u = np.full(m, float(ub), dtype=np.float64)

    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, eps_abs=1e-9, eps_rel=1e-9, warm_start=True)
    res = solver.solve()
    solved = {osqp.constant("OSQP_SOLVED"), osqp.constant("OSQP_SOLVED_INACCURATE")}
    if res.x is None or int(res.info.status_val) not in solved:
        raise RuntimeError("smoothed-gap subproblem failed")
    x_star = np.asarray(res.x, dtype=np.float64).reshape(-1)
    fval = 0.5 * float(x_star @ (qp_Q @ x_star)) + float(q @ x_star)
    return x_star, float(fval)


def smoothed_duality_gap(np_prob: NumpyProblem, x_curr: np.ndarray, y_curr: np.ndarray, xi: float) -> float:
    _, fval = smooth_gap_qp_solve(np_prob, x_curr, y_curr, xi)

    g_xpart = fval + xi * float(np.dot(x_curr, x_curr)) - float(np.dot(y_curr, np_prob.c))
    g_xpart = -g_xpart

    g_x = np.zeros_like(np_prob.c)
    for j, q_j in enumerate(np_prob.Q):
        g_x[j] = 0.5 * x_curr @ (q_j @ x_curr) + np_prob.d[j] @ x_curr - np_prob.c[j]

    tmp = y_curr + g_x / (2.0 * xi)
    g_ypart = xi * np.sum(np.minimum(tmp, 0.0) ** 2) - np.dot(g_x, g_x) / (4.0 * xi) - np.dot(g_x, y_curr)
    g_ypart = -g_ypart
    g_ypart = g_ypart + 0.5 * x_curr @ (np_prob.A @ x_curr) + np_prob.b @ x_curr

    return float(g_xpart + g_ypart)


def to_numpy(v: torch.Tensor) -> np.ndarray:
    return v.detach().cpu().numpy()


def scalar(v: torch.Tensor | float) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)
