from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp
import torch

from apdb_real_data_python.operators import inner_prod_matvec_torch, proj_pos_plane_torch, projsplx_torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


@dataclass
class SolverInputTorch:
    g_tr_class: list[torch.Tensor]
    g_tr_max_norm: float
    y_train: torch.Tensor
    trace_kernels: torch.Tensor
    reg_param: float
    x0: torch.Tensor
    y0: torch.Tensor
    optsol: torch.Tensor


@dataclass
class SolverResult:
    rel_err_x: np.ndarray
    time_period: np.ndarray
    iter_epoch: np.ndarray
    restart_iter_epoch: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    restart_time_period: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))


@dataclass
class SharedParams:
    c: float
    tau: float
    gamma: float
    eta: float
    c_alpha: float
    delta: float
    sigma_old: float
    tau_old: float
    alpha_old: float


class ProgressReporter(Protocol):
    def update(self, *, iter_idx: int, rel_err: float, elapsed: float) -> None:
        ...

    def close(self) -> None:
        ...


class _NullProgressReporter:
    def update(self, *, iter_idx: int, rel_err: float, elapsed: float) -> None:
        return

    def close(self) -> None:
        return


class _TextProgressReporter:
    def __init__(self, *, name: str, refresh_sec: float, every_n: int) -> None:
        self.name = name
        self.refresh_sec = refresh_sec
        self.every_n = every_n
        self._last_print = -1e9

    def update(self, *, iter_idx: int, rel_err: float, elapsed: float) -> None:
        if iter_idx % self.every_n != 0:
            return
        if (elapsed - self._last_print) >= self.refresh_sec:
            print(f"[{self.name}] iter={iter_idx} rel_err={rel_err:.3e} elapsed={elapsed:.1f}s", flush=True)
            self._last_print = elapsed

    def close(self) -> None:
        return


class _TqdmProgressReporter:
    def __init__(self, *, name: str, max_iter: int, refresh_sec: float, every_n: int) -> None:
        self._bar = tqdm(total=max_iter, desc=name, mininterval=refresh_sec, miniters=every_n, leave=True)
        self._every_n = every_n
        self._last_iter_seen = 0
        self._pending_step = 0
        self._last_rel_err = 0.0
        self._last_elapsed = 0.0

    def update(self, *, iter_idx: int, rel_err: float, elapsed: float) -> None:
        step = max(0, int(iter_idx) - self._last_iter_seen)
        self._last_iter_seen = int(iter_idx)
        self._pending_step += step
        self._last_rel_err = rel_err
        self._last_elapsed = elapsed
        if int(iter_idx) % self._every_n != 0:
            return
        if self._pending_step > 0:
            self._bar.update(self._pending_step)
            self._pending_step = 0
        self._bar.set_postfix(rel_err=f"{rel_err:.3e}", elapsed=f"{elapsed:.1f}s")

    def close(self) -> None:
        if self._pending_step > 0:
            self._bar.update(self._pending_step)
            self._pending_step = 0
            self._bar.set_postfix(rel_err=f"{self._last_rel_err:.3e}", elapsed=f"{self._last_elapsed:.1f}s")
        self._bar.close()


def make_progress_reporter(
    *,
    name: str,
    max_iter: int,
    progress: dict | None,
) -> ProgressReporter:
    cfg = progress or {}
    mode = str(cfg.get("mode", "off"))
    refresh_sec = float(cfg.get("refresh_sec", 0.5))
    every_n = int(cfg.get("every_n", 20))
    if refresh_sec <= 0.0:
        refresh_sec = 0.5
    if every_n <= 0:
        every_n = 20

    if mode == "off":
        return _NullProgressReporter()

    use_tqdm = False
    if mode == "on":
        use_tqdm = tqdm is not None
    elif mode == "auto":
        use_tqdm = tqdm is not None and bool(getattr(sys.stderr, "isatty", lambda: False)())

    if use_tqdm:
        return _TqdmProgressReporter(name=name, max_iter=max_iter, refresh_sec=refresh_sec, every_n=every_n)
    return _TextProgressReporter(name=name, refresh_sec=refresh_sec, every_n=every_n)


def to_torch_input(input_data: dict, *, device: str) -> SolverInputTorch:
    dev = torch.device(device)
    g_tr_class = [torch.as_tensor(g, dtype=torch.float64, device=dev) for g in input_data["G_tr_class"]]
    y_train = torch.as_tensor(input_data["Ytrain"], dtype=torch.float64, device=dev).reshape(-1)
    trace_kernels = torch.as_tensor(input_data["trace_kernels"], dtype=torch.float64, device=dev).reshape(-1)
    x0 = torch.as_tensor(input_data["x0"], dtype=torch.float64, device=dev).reshape(-1)
    y0 = torch.as_tensor(input_data["y0"], dtype=torch.float64, device=dev).reshape(-1)
    optsol = torch.as_tensor(input_data["optsol"], dtype=torch.float64, device=dev).reshape(-1)
    return SolverInputTorch(
        g_tr_class=g_tr_class,
        g_tr_max_norm=float(input_data["G_tr_max_norm"]),
        y_train=y_train,
        trace_kernels=trace_kernels,
        reg_param=float(input_data["reg_param"]),
        x0=x0,
        y0=y0,
        optsol=optsol,
    )


def validate_apdb_strong_convex_settings(inp: SolverInputTorch, *, mu: float) -> None:
    if not np.isclose(float(mu), 2.0, rtol=0.0, atol=1e-12):
        raise ValueError("APDB-yx strong-convex splitting requires mu=2.")
    if not np.isclose(inp.reg_param, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("APDB-yx strong-convex splitting requires reg_param=1.")


def init_shared_params(inp: SolverInputTorch, *, strong_convexity: float = 0.0) -> SharedParams:
    num_kern = int(inp.trace_kernels.numel())
    coef = 0.5
    b_bound = 15.0
    l_yx = coef * 2.0 * num_kern * b_bound * inp.g_tr_max_norm
    linearized_quadratic_curvature = 2.0 * inp.reg_param - float(strong_convexity)
    l_xx = coef * (linearized_quadratic_curvature + 6.0 * inp.g_tr_max_norm)
    tau = 10.0 / (l_xx + l_yx)
    gamma = 1.0 / l_yx / tau
    c = float(torch.sum(inp.trace_kernels).item())
    sigma_old = gamma * tau
    c_alpha = 0.4
    return SharedParams(
        c=c,
        tau=tau,
        gamma=gamma,
        eta=0.7,
        c_alpha=c_alpha,
        delta=0.5,
        sigma_old=sigma_old,
        tau_old=tau,
        alpha_old=c_alpha / sigma_old,
    )


def _make_osqp_solver(osqp_module, *, device_type: str):
    available = getattr(osqp_module, "algebras_available", None)
    if device_type == "cuda":
        if available is None or "cuda" not in available():
            raise RuntimeError(
                "GPU adaptive restart requires the OSQP CUDA backend. "
                "Install osqp[cu12] and its CUDA runtime dependencies, then check "
                "that 'cuda' is in osqp.algebras_available()."
            )
        return osqp_module.OSQP(algebra="cuda")
    if device_type == "cpu":
        # OSQP 0.6 has no algebra selector; OSQP 1.x may otherwise default to CUDA.
        return osqp_module.OSQP(algebra="builtin") if available is not None else osqp_module.OSQP()
    raise ValueError(f"Unsupported OSQP device type: {device_type}")


def run_x_qp_for_gxi(
    *,
    x_curr: torch.Tensor,
    y_curr: torch.Tensor,
    inp: SolverInputTorch,
    xi: float,
    c: float,
) -> float:
    try:
        import osqp
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("OSQP is required for G_xi subproblem solve") from exc

    solver = _make_osqp_solver(osqp, device_type=x_curr.device.type)
    x_np = x_curr.detach().cpu().numpy().astype(np.float64)
    y_np = y_curr.detach().cpu().numpy().astype(np.float64)
    ytrain_np = inp.y_train.detach().cpu().numpy().astype(np.float64)
    traces = inp.trace_kernels.detach().cpu().numpy().astype(np.float64)

    m = x_np.shape[0]
    q_mat = (2.0 * inp.reg_param + 2.0 * xi) * np.eye(m, dtype=np.float64)
    for j, gj in enumerate(inp.g_tr_class):
        q_mat = q_mat + 2.0 * y_np[j] * c / traces[j] * gj.detach().cpu().numpy()
    q_mat = 0.5 * (q_mat + q_mat.T)
    q_vec = -2.0 * np.ones(m, dtype=np.float64) - 2.0 * xi * x_np

    P = sp.csc_matrix(q_mat)
    q = np.asarray(q_vec, dtype=np.float64).reshape(-1)
    A_eq = sp.csc_matrix(ytrain_np.reshape(1, -1))
    A_ineq = sp.eye(m, dtype=np.float64, format="csc")
    A = sp.vstack([A_eq, A_ineq], format="csc")
    l = np.concatenate([np.array([0.0], dtype=np.float64), np.zeros(m, dtype=np.float64)])
    u = np.concatenate([np.array([0.0], dtype=np.float64), np.full(m, np.inf, dtype=np.float64)])

    solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, eps_abs=1e-8, eps_rel=1e-8, warm_start=True)
    res = solver.solve()
    solved = {osqp.constant("OSQP_SOLVED"), osqp.constant("OSQP_SOLVED_INACCURATE")}
    if res.x is None or int(res.info.status_val) not in solved:
        raise RuntimeError("G_xi subproblem solve failed.")
    x_sol = np.asarray(res.x, dtype=np.float64).reshape(-1)
    qp_value = 0.5 * float(x_sol @ (q_mat @ x_sol)) + float(q @ x_sol)

    g_xpart = -float(qp_value + xi * np.dot(x_np, x_np))

    grad2, _ = inner_prod_matvec_torch(inp.g_tr_class, x_curr, return_matvec=True)
    g_xx = (c / inp.trace_kernels) * grad2
    tmp_vec = y_curr + g_xx / (2.0 * xi)
    tmp_proj = projsplx_torch(tmp_vec)

    g_ypart = xi * torch.norm(tmp_proj - tmp_vec) ** 2 - torch.norm(g_xx) ** 2 / (4.0 * xi) - torch.dot(g_xx, y_curr)
    g_ypart = -g_ypart + inp.reg_param * torch.norm(x_curr) ** 2 - 2.0 * torch.sum(x_curr)
    return float(g_xpart + g_ypart.item())


def append_metrics(
    rel_err: list[float],
    time_curve: list[float],
    iter_curve: list[float],
    *,
    x: torch.Tensor,
    optsol: torch.Tensor,
    elapsed: float,
    iter_idx: int,
) -> None:
    norm_optsol = 1.0 + float(torch.norm(optsol).item())
    rel = float(torch.norm(x - optsol).item()) / norm_optsol
    rel_err.append(rel)
    time_curve.append(elapsed)
    iter_curve.append(float(iter_idx))


def finish_result(
    rel_err: list[float], time_curve: list[float], iter_curve: list[float]
) -> SolverResult:
    return SolverResult(
        rel_err_x=np.asarray(rel_err, dtype=np.float64),
        time_period=np.asarray(time_curve, dtype=np.float64),
        iter_epoch=np.asarray(iter_curve, dtype=np.float64),
    )
