from types import SimpleNamespace

import numpy as np
import torch

from apdb_real_data_python.solvers.common import SharedParams


def _minimal_input() -> dict:
    return {
        "G_tr_class": [np.zeros((1, 1), dtype=np.float64)],
        "G_tr_max_norm": 1.0,
        "Ytrain": np.array([1.0], dtype=np.float64),
        "trace_kernels": np.array([1.0], dtype=np.float64),
        "reg_param": 1.0,
        "x0": np.full(1, 0.25, dtype=np.float64),
        "y0": np.zeros(1, dtype=np.float64),
        "optsol": np.ones(1, dtype=np.float64),
    }


def _params() -> SharedParams:
    return SharedParams(
        c=1.0,
        tau=1.0,
        gamma=1.0,
        eta=0.7,
        c_alpha=0.4,
        delta=0.5,
        sigma_old=1.0,
        tau_old=1.0,
        alpha_old=0.4,
    )


def _patch_iterations(monkeypatch, module, grad3_points: list[float]) -> list[float]:
    projected_points: list[float] = []
    monkeypatch.setattr(module, "init_shared_params", lambda inp, **kwargs: _params())

    def _inner_prod(g_list, x, return_matvec=False):
        if not return_matvec:
            grad3_points.append(float(x.detach().cpu().reshape(-1)[0].item()))
        grad2 = torch.zeros(1, dtype=torch.float64, device=x.device)
        grad1 = torch.zeros((1, 1), dtype=torch.float64, device=x.device) if return_matvec else None
        return grad2, grad1

    monkeypatch.setattr(module, "inner_prod_matvec_torch", _inner_prod)
    monkeypatch.setattr(module, "projsplx_torch", lambda y: y.new_zeros(y.shape))
    def _projection(x, y):
        projected_points.append(float(x.detach().cpu().reshape(-1)[0].item()))
        return x

    monkeypatch.setattr(module, "proj_pos_plane_torch", _projection)
    monkeypatch.setattr(
        module,
        "make_progress_reporter",
        lambda **kwargs: SimpleNamespace(update=lambda **kw: None, close=lambda: None),
    )
    return projected_points


def test_apdb_yx_keeps_initial_dual_anchor(monkeypatch):
    import apdb_real_data_python.solvers.apdb_yx as solver

    grad3_points: list[float] = []
    _patch_iterations(monkeypatch, solver, grad3_points)

    solver.run_apdb_yx(
        _minimal_input(),
        stop_criteria=-1.0,
        max_iter=3,
        non_mono=0,
    )

    np.testing.assert_allclose(grad3_points, np.array([0.25]))


def test_rapdb_yx_fixed_restart_keeps_initial_dual_anchor(monkeypatch):
    import apdb_real_data_python.solvers.rapdb_yx as solver

    grad3_points: list[float] = []
    _patch_iterations(monkeypatch, solver, grad3_points)

    result = solver.run_rapdb_yx(
        _minimal_input(),
        stop_criteria=-1.0,
        max_iter=3,
        k_restart=1,
        non_mono=0,
        mu=2.0,
    )

    np.testing.assert_allclose(grad3_points, np.array([0.25]))
    np.testing.assert_array_equal(result.restart_iter_epoch, np.array([1.0, 2.0, 3.0]))


def test_rapdb_yx_ada_adaptive_restart_keeps_initial_dual_anchor(monkeypatch):
    import apdb_real_data_python.solvers.rapdb_yx_ada as solver

    grad3_points: list[float] = []
    _patch_iterations(monkeypatch, solver, grad3_points)
    g_xi_values = iter([2.0, 0.0])
    monkeypatch.setattr(solver, "run_x_qp_for_gxi", lambda **kwargs: next(g_xi_values, 0.0))

    result = solver.run_rapdb_yx_ada(
        _minimal_input(),
        stop_criteria=-1.0,
        max_iter=51,
        k_restart=1,
        non_mono=0,
        mu=2.0,
    )

    np.testing.assert_allclose(grad3_points, np.array([0.25]))
    np.testing.assert_array_equal(result.restart_iter_epoch, np.array([50.0]))


def test_rapdb_yx_ada_does_not_allocate_weighted_restart_accumulators(monkeypatch):
    import apdb_real_data_python.solvers.rapdb_yx_ada as solver

    grad3_points: list[float] = []
    _patch_iterations(monkeypatch, solver, grad3_points)

    def _reject_zeros_like(*args, **kwargs):
        raise AssertionError("adaptive restart allocated weighted-only accumulators")

    monkeypatch.setattr(solver.torch, "zeros_like", _reject_zeros_like)

    solver.run_rapdb_yx_ada(
        _minimal_input(),
        stop_criteria=-1.0,
        max_iter=1,
        k_restart=1,
        non_mono=0,
        mu=2.0,
    )
