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
        "reg_param": 0.0,
        "x0": np.zeros(1, dtype=np.float64),
        "y0": np.zeros(1, dtype=np.float64),
        "optsol": np.ones(1, dtype=np.float64),
    }


def _params() -> SharedParams:
    return SharedParams(
        c=1.0,
        tau=1.0,
        gamma=2.0,
        eta=0.7,
        c_alpha=0.4,
        delta=0.5,
        sigma_old=2.0,
        tau_old=1.0,
        alpha_old=0.2,
    )


def _patch_deterministic_iteration(monkeypatch, module, simplex_inputs: list[float]) -> None:
    monkeypatch.setattr(module, "init_shared_params", lambda inp: _params())

    def _inner_prod(g_list, x, return_matvec=False):
        grad2 = torch.ones(1, dtype=torch.float64, device=x.device)
        grad1 = torch.zeros((1, 1), dtype=torch.float64, device=x.device) if return_matvec else None
        return grad2, grad1

    def _projsplx(y):
        simplex_inputs.append(float(y.detach().cpu().reshape(-1)[0].item()))
        return torch.zeros_like(y)

    monkeypatch.setattr(module, "inner_prod_matvec_torch", _inner_prod)
    monkeypatch.setattr(module, "projsplx_torch", _projsplx)
    monkeypatch.setattr(module, "proj_pos_plane_torch", lambda x, y: x)
    monkeypatch.setattr(module, "make_progress_reporter", lambda **kwargs: SimpleNamespace(update=lambda **kw: None, close=lambda: None))


def test_rapdb_yx_resets_gamma_on_restart(monkeypatch):
    import apdb_real_data_python.solvers.rapdb_yx as solver

    simplex_inputs: list[float] = []
    _patch_deterministic_iteration(monkeypatch, solver, simplex_inputs)

    solver.run_rapdb_yx(
        _minimal_input(),
        stop_criteria=-1.0,
        max_iter=2,
        k_restart=1,
        non_mono=0,
        mu=1.0,
    )

    assert simplex_inputs[1] == 2.0


def test_rapdb_yx_ada_resets_gamma_on_restart(monkeypatch):
    import apdb_real_data_python.solvers.rapdb_yx_ada as solver

    simplex_inputs: list[float] = []
    _patch_deterministic_iteration(monkeypatch, solver, simplex_inputs)
    g_xi_values = iter([2.0, 0.0])
    monkeypatch.setattr(solver, "run_x_qp_for_gxi", lambda **kwargs: next(g_xi_values, 0.0))

    solver.run_rapdb_yx_ada(
        _minimal_input(),
        stop_criteria=-1.0,
        max_iter=51,
        k_restart=1,
        non_mono=0,
        mu=1.0,
    )

    assert simplex_inputs[50] == 2.0
