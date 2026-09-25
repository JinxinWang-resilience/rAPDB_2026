from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from apdb_real_data_python.solvers.common import SharedParams, init_shared_params, to_torch_input


SOLVERS = (
    ("apdb_real_data_python.solvers.apdb_yx", "run_apdb_yx"),
    ("apdb_real_data_python.solvers.rapdb_yx", "run_rapdb_yx"),
    ("apdb_real_data_python.solvers.rapdb_yx_ada", "run_rapdb_yx_ada"),
)


def _minimal_input(*, reg_param: float = 1.0) -> dict:
    return {
        "G_tr_class": [np.zeros((1, 1), dtype=np.float64)],
        "G_tr_max_norm": 1.0,
        "Ytrain": np.array([1.0], dtype=np.float64),
        "trace_kernels": np.array([1.0], dtype=np.float64),
        "reg_param": reg_param,
        "x0": np.array([3.0], dtype=np.float64),
        "y0": np.zeros(1, dtype=np.float64),
        "optsol": np.array([10.0], dtype=np.float64),
    }


def _params() -> SharedParams:
    return SharedParams(
        c=1.0,
        tau=0.5,
        gamma=1.0,
        eta=0.7,
        c_alpha=0.4,
        delta=0.5,
        sigma_old=0.5,
        tau_old=0.5,
        alpha_old=0.8,
    )


def _run_one_iteration(monkeypatch, module_name: str, function_name: str) -> list[float]:
    module = importlib.import_module(module_name)
    projection_inputs: list[float] = []

    monkeypatch.setattr(module, "init_shared_params", lambda inp, **kwargs: _params())

    def _inner_prod(g_list, x, return_matvec=False):
        grad2 = torch.zeros(1, dtype=torch.float64, device=x.device)
        grad1 = torch.zeros((1, 1), dtype=torch.float64, device=x.device) if return_matvec else None
        return grad2, grad1

    def _projection(candidate, labels):
        projection_inputs.append(float(candidate.detach().cpu().item()))
        return candidate

    monkeypatch.setattr(module, "inner_prod_matvec_torch", _inner_prod)
    monkeypatch.setattr(module, "projsplx_torch", lambda y: torch.zeros_like(y))
    monkeypatch.setattr(module, "proj_pos_plane_torch", _projection)
    monkeypatch.setattr(
        module,
        "make_progress_reporter",
        lambda **kwargs: SimpleNamespace(update=lambda **kw: None, close=lambda: None),
    )

    kwargs = {
        "stop_criteria": -1.0,
        "max_iter": 1,
        "non_mono": 0,
        "mu": 2.0,
    }
    if function_name != "run_apdb_yx":
        kwargs["k_restart"] = 10
    getattr(module, function_name)(_minimal_input(), **kwargs)
    return projection_inputs


@pytest.mark.parametrize(("module_name", "function_name"), SOLVERS)
def test_x_step_proximally_handles_strong_convex_quadratic(
    monkeypatch, module_name: str, function_name: str
):
    projection_inputs = _run_one_iteration(monkeypatch, module_name, function_name)

    # grad(tilde Phi)=-2, x=3 and tau=0.5, hence (3-0.5*(-2))/(1+2*0.5)=2.
    assert projection_inputs[0] == pytest.approx(2.0)


@pytest.mark.parametrize(("module_name", "function_name"), SOLVERS)
def test_line_search_does_not_backtrack_on_removed_quadratic_remainder(
    monkeypatch, module_name: str, function_name: str
):
    projection_inputs = _run_one_iteration(monkeypatch, module_name, function_name)

    # With zero coupling remainder, the first trial must pass.  Keeping +||dx||^2
    # in e_val would incorrectly reject tau=0.5 and call the projection again.
    assert len(projection_inputs) == 1


def test_shared_params_remove_strong_convex_curvature_only_for_apdb():
    inp = to_torch_input(_minimal_input(), device="cpu")

    egm_params = init_shared_params(inp)
    apdb_params = init_shared_params(inp, strong_convexity=2.0)

    assert egm_params.tau == pytest.approx(10.0 / (15.0 + 4.0))
    assert apdb_params.tau == pytest.approx(10.0 / (15.0 + 3.0))


@pytest.mark.parametrize(("module_name", "function_name"), SOLVERS)
@pytest.mark.parametrize(
    ("input_data", "mu", "message"),
    (
        (_minimal_input(), 1.0, "mu=2"),
        (_minimal_input(reg_param=0.5), 2.0, "reg_param=1"),
    ),
)
def test_apdb_solvers_reject_parameters_outside_fixed_paper_setting(
    module_name: str,
    function_name: str,
    input_data: dict,
    mu: float,
    message: str,
):
    module = importlib.import_module(module_name)
    kwargs = {
        "stop_criteria": -1.0,
        "max_iter": 0,
        "non_mono": 0,
        "mu": mu,
    }
    if function_name != "run_apdb_yx":
        kwargs["k_restart"] = 10

    with pytest.raises(ValueError, match=message):
        getattr(module, function_name)(input_data, **kwargs)
