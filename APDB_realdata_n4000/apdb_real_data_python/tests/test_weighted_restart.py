from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from apdb_real_data_python.solvers.common import SharedParams


def _minimal_input() -> dict:
    return {
        "G_tr_class": [np.zeros((1, 1), dtype=np.float64)],
        "G_tr_max_norm": 1.0,
        "Ytrain": np.array([1.0], dtype=np.float64),
        "trace_kernels": np.array([1.0], dtype=np.float64),
        "reg_param": 1.0,
        "x0": np.zeros(1, dtype=np.float64),
        "y0": np.zeros(1, dtype=np.float64),
        "optsol": np.full(1, 10.0, dtype=np.float64),
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


def _patch_two_step_epoch(monkeypatch, solver, *, forced_points=(1.0, 3.0)):
    anchor_points: list[float] = []
    projection_inputs: list[float] = []

    monkeypatch.setattr(solver, "init_shared_params", lambda inp, **kwargs: _params())

    def _inner_prod(g_list, x, return_matvec=False):
        if return_matvec:
            return torch.zeros(1, dtype=torch.float64), torch.zeros((1, 1), dtype=torch.float64)
        anchor_points.append(float(x.item()))
        return torch.zeros(1, dtype=torch.float64), None

    projection_count = 0

    def _projection(candidate, labels):
        nonlocal projection_count
        projection_inputs.append(float(candidate.item()))
        if projection_count < len(forced_points):
            out = torch.full_like(candidate, forced_points[projection_count])
        else:
            out = candidate
        projection_count += 1
        return out

    monkeypatch.setattr(solver, "inner_prod_matvec_torch", _inner_prod)
    monkeypatch.setattr(solver, "projsplx_torch", lambda y: y.new_zeros(y.shape))
    monkeypatch.setattr(solver, "proj_pos_plane_torch", _projection)
    monkeypatch.setattr(
        solver,
        "make_progress_reporter",
        lambda **kwargs: SimpleNamespace(update=lambda **kw: None, close=lambda: None),
    )
    return anchor_points, projection_inputs


def test_weighted_restart_uses_sigma_weighted_epoch_average(monkeypatch):
    import apdb_real_data_python.solvers.rapdb_yx as solver

    anchor_points, projection_inputs = _patch_two_step_epoch(monkeypatch, solver)

    result = solver.run_rapdb_yx(
        _minimal_input(),
        stop_criteria=-1.0,
        max_iter=3,
        k_restart=2,
        non_mono=0,
        mu=2.0,
        restart_point="weighted",
    )

    expected_average = (1.0 + np.sqrt(3.0) * 3.0) / (1.0 + np.sqrt(3.0))
    assert anchor_points == pytest.approx([0.0])
    np.testing.assert_array_equal(result.restart_iter_epoch, np.array([2.0]))

    # tau is reset to 1 before the third step, so its pre-projection point is
    # (x_bar - tau*(-2))/(1 + mu*tau).
    assert projection_inputs[2] == pytest.approx((expected_average + 2.0) / 3.0)


def test_default_restart_point_preserves_last_iterate_behavior(monkeypatch):
    import apdb_real_data_python.solvers.rapdb_yx as solver

    anchor_points, projection_inputs = _patch_two_step_epoch(monkeypatch, solver)

    result = solver.run_rapdb_yx(
        _minimal_input(),
        stop_criteria=-1.0,
        max_iter=3,
        k_restart=2,
        non_mono=0,
        mu=2.0,
    )

    assert anchor_points == pytest.approx([0.0])
    assert projection_inputs[2] == pytest.approx((3.0 + 2.0) / 3.0)
    np.testing.assert_array_equal(result.restart_iter_epoch, np.array([2.0]))


def test_weighted_restart_clears_epoch_accumulators_and_resets_state(monkeypatch):
    import apdb_real_data_python.solvers.rapdb_yx as solver

    anchor_points, projection_inputs = _patch_two_step_epoch(
        monkeypatch,
        solver,
        forced_points=(1.0, 3.0, 10.0, 14.0),
    )

    result = solver.run_rapdb_yx(
        _minimal_input(),
        stop_criteria=-1.0,
        max_iter=5,
        k_restart=2,
        non_mono=0,
        mu=2.0,
        restart_point="weighted",
    )

    root_three = np.sqrt(3.0)
    first_average = (1.0 + root_three * 3.0) / (1.0 + root_three)
    second_average = (10.0 + root_three * 14.0) / (1.0 + root_three)
    assert anchor_points == pytest.approx([0.0])
    assert projection_inputs[2] == pytest.approx((first_average + 2.0) / 3.0)
    assert projection_inputs[4] == pytest.approx((second_average + 2.0) / 3.0)
    np.testing.assert_array_equal(result.restart_iter_epoch, np.array([2.0, 4.0]))


def test_last_restart_does_not_allocate_weighted_restart_accumulators(monkeypatch):
    import apdb_real_data_python.solvers.rapdb_yx as solver

    _patch_two_step_epoch(monkeypatch, solver)

    def _reject_zeros_like(*args, **kwargs):
        raise AssertionError("last restart allocated weighted-only accumulators")

    monkeypatch.setattr(solver.torch, "zeros_like", _reject_zeros_like)

    solver.run_rapdb_yx(
        _minimal_input(),
        stop_criteria=-1.0,
        max_iter=1,
        k_restart=1,
        non_mono=0,
        mu=2.0,
    )


def test_fixed_restart_rejects_unknown_restart_point():
    import apdb_real_data_python.solvers.rapdb_yx as solver

    with pytest.raises(ValueError, match="restart_point"):
        solver.run_rapdb_yx(
            _minimal_input(),
            stop_criteria=-1.0,
            max_iter=0,
            k_restart=2,
            non_mono=0,
            mu=2.0,
            restart_point="unknown",
        )
