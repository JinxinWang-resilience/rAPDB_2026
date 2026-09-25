from __future__ import annotations

import numpy as np

from apdb_python import plotting


def test_strict_line_styles_group_yx_as_dashed_and_xy_as_solid():
    assert plotting._line_style_for_algorithm("APDB-yx", "strict") == "--"
    assert plotting._line_style_for_algorithm("rAPDB-yx-ada", "strict") == "--"
    assert plotting._line_style_for_algorithm("rAPDB-yx", "strict") == "--"

    assert plotting._line_style_for_algorithm("APDB-xy", "strict") == "-"
    assert plotting._line_style_for_algorithm("rAPDB-xy-ada", "strict") == "-"
    assert plotting._line_style_for_algorithm("rAPDB-xy", "strict") == "-"


def test_fixed_restart_marker_points_use_mean_oracle_and_curve_y_values():
    oracle_list = [
        np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        np.array([11.0, 21.0, 31.0, 41.0, 51.0]),
    ]
    rel_list = [
        np.array([1.0, 0.8, 0.6, 0.4, 0.2]),
        np.array([1.1, 0.9, 0.7, 0.5, 0.3]),
    ]
    iter_list = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ]
    curve_x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    curve_y = np.array([1.0, 0.9, 0.7, 0.5, 0.3])

    x, y = plotting._fixed_restart_marker_points(
        oracle_list=oracle_list,
        iter_list=iter_list,
        restart_period=2,
        numsim=2,
        curve_x=curve_x,
        curve_y=curve_y,
    )

    np.testing.assert_allclose(x, [20.5, 40.5])
    np.testing.assert_allclose(y, np.interp(x, curve_x, curve_y))


def test_restart_event_marker_points_use_recorded_oracles():
    x, y = plotting._restart_event_marker_points(
        event_oracle_list=[
            np.array([20.0, 40.0]),
            np.array([22.0, 42.0]),
        ],
        numsim=2,
        curve_x=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        curve_y=np.array([1.0, 0.8, 0.6, 0.4, 0.2]),
        log_y=False,
    )

    np.testing.assert_allclose(x, [21.0, 41.0])
    np.testing.assert_allclose(y, np.interp(x, [10.0, 20.0, 30.0, 40.0, 50.0], [1.0, 0.8, 0.6, 0.4, 0.2]))


def test_fixed_restart_marker_points_skip_nonpositive_curve_values():
    x, y = plotting._fixed_restart_marker_points(
        oracle_list=[np.array([10.0, 20.0, 30.0])],
        iter_list=[np.array([1.0, 2.0, 3.0])],
        restart_period=2,
        numsim=1,
        curve_x=np.array([10.0, 20.0, 30.0]),
        curve_y=np.array([1.0, 0.0, 0.5]),
        mask_nonpositive=True,
    )

    assert x.size == 0
    assert y.size == 0


def test_fixed_restart_marker_points_can_use_log_y_interpolation():
    x, y = plotting._fixed_restart_marker_points(
        oracle_list=[np.array([10.0, 20.0, 30.0])],
        iter_list=[np.array([1.0, 2.0, 3.0])],
        restart_period=2,
        numsim=1,
        curve_x=np.array([10.0, 30.0]),
        curve_y=np.array([1.0, 100.0]),
        log_y=True,
    )

    np.testing.assert_allclose(x, [20.0])
    np.testing.assert_allclose(y, [10.0])


def test_fixed_restart_marker_points_do_not_cross_nan_gap():
    x, y = plotting._fixed_restart_marker_points(
        oracle_list=[np.array([10.0, 20.0, 30.0, 40.0])],
        iter_list=[np.array([1.0, 2.0, 3.0, 4.0])],
        restart_period=2,
        numsim=1,
        curve_x=np.array([10.0, 15.0, 25.0, 30.0, 40.0]),
        curve_y=np.array([1.0, 0.8, np.nan, 0.4, 0.2]),
        log_y=True,
    )

    np.testing.assert_allclose(x, [40.0])
    np.testing.assert_allclose(y, [0.2])


def test_fixed_restart_marker_points_keep_marker_inside_finite_segment():
    x, y = plotting._fixed_restart_marker_points(
        oracle_list=[np.array([10.0, 20.0, 30.0, 40.0])],
        iter_list=[np.array([1.0, 2.0, 3.0, 4.0])],
        restart_period=2,
        numsim=1,
        curve_x=np.array([10.0, 20.0, 30.0, np.nan, 40.0]),
        curve_y=np.array([1.0, 0.5, 0.25, np.nan, 0.125]),
        log_y=True,
    )

    np.testing.assert_allclose(x, [20.0, 40.0])
    np.testing.assert_allclose(y, [0.5, 0.125])


def test_strict_fixed_restart_periods_match_run_configuration():
    assert plotting._fixed_restart_period("rAPDB-yx", mode_idx=0) == 800
    assert plotting._fixed_restart_period("rAPDB-yx", mode_idx=1) == 400
    assert plotting._fixed_restart_period("rAPDB-xy", mode_idx=0) == 2000
    assert plotting._fixed_restart_period("rAPDB-xy", mode_idx=1) == 1000
    assert plotting._fixed_restart_period("rAPDB-yx-ada", mode_idx=0) is None
