import numpy as np
from scipy.io import savemat

from apdb_real_data_python.plot_cpu import (
    _event_markevery,
    AXIS_FONTSIZE,
    CPU_TIME_X_LABEL,
    GPU_TIME_X_LABEL,
    GPU_TIME_X_LIMIT,
    ITERATION_X_LIMIT,
    LABEL_FONTSIZE,
    LEGEND_FRAME_ALPHA,
    LEGEND_FONTSIZE,
    MARKER_SIZE,
    REL_ERROR_Y_LABEL,
    TIME_X_LIMIT,
    _build_arg_parser,
    _legend_label,
    load_cpu_curves_mat,
    _marker_kwargs,
    _plot_specs,
)


def test_legend_label_for_egm_has_no_mode_suffix():
    assert _legend_label("EGM", "-") == "EGM"
    assert _legend_label("APDB-yx", "mono", 2.0) == "APDB-yx (mono)"
    assert _legend_label("APDB-yx", "nonmono", 2.0) == "APDB-yx (non-mono)"


def test_keeps_mu2_mono_and_nonmono_curves_only():
    specs = _plot_specs()
    mono_specs = [s for s in specs if s["mode"] == "mono"]
    nonmono_specs = [s for s in specs if s["mode"] == "nonmono"]
    egm_specs = [s for s in specs if s["alg"] == "EGM"]

    assert len(mono_specs) == 3
    assert len(nonmono_specs) == 3
    assert egm_specs[0]["color"] == "tab:purple"
    assert all(s["linestyle"] == "--" for s in mono_specs)
    assert all(s["linestyle"] == "-" for s in nonmono_specs)
    assert [s for s in mono_specs if s["alg"] == "APDB-yx"][0]["marker"] is None
    assert [s for s in nonmono_specs if s["alg"] == "APDB-yx"][0]["marker"] is None
    assert [s for s in mono_specs if s["alg"] == "rAPDB-yx-ada"][0]["marker"] is None
    assert [s for s in nonmono_specs if s["alg"] == "rAPDB-yx-ada"][0]["marker"] is None
    assert [s for s in mono_specs if s["alg"] == "rAPDB-yx"][0]["marker"] == "x"
    assert [s for s in nonmono_specs if s["alg"] == "rAPDB-yx"][0]["marker"] == "x"
    assert [s for s in mono_specs if s["alg"] == "rAPDB-yx"][0]["restart_every"] == 1000
    assert [s for s in nonmono_specs if s["alg"] == "rAPDB-yx"][0]["restart_every"] == 200
    assert {s["mu"] for s in specs if s["alg"] != "EGM"} == {2.0}
    assert all(float(s.get("alpha", 1.0)) >= 0.9 for s in specs)


def test_rapdb_yx_cross_markers_use_restart_iterations_only():
    specs = _plot_specs()
    mono_spec = [s for s in specs if s["alg"] == "rAPDB-yx" and s["mode"] == "mono"][0]
    nonmono_spec = [s for s in specs if s["alg"] == "rAPDB-yx" and s["mode"] == "nonmono"][0]

    assert _marker_kwargs(mono_spec, [1, 1000, 1001, 2000, 2001])["markevery"] == [1, 3]
    assert _marker_kwargs(nonmono_spec, [1, 199, 200, 201, 400])["markevery"] == [2, 4]


def test_non_k_multiple_restart_event_takes_priority_over_fallback_indices():
    mono_spec = [
        s for s in _plot_specs() if s["alg"] == "rAPDB-yx" and s["mode"] == "mono"
    ][0]
    iterations = [1, 750, 1000, 2000, 2001]

    kwargs = _marker_kwargs(
        mono_spec,
        iterations,
        event_values=[750],
    )

    assert kwargs["markevery"] == [1]


def _record_plot_marker_calls(monkeypatch, tmp_path, curve):
    import apdb_real_data_python.plot_cpu as plot_cpu

    calls = []
    original = plot_cpu._marker_kwargs

    def _record(spec, x_values, event_values=None, **kwargs):
        result = original(spec, x_values, event_values, **kwargs)
        calls.append(
            {
                "x_values": np.asarray(x_values, dtype=np.float64),
                "event_values": np.asarray(
                    [] if event_values is None else event_values,
                    dtype=np.float64,
                ),
                "kwargs": kwargs,
                "result": result,
            }
        )
        return result

    monkeypatch.setattr(plot_cpu, "_marker_kwargs", _record)
    plot_cpu.plot_cpu_curves({"rAPDB-yx|mono|mu=2": curve}, tmp_path)
    return calls


def test_time_axis_without_restart_events_uses_iteration_k_multiples(monkeypatch, tmp_path):
    iterations = np.asarray([1, 1000, 1001, 2000, 2001], dtype=np.float64)
    curve = {
        "iter": iterations,
        "time": np.asarray([0.1, 0.5, 0.7, 1.2, 1.4], dtype=np.float64),
        "rel": np.asarray([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], dtype=np.float64),
    }

    calls = _record_plot_marker_calls(monkeypatch, tmp_path, curve)

    assert len(calls) == 2
    for call in calls:
        np.testing.assert_allclose(call["x_values"], iterations)
        assert call["result"]["markevery"] == [1, 3]


def test_restart_time_rounding_does_not_change_time_axis_marker_indices(monkeypatch, tmp_path):
    iterations = np.asarray([1, 600, 750, 1200, 1201], dtype=np.float64)
    curve = {
        "iter": iterations,
        "time": np.asarray([0.1, 0.5, 0.7, 1.2, 1.4], dtype=np.float64),
        "rel": np.asarray([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], dtype=np.float64),
        "restart_iter": np.asarray([750], dtype=np.float64),
        "restart_time": np.asarray([0.7001], dtype=np.float64),
    }

    calls = _record_plot_marker_calls(monkeypatch, tmp_path, curve)

    assert len(calls) == 2
    for call in calls:
        np.testing.assert_allclose(call["x_values"], iterations)
        np.testing.assert_allclose(call["event_values"], [750])
        assert call["result"]["markevery"] == [2]


def test_plot_limits_and_marker_size_match_requested_style():
    assert MARKER_SIZE > 6
    assert ITERATION_X_LIMIT == (0, 8000)
    assert TIME_X_LIMIT == (0, 600)
    assert GPU_TIME_X_LIMIT == (0, 200)
    assert CPU_TIME_X_LABEL == "CPU time (s)"
    assert GPU_TIME_X_LABEL == "GPU time (s)"


def test_text_sizes_match_strict_comparison_plot_style():
    assert AXIS_FONTSIZE == 16
    assert LABEL_FONTSIZE == 16
    assert LEGEND_FONTSIZE == 13
    assert LEGEND_FRAME_ALPHA == 0.55


def test_gpu_curves_drive_right_axis_and_legend_style(monkeypatch, tmp_path):
    import apdb_real_data_python.plot_cpu as plot_cpu

    cpu_curve = {
        "iter": np.asarray([1.0, 2.0]),
        "time": np.asarray([10.0, 20.0]),
        "rel": np.asarray([1e-1, 1e-2]),
    }
    gpu_curve = {
        "iter": np.asarray([1.0, 2.0]),
        "time": np.asarray([3.0, 4.0]),
        "rel": np.asarray([1e-1, 1e-2]),
    }
    closed_figures = []
    monkeypatch.setattr(plot_cpu.plt, "close", closed_figures.append)

    plot_cpu.plot_cpu_curves(
        {"EGM|-": cpu_curve},
        tmp_path,
        gpu_curves={"EGM|-": gpu_curve},
    )

    fig = closed_figures[0]
    left_axis, right_axis = fig.axes
    np.testing.assert_allclose(left_axis.lines[0].get_xdata(), cpu_curve["iter"])
    np.testing.assert_allclose(right_axis.lines[0].get_xdata(), gpu_curve["time"])
    assert right_axis.get_xlabel() == GPU_TIME_X_LABEL
    np.testing.assert_allclose(right_axis.get_xlim(), GPU_TIME_X_LIMIT)
    for axis in (left_axis, right_axis):
        legend = axis.get_legend()
        assert legend.get_frame().get_alpha() == LEGEND_FRAME_ALPHA
        assert {text.get_fontsize() for text in legend.get_texts()} == {LEGEND_FONTSIZE}


def test_cpu_only_plot_keeps_cpu_time_fallback(monkeypatch, tmp_path):
    import apdb_real_data_python.plot_cpu as plot_cpu

    curve = {
        "iter": np.asarray([1.0, 2.0]),
        "time": np.asarray([10.0, 20.0]),
        "rel": np.asarray([1e-1, 1e-2]),
    }
    closed_figures = []
    monkeypatch.setattr(plot_cpu.plt, "close", closed_figures.append)

    plot_cpu.plot_cpu_curves({"EGM|-": curve}, tmp_path)

    right_axis = closed_figures[0].axes[1]
    np.testing.assert_allclose(right_axis.lines[0].get_xdata(), curve["time"])
    assert right_axis.get_xlabel() == CPU_TIME_X_LABEL
    np.testing.assert_allclose(right_axis.get_xlim(), TIME_X_LIMIT)


def test_gpu_plot_shows_full_time_and_mismatched_reference(monkeypatch, tmp_path):
    import apdb_real_data_python.plot_cpu as plot_cpu

    curve = {"iter": np.array([1.0, 2.0]), "time": np.array([3.0, 340.0]), "rel": np.array([0.1, 0.01])}
    figures = []
    monkeypatch.setattr(plot_cpu.plt, "close", figures.append)

    plot_cpu.plot_cpu_curves({"EGM|-": curve}, tmp_path, gpu_curves={"EGM|-": curve}, reference_match=False)

    left, right = figures[0].axes
    np.testing.assert_allclose(left.get_xlim(), (0, 8000))
    np.testing.assert_allclose(right.get_xlim(), (0, 400))
    assert "original MOSEK reference" in left.get_title()
    assert "recomputed MOSEK reference" in right.get_title()


def test_plot_cli_accepts_optional_gpu_mat_file():
    args = _build_arg_parser().parse_args(
        ["experiment_outputs_cpu.mat", "--gpu-mat", "experiment_outputs_gpu.mat"]
    )

    assert args.mat_file == "experiment_outputs_cpu.mat"
    assert args.gpu_mat == "experiment_outputs_gpu.mat"


def test_relative_error_label_uses_x_star_not_asterisk():
    assert REL_ERROR_Y_LABEL == r"$\|x^k-x^\star\|/(1+\|x^\star\|)$"


def test_event_markevery_marks_recorded_restart_events():
    x_values = [1.0, 2.0, 3.0, 4.0]
    event_values = [2.0, 4.0]

    assert _event_markevery(x_values, event_values) == [1, 3]
    assert _event_markevery(x_values, []) is None


def test_load_cpu_curves_reads_fixed_restart_event_arrays(tmp_path):
    mat_path = tmp_path / "experiment_outputs_cpu.mat"
    payload = {
        "iter_egm": np.asarray([1.0]),
        "time_egm": np.asarray([0.1]),
        "rel_err_egm": np.asarray([1e-2]),
    }
    for raw_name in ["apdb", "rapdb", "rapdb_ada"]:
        for mode_idx in [0, 1]:
            payload[f"iter_{raw_name}_mu2_{mode_idx}"] = np.asarray([1.0, 2.0])
            payload[f"time_{raw_name}_mu2_{mode_idx}"] = np.asarray([0.5, 1.5])
            payload[f"rel_err_{raw_name}_mu2_{mode_idx}"] = np.asarray([1e-2, 1e-4])
    payload["restart_iter_rapdb_mu2_0"] = np.asarray([2.0])
    payload["restart_time_rapdb_mu2_0"] = np.asarray([1.5])
    savemat(mat_path, payload)

    curves = load_cpu_curves_mat(mat_path)
    curve = curves["rAPDB-yx|mono|mu=2"]

    np.testing.assert_allclose(curve["restart_iter"], [2.0])
    np.testing.assert_allclose(curve["restart_time"], [1.5])
