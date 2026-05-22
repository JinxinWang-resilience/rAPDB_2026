from apdb_real_data_python.plot_cpu import (
    AXIS_FONTSIZE,
    CPU_TIME_X_LABEL,
    ITERATION_X_LIMIT,
    LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    MARKER_SIZE,
    REL_ERROR_Y_LABEL,
    TIME_X_LIMIT,
    _legend_label,
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
    assert all(s["marker"] is not None for s in mono_specs)
    assert all(s["marker"] is None for s in nonmono_specs)
    assert [s for s in mono_specs if s["alg"] == "rAPDB-yx"][0]["marker"] == "x"
    assert {s["mu"] for s in specs if s["alg"] != "EGM"} == {2.0}
    assert all(float(s.get("alpha", 1.0)) >= 0.9 for s in specs)


def test_plot_limits_and_marker_size_match_requested_style():
    assert MARKER_SIZE > 6
    assert ITERATION_X_LIMIT == (0, 8000)
    assert TIME_X_LIMIT == (0, 600)
    assert CPU_TIME_X_LABEL == "CPU time (s)"


def test_text_sizes_match_strict_comparison_plot_style():
    assert AXIS_FONTSIZE == 16
    assert LABEL_FONTSIZE == 16
    assert LEGEND_FONTSIZE == 11


def test_relative_error_label_uses_x_star_not_asterisk():
    assert REL_ERROR_Y_LABEL == r"$\|x^k-x^\star\|/(1+\|x^\star\|)$"
