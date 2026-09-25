from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import interp1d


def _plot_main(
    ax: plt.Axes,
    oracle_list: list[np.ndarray],
    rel_list: list[np.ndarray],
    color: tuple[float, float, float],
    linestyle: str,
    numsim: int,
    marker: str | None = None,
    gap: int = 0,
    marker_offset: int = 0,
    linewidth: float = 2.5,
    markersize: float = 11.0,
    mask_nonpositive: bool = False,
    return_curve: bool = False,
):
    lengths = [len(o) for o in oracle_list if len(o)]
    maxima = [np.max(o) for o in oracle_list if len(o)]
    if not lengths or not maxima:
        if return_curve:
            return None, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        return None

    x_axis = np.linspace(1.0, float(max(maxima)), int(max(lengths)))
    interped = []
    ind_f = []

    for sim in range(numsim):
        o = np.asarray(oracle_list[sim], dtype=np.float64)
        r = np.asarray(rel_list[sim], dtype=np.float64)
        if o.size == 0 or r.size == 0:
            interped.append(np.full_like(x_axis, np.nan))
            ind_f.append(len(x_axis) - 1)
            continue

        f = interp1d(o, r, kind="nearest", fill_value="extrapolate", bounds_error=False)
        values = np.asarray(f(x_axis), dtype=np.float64)
        search_start = min(40, len(values) - 1)
        pos_idx = np.where(values[search_start:] > 0)[0]
        idx = search_start + int(pos_idx[0]) if pos_idx.size else len(values) - 1
        ind_f.append(idx)
        interped.append(values)

    ind_f_min = min(ind_f)
    for sim in range(numsim):
        idx = ind_f[sim]
        interped[sim][ind_f_min : idx + 1] = interped[sim][idx]

    y = np.nanmean(np.stack(interped, axis=1), axis=1)
    if mask_nonpositive:
        y = np.where(y > 0.0, y, np.nan)

    kwargs = {
        "color": color,
        "linewidth": linewidth,
        "linestyle": linestyle,
    }
    if marker is not None and gap > 0:
        kwargs["marker"] = marker
        kwargs["markersize"] = markersize
        kwargs["markevery"] = list(range(marker_offset, len(x_axis), gap))

    (line,) = ax.semilogy(x_axis, y, **kwargs)
    if return_curve:
        return line, x_axis, y
    return line


def _line_style_for_algorithm(algorithm: str, plot_style: str) -> str:
    if plot_style == "strict":
        if "-yx" in algorithm:
            return "--"
        if "-xy" in algorithm:
            return "-"

    defaults = {
        "APDB-yx": "-",
        "rAPDB-yx-ada": ":",
        "rAPDB-yx": ":",
        "APDB-xy": "--",
        "rAPDB-xy-ada": "-",
        "rAPDB-xy": "-",
    }
    return defaults[algorithm]


def _fixed_restart_period(algorithm: str, mode_idx: int) -> int | None:
    if algorithm == "rAPDB-yx":
        return 800 if mode_idx == 0 else 400
    if algorithm == "rAPDB-xy":
        return 2000 if mode_idx == 0 else 1000
    return None


def _fixed_restart_marker_points(
    oracle_list: list[np.ndarray],
    iter_list: list[np.ndarray],
    restart_period: int,
    numsim: int,
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    mask_nonpositive: bool = False,
    log_y: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    curve_x = np.asarray(curve_x, dtype=np.float64).reshape(-1)
    curve_y = np.asarray(curve_y, dtype=np.float64).reshape(-1)
    curve_n = min(curve_x.size, curve_y.size)
    if curve_n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    curve_x = curve_x[:curve_n]
    curve_y = curve_y[:curve_n]
    curve_mask = np.isfinite(curve_x) & np.isfinite(curve_y)
    if log_y:
        curve_mask &= curve_y > 0.0
    if not np.any(curve_mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start: int | None = None
    for idx, is_valid in enumerate(curve_mask):
        if is_valid and start is None:
            start = idx
        elif not is_valid and start is not None:
            if idx - start > 0:
                segments.append((curve_x[start:idx], curve_y[start:idx]))
            start = None
    if start is not None:
        segments.append((curve_x[start:], curve_y[start:]))
    if not segments:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    max_iters = []
    for sim in range(numsim):
        iters = np.asarray(iter_list[sim], dtype=np.float64).reshape(-1)
        finite_iters = iters[np.isfinite(iters)]
        if finite_iters.size:
            max_iters.append(float(np.max(finite_iters)))
    if not max_iters:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    x_points: list[float] = []
    y_points: list[float] = []
    max_restart_iter = int(np.floor(max(max_iters) / restart_period) * restart_period)

    for restart_iter in range(restart_period, max_restart_iter + 1, restart_period):
        restart_oracles: list[float] = []
        for sim in range(numsim):
            oracle = np.asarray(oracle_list[sim], dtype=np.float64).reshape(-1)
            iters = np.asarray(iter_list[sim], dtype=np.float64).reshape(-1)
            n = min(oracle.size, iters.size)
            if n == 0:
                continue

            oracle = oracle[:n]
            iters = iters[:n]
            restart_idx = np.where(np.isclose(iters, float(restart_iter)))[0]
            if restart_idx.size == 0:
                continue

            x_val = float(oracle[int(restart_idx[0])])
            if np.isfinite(x_val):
                restart_oracles.append(x_val)

        if not restart_oracles:
            continue

        x_mean = float(np.mean(restart_oracles))
        y_on_curve = np.nan
        for segment_x, segment_y in segments:
            order = np.argsort(segment_x)
            segment_x = segment_x[order]
            segment_y = segment_y[order]
            if x_mean < segment_x[0] or x_mean > segment_x[-1]:
                continue
            if log_y:
                y_on_curve = float(np.exp(np.interp(x_mean, segment_x, np.log(segment_y))))
            else:
                y_on_curve = float(np.interp(x_mean, segment_x, segment_y))
            break

        if not np.isfinite(y_on_curve):
            continue
        if not np.isfinite(y_on_curve) or (mask_nonpositive and y_on_curve <= 0.0):
            continue

        x_points.append(x_mean)
        y_points.append(y_on_curve)

    return np.asarray(x_points, dtype=np.float64), np.asarray(y_points, dtype=np.float64)


def _curve_segments(
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    log_y: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    curve_x = np.asarray(curve_x, dtype=np.float64).reshape(-1)
    curve_y = np.asarray(curve_y, dtype=np.float64).reshape(-1)
    curve_n = min(curve_x.size, curve_y.size)
    if curve_n == 0:
        return []

    curve_x = curve_x[:curve_n]
    curve_y = curve_y[:curve_n]
    curve_mask = np.isfinite(curve_x) & np.isfinite(curve_y)
    if log_y:
        curve_mask &= curve_y > 0.0

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start: int | None = None
    for idx, is_valid in enumerate(curve_mask):
        if is_valid and start is None:
            start = idx
        elif not is_valid and start is not None:
            if idx - start > 0:
                segments.append((curve_x[start:idx], curve_y[start:idx]))
            start = None
    if start is not None:
        segments.append((curve_x[start:], curve_y[start:]))
    return segments


def _interp_on_curve_segments(
    x_val: float,
    segments: list[tuple[np.ndarray, np.ndarray]],
    log_y: bool = False,
) -> float:
    for segment_x, segment_y in segments:
        order = np.argsort(segment_x)
        segment_x = segment_x[order]
        segment_y = segment_y[order]
        if x_val < segment_x[0] or x_val > segment_x[-1]:
            continue
        if log_y:
            return float(np.exp(np.interp(x_val, segment_x, np.log(segment_y))))
        return float(np.interp(x_val, segment_x, segment_y))
    return float("nan")


def _restart_event_marker_points(
    event_oracle_list: list[np.ndarray],
    numsim: int,
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    mask_nonpositive: bool = False,
    log_y: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    segments = _curve_segments(curve_x, curve_y, log_y=log_y)
    if not segments:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    max_events = 0
    for sim in range(numsim):
        events = np.asarray(event_oracle_list[sim], dtype=np.float64).reshape(-1)
        max_events = max(max_events, int(events.size))
    if max_events == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    x_points: list[float] = []
    y_points: list[float] = []
    for event_idx in range(max_events):
        event_oracles: list[float] = []
        for sim in range(numsim):
            events = np.asarray(event_oracle_list[sim], dtype=np.float64).reshape(-1)
            if event_idx >= events.size:
                continue
            x_val = float(events[event_idx])
            if np.isfinite(x_val):
                event_oracles.append(x_val)
        if not event_oracles:
            continue

        x_mean = float(np.mean(event_oracles))
        y_on_curve = _interp_on_curve_segments(x_mean, segments, log_y=log_y)
        if not np.isfinite(y_on_curve) or (mask_nonpositive and y_on_curve <= 0.0):
            continue
        x_points.append(x_mean)
        y_points.append(y_on_curve)

    return np.asarray(x_points, dtype=np.float64), np.asarray(y_points, dtype=np.float64)


def _plot_fixed_restart_markers(
    ax: plt.Axes,
    oracle_list: list[np.ndarray],
    iter_list: list[np.ndarray],
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    color: tuple[float, float, float],
    restart_period: int | None,
    numsim: int,
    markersize: float,
    mask_nonpositive: bool = False,
    log_y: bool = False,
) -> Line2D | None:
    if restart_period is None:
        return None

    x, y = _fixed_restart_marker_points(
        oracle_list=oracle_list,
        iter_list=iter_list,
        restart_period=restart_period,
        numsim=numsim,
        curve_x=curve_x,
        curve_y=curve_y,
        mask_nonpositive=mask_nonpositive,
        log_y=log_y,
    )
    if x.size == 0:
        return None

    (line,) = ax.semilogy(
        x,
        y,
        linestyle="None",
        marker="x",
        color=color,
        markersize=max(6.0, markersize * 0.65),
        markeredgewidth=1.4,
    )
    return line


def _plot_restart_event_markers(
    ax: plt.Axes,
    event_oracle_list: list[np.ndarray] | None,
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    color: tuple[float, float, float],
    numsim: int,
    markersize: float,
    mask_nonpositive: bool = False,
    log_y: bool = False,
) -> Line2D | None:
    if event_oracle_list is None:
        return None

    x, y = _restart_event_marker_points(
        event_oracle_list=event_oracle_list,
        numsim=numsim,
        curve_x=curve_x,
        curve_y=curve_y,
        mask_nonpositive=mask_nonpositive,
        log_y=log_y,
    )
    if x.size == 0:
        return None

    (line,) = ax.semilogy(
        x,
        y,
        linestyle="None",
        marker="x",
        color=color,
        markersize=max(6.0, markersize * 0.65),
        markeredgewidth=1.4,
    )
    return line


def _legend_handle_with_marker(line: Line2D, marker_line: Line2D | None) -> Line2D:
    if marker_line is None:
        return line
    return Line2D(
        [],
        [],
        color=line.get_color(),
        linewidth=line.get_linewidth(),
        linestyle=line.get_linestyle(),
        marker=marker_line.get_marker(),
        markersize=marker_line.get_markersize(),
        markeredgewidth=marker_line.get_markeredgewidth(),
    )


def _style_config(plot_style: str) -> dict:
    if plot_style == "strict":
        return {
            "name": plot_style,
            "linewidth": 2.2,
            "markersize": 12.0,
            "nonmono_markers": [("o", 500, 0), ("^", 700, 0), ("x", 800, 0)],
            "infeas_legend_last": "rAPDB-xy",
            "restart_color": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
            "axis_fontsize": 16,
            "axis_linewidth": 1.2,
            "label_fontsize": 16,
            "top_legend_fontsize": 14,
            "bottom_right_legend_fontsize": 14,
            "legend_fontsize": 11,
            "legend_loc": "upper right",
            "grid": True,
            "x_label": "Number of gradient computations",
            "subopt_label": r"$|f(x^k)-f^\star|/(1+|f^\star|)$",
            "infeas_label": r"$\frac{1}{m}\sum_{i=1}^m \max\{g_i(x^k),0\}$",
        }
    if plot_style == "custom":
        return {
            "name": plot_style,
            "linewidth": 2.5,
            "markersize": 11.0,
            "infeas_legend_last": "rAPDB-xy",
            "restart_color": (0.635, 0.078, 0.184),
            "axis_fontsize": 10,
            "axis_linewidth": 1.0,
            "label_fontsize": 10,
            "top_legend_fontsize": 9,
            "bottom_right_legend_fontsize": 9,
            "legend_fontsize": 9,
            "legend_loc": "best",
            "grid": True,
            "x_label": "Number of gradient computations",
            "subopt_label": "|f(x^k)-f*|/(1+|f*|)",
            "infeas_label": "mean(max(g_i(x^k), 0))",
        }
    raise ValueError(f"Unknown plot_style: {plot_style}")


def _apply_axis_style(ax: plt.Axes, style_cfg: dict) -> None:
    ax.grid(bool(style_cfg.get("grid", True)))
    ax.tick_params(labelsize=float(style_cfg.get("axis_fontsize", 10)))
    axis_lw = float(style_cfg.get("axis_linewidth", 1.0))
    for spine in ax.spines.values():
        spine.set_linewidth(axis_lw)


def _plot_one_style(results: dict, numsim: int, style_cfg: dict) -> plt.Figure:
    c1 = (0.000, 0.447, 0.741)
    c4 = (0.494, 0.184, 0.556)
    c5 = (0.466, 0.674, 0.188)
    c7 = style_cfg["restart_color"]
    plot_style = style_cfg["name"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=140)

    mode_idx = 0
    ax = axs[0, 0]
    p1 = _plot_main(
        ax, results["oracle_egm"], results["rel_subopt_egm"], c4, "--", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p2 = _plot_main(
        ax, results["oracle_1"][mode_idx], results["rel_subopt_1"][mode_idx], c1, _line_style_for_algorithm("APDB-yx", plot_style), numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p3, p3_x, p3_y = _plot_main(
        ax, results["oracle_1_ada"][mode_idx], results["rel_subopt_1_ada"][mode_idx], c5, _line_style_for_algorithm("rAPDB-yx-ada", plot_style), numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"], return_curve=True
    )
    p4, p4_x, p4_y = _plot_main(
        ax, results["oracle_r1"][mode_idx], results["rel_subopt_r1"][mode_idx], c7, _line_style_for_algorithm("rAPDB-yx", plot_style), numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"], return_curve=True
    )
    p5 = _plot_main(
        ax, results["oracle_xy_1"][mode_idx], results["rel_subopt_xy_1"][mode_idx], c1, _line_style_for_algorithm("APDB-xy", plot_style), numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p6, p6_x, p6_y = _plot_main(
        ax, results["oracle_xy_ada1"][mode_idx], results["rel_subopt_xy_ada1"][mode_idx], c5, _line_style_for_algorithm("rAPDB-xy-ada", plot_style), numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"], return_curve=True
    )
    p7, p7_x, p7_y = _plot_main(
        ax, results["oracle_xy_r1"][mode_idx], results["rel_subopt_xy_r1"][mode_idx], c7, _line_style_for_algorithm("rAPDB-xy", plot_style), numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"], return_curve=True
    )
    p4_marker = _plot_fixed_restart_markers(
        ax, results["oracle_r1"][mode_idx], results["iter_epoch_r1"][mode_idx], p4_x, p4_y, c7, _fixed_restart_period("rAPDB-yx", mode_idx), numsim, style_cfg["markersize"], log_y=True
    )
    p3_marker = _plot_restart_event_markers(
        ax, results.get("restart_oracle_1_ada", [None, None])[mode_idx], p3_x, p3_y, c5, numsim, style_cfg["markersize"], log_y=True
    )
    p7_marker = _plot_fixed_restart_markers(
        ax, results["oracle_xy_r1"][mode_idx], results["iter_epoch_xy_r1"][mode_idx], p7_x, p7_y, c7, _fixed_restart_period("rAPDB-xy", mode_idx), numsim, style_cfg["markersize"], log_y=True
    )
    p6_marker = _plot_restart_event_markers(
        ax, results.get("restart_oracle_xy_ada1", [None, None])[mode_idx], p6_x, p6_y, c5, numsim, style_cfg["markersize"], log_y=True
    )
    _apply_axis_style(ax, style_cfg)
    ax.set_xlabel(style_cfg["x_label"], fontsize=style_cfg["label_fontsize"])
    ax.set_ylabel(style_cfg["subopt_label"], fontsize=style_cfg["label_fontsize"])
    ax.legend(
        [p1, p2, _legend_handle_with_marker(p3, p3_marker), _legend_handle_with_marker(p4, p4_marker), p5, _legend_handle_with_marker(p6, p6_marker), _legend_handle_with_marker(p7, p7_marker)],
        ["EGM", "APDB-yx", "rAPDB-yx-ada", "rAPDB-yx", "APDB-xy", "rAPDB-xy-ada", "rAPDB-xy"],
        fontsize=style_cfg["top_legend_fontsize"],
        loc=style_cfg["legend_loc"],
    )

    ax = axs[0, 1]
    p1 = _plot_main(
        ax,
        results["oracle_egm"],
        results["rel_infeas_err_egm"],
        c4,
        "--",
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p2 = _plot_main(
        ax,
        results["oracle_1"][mode_idx],
        results["rel_infeas_1"][mode_idx],
        c1,
        _line_style_for_algorithm("APDB-yx", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p3, p3_x, p3_y = _plot_main(
        ax,
        results["oracle_1_ada"][mode_idx],
        results["rel_infeas_1_ada"][mode_idx],
        c5,
        _line_style_for_algorithm("rAPDB-yx-ada", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
        return_curve=True,
    )
    p4, p4_x, p4_y = _plot_main(
        ax,
        results["oracle_r1"][mode_idx],
        results["rel_infeas_r1"][mode_idx],
        c7,
        _line_style_for_algorithm("rAPDB-yx", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
        return_curve=True,
    )
    p5 = _plot_main(
        ax,
        results["oracle_xy_1"][mode_idx],
        results["rel_infeas_xy_1"][mode_idx],
        c1,
        _line_style_for_algorithm("APDB-xy", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p6, p6_x, p6_y = _plot_main(
        ax,
        results["oracle_xy_ada1"][mode_idx],
        results["rel_infeas_xy_ada1"][mode_idx],
        c5,
        _line_style_for_algorithm("rAPDB-xy-ada", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
        return_curve=True,
    )
    p7, p7_x, p7_y = _plot_main(
        ax,
        results["oracle_xy_r1"][mode_idx],
        results["rel_infeas_xy_r1"][mode_idx],
        c7,
        _line_style_for_algorithm("rAPDB-xy", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
        return_curve=True,
    )
    p4_marker = _plot_fixed_restart_markers(
        ax, results["oracle_r1"][mode_idx], results["iter_epoch_r1"][mode_idx], p4_x, p4_y, c7, _fixed_restart_period("rAPDB-yx", mode_idx), numsim, style_cfg["markersize"], mask_nonpositive=True, log_y=True
    )
    p3_marker = _plot_restart_event_markers(
        ax, results.get("restart_oracle_1_ada", [None, None])[mode_idx], p3_x, p3_y, c5, numsim, style_cfg["markersize"], mask_nonpositive=True, log_y=True
    )
    p7_marker = _plot_fixed_restart_markers(
        ax, results["oracle_xy_r1"][mode_idx], results["iter_epoch_xy_r1"][mode_idx], p7_x, p7_y, c7, _fixed_restart_period("rAPDB-xy", mode_idx), numsim, style_cfg["markersize"], mask_nonpositive=True, log_y=True
    )
    p6_marker = _plot_restart_event_markers(
        ax, results.get("restart_oracle_xy_ada1", [None, None])[mode_idx], p6_x, p6_y, c5, numsim, style_cfg["markersize"], mask_nonpositive=True, log_y=True
    )
    _apply_axis_style(ax, style_cfg)
    ax.set_xlabel(style_cfg["x_label"], fontsize=style_cfg["label_fontsize"])
    ax.set_ylabel(style_cfg["infeas_label"], fontsize=style_cfg["label_fontsize"])
    ax.legend(
        [p1, p2, _legend_handle_with_marker(p3, p3_marker), _legend_handle_with_marker(p4, p4_marker), p5, _legend_handle_with_marker(p6, p6_marker), _legend_handle_with_marker(p7, p7_marker)],
        ["EGM", "APDB-yx", "rAPDB-yx-ada", "rAPDB-yx", "APDB-xy", "rAPDB-xy-ada", style_cfg["infeas_legend_last"]],
        fontsize=style_cfg["top_legend_fontsize"],
        loc=style_cfg["legend_loc"],
    )

    mode_idx = 1
    ax = axs[1, 0]
    p2 = _plot_main(
        ax, results["oracle_1"][mode_idx], results["rel_subopt_1"][mode_idx], c1, _line_style_for_algorithm("APDB-yx", plot_style), numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p3, p3_x, p3_y = _plot_main(
        ax, results["oracle_1_ada"][mode_idx], results["rel_subopt_1_ada"][mode_idx], c5, _line_style_for_algorithm("rAPDB-yx-ada", plot_style), numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"], return_curve=True
    )
    p4, p4_x, p4_y = _plot_main(
        ax, results["oracle_r1"][mode_idx], results["rel_subopt_r1"][mode_idx], c7, _line_style_for_algorithm("rAPDB-yx", plot_style), numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"], return_curve=True
    )
    p5 = _plot_main(
        ax,
        results["oracle_xy_1"][mode_idx],
        results["rel_subopt_xy_1"][mode_idx],
        c1,
        _line_style_for_algorithm("APDB-xy", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
    )
    p6, p6_x, p6_y = _plot_main(
        ax,
        results["oracle_xy_ada1"][mode_idx],
        results["rel_subopt_xy_ada1"][mode_idx],
        c5,
        _line_style_for_algorithm("rAPDB-xy-ada", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        return_curve=True,
    )
    p7, p7_x, p7_y = _plot_main(
        ax,
        results["oracle_xy_r1"][mode_idx],
        results["rel_subopt_xy_r1"][mode_idx],
        c7,
        _line_style_for_algorithm("rAPDB-xy", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        return_curve=True,
    )
    p4_marker = _plot_fixed_restart_markers(
        ax, results["oracle_r1"][mode_idx], results["iter_epoch_r1"][mode_idx], p4_x, p4_y, c7, _fixed_restart_period("rAPDB-yx", mode_idx), numsim, style_cfg["markersize"], log_y=True
    )
    p3_marker = _plot_restart_event_markers(
        ax, results.get("restart_oracle_1_ada", [None, None])[mode_idx], p3_x, p3_y, c5, numsim, style_cfg["markersize"], log_y=True
    )
    p7_marker = _plot_fixed_restart_markers(
        ax, results["oracle_xy_r1"][mode_idx], results["iter_epoch_xy_r1"][mode_idx], p7_x, p7_y, c7, _fixed_restart_period("rAPDB-xy", mode_idx), numsim, style_cfg["markersize"], log_y=True
    )
    p6_marker = _plot_restart_event_markers(
        ax, results.get("restart_oracle_xy_ada1", [None, None])[mode_idx], p6_x, p6_y, c5, numsim, style_cfg["markersize"], log_y=True
    )
    _apply_axis_style(ax, style_cfg)
    ax.set_xlabel(style_cfg["x_label"], fontsize=style_cfg["label_fontsize"])
    ax.set_ylabel(style_cfg["subopt_label"], fontsize=style_cfg["label_fontsize"])
    ax.legend(
        [p2, _legend_handle_with_marker(p3, p3_marker), _legend_handle_with_marker(p4, p4_marker), p5, _legend_handle_with_marker(p6, p6_marker), _legend_handle_with_marker(p7, p7_marker)],
        ["APDB-yx", "rAPDB-yx-ada", "rAPDB-yx", "APDB-xy", "rAPDB-xy-ada", "rAPDB-xy"],
        fontsize=style_cfg["legend_fontsize"],
        loc=style_cfg["legend_loc"],
    )

    ax = axs[1, 1]
    p2 = _plot_main(
        ax,
        results["oracle_1"][mode_idx],
        results["rel_infeas_1"][mode_idx],
        c1,
        _line_style_for_algorithm("APDB-yx", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p3, p3_x, p3_y = _plot_main(
        ax,
        results["oracle_1_ada"][mode_idx],
        results["rel_infeas_1_ada"][mode_idx],
        c5,
        _line_style_for_algorithm("rAPDB-yx-ada", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
        return_curve=True,
    )
    p4, p4_x, p4_y = _plot_main(
        ax,
        results["oracle_r1"][mode_idx],
        results["rel_infeas_r1"][mode_idx],
        c7,
        _line_style_for_algorithm("rAPDB-yx", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
        return_curve=True,
    )
    p5 = _plot_main(
        ax,
        results["oracle_xy_1"][mode_idx],
        results["rel_infeas_xy_1"][mode_idx],
        c1,
        _line_style_for_algorithm("APDB-xy", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p6, p6_x, p6_y = _plot_main(
        ax,
        results["oracle_xy_ada1"][mode_idx],
        results["rel_infeas_xy_ada1"][mode_idx],
        c5,
        _line_style_for_algorithm("rAPDB-xy-ada", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
        return_curve=True,
    )
    p7, p7_x, p7_y = _plot_main(
        ax,
        results["oracle_xy_r1"][mode_idx],
        results["rel_infeas_xy_r1"][mode_idx],
        c7,
        _line_style_for_algorithm("rAPDB-xy", plot_style),
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
        return_curve=True,
    )
    p4_marker = _plot_fixed_restart_markers(
        ax, results["oracle_r1"][mode_idx], results["iter_epoch_r1"][mode_idx], p4_x, p4_y, c7, _fixed_restart_period("rAPDB-yx", mode_idx), numsim, style_cfg["markersize"], mask_nonpositive=True, log_y=True
    )
    p3_marker = _plot_restart_event_markers(
        ax, results.get("restart_oracle_1_ada", [None, None])[mode_idx], p3_x, p3_y, c5, numsim, style_cfg["markersize"], mask_nonpositive=True, log_y=True
    )
    p7_marker = _plot_fixed_restart_markers(
        ax, results["oracle_xy_r1"][mode_idx], results["iter_epoch_xy_r1"][mode_idx], p7_x, p7_y, c7, _fixed_restart_period("rAPDB-xy", mode_idx), numsim, style_cfg["markersize"], mask_nonpositive=True, log_y=True
    )
    p6_marker = _plot_restart_event_markers(
        ax, results.get("restart_oracle_xy_ada1", [None, None])[mode_idx], p6_x, p6_y, c5, numsim, style_cfg["markersize"], mask_nonpositive=True, log_y=True
    )
    _apply_axis_style(ax, style_cfg)
    ax.set_xlabel(style_cfg["x_label"], fontsize=style_cfg["label_fontsize"])
    ax.set_ylabel(style_cfg["infeas_label"], fontsize=style_cfg["label_fontsize"])
    if plot_style == "strict":
        ax.set_ylim(axs[1, 0].get_ylim())
    ax.legend(
        [p2, _legend_handle_with_marker(p3, p3_marker), _legend_handle_with_marker(p4, p4_marker), p5, _legend_handle_with_marker(p6, p6_marker), _legend_handle_with_marker(p7, p7_marker)],
        ["APDB-yx", "rAPDB-yx-ada", "rAPDB-yx", "APDB-xy", "rAPDB-xy-ada", style_cfg["infeas_legend_last"]],
        fontsize=style_cfg["bottom_right_legend_fontsize"],
        loc=style_cfg["legend_loc"],
    )

    fig.tight_layout()
    return fig


def plot_results(
    results: dict,
    numsim: int,
    n: int,
    m: int,
    device: str,
    outdir: str | Path,
    plot_style: str = "custom",
) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if plot_style not in {"strict", "custom", "both"}:
        raise ValueError("plot_style must be one of: strict, custom, both")

    if plot_style == "both":
        styles = ["strict", "custom"]
        use_suffix = True
    else:
        styles = [plot_style]
        use_suffix = False

    out_paths: list[Path] = []
    for style in styles:
        style_cfg = _style_config(style)
        fig = _plot_one_style(results, numsim, style_cfg)
        stem = f"result_apdb_n{n}_m{m}_numsim{numsim}_{device}"
        if use_suffix:
            fig_path = outdir / f"{stem}_{style}.png"
        else:
            fig_path = outdir / f"{stem}.png"
        fig.savefig(fig_path)
        out_paths.append(fig_path)
        if style == "strict":
            pdf_path = fig_path.with_suffix(".pdf")
            fig.savefig(pdf_path)
            out_paths.append(pdf_path)
        plt.close(fig)
    return out_paths
