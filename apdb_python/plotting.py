from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
):
    lengths = [len(o) for o in oracle_list if len(o)]
    maxima = [np.max(o) for o in oracle_list if len(o)]
    if not lengths or not maxima:
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
    return line


def _style_config(plot_style: str) -> dict:
    if plot_style == "strict":
        return {
            "linewidth": 2.2,
            "markersize": 12.0,
            "nonmono_markers": [("o", 500, 0), ("^", 700, 0), ("+", 800, 0)],
            "infeas_legend_last": "rAPDB",
            "axis_fontsize": 16,
            "axis_linewidth": 1.2,
            "label_fontsize": 16,
            "legend_fontsize": 11,
            "legend_loc": "upper right",
            "grid": True,
            "x_label": "Number of gradient computations",
            "subopt_label": r"$|f(x^k)-f^\star|/(1+|f^\star|)$",
            "infeas_label": r"$\frac{1}{m}\sum_{i=1}^m \max\{g_i(x^k),0\}$",
        }
    if plot_style == "custom":
        return {
            "linewidth": 2.5,
            "markersize": 11.0,
            "nonmono_markers": [("o", 300, 0), ("^", 500, 140), ("X", 700, 280)],
            "infeas_legend_last": "rAPDB-xy",
            "axis_fontsize": 10,
            "axis_linewidth": 1.0,
            "label_fontsize": 10,
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
    c7 = (0.635, 0.078, 0.184)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=140)

    mode_idx = 0
    ax = axs[0, 0]
    p1 = _plot_main(
        ax, results["oracle_egm"], results["rel_subopt_egm"], c4, "--", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p2 = _plot_main(
        ax, results["oracle_1"][mode_idx], results["rel_subopt_1"][mode_idx], c1, "-", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p3 = _plot_main(
        ax, results["oracle_1_ada"][mode_idx], results["rel_subopt_1_ada"][mode_idx], c5, ":", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p4 = _plot_main(
        ax, results["oracle_r1"][mode_idx], results["rel_subopt_r1"][mode_idx], c7, ":", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p5 = _plot_main(
        ax, results["oracle_xy_1"][mode_idx], results["rel_subopt_xy_1"][mode_idx], c1, "--", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p6 = _plot_main(
        ax, results["oracle_xy_ada1"][mode_idx], results["rel_subopt_xy_ada1"][mode_idx], c5, "-", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p7 = _plot_main(
        ax, results["oracle_xy_r1"][mode_idx], results["rel_subopt_xy_r1"][mode_idx], c7, "-", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    _apply_axis_style(ax, style_cfg)
    ax.set_xlabel(style_cfg["x_label"], fontsize=style_cfg["label_fontsize"])
    ax.set_ylabel(style_cfg["subopt_label"], fontsize=style_cfg["label_fontsize"])
    ax.legend(
        [p1, p2, p3, p4, p5, p6, p7],
        ["EGM", "APDB-yx", "rAPDB-yx-ada", "rAPDB-yx", "APDB-xy", "rAPDB-xy-ada", "rAPDB-xy"],
        fontsize=style_cfg["legend_fontsize"],
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
        "-",
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p3 = _plot_main(
        ax,
        results["oracle_1_ada"][mode_idx],
        results["rel_infeas_1_ada"][mode_idx],
        c5,
        ":",
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p4 = _plot_main(
        ax,
        results["oracle_r1"][mode_idx],
        results["rel_infeas_r1"][mode_idx],
        c7,
        ":",
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p5 = _plot_main(
        ax,
        results["oracle_xy_1"][mode_idx],
        results["rel_infeas_xy_1"][mode_idx],
        c1,
        "--",
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p6 = _plot_main(
        ax,
        results["oracle_xy_ada1"][mode_idx],
        results["rel_infeas_xy_ada1"][mode_idx],
        c5,
        "-",
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p7 = _plot_main(
        ax,
        results["oracle_xy_r1"][mode_idx],
        results["rel_infeas_xy_r1"][mode_idx],
        c7,
        "-",
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    _apply_axis_style(ax, style_cfg)
    ax.set_xlabel(style_cfg["x_label"], fontsize=style_cfg["label_fontsize"])
    ax.set_ylabel(style_cfg["infeas_label"], fontsize=style_cfg["label_fontsize"])
    ax.legend(
        [p1, p2, p3, p4, p5, p6, p7],
        ["EGM", "APDB-yx", "rAPDB-yx-ada", "rAPDB-yx", "APDB-xy", "rAPDB-xy-ada", style_cfg["infeas_legend_last"]],
        fontsize=style_cfg["legend_fontsize"],
        loc=style_cfg["legend_loc"],
    )

    mode_idx = 1
    ax = axs[1, 0]
    m5, g5, o5 = style_cfg["nonmono_markers"][0]
    m6, g6, o6 = style_cfg["nonmono_markers"][1]
    m7, g7, o7 = style_cfg["nonmono_markers"][2]
    p2 = _plot_main(
        ax, results["oracle_1"][mode_idx], results["rel_subopt_1"][mode_idx], c1, "-", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p3 = _plot_main(
        ax, results["oracle_1_ada"][mode_idx], results["rel_subopt_1_ada"][mode_idx], c5, ":", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p4 = _plot_main(
        ax, results["oracle_r1"][mode_idx], results["rel_subopt_r1"][mode_idx], c7, ":", numsim, linewidth=style_cfg["linewidth"], markersize=style_cfg["markersize"]
    )
    p5 = _plot_main(
        ax,
        results["oracle_xy_1"][mode_idx],
        results["rel_subopt_xy_1"][mode_idx],
        c1,
        "--",
        numsim,
        marker=m5,
        gap=g5,
        marker_offset=o5,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
    )
    p6 = _plot_main(
        ax,
        results["oracle_xy_ada1"][mode_idx],
        results["rel_subopt_xy_ada1"][mode_idx],
        c5,
        "-",
        numsim,
        marker=m6,
        gap=g6,
        marker_offset=o6,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
    )
    p7 = _plot_main(
        ax,
        results["oracle_xy_r1"][mode_idx],
        results["rel_subopt_xy_r1"][mode_idx],
        c7,
        "-",
        numsim,
        marker=m7,
        gap=g7,
        marker_offset=o7,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
    )
    _apply_axis_style(ax, style_cfg)
    ax.set_xlabel(style_cfg["x_label"], fontsize=style_cfg["label_fontsize"])
    ax.set_ylabel(style_cfg["subopt_label"], fontsize=style_cfg["label_fontsize"])
    ax.legend(
        [p2, p3, p4, p5, p6, p7],
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
        "-",
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p3 = _plot_main(
        ax,
        results["oracle_1_ada"][mode_idx],
        results["rel_infeas_1_ada"][mode_idx],
        c5,
        ":",
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p4 = _plot_main(
        ax,
        results["oracle_r1"][mode_idx],
        results["rel_infeas_r1"][mode_idx],
        c7,
        ":",
        numsim,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p5 = _plot_main(
        ax,
        results["oracle_xy_1"][mode_idx],
        results["rel_infeas_xy_1"][mode_idx],
        c1,
        "--",
        numsim,
        marker=m5,
        gap=g5,
        marker_offset=o5,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p6 = _plot_main(
        ax,
        results["oracle_xy_ada1"][mode_idx],
        results["rel_infeas_xy_ada1"][mode_idx],
        c5,
        "-",
        numsim,
        marker=m6,
        gap=g6,
        marker_offset=o6,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    p7 = _plot_main(
        ax,
        results["oracle_xy_r1"][mode_idx],
        results["rel_infeas_xy_r1"][mode_idx],
        c7,
        "-",
        numsim,
        marker=m7,
        gap=g7,
        marker_offset=o7,
        linewidth=style_cfg["linewidth"],
        markersize=style_cfg["markersize"],
        mask_nonpositive=True,
    )
    _apply_axis_style(ax, style_cfg)
    ax.set_xlabel(style_cfg["x_label"], fontsize=style_cfg["label_fontsize"])
    ax.set_ylabel(style_cfg["infeas_label"], fontsize=style_cfg["label_fontsize"])
    ax.legend(
        [p2, p3, p4, p5, p6, p7],
        ["APDB-yx", "rAPDB-yx-ada", "rAPDB-yx", "APDB-xy", "rAPDB-xy-ada", style_cfg["infeas_legend_last"]],
        fontsize=style_cfg["legend_fontsize"],
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
        plt.close(fig)
        out_paths.append(fig_path)
    return out_paths
