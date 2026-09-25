from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


MARKER_SIZE = 9
ITERATION_X_LIMIT = (0, 8000)
TIME_X_LIMIT = (0, 600)
GPU_TIME_X_LIMIT = (0, 200)
CPU_TIME_X_LABEL = "CPU time (s)"
GPU_TIME_X_LABEL = "GPU time (s)"
REL_ERROR_Y_LABEL = r"$\|x^k-x^\star\|/(1+\|x^\star\|)$"
AXIS_FONTSIZE = 16
LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 13
LEGEND_FRAME_ALPHA = 0.55


def _legend_label(alg: str, mode: str, mu: float | None = None) -> str:
    if alg == "EGM":
        return "EGM"
    display_mode = "non-mono" if mode == "nonmono" else mode
    return f"{alg} ({display_mode})"


def _plot_specs() -> list[dict[str, str | float | int | None]]:
    base = [
        {"alg": "EGM", "mode": "-", "linestyle": "--", "color": "tab:purple", "marker": None},
    ]
    for mu, alpha in [(2.0, 1.0)]:
        base.extend(
            [
                {"alg": "APDB-yx", "mode": "mono", "mu": mu, "linestyle": "--", "color": "tab:blue", "marker": None, "alpha": alpha},
                {
                    "alg": "rAPDB-yx-ada",
                    "mode": "mono",
                    "mu": mu,
                    "linestyle": "--",
                    "color": "tab:green",
                    "marker": None,
                    "alpha": alpha,
                },
                {
                    "alg": "rAPDB-yx",
                    "mode": "mono",
                    "mu": mu,
                    "linestyle": "--",
                    "color": "tab:red",
                    "marker": "x",
                    "restart_every": 1000,
                    "alpha": alpha,
                },
                {"alg": "APDB-yx", "mode": "nonmono", "mu": mu, "linestyle": "-", "color": "tab:blue", "marker": None, "alpha": alpha},
                {
                    "alg": "rAPDB-yx-ada",
                    "mode": "nonmono",
                    "mu": mu,
                    "linestyle": "-",
                    "color": "tab:green",
                    "marker": None,
                    "alpha": alpha,
                },
                {
                    "alg": "rAPDB-yx",
                    "mode": "nonmono",
                    "mu": mu,
                    "linestyle": "-",
                    "color": "tab:red",
                    "marker": "x",
                    "restart_every": 200,
                    "alpha": alpha,
                },
            ]
        )
    return base


def _format_mu_label(mu: float) -> str:
    text = f"{float(mu):g}"
    return text.replace("-", "m").replace(".", "p")


def _event_markevery(x_values, event_values) -> list[int] | None:
    values = np.asarray(x_values, dtype=np.float64).reshape(-1)
    events = np.asarray(event_values, dtype=np.float64).reshape(-1)
    if values.size == 0 or events.size == 0:
        return None

    indices: list[int] = []
    for event in events:
        if not np.isfinite(event):
            continue
        matches = np.flatnonzero(np.isclose(values, event))
        if matches.size:
            indices.append(int(matches[0]))
    return indices or None


def _apply_axis_text_style(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=AXIS_FONTSIZE)


def _apply_legend_style(ax: plt.Axes) -> None:
    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        loc="best",
        facecolor="white",
        framealpha=LEGEND_FRAME_ALPHA,
    )


def _marker_kwargs(spec: dict, x_values, event_values=None) -> dict:
    event_markevery = _event_markevery(x_values, [] if event_values is None else event_values)
    if event_markevery:
        return {"marker": "x", "markersize": MARKER_SIZE, "markevery": event_markevery}

    marker = spec["marker"]
    if marker is None:
        return {}

    kwargs = {"marker": marker, "markersize": MARKER_SIZE}
    restart_every = spec.get("restart_every")
    if restart_every is not None:
        values = np.asarray(x_values, dtype=np.float64)
        every = float(restart_every)
        restart_mask = (values > 0) & np.isclose(np.mod(values, every), 0.0)
        kwargs["markevery"] = np.flatnonzero(restart_mask).tolist()
    else:
        kwargs["markevery"] = max(1, len(x_values) // 12)
    return kwargs


def plot_cpu_curves(
    curves: dict,
    outdir: str | Path,
    *,
    gpu_curves: dict | None = None,
) -> tuple[Path, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 5.5))

    specs = _plot_specs()

    ax1 = fig.add_subplot(1, 2, 1)
    for spec in specs:
        alg = str(spec["alg"])
        mode = str(spec["mode"])
        mu = spec.get("mu")
        key = f"{alg}|{mode}" if mu is None else f"{alg}|{mode}|mu={_format_mu_label(float(mu))}"
        if key not in curves:
            continue
        it = curves[key]["iter"]
        re = curves[key]["rel"]
        kwargs = _marker_kwargs(spec, it, curves[key].get("restart_iter", []))
        ax1.semilogy(
            it,
            re,
            str(spec["linestyle"]),
            color=str(spec["color"]),
            alpha=float(spec.get("alpha", 1.0)),
            linewidth=2.0,
            label=_legend_label(alg, mode, None if mu is None else float(mu)),
            **kwargs,
        )

    ax1.grid(True)
    ax1.set_xlim(*ITERATION_X_LIMIT)
    _apply_axis_text_style(ax1)
    ax1.set_xlabel("Iteration", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel(REL_ERROR_Y_LABEL, fontsize=LABEL_FONTSIZE)
    _apply_legend_style(ax1)

    ax2 = fig.add_subplot(1, 2, 2)
    timing_curves = curves if gpu_curves is None else gpu_curves
    for spec in specs:
        alg = str(spec["alg"])
        mode = str(spec["mode"])
        mu = spec.get("mu")
        key = f"{alg}|{mode}" if mu is None else f"{alg}|{mode}|mu={_format_mu_label(float(mu))}"
        if key not in timing_curves:
            continue
        tm = timing_curves[key]["time"]
        it = timing_curves[key]["iter"]
        re = timing_curves[key]["rel"]
        kwargs = _marker_kwargs(
            spec,
            it,
            timing_curves[key].get("restart_iter", []),
        )
        ax2.semilogy(
            tm,
            re,
            str(spec["linestyle"]),
            color=str(spec["color"]),
            alpha=float(spec.get("alpha", 1.0)),
            linewidth=2.0,
            label=_legend_label(alg, mode, None if mu is None else float(mu)),
            **kwargs,
        )

    ax2.grid(True)
    timing_x_limit = TIME_X_LIMIT if gpu_curves is None else GPU_TIME_X_LIMIT
    timing_x_label = CPU_TIME_X_LABEL if gpu_curves is None else GPU_TIME_X_LABEL
    ax2.set_xlim(*timing_x_limit)
    _apply_axis_text_style(ax2)
    ax2.set_xlabel(timing_x_label, fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel(REL_ERROR_Y_LABEL, fontsize=LABEL_FONTSIZE)
    _apply_legend_style(ax2)

    png = outdir / "comparison_plot_cpu.png"
    pdf = outdir / "comparison_plot_cpu.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def load_cpu_curves_mat(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    curves: dict[str, dict[str, np.ndarray]] = {
        "EGM|-": {
            "iter": np.asarray(payload["iter_egm"], dtype=np.float64),
            "time": np.asarray(payload["time_egm"], dtype=np.float64),
            "rel": np.asarray(payload["rel_err_egm"], dtype=np.float64),
        }
    }

    for alg, raw_name in [
        ("APDB-yx", "apdb"),
        ("rAPDB-yx", "rapdb"),
        ("rAPDB-yx-ada", "rapdb_ada"),
    ]:
        for mode_idx, mode in [(0, "mono"), (1, "nonmono")]:
            key = f"{alg}|{mode}|mu=2"
            curves[key] = {
                "iter": np.asarray(payload[f"iter_{raw_name}_mu2_{mode_idx}"], dtype=np.float64),
                "time": np.asarray(payload[f"time_{raw_name}_mu2_{mode_idx}"], dtype=np.float64),
                "rel": np.asarray(payload[f"rel_err_{raw_name}_mu2_{mode_idx}"], dtype=np.float64),
            }
            restart_iter_key = f"restart_iter_{raw_name}_mu2_{mode_idx}"
            restart_time_key = f"restart_time_{raw_name}_mu2_{mode_idx}"
            if restart_iter_key in payload:
                curves[key]["restart_iter"] = np.asarray(payload[restart_iter_key], dtype=np.float64).reshape(-1)
            if restart_time_key in payload:
                curves[key]["restart_time"] = np.asarray(payload[restart_time_key], dtype=np.float64).reshape(-1)

    return curves


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redraw comparison plot from saved CPU and optional GPU outputs"
    )
    parser.add_argument("mat_file", type=str, help="path to experiment_outputs_cpu.mat")
    parser.add_argument(
        "--gpu-mat",
        type=str,
        default=None,
        help="optional path to experiment_outputs_gpu.mat for the right-hand timing plot",
    )
    parser.add_argument("--outdir", type=str, default=None, help="output directory; defaults to mat file parent")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    mat_file = Path(args.mat_file)
    outdir = Path(args.outdir) if args.outdir is not None else mat_file.parent
    gpu_curves = None if args.gpu_mat is None else load_cpu_curves_mat(args.gpu_mat)
    plot_cpu_curves(load_cpu_curves_mat(mat_file), outdir, gpu_curves=gpu_curves)


if __name__ == "__main__":
    main()
