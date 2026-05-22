from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


MARKER_SIZE = 9
ITERATION_X_LIMIT = (0, 8000)
TIME_X_LIMIT = (0, 600)
CPU_TIME_X_LABEL = "CPU time (s)"
REL_ERROR_Y_LABEL = r"$\|x^k-x^\star\|/(1+\|x^\star\|)$"
AXIS_FONTSIZE = 16
LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 11


def _legend_label(alg: str, mode: str, mu: float | None = None) -> str:
    if alg == "EGM":
        return "EGM"
    display_mode = "non-mono" if mode == "nonmono" else mode
    return f"{alg} ({display_mode})"


def _plot_specs() -> list[dict[str, str | float | None]]:
    base = [
        {"alg": "EGM", "mode": "-", "linestyle": "--", "color": "tab:purple", "marker": None},
    ]
    for mu, alpha in [(2.0, 1.0)]:
        base.extend(
            [
                {"alg": "APDB-yx", "mode": "mono", "mu": mu, "linestyle": "-", "color": "tab:blue", "marker": "o", "alpha": alpha},
                {
                    "alg": "rAPDB-yx-ada",
                    "mode": "mono",
                    "mu": mu,
                    "linestyle": ":",
                    "color": "tab:green",
                    "marker": "^",
                    "alpha": alpha,
                },
                {"alg": "rAPDB-yx", "mode": "mono", "mu": mu, "linestyle": ":", "color": "tab:red", "marker": "x", "alpha": alpha},
                {"alg": "APDB-yx", "mode": "nonmono", "mu": mu, "linestyle": "--", "color": "tab:blue", "marker": None, "alpha": alpha},
                {
                    "alg": "rAPDB-yx-ada",
                    "mode": "nonmono",
                    "mu": mu,
                    "linestyle": "-",
                    "color": "tab:green",
                    "marker": None,
                    "alpha": alpha,
                },
                {"alg": "rAPDB-yx", "mode": "nonmono", "mu": mu, "linestyle": "-", "color": "tab:red", "marker": None, "alpha": alpha},
            ]
        )
    return base


def _format_mu_label(mu: float) -> str:
    text = f"{float(mu):g}"
    return text.replace("-", "m").replace(".", "p")


def _apply_axis_text_style(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=AXIS_FONTSIZE)


def plot_cpu_curves(curves: dict, outdir: str | Path) -> tuple[Path, Path]:
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
        marker = spec["marker"]
        kwargs = {}
        if marker is not None:
            kwargs["marker"] = marker
            kwargs["markersize"] = MARKER_SIZE
            kwargs["markevery"] = max(1, len(it) // 12)
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
    ax1.legend(fontsize=LEGEND_FONTSIZE, loc="best")

    ax2 = fig.add_subplot(1, 2, 2)
    for spec in specs:
        alg = str(spec["alg"])
        mode = str(spec["mode"])
        mu = spec.get("mu")
        key = f"{alg}|{mode}" if mu is None else f"{alg}|{mode}|mu={_format_mu_label(float(mu))}"
        if key not in curves:
            continue
        tm = curves[key]["time"]
        re = curves[key]["rel"]
        marker = spec["marker"]
        kwargs = {}
        if marker is not None:
            kwargs["marker"] = marker
            kwargs["markersize"] = MARKER_SIZE
            kwargs["markevery"] = max(1, len(tm) // 12)
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
    ax2.set_xlim(*TIME_X_LIMIT)
    _apply_axis_text_style(ax2)
    ax2.set_xlabel(CPU_TIME_X_LABEL, fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel(REL_ERROR_Y_LABEL, fontsize=LABEL_FONTSIZE)
    ax2.legend(fontsize=LEGEND_FONTSIZE, loc="best")

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

    return curves


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Redraw CPU comparison plot from saved experiment_outputs_cpu.mat")
    parser.add_argument("mat_file", type=str, help="path to experiment_outputs_cpu.mat")
    parser.add_argument("--outdir", type=str, default=None, help="output directory; defaults to mat file parent")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    mat_file = Path(args.mat_file)
    outdir = Path(args.outdir) if args.outdir is not None else mat_file.parent
    plot_cpu_curves(load_cpu_curves_mat(mat_file), outdir)


if __name__ == "__main__":
    main()
