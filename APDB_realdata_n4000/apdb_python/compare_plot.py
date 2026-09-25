from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from apdb_python.plotting import plot_results


def _cell_row(cell: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(cell[0, i]).reshape(-1) for i in range(cell.shape[1])]


def _cell_2d(cell: np.ndarray) -> list[list[np.ndarray]]:
    return [[np.asarray(cell[r, c]).reshape(-1) for c in range(cell.shape[1])] for r in range(cell.shape[0])]


def load_results_mat(mat_path: str | Path) -> dict[str, Any]:
    mat = loadmat(mat_path)
    required_1d = ["oracle_egm", "rel_subopt_egm", "rel_infeas_err_egm"]
    required_2d = [
        "oracle_1",
        "rel_subopt_1",
        "rel_infeas_1",
        "oracle_1_ada",
        "rel_subopt_1_ada",
        "rel_infeas_1_ada",
        "oracle_r1",
        "rel_subopt_r1",
        "rel_infeas_r1",
        "oracle_xy_1",
        "rel_subopt_xy_1",
        "rel_infeas_xy_1",
        "oracle_xy_ada1",
        "rel_subopt_xy_ada1",
        "rel_infeas_xy_ada1",
        "oracle_xy_r1",
        "rel_subopt_xy_r1",
        "rel_infeas_xy_r1",
    ]

    missing = [k for k in required_1d + required_2d if k not in mat]
    if missing:
        raise KeyError(f"Missing required keys in {mat_path}: {missing}")

    out: dict[str, Any] = {}
    for key in required_1d:
        out[key] = _cell_row(mat[key])
    for key in required_2d:
        out[key] = _cell_2d(mat[key])
    return out


def generate_compare_plots(
    matlab_mat: str | Path,
    python_mat: str | Path,
    outdir: str | Path,
    n: int,
    m: int,
    numsim: int | None = None,
) -> tuple[Path, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    matlab_results = load_results_mat(matlab_mat)
    python_results = load_results_mat(python_mat)

    available = min(len(matlab_results["oracle_egm"]), len(python_results["oracle_egm"]))
    if numsim is None:
        numsim = available
    else:
        numsim = min(numsim, available)

    matlab_tmp = plot_results(
        matlab_results,
        numsim=numsim,
        n=n,
        m=m,
        device="matlab",
        outdir=outdir,
        plot_style="strict",
    )[0]
    python_tmp = plot_results(
        python_results,
        numsim=numsim,
        n=n,
        m=m,
        device="python",
        outdir=outdir,
        plot_style="strict",
    )[0]

    matlab_out = outdir / f"compare_matlab_n{n}_m{m}_numsim{numsim}.png"
    python_out = outdir / f"compare_python_n{n}_m{m}_numsim{numsim}.png"

    matlab_tmp.replace(matlab_out)
    python_tmp.replace(python_out)
    return matlab_out, python_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MATLAB/Python comparison plots from .mat outputs")
    parser.add_argument("--matlab-mat", required=True, type=str)
    parser.add_argument("--python-mat", required=True, type=str)
    parser.add_argument("--outdir", required=True, type=str)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--numsim", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    mat_png, py_png = generate_compare_plots(
        matlab_mat=args.matlab_mat,
        python_mat=args.python_mat,
        outdir=args.outdir,
        n=args.n,
        m=args.m,
        numsim=args.numsim,
    )
    print(mat_png)
    print(py_png)


if __name__ == "__main__":
    main()
