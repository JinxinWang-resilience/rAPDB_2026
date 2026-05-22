from __future__ import annotations

import argparse
from pathlib import Path

from apdb_real_data_python.data_pipeline import load_sido_real_problem
from apdb_real_data_python.runner import run_real_data_experiment


def _parse_tau_grid(grid_text: str) -> tuple[float, ...]:
    parts = [p.strip() for p in str(grid_text).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("egm-tune-grid must contain at least one positive float.")
    vals: list[float] = []
    for part in parts:
        try:
            v = float(part)
        except ValueError as exc:  # pragma: no cover - argparse handles message
            raise argparse.ArgumentTypeError(f"Invalid float in egm-tune-grid: {part}") from exc
        if v <= 0.0:
            raise argparse.ArgumentTypeError("All egm-tune-grid values must be positive.")
        vals.append(v)
    return tuple(vals)


def _parse_mu_values(mu_text: str) -> tuple[float, ...]:
    parts = [p.strip() for p in str(mu_text).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("mu-values must contain at least one nonnegative float.")
    vals: list[float] = []
    for part in parts:
        try:
            v = float(part)
        except ValueError as exc:  # pragma: no cover - argparse handles message
            raise argparse.ArgumentTypeError(f"Invalid float in mu-values: {part}") from exc
        if v < 0.0:
            raise argparse.ArgumentTypeError("All mu-values must be nonnegative.")
        vals.append(v)
    return tuple(vals)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone APDB real-data experiment runner")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--run-all", action="store_true", help="always run CPU first, then GPU if available")
    parser.add_argument("--max-iter", type=int, default=8000)
    parser.add_argument("--stop-criteria", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--reg-param", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="apdb_real_data_python/outputs")
    parser.add_argument("--data-dir", type=str, default="apdb_real_data_matlab")
    parser.add_argument(
        "--progress",
        choices=["on", "off", "auto"],
        default="auto",
        help="show optimization progress bars/logs",
    )
    parser.add_argument(
        "--progress-refresh",
        type=float,
        default=0.5,
        help="progress refresh interval in seconds",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="progress update interval in iterations",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=["legacy_matlab", "current"],
        default="legacy_matlab",
        help="legacy_matlab reproduces MATLAB preprocessing/split behavior more strictly",
    )
    parser.add_argument("--strict-mosek", action="store_true", default=True)
    parser.add_argument("--mosek-tol", type=float, default=1e-9)
    parser.add_argument("--egm-tune", action="store_true", default=False)
    parser.add_argument("--egm-tau-scale", type=float, default=None)
    parser.add_argument("--egm-tune-max-iter", type=int, default=1000)
    parser.add_argument(
        "--egm-tune-grid",
        type=str,
        default="0.05,0.1,0.2,0.5",
        help="comma-separated tau scales for EGM tuning",
    )
    parser.add_argument(
        "--mu-values",
        type=str,
        default="0,2",
        help="comma-separated mu values for APDB-yx variants",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    problem = load_sido_real_problem(
        Path(args.data_dir),
        n_samples=args.n_samples,
        reg_param=args.reg_param,
        seed=args.seed,
        alignment_mode=args.alignment_mode,
    )

    run_real_data_experiment(
        problem=problem,
        outdir=Path(args.outdir),
        run_all=bool(args.run_all),
        device=args.device,
        max_iter=int(args.max_iter),
        stop_criteria=float(args.stop_criteria),
        strict_mosek=bool(args.strict_mosek),
        mosek_tol=float(args.mosek_tol),
        progress_mode=str(args.progress),
        progress_refresh=float(args.progress_refresh),
        progress_every=int(args.progress_every),
        egm_tau_scale=(None if args.egm_tau_scale is None else float(args.egm_tau_scale)),
        egm_tune=bool(args.egm_tune),
        egm_tune_max_iter=int(args.egm_tune_max_iter),
        egm_tune_grid=_parse_tau_grid(args.egm_tune_grid),
        mu_values=_parse_mu_values(args.mu_values),
    )


if __name__ == "__main__":
    main()
