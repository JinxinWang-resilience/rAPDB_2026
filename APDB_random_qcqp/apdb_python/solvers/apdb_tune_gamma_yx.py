from __future__ import annotations

import traceback
from collections.abc import Callable

import numpy as np


def _terminal_subopt(out: dict) -> float:
    sub = np.asarray(out.get("rel_subopt_err", np.array([], dtype=np.float64)), dtype=np.float64).reshape(-1)
    if sub.size == 0:
        raise ValueError("rel_subopt_err is empty")
    val = float(sub[-1])
    if not np.isfinite(val):
        raise ValueError(f"non-finite terminal subopt: {val}")
    return val


def tune_gamma_yx_family(
    *,
    runner: Callable[..., dict],
    input_data: dict,
    optval: float,
    x0: np.ndarray,
    y0: np.ndarray,
    epsilon: float,
    max_iter: int,
    gamma_grid: np.ndarray | None = None,
    tune_iter: int | None = None,
    device: str = "cpu",
    sim_idx: int | None = None,
) -> tuple[float, dict]:
    if gamma_grid is None:
        gamma_grid = np.asarray([1.0, 3.0, 6.0, 9.0], dtype=np.float64)
    else:
        gamma_grid = np.asarray(gamma_grid, dtype=np.float64).reshape(-1)

    if tune_iter is None:
        tune_iter = int(min(2000, max_iter))
    tune_iter = max(1, int(tune_iter))

    score_list = np.full(gamma_grid.shape, np.inf, dtype=np.float64)
    subopt_mode0 = np.full(gamma_grid.shape, np.inf, dtype=np.float64)

    sim_txt = "?" if sim_idx is None else str(int(sim_idx) + 1)
    for idx, gamma in enumerate(gamma_grid):
        mode = 0
        try:
            out = runner(
                input_data=input_data,
                optval=optval,
                x0=x0,
                y0=y0,
                stop_criteria=epsilon,
                max_iter=tune_iter,
                non_mono=mode,
                device=device,
                gamma0=float(gamma),
            )
            subopt_mode0[idx] = _terminal_subopt(out)
        except Exception as exc:  # noqa: BLE001
            print(
                f"YX gamma tuning failed: sim={sim_txt}, gamma={float(gamma)}, "
                f"mode={int(mode)}, exc={type(exc).__name__}: {exc}"
            )
            print(traceback.format_exc())
            raise RuntimeError("failed during YX gamma tuning") from exc
        score_list[idx] = float(subopt_mode0[idx])

    best_idx = int(np.argmin(score_list))
    info = {
        "gamma_grid": gamma_grid,
        "scores": score_list,
        "subopt_mode0": subopt_mode0,
        "best_score": float(score_list[best_idx]),
        "best_subopt_mode0": float(subopt_mode0[best_idx]),
    }
    return float(gamma_grid[best_idx]), info
