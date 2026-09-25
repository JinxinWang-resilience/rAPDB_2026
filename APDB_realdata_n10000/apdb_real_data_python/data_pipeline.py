from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse

from apdb_real_data_python.kernels import kernel_gaussian, kernel_linear, kernel_poly

LOW_STD_EPS = 1e-8


@dataclass
class KernelData:
    y_train: np.ndarray
    y_test: np.ndarray
    index_train: np.ndarray
    index_test: np.ndarray
    kernel_tr_class: list[np.ndarray]
    kernel_test_class: list[np.ndarray]
    g_tr_class: list[np.ndarray]
    g_tr_max_norm: float
    trace_kernels: np.ndarray


@dataclass
class RealDataProblem:
    input_data: KernelData
    reg_param: float
    x0: np.ndarray
    y0: np.ndarray
    seed: int
    data_name: str
    n_samples: int
    alignment_mode: str


def _standardize_features(x: np.ndarray) -> np.ndarray:
    return _standardize_columns(x, ddof=0, drop_nan_cols=True, protect_zero_std=True)


def _standardize_columns(
    x: np.ndarray,
    *,
    ddof: int,
    drop_nan_cols: bool,
    protect_zero_std: bool,
) -> np.ndarray:
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=ddof)
    low_std_cols = sigma < LOW_STD_EPS
    if protect_zero_std or np.any(low_std_cols):
        # Avoid divide-by-zero warnings while keeping legacy column-drop behavior.
        sigma = np.where(low_std_cols, 1.0, sigma)
    z = (x - mu) / sigma
    if drop_nan_cols:
        bad_cols = np.any(np.isnan(z), axis=0) | low_std_cols
        z = z[:, ~bad_cols]
    return z


def _standardize_like_demo(x: np.ndarray) -> np.ndarray:
    # MATLAB demo.m: std(X,0,1), then drop NaN columns.
    return _standardize_columns(x, ddof=1, drop_nan_cols=True, protect_zero_std=False)


def _standardize_like_kernel_data_generator(x: np.ndarray) -> np.ndarray:
    # MATLAB kernel_data_generator.m: std(X,1), no explicit NaN drop.
    return _standardize_columns(x, ddof=0, drop_nan_cols=False, protect_zero_std=False)


def _split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    ratio: float = 0.8,
    rng_mode: str = "matlab_legacy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rng_mode == "matlab_legacy":
        # MATLAB randperm in this workflow is best matched by sorting MT19937 uniforms.
        rng = np.random.RandomState(seed)
        idx = np.argsort(rng.rand(x.shape[0]), kind="mergesort")
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(x.shape[0])
    m = int(np.floor(x.shape[0] * ratio))
    tr = idx[:m]
    te = idx[m:]
    return x[tr], x[te], y[tr], y[te], tr, te


def kernel_data_generator(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 17,
    *,
    rng_mode: str = "matlab_legacy",
) -> KernelData:
    x = _standardize_like_kernel_data_generator(x)
    x_train, x_test, y_train, y_test, idx_train, idx_test = _split_train_test(
        x,
        y.reshape(-1),
        seed=seed,
        ratio=0.8,
        rng_mode=rng_mode,
    )

    x_joint = np.vstack([x_train, x_test])
    k1 = kernel_poly(x_joint, degree=2)
    k2 = kernel_gaussian(x_joint, sigma=0.5)
    k3 = kernel_linear(x_joint)

    m = x_train.shape[0]
    kernels = [k1, k2, k3]
    traces = np.asarray([np.trace(k) for k in kernels], dtype=np.float64)

    dy = np.diag(y_train)
    kernel_tr_class = [k[:m, :m] for k in kernels]
    kernel_test_class = [k[:m, m:] for k in kernels]
    g_tr_class = [dy @ kt @ dy for kt in kernel_tr_class]
    g_norms = [np.linalg.norm(g) for g in g_tr_class]

    return KernelData(
        y_train=y_train.astype(np.float64),
        y_test=y_test.astype(np.float64),
        index_train=np.asarray(idx_train, dtype=np.int64),
        index_test=np.asarray(idx_test, dtype=np.int64),
        kernel_tr_class=[k.astype(np.float64) for k in kernel_tr_class],
        kernel_test_class=[k.astype(np.float64) for k in kernel_test_class],
        g_tr_class=[g.astype(np.float64) for g in g_tr_class],
        g_tr_max_norm=float(np.max(g_norms)),
        trace_kernels=traces,
    )


def load_sido_real_problem(
    base_dir: str | Path,
    *,
    n_samples: int = 1000,
    reg_param: float = 1.0,
    seed: int = 17,
    alignment_mode: str = "legacy_matlab",
) -> RealDataProblem:
    base = Path(base_dir)
    mat_path = base / "sido0_train.mat"
    tgt_path = base / "sido0_train.targets"

    if not mat_path.exists():
        raise FileNotFoundError(f"Missing data file: {mat_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Missing target file: {tgt_path}")

    mat = loadmat(mat_path)
    if "X" not in mat:
        raise KeyError(f"Expected variable 'X' in {mat_path}")

    x_raw = mat["X"]
    if issparse(x_raw):
        x_full = x_raw.toarray().astype(np.float64)
    else:
        x_full = np.asarray(x_raw, dtype=np.float64)
    y_full = np.loadtxt(tgt_path, dtype=np.float64).reshape(-1)

    n_use = min(n_samples, x_full.shape[0], y_full.shape[0])
    x = x_full[:n_use, :]
    y = y_full[:n_use]

    if alignment_mode == "legacy_matlab":
        x = _standardize_like_demo(x)
        kd = kernel_data_generator(x, y, seed=seed, rng_mode="matlab_legacy")
    else:
        kd = kernel_data_generator(x, y, seed=seed, rng_mode="numpy_default")

    x0 = np.zeros_like(kd.y_train, dtype=np.float64)
    y0 = np.zeros(kd.trace_kernels.shape[0], dtype=np.float64)

    return RealDataProblem(
        input_data=kd,
        reg_param=float(reg_param),
        x0=x0,
        y0=y0,
        seed=seed,
        data_name="sido0_train",
        n_samples=n_use,
        alignment_mode=alignment_mode,
    )
