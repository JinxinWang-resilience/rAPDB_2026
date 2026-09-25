from __future__ import annotations

import numpy as np


def _normalize_kernel(kern: np.ndarray) -> np.ndarray:
    diag = np.diag(kern)
    den = np.sqrt(np.outer(diag, diag))
    den = np.where(den > 0.0, den, 1.0)
    return kern / den


def kernel_poly(x: np.ndarray, degree: int = 2) -> np.ndarray:
    kern = (1.0 + x @ x.T) ** degree
    return _normalize_kernel(kern)


def kernel_gaussian(x: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    sq = np.sum(x * x, axis=1)
    pair = sq[:, None] - 2.0 * (x @ x.T) + sq[None, :]
    kern = np.exp(-0.5 * pair / sigma)
    return _normalize_kernel(kern)


def kernel_linear(x: np.ndarray) -> np.ndarray:
    kern = x @ x.T
    return _normalize_kernel(kern)
