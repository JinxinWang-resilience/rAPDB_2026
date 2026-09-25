import numpy as np
import torch
import warnings

from apdb_real_data_python.data_pipeline import (
    _split_train_test,
    _standardize_like_demo,
    _standardize_like_kernel_data_generator,
)
from apdb_real_data_python.operators import proj_pos_plane_torch


def _proj_pos_plane_reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    idx_plus = np.where(y == 1)[0]
    idx_minus = np.where(y == -1)[0]
    x_plus = x[idx_plus]
    x_minus = x[idx_minus]
    x_sort = np.sort(np.concatenate([x_plus, -x_minus]))

    def g(lam: float) -> float:
        return float(np.sum(np.maximum(x_plus - lam, 0.0)) - np.sum(np.maximum(x_minus + lam, 0.0)))

    x0 = 0.0
    g_old = g(x0)
    if g_old == 0.0:
        dual = 0.0
    elif g_old > 0.0:
        xpp = x_sort[x_sort > 0.0]
        x_old = x0
        dual = x_old
        found = False
        for x_new in xpp:
            gg = g(float(x_new))
            if g_old >= 0.0 and gg <= 0.0:
                if g_old == gg:
                    dual = x_old
                else:
                    dual = x_old - g_old * (float(x_new) - x_old) / (gg - g_old)
                found = True
                break
            x_old = float(x_new)
            g_old = gg
        if not found:
            x_new = x_old + 1.0
            gg = g(x_new)
            dual = x_old - g_old * (x_new - x_old) / (gg - g_old)
    else:
        xmm = x_sort[x_sort < 0.0]
        x_old = x0
        dual = x_old
        found = False
        for x_new in xmm[::-1]:
            gg = g(float(x_new))
            if g_old <= 0.0 and gg >= 0.0:
                if g_old == gg:
                    dual = x_old
                else:
                    dual = x_old - g_old * (float(x_new) - x_old) / (gg - g_old)
                found = True
                break
            x_old = float(x_new)
            g_old = gg
        if not found:
            x_new = x_old - 1.0
            gg = g(x_new)
            dual = x_old - g_old * (x_new - x_old) / (gg - g_old)

    z = np.zeros_like(x)
    z[idx_plus] = np.maximum(x_plus - dual, 0.0)
    z[idx_minus] = np.maximum(x_minus + dual, 0.0)
    return z


def test_demo_standardization_drops_constant_columns():
    x = np.array(
        [
            [1.0, 2.0, 3.0, 10.0],
            [1.0, 3.0, 6.0, 10.0],
            [1.0, 4.0, 9.0, 10.0],
        ],
        dtype=np.float64,
    )
    z = _standardize_like_demo(x)
    # Col-0 and col-3 are constant and should be removed by NaN-column filtering.
    assert z.shape[1] == 2
    assert np.isfinite(z).all()


def test_demo_standardization_drops_low_variance_columns():
    base = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    low_var = np.array([1.0, 1.0 + 1e-9, 1.0 + 2e-9], dtype=np.float64)
    x = np.column_stack([base, low_var, base * 2.0])

    z = _standardize_like_demo(x)

    # The middle column has tiny std (<1e-8) and should be removed.
    assert z.shape[1] == 2
    assert np.isfinite(z).all()


def test_demo_standardization_no_divide_warning_on_constant_columns():
    x = np.array(
        [
            [1.0, 2.0, 3.0, 10.0],
            [1.0, 3.0, 6.0, 10.0],
            [1.0, 4.0, 9.0, 10.0],
        ],
        dtype=np.float64,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        z = _standardize_like_demo(x)

    assert np.isfinite(z).all()
    assert not any("invalid value encountered in divide" in str(w.message) for w in caught)


def test_split_uses_legacy_mt_permutation():
    x = np.arange(24, dtype=np.float64).reshape(12, 2)
    y = np.arange(12, dtype=np.float64)
    _, _, _, _, idx_tr, idx_te = _split_train_test(x, y, seed=17, ratio=0.8, rng_mode="matlab_legacy")
    rs = np.random.RandomState(17)
    expected = np.argsort(rs.rand(12), kind="mergesort")
    m = int(np.floor(12 * 0.8))
    assert np.array_equal(idx_tr, expected[:m])
    assert np.array_equal(idx_te, expected[m:])


def test_kernel_generator_second_standardization_matches_matlab_style():
    x = np.array([[2.0, 3.0], [4.0, 8.0], [6.0, 13.0], [8.0, 18.0]], dtype=np.float64)
    z = _standardize_like_kernel_data_generator(x)
    mu = np.mean(z, axis=0)
    # MATLAB std(X,1) normalization implies centered columns.
    assert np.allclose(mu, 0.0, atol=1e-12)


def test_proj_pos_plane_matches_reference():
    x = np.array([2.3, -0.1, 1.7, 0.5, 3.2, -2.0, 0.2], dtype=np.float64)
    y = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64)
    ref = _proj_pos_plane_reference(x, y)

    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    out = proj_pos_plane_torch(xt, yt).cpu().numpy()

    assert np.allclose(out, ref, atol=1e-12, rtol=0.0)
