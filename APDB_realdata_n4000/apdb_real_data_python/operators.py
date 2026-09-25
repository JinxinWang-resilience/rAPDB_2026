from __future__ import annotations

import numpy as np
import torch


def _projsplx_numpy(y: np.ndarray) -> np.ndarray:
    m = int(y.size)
    bget = False
    s = np.sort(y)[::-1]
    tmpsum = 0.0
    tmax = 0.0
    for ii in range(1, m):
        tmpsum = tmpsum + s[ii - 1]
        tmax = (tmpsum - 1.0) / float(ii)
        if tmax >= s[ii]:
            bget = True
            break
    if not bget:
        tmax = (tmpsum + s[m - 1] - 1.0) / float(m)
    return np.maximum(y - tmax, 0.0)


def _proj_pos_plane_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        dual_star = 0.0
    elif g_old > 0.0:
        xpp = x_sort[x_sort > 0.0]
        x_old = x0
        flag = False
        for x_new in xpp:
            gg = g(float(x_new))
            if g_old >= 0.0 and gg <= 0.0:
                if g_old == gg:
                    dual_star = x_old
                else:
                    dual_star = x_old - g_old * (float(x_new) - x_old) / (gg - g_old)
                flag = True
                break
            x_old = float(x_new)
            g_old = gg
        if not flag:
            x_new = x_old + 1.0
            gg = g(x_new)
            dual_star = x_old - g_old * (x_new - x_old) / (gg - g_old)
    else:
        xmm = x_sort[x_sort < 0.0]
        x_old = x0
        flag = False
        for x_new in xmm[::-1]:
            gg = g(float(x_new))
            if g_old <= 0.0 and gg >= 0.0:
                if g_old == gg:
                    dual_star = x_old
                else:
                    dual_star = x_old - g_old * (float(x_new) - x_old) / (gg - g_old)
                flag = True
                break
            x_old = float(x_new)
            g_old = gg
        if not flag:
            x_new = x_old - 1.0
            gg = g(x_new)
            dual_star = x_old - g_old * (x_new - x_old) / (gg - g_old)

    z = np.zeros_like(x)
    z[idx_plus] = np.maximum(x_plus - dual_star, 0.0)
    z[idx_minus] = np.maximum(x_minus + dual_star, 0.0)
    return z


def projsplx_torch(y: torch.Tensor) -> torch.Tensor:
    """MATLAB-compatible simplex projection from projsplx.m."""
    y_flat = y.reshape(-1)
    y_np = y_flat.detach().cpu().numpy().astype(np.float64)
    out_np = _projsplx_numpy(y_np)
    out = torch.as_tensor(out_np, dtype=y_flat.dtype, device=y_flat.device)
    return out.reshape(y.shape) if y.ndim > 1 else out


def proj_pos_plane_torch(x: torch.Tensor, y: torch.Tensor, iters: int = 80) -> torch.Tensor:
    """MATLAB-compatible projection from proj_pos_plane.m."""
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    x_np = x_flat.detach().cpu().numpy().astype(np.float64)
    y_np = y_flat.detach().cpu().numpy().astype(np.float64)
    out_np = _proj_pos_plane_numpy(x_np, y_np)
    out = torch.as_tensor(out_np, dtype=x_flat.dtype, device=x_flat.device)
    return out.reshape(x.shape) if x.ndim > 1 else out


def inner_prod_matvec_torch(
    g_list: list[torch.Tensor], x: torch.Tensor, return_matvec: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute y2_i = x^T G_i x and optionally y1 columns = G_i x."""
    y2_terms = []
    y1_cols = []
    for g in g_list:
        gx = g @ x
        if return_matvec:
            y1_cols.append(gx)
        y2_terms.append(torch.dot(x, gx))

    y2 = torch.stack(y2_terms)
    y1 = torch.stack(y1_cols, dim=1) if return_matvec else None
    return y2, y1
