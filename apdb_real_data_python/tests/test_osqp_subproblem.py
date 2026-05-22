import importlib

import numpy as np
import pytest
import torch


def test_run_x_qp_for_gxi_uses_non_cvxpy_backend():
    common = importlib.import_module("apdb_real_data_python.solvers.common")
    assert "cp" not in common.__dict__

    osqp = pytest.importorskip("osqp")
    assert osqp is not None

    g_mat = torch.eye(4, dtype=torch.float64)
    inp = common.SolverInputTorch(
        g_tr_class=[g_mat.clone(), 1.2 * g_mat.clone()],
        g_tr_max_norm=1.0,
        y_train=torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64),
        trace_kernels=torch.tensor([1.0, 1.0], dtype=torch.float64),
        reg_param=1.0,
        x0=torch.zeros(4, dtype=torch.float64),
        y0=torch.zeros(2, dtype=torch.float64),
        optsol=torch.zeros(4, dtype=torch.float64),
    )

    g_xi = common.run_x_qp_for_gxi(
        x_curr=torch.zeros(4, dtype=torch.float64),
        y_curr=torch.zeros(2, dtype=torch.float64),
        inp=inp,
        xi=0.04,
        c=2.0,
    )
    assert np.isfinite(g_xi)
