import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    ("device_type", "expected_algebra"),
    [("cpu", "builtin"), ("cuda", "cuda")],
)
def test_osqp_solver_uses_matching_device_backend(device_type, expected_algebra):
    common = importlib.import_module("apdb_real_data_python.solvers.common")
    selected = []
    fake_osqp = SimpleNamespace(
        algebras_available=lambda: ["builtin", "cuda"],
        OSQP=lambda **kwargs: selected.append(kwargs) or object(),
    )

    common._make_osqp_solver(fake_osqp, device_type=device_type)

    assert selected == [{"algebra": expected_algebra}]


def test_gpu_osqp_requires_cuda_backend():
    common = importlib.import_module("apdb_real_data_python.solvers.common")
    fake_osqp = SimpleNamespace(
        algebras_available=lambda: ["builtin"],
        OSQP=lambda **kwargs: pytest.fail("CPU OSQP must not be used for a GPU run"),
    )

    with pytest.raises(RuntimeError, match=r"osqp\[cu12\]"):
        common._make_osqp_solver(fake_osqp, device_type="cuda")


def test_cpu_osqp_supports_legacy_package_without_backend_selection():
    common = importlib.import_module("apdb_real_data_python.solvers.common")
    selected = []
    fake_osqp = SimpleNamespace(OSQP=lambda **kwargs: selected.append(kwargs) or object())

    common._make_osqp_solver(fake_osqp, device_type="cpu")

    assert selected == [{}]


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
