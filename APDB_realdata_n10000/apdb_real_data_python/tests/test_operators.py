import torch

from apdb_real_data_python.operators import projsplx_torch, proj_pos_plane_torch


def test_projsplx_torch_projects_to_simplex():
    y = torch.tensor([0.2, -1.0, 3.0, 0.4], dtype=torch.float64)
    x = projsplx_torch(y)

    assert torch.all(x >= -1e-12)
    assert torch.isclose(torch.sum(x), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)


def test_proj_pos_plane_torch_projects_to_feasible_set():
    x = torch.tensor([1.2, 0.3, -0.7, 2.1, -0.5], dtype=torch.float64)
    y = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
    z = proj_pos_plane_torch(x, y)

    assert torch.all(z >= -1e-12)
    assert torch.isclose(torch.dot(y, z), torch.tensor(0.0, dtype=torch.float64), atol=1e-8)
