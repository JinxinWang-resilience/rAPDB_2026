import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from apdb_real_data_python.data_pipeline import KernelData, RealDataProblem


def _problem(g_scale=1.0):
    kd = KernelData(
        y_train=np.array([1.0, -1.0]),
        y_test=np.array([1.0]),
        index_train=np.array([0, 1]),
        index_test=np.array([2]),
        kernel_tr_class=[],
        kernel_test_class=[],
        g_tr_class=[g_scale * np.eye(2)],
        g_tr_max_norm=g_scale,
        trace_kernels=np.array([2.0]),
    )
    return RealDataProblem(kd, 1.0, np.zeros(2), np.zeros(1), 17, "sido0_train", 3, "legacy_matlab")


def test_reference_check_detects_changed_kernel():
    from apdb_real_data_python.complete_gpu import reference_check

    alpha = np.array([1.0, 1.0])
    assert reference_check(_problem(), alpha, 0.0)[0]
    assert not reference_check(_problem(g_scale=2.0), alpha, 0.0)[0]


def test_cuda_absence_fails_before_writing(tmp_path, monkeypatch):
    import apdb_real_data_python.complete_gpu as completion

    monkeypatch.setattr(completion.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        completion.complete_missing_gpu(tmp_path, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_gpu_completion_reuses_cpu_reference_and_keeps_eight_summary_rows(tmp_path, monkeypatch):
    import apdb_real_data_python.complete_gpu as completion

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    manifest = {
        "n_samples": 3,
        "train_size": 2,
        "seed": 17,
        "alignment_mode": "legacy_matlab",
        "max_iter": 5,
        "stop_criteria": 1e-6,
        "progress_mode": "off",
        "progress_refresh": 0.5,
        "progress_every": 20,
        "egm_tau_scale": None,
        "egm_tune": False,
        "egm_tune_max_iter": 1000,
        "egm_tune_grid": [0.05],
        "mu_values": [2.0],
        "mosek_tol_requested": 1e-9,
        "gpu_available": False,
        "gpu_used": False,
        "restart_config": {"rapdb_yx": {"mono": 1000, "nonmono": 200}, "rapdb_yx_ada": {"mono": 1000, "nonmono": 200}},
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cpu_rows = pd.DataFrame([
        {"Algorithm": "MOSEK(CVX)", "Mode": "-", "Mu": "-", "Iterations": np.nan, "FinalResidual": np.nan, "TimeSeconds": 8.0},
        {"Algorithm": "EGM", "Mode": "-", "Mu": "-", "Iterations": 5.0, "FinalResidual": 0.1, "TimeSeconds": 7.0},
        {"Algorithm": "APDB-yx", "Mode": "mono", "Mu": "2.0", "Iterations": 3.0, "FinalResidual": 1e-6, "TimeSeconds": 6.0},
    ])
    cpu_rows.to_csv(output_dir / "summary_table_cpu.csv", index=False)
    savemat(output_dir / "experiment_outputs_cpu.mat", {"optsol": np.array([1.0, 1.0]), "optval": 0.0, "time_cvx": 8.0, "rel_err_egm": np.array([0.25]), "egm_tau_scale_used": 0.1})
    before = (output_dir / "experiment_outputs_cpu.mat").read_bytes()

    monkeypatch.setattr(completion.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(completion.torch.cuda, "get_device_name", lambda *_: "test GPU")
    monkeypatch.setattr(completion, "_require_run_parameters", lambda *_: None)
    monkeypatch.setattr(completion, "load_sido_real_problem", lambda *args, **kwargs: _problem())
    monkeypatch.setattr(completion, "input_hashes", lambda *_: {"sido0_train.mat": "a", "sido0_train.targets": "b"})
    monkeypatch.setattr(completion, "run_egm", lambda *args, **kwargs: SimpleNamespace(rel_err_x=np.array([0.25])))
    gpu_rows = pd.DataFrame([
        {"Algorithm": "MOSEK(CVX)", "Mode": "-", "Mu": "-", "Iterations": np.nan, "FinalResidual": np.nan, "TimeSeconds": 8.0},
        {"Algorithm": "EGM", "Mode": "-", "Mu": "-", "Iterations": 5.0, "FinalResidual": 0.1, "TimeSeconds": 3.0},
        {"Algorithm": "APDB-yx", "Mode": "mono", "Mu": 2.0, "Iterations": 3.0, "FinalResidual": 1e-6, "TimeSeconds": 2.0},
    ])
    calls = []
    def fake_run(**kwargs):
        calls.append(kwargs)
        return gpu_rows, {"iter_egm": np.array([1.0]), "time_egm": np.array([3.0]), "rel_err_egm": np.array([0.1])}, {
            "EGM|-": {"iter": np.array([1.0]), "time": np.array([3.0]), "rel": np.array([0.1])},
            "APDB-yx|mono|mu=2": {"iter": np.array([1.0]), "time": np.array([2.0]), "rel": np.array([1e-6])},
        }
    monkeypatch.setattr(completion, "_single_device_run", fake_run)
    monkeypatch.setattr(completion, "load_cpu_curves_mat", lambda *_: {"EGM|-": {"iter": np.array([1.0]), "time": np.array([7.0]), "rel": np.array([0.1])}})
    def fake_plot(cpu_curves, outdir, *, gpu_curves, reference_match):
        assert gpu_curves["EGM|-"]["time"][0] == 3.0
        assert reference_match is True
        png, pdf = Path(outdir) / "comparison_plot_cpu.png", Path(outdir) / "comparison_plot_cpu.pdf"
        png.write_bytes(b"png")
        pdf.write_bytes(b"pdf")
        return png, pdf
    monkeypatch.setattr(completion, "plot_cpu_curves", fake_plot)

    completion.complete_missing_gpu(output_dir, tmp_path)

    assert len(calls) == 1
    assert calls[0]["device"] == "cuda"
    np.testing.assert_array_equal(calls[0]["optsol"], [1.0, 1.0])
    assert (output_dir / "experiment_outputs_cpu.mat").read_bytes() == before
    merged = pd.read_csv(output_dir / "summary_cpu_gpu.csv")
    assert len(merged) == 3
    assert merged.loc[merged.Algorithm == "APDB-yx", "Time_GPU"].iloc[0] == 2.0
    assert pd.isna(merged.loc[merged.Algorithm == "MOSEK(CVX)", "Time_GPU"].iloc[0])
    assert json.loads((output_dir / "run_manifest.json").read_text())["gpu_used"] is True
    completion.complete_missing_gpu(output_dir, tmp_path)
    assert len(calls) == 1


def test_changed_data_reruns_only_mosek_and_gpu(tmp_path, monkeypatch):
    import apdb_real_data_python.complete_gpu as completion

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    manifest = {"n_samples": 3, "train_size": 2, "seed": 17, "alignment_mode": "legacy_matlab", "max_iter": 5, "stop_criteria": 1e-6, "progress_mode": "off", "progress_refresh": 0.5, "progress_every": 20, "egm_tau_scale": None, "egm_tune": False, "egm_tune_max_iter": 1000, "egm_tune_grid": [0.05], "mu_values": [2.0], "mosek_tol_requested": 1e-9, "mosek_tol_effective": 1e-11, "gpu_used": False}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    savemat(output_dir / "experiment_outputs_cpu.mat", {"optsol": np.array([1.0, 1.0]), "optval": 0.0, "time_cvx": 8.0, "rel_err_egm": np.array([0.25]), "egm_tau_scale_used": 0.1})
    (output_dir / "summary_table_cpu.csv").write_text("Algorithm,Mode,Mu,Iterations,FinalResidual,TimeSeconds\nMOSEK(CVX),-,-,,,8.0\nEGM,-,-,5,0.1,7.0\n", encoding="utf-8")
    before = (output_dir / "experiment_outputs_cpu.mat").read_bytes()
    monkeypatch.setattr(completion.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(completion.torch.cuda, "get_device_name", lambda *_: "test GPU")
    monkeypatch.setattr(completion, "_require_run_parameters", lambda *_: None)
    monkeypatch.setattr(completion, "load_sido_real_problem", lambda *args, **kwargs: _problem(g_scale=2.0))
    monkeypatch.setattr(completion, "input_hashes", lambda *_: {})
    solve_calls = []
    def fake_solve(*args, **kwargs):
        solve_calls.append(kwargs)
        return np.array([2.0, 2.0]), 2.0, 9.0
    monkeypatch.setattr(completion, "solve_reference_mosek", fake_solve)
    run_calls = []
    def fake_run(**kwargs):
        run_calls.append(kwargs)
        rows = pd.DataFrame([
            {"Algorithm": "MOSEK(CVX)", "Mode": "-", "Mu": "-", "Iterations": np.nan, "FinalResidual": np.nan, "TimeSeconds": 9.0},
            {"Algorithm": "EGM", "Mode": "-", "Mu": "-", "Iterations": 5.0, "FinalResidual": 0.1, "TimeSeconds": 3.0},
        ])
        return rows, {"iter_egm": np.array([1.0]), "time_egm": np.array([3.0]), "rel_err_egm": np.array([0.1])}, {"EGM|-": {"iter": np.array([1.0]), "time": np.array([3.0]), "rel": np.array([0.1])}}
    monkeypatch.setattr(completion, "_single_device_run", fake_run)
    monkeypatch.setattr(completion, "load_cpu_curves_mat", lambda *_: {"EGM|-": {"iter": np.array([1.0]), "time": np.array([7.0]), "rel": np.array([0.1])}})
    plot_flags = []
    def fake_plot(cpu_curves, outdir, *, gpu_curves, reference_match):
        plot_flags.append(reference_match)
        png, pdf = Path(outdir) / "comparison_plot_cpu.png", Path(outdir) / "comparison_plot_cpu.pdf"
        png.write_bytes(b"png")
        pdf.write_bytes(b"pdf")
        return png, pdf
    monkeypatch.setattr(completion, "plot_cpu_curves", fake_plot)

    completion.complete_missing_gpu(output_dir, tmp_path)

    assert len(solve_calls) == 1
    assert solve_calls[0]["mosek_tol"] == 1e-11
    assert len(run_calls) == 1
    np.testing.assert_array_equal(run_calls[0]["optsol"], [2.0, 2.0])
    assert plot_flags == [False]
    assert (output_dir / "experiment_outputs_cpu.mat").read_bytes() == before
    merged = pd.read_csv(output_dir / "summary_cpu_gpu.csv")
    assert merged["ReferenceMatch"].eq(False).all()
    new_manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert new_manifest["gpu_completion"]["reference_match"] is False


def test_gpu_failure_does_not_publish_completion(tmp_path, monkeypatch):
    import apdb_real_data_python.complete_gpu as completion

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    manifest = {"n_samples": 3, "train_size": 2, "seed": 17, "alignment_mode": "legacy_matlab", "max_iter": 5, "stop_criteria": 1e-6, "mu_values": [2.0], "gpu_used": False}
    original = json.dumps(manifest).encode()
    (output_dir / "run_manifest.json").write_bytes(original)
    (output_dir / "summary_table_cpu.csv").write_text("Algorithm,Mode,Mu,Iterations,FinalResidual,TimeSeconds\nMOSEK(CVX),-,-,,,8\nEGM,-,-,5,0.1,7\n")
    savemat(output_dir / "experiment_outputs_cpu.mat", {"optsol": [1.0, 1.0], "optval": 0.0, "time_cvx": 8.0, "rel_err_egm": [0.25], "egm_tau_scale_used": 0.1})
    cpu_bytes = (output_dir / "experiment_outputs_cpu.mat").read_bytes()
    monkeypatch.setattr(completion.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(completion, "_require_run_parameters", lambda *_: None)
    monkeypatch.setattr(completion, "input_hashes", lambda *_: {})
    monkeypatch.setattr(completion, "load_sido_real_problem", lambda *args, **kwargs: _problem())
    monkeypatch.setattr(completion, "run_egm", lambda *args, **kwargs: SimpleNamespace(rel_err_x=np.array([0.25])))
    def fail_gpu(**kwargs):
        raise RuntimeError("simulated GPU failure")
    monkeypatch.setattr(completion, "_single_device_run", fail_gpu)

    with pytest.raises(RuntimeError, match="simulated GPU failure"):
        completion.complete_missing_gpu(output_dir, tmp_path)

    assert (output_dir / "run_manifest.json").read_bytes() == original
    assert (output_dir / "experiment_outputs_cpu.mat").read_bytes() == cpu_bytes
    assert not (output_dir / "experiment_outputs_gpu.mat").exists()


def test_launcher_calls_only_gpu_completion():
    root = Path(__file__).resolve().parents[2]
    text = (root / "run_apdb_real_data_n10000_complete_gpu.sh").read_text()
    assert "apdb_real_data_python.complete_gpu" in text
    assert "main_real_data" not in text
    assert 'OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"' in text


def test_original_run_parameters_are_checked():
    from apdb_real_data_python.complete_gpu import _require_run_parameters
    from apdb_real_data_python.runner import RESTART_CONFIG

    manifest = {
        "data": "sido0_train", "run_all": True, "n_samples": 12500, "train_size": 10000,
        "seed": 17, "alignment_mode": "legacy_matlab", "max_iter": 14999,
        "stop_criteria": 1e-6, "mu_values": [2.0], "mosek_tol_effective": 1e-11,
        "restart_config": RESTART_CONFIG, "x_old_policy": "frozen_x0",
    }
    _require_run_parameters(manifest)
    with pytest.raises(ValueError, match="sample"):
        _require_run_parameters({**manifest, "n_samples": 12501})
