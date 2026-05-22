from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apdb_real_data_python.data_pipeline import KernelData, RealDataProblem
from apdb_real_data_python.runner import _single_device_run


@dataclass
class _DummyResult:
    rel_err_x: np.ndarray
    time_period: np.ndarray
    iter_epoch: np.ndarray


def _dummy_solver_result() -> _DummyResult:
    return _DummyResult(
        rel_err_x=np.asarray([1e-2], dtype=np.float64),
        time_period=np.asarray([0.01], dtype=np.float64),
        iter_epoch=np.asarray([1.0], dtype=np.float64),
    )


def _build_problem() -> RealDataProblem:
    kd = KernelData(
        y_train=np.asarray([1.0, -1.0], dtype=np.float64),
        y_test=np.asarray([1.0], dtype=np.float64),
        index_train=np.asarray([0, 1], dtype=np.int64),
        index_test=np.asarray([2], dtype=np.int64),
        kernel_tr_class=[np.eye(2, dtype=np.float64) for _ in range(3)],
        kernel_test_class=[np.ones((2, 1), dtype=np.float64) for _ in range(3)],
        g_tr_class=[np.eye(2, dtype=np.float64) for _ in range(3)],
        g_tr_max_norm=1.0,
        trace_kernels=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
    )
    return RealDataProblem(
        input_data=kd,
        reg_param=1.0,
        x0=np.zeros(2, dtype=np.float64),
        y0=np.zeros(3, dtype=np.float64),
        seed=17,
        data_name="dummy",
        n_samples=3,
        alignment_mode="legacy_matlab",
    )


def test_single_device_run_uses_distinct_restart_k_for_rapdb_variants(monkeypatch):
    import apdb_real_data_python.runner as runner

    calls_r = []
    calls_ra = []
    calls_egm = []
    calls_apdb = []

    def _stub_egm(*args, **kwargs):
        calls_egm.append(kwargs)
        return _dummy_solver_result()

    def _stub_apdb(*args, **kwargs):
        calls_apdb.append(kwargs)
        return _dummy_solver_result()

    def _stub_r(*args, **kwargs):
        calls_r.append(kwargs)
        return _dummy_solver_result()

    def _stub_ra(*args, **kwargs):
        calls_ra.append(kwargs)
        return _dummy_solver_result()

    monkeypatch.setattr(runner, "run_egm", _stub_egm)
    monkeypatch.setattr(runner, "run_apdb_yx", _stub_apdb)
    monkeypatch.setattr(runner, "run_rapdb_yx", _stub_r)
    monkeypatch.setattr(runner, "run_rapdb_yx_ada", _stub_ra)

    problem = _build_problem()
    progress_cfg = {"mode": "off", "refresh_sec": 0.5, "every_n": 20}
    _single_device_run(
        problem=problem,
        optsol=np.zeros(2, dtype=np.float64),
        optval=0.0,
        time_mosek=0.0,
        max_iter=5,
        stop_criteria=1e-8,
        device="cpu",
        progress_cfg=progress_cfg,
    )

    assert len(calls_egm) == 1
    assert len(calls_apdb) == 4
    assert len(calls_r) == 4
    assert len(calls_ra) == 4

    assert [c["mu"] for c in calls_apdb] == [0.0, 0.0, 2.0, 2.0]
    assert [c["mu"] for c in calls_r] == [0.0, 0.0, 2.0, 2.0]
    assert [c["mu"] for c in calls_ra] == [0.0, 0.0, 2.0, 2.0]

    # mode 0 -> mono
    assert calls_r[0]["non_mono"] == 0
    assert calls_ra[0]["non_mono"] == 0
    assert calls_r[0]["k_restart"] == 600
    assert calls_ra[0]["k_restart"] == 1000

    # mode 1 -> nonmono
    assert calls_r[1]["non_mono"] == 1
    assert calls_ra[1]["non_mono"] == 1
    assert calls_r[1]["k_restart"] == 200
    assert calls_ra[1]["k_restart"] == 200

    # progress config propagated into all solver calls
    assert calls_egm[0]["progress"]["mode"] == "off"
    assert calls_egm[0]["progress"]["every_n"] == 20
    assert calls_apdb[0]["progress"]["mode"] == "off"
    assert calls_r[0]["progress"]["mode"] == "off"
    assert calls_ra[0]["progress"]["mode"] == "off"


def test_single_device_run_adds_mu_dimension_to_rows_and_curves(monkeypatch):
    import apdb_real_data_python.runner as runner

    monkeypatch.setattr(runner, "run_egm", lambda *args, **kwargs: _dummy_solver_result())
    monkeypatch.setattr(runner, "run_apdb_yx", lambda *args, **kwargs: _dummy_solver_result())
    monkeypatch.setattr(runner, "run_rapdb_yx", lambda *args, **kwargs: _dummy_solver_result())
    monkeypatch.setattr(runner, "run_rapdb_yx_ada", lambda *args, **kwargs: _dummy_solver_result())

    df, raw, curves = runner._single_device_run(
        problem=_build_problem(),
        optsol=np.zeros(2, dtype=np.float64),
        optval=0.0,
        time_mosek=0.0,
        max_iter=5,
        stop_criteria=1e-8,
        device="cpu",
        progress_cfg={"mode": "off", "refresh_sec": 0.5, "every_n": 20},
    )

    assert "Mu" in df.columns
    assert df[df["Algorithm"] == "EGM"].iloc[0]["Mu"] == "-"
    apdb_rows = df[df["Algorithm"].isin(["APDB-yx", "rAPDB-yx", "rAPDB-yx-ada"])]
    assert set(apdb_rows["Mu"]) == {0.0, 2.0}
    assert "APDB-yx|mono|mu=0" in curves
    assert "APDB-yx|mono|mu=2" in curves
    assert "rel_err_apdb_mu0_0" in raw
    assert "rel_err_apdb_mu2_0" in raw


def test_single_device_run_tunes_egm_tau_scale_when_enabled(monkeypatch):
    import apdb_real_data_python.runner as runner

    calls_egm = []
    calls_apdb = []
    calls_r = []
    calls_ra = []

    def _stub_egm(*args, **kwargs):
        calls_egm.append(kwargs.copy())
        tau_scale = float(kwargs.get("tau_scale", 0.1))
        resid = abs(tau_scale - 0.2) + 1e-4 * float(kwargs["max_iter"])
        return _DummyResult(
            rel_err_x=np.asarray([resid], dtype=np.float64),
            time_period=np.asarray([tau_scale], dtype=np.float64),
            iter_epoch=np.asarray([1.0], dtype=np.float64),
        )

    def _stub_apdb(*args, **kwargs):
        calls_apdb.append(kwargs)
        return _dummy_solver_result()

    def _stub_r(*args, **kwargs):
        calls_r.append(kwargs)
        return _dummy_solver_result()

    def _stub_ra(*args, **kwargs):
        calls_ra.append(kwargs)
        return _dummy_solver_result()

    monkeypatch.setattr(runner, "run_egm", _stub_egm)
    monkeypatch.setattr(runner, "run_apdb_yx", _stub_apdb)
    monkeypatch.setattr(runner, "run_rapdb_yx", _stub_r)
    monkeypatch.setattr(runner, "run_rapdb_yx_ada", _stub_ra)

    problem = _build_problem()
    progress_cfg = {"mode": "off", "refresh_sec": 0.5, "every_n": 20}
    runner._single_device_run(
        problem=problem,
        optsol=np.zeros(2, dtype=np.float64),
        optval=0.0,
        time_mosek=0.0,
        max_iter=5,
        stop_criteria=1e-8,
        device="cpu",
        progress_cfg=progress_cfg,
        egm_tau_scale=None,
        egm_tune=True,
        egm_tune_max_iter=7,
        egm_tune_grid=(0.05, 0.1, 0.2, 0.5),
    )

    assert len(calls_egm) == 5
    assert [float(c["tau_scale"]) for c in calls_egm[:-1]] == [0.05, 0.1, 0.2, 0.5]
    assert all(int(c["max_iter"]) == 7 for c in calls_egm[:-1])
    assert float(calls_egm[-1]["tau_scale"]) == 0.2
    assert int(calls_egm[-1]["max_iter"]) == 5
    assert len(calls_apdb) == 4
    assert len(calls_r) == 4
    assert len(calls_ra) == 4


def test_single_device_run_uses_manual_egm_tau_scale_without_tuning(monkeypatch):
    import apdb_real_data_python.runner as runner

    calls_egm = []

    def _stub_egm(*args, **kwargs):
        calls_egm.append(kwargs.copy())
        return _dummy_solver_result()

    monkeypatch.setattr(runner, "run_egm", _stub_egm)
    monkeypatch.setattr(runner, "run_apdb_yx", lambda *args, **kwargs: _dummy_solver_result())
    monkeypatch.setattr(runner, "run_rapdb_yx", lambda *args, **kwargs: _dummy_solver_result())
    monkeypatch.setattr(runner, "run_rapdb_yx_ada", lambda *args, **kwargs: _dummy_solver_result())

    problem = _build_problem()
    progress_cfg = {"mode": "off", "refresh_sec": 0.5, "every_n": 20}
    runner._single_device_run(
        problem=problem,
        optsol=np.zeros(2, dtype=np.float64),
        optval=0.0,
        time_mosek=0.0,
        max_iter=5,
        stop_criteria=1e-8,
        device="cpu",
        progress_cfg=progress_cfg,
        egm_tau_scale=0.33,
        egm_tune=True,
        egm_tune_max_iter=7,
        egm_tune_grid=(0.05, 0.1, 0.2, 0.5),
    )

    assert len(calls_egm) == 1
    assert float(calls_egm[0]["tau_scale"]) == 0.33
    assert int(calls_egm[0]["max_iter"]) == 5
