import importlib
import json

import numpy as np
import pytest
import torch
from scipy.io import savemat


def _toy_problem(m: int = 4, n: int = 2):
    rng = np.random.default_rng(0)
    a = np.eye(m) * 0.5
    q_list = [np.eye(m) * (0.2 + 0.1 * i) for i in range(n)]
    b = rng.standard_normal(m)
    d = rng.standard_normal((n, m))
    c = np.ones(n) * 0.5
    return {
        "A": a,
        "Q": q_list,
        "b": b,
        "d": d,
        "c": c,
        "sc": 0.0,
    }


def test_core_projection_and_constraints_shape():
    core = importlib.import_module("apdb_python.core")

    x = torch.tensor([-20.0, 0.5, 30.0], dtype=torch.float64)
    clipped = core.project_box(x)
    assert torch.allclose(clipped, torch.tensor([-10.0, 0.5, 10.0], dtype=torch.float64))

    problem = _toy_problem(m=3, n=2)
    xt = torch.zeros(3, dtype=torch.float64)
    vals = core.constraint_values(xt, core.to_torch_problem(problem, device="cpu"))
    assert vals.shape == (2,)


def test_smoothed_gap_qp_solve_uses_non_cvxpy_backend():
    core = importlib.import_module("apdb_python.core")
    assert "cp" not in core.__dict__

    osqp = pytest.importorskip("osqp")
    assert osqp is not None

    problem = _toy_problem(m=3, n=2)
    np_prob = core.to_numpy_problem(problem)
    x0 = np.zeros(3, dtype=np.float64)
    y0 = np.zeros(2, dtype=np.float64)

    x_star, fval = core.smooth_gap_qp_solve(np_prob, x0, y0, xi=0.04)
    assert x_star.shape == (3,)
    assert np.all(np.isfinite(x_star))
    assert np.isfinite(fval)


def test_egm_smoke_outputs_are_consistent():
    egm = importlib.import_module("apdb_python.solvers.egm")

    problem = _toy_problem()
    x0 = np.zeros(4)
    y0 = np.zeros(2)

    out = egm.run_egm(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=1e-8,
        max_iter=10,
        opts={"tau": 1e-2, "epoch": 1},
        device="cpu",
    )

    assert len(out["rel_infeas_err"]) == len(out["rel_subopt_err"]) == len(out["oracle"])
    assert out["oracle"][-1] >= 2


def test_main_parser_has_numsim_argument():
    main_mod = importlib.import_module("apdb_python.main_qcqp")
    parser = main_mod.build_arg_parser()
    args = parser.parse_args(
        [
            "--numsim",
            "2",
            "--problem-mat",
            "case.mat",
            "--plot-style",
            "strict",
            "--strict-mosek",
            "--run-all",
            "--no-mosek-high-accuracy",
            "--mosek-tol",
            "1e-11",
        ]
    )
    default_args = parser.parse_args([])
    assert args.numsim == 2
    assert args.problem_mat == "case.mat"
    assert args.plot_style == "strict"
    assert args.strict_mosek is True
    assert args.run_all is True
    assert args.no_mosek_high_accuracy is True
    assert args.mosek_tol == pytest.approx(1e-11)
    assert default_args.m == 90
    assert default_args.run_all is False
    assert default_args.no_mosek_high_accuracy is False
    assert default_args.mosek_tol == pytest.approx(1e-10)


def test_data_gen_uses_matlab_style_mt19937():
    data_gen = importlib.import_module("apdb_python.data_gen")
    rng = data_gen.make_rng(123)
    assert rng.bit_generator.__class__.__name__ == "MT19937"


def test_load_problem_from_mat_roundtrip(tmp_path):
    io_utils = importlib.import_module("apdb_python.io_utils")

    mat_path = tmp_path / "case.mat"
    payload = {
        "A": np.eye(3, dtype=np.float64),
        "Q": np.array([[np.eye(3, dtype=np.float64)], [2.0 * np.eye(3, dtype=np.float64)]], dtype=object),
        "b": np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
        "d": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        "c": np.array([[1.5], [2.5]], dtype=np.float64),
        "sc": np.array([[0.0]], dtype=np.float64),
        "x0": np.array([[0.0], [0.1], [0.2]], dtype=np.float64),
        "y0": np.array([[0.0], [0.0]], dtype=np.float64),
        "optval": np.array([[9.0]], dtype=np.float64),
    }
    savemat(mat_path, payload)

    loaded = io_utils.load_problem_from_mat(mat_path)
    assert loaded["problem"]["A"].shape == (3, 3)
    assert len(loaded["problem"]["Q"]) == 2
    assert loaded["x0"].shape == (3,)
    assert loaded["y0"].shape == (2,)
    assert loaded["optval"] == pytest.approx(9.0)


def test_reference_solver_uses_psd_wrap(monkeypatch):
    ref = importlib.import_module("apdb_python.solvers.reference_cvxpy")
    cp = importlib.import_module("cvxpy")

    calls = []
    orig = cp.psd_wrap

    def _wrapped(m):
        calls.append(np.asarray(m).shape)
        return orig(m)

    monkeypatch.setattr(cp, "psd_wrap", _wrapped)

    problem = {
        "A": np.eye(3, dtype=np.float64),
        "Q": [0.5 * np.eye(3, dtype=np.float64), 0.8 * np.eye(3, dtype=np.float64)],
        "b": np.zeros(3, dtype=np.float64),
        "d": np.zeros((2, 3), dtype=np.float64),
        "c": np.ones(2, dtype=np.float64),
        "sc": 0.0,
    }
    ref.solve_qcqp_reference(problem)
    assert len(calls) >= 1 + len(problem["Q"])


def test_reference_solver_high_accuracy_default_passes_mosek_kwargs(monkeypatch):
    ref = importlib.import_module("apdb_python.solvers.reference_cvxpy")
    cp = importlib.import_module("cvxpy")

    solve_calls = []
    orig_solve = cp.Problem.solve

    def _fake_solve(self, *args, **kwargs):
        solve_calls.append(kwargs.copy())
        for v in self.variables():
            v.value = np.zeros(v.shape, dtype=np.float64)
        self._value = 0.0
        return 0.0

    monkeypatch.setattr(cp.Problem, "solve", _fake_solve)

    problem = {
        "A": np.eye(3, dtype=np.float64),
        "Q": [0.5 * np.eye(3, dtype=np.float64), 0.8 * np.eye(3, dtype=np.float64)],
        "b": np.zeros(3, dtype=np.float64),
        "d": np.zeros((2, 3), dtype=np.float64),
        "c": np.ones(2, dtype=np.float64),
        "sc": 0.0,
    }
    _, _, _ = ref.solve_qcqp_reference(problem)
    assert len(solve_calls) == 1
    call = solve_calls[0]
    assert call["solver"] == cp.MOSEK
    assert "eps" not in call
    assert call["mosek_params"]["MSK_DPAR_INTPNT_CO_TOL_PFEAS"] == pytest.approx(1e-10)
    assert call["mosek_params"]["MSK_DPAR_INTPNT_CO_TOL_DFEAS"] == pytest.approx(1e-10)
    assert call["mosek_params"]["MSK_DPAR_INTPNT_CO_TOL_REL_GAP"] == pytest.approx(1e-10)

    monkeypatch.setattr(cp.Problem, "solve", orig_solve)


def test_reference_solver_can_disable_high_accuracy(monkeypatch):
    ref = importlib.import_module("apdb_python.solvers.reference_cvxpy")
    cp = importlib.import_module("cvxpy")

    solve_calls = []
    orig_solve = cp.Problem.solve

    def _fake_solve(self, *args, **kwargs):
        solve_calls.append(kwargs.copy())
        for v in self.variables():
            v.value = np.zeros(v.shape, dtype=np.float64)
        self._value = 0.0
        return 0.0

    monkeypatch.setattr(cp.Problem, "solve", _fake_solve)

    problem = {
        "A": np.eye(3, dtype=np.float64),
        "Q": [0.5 * np.eye(3, dtype=np.float64), 0.8 * np.eye(3, dtype=np.float64)],
        "b": np.zeros(3, dtype=np.float64),
        "d": np.zeros((2, 3), dtype=np.float64),
        "c": np.ones(2, dtype=np.float64),
        "sc": 0.0,
    }
    _, _, _ = ref.solve_qcqp_reference(problem, high_accuracy=False)
    assert len(solve_calls) == 1
    call = solve_calls[0]
    assert call["solver"] == cp.MOSEK
    assert "eps" not in call
    assert "mosek_params" not in call

    monkeypatch.setattr(cp.Problem, "solve", orig_solve)


def test_tune_gamma_xy_family_uses_terminal_subopt_mode0_only():
    tune_mod = importlib.import_module("apdb_python.solvers.apdb_tune_gamma_xy")

    def _runner(
        input_data, optval, x0, y0, stop_criteria, max_iter, non_mono, device="cpu", gamma0=1.0, ls_debug=False
    ):
        if int(non_mono) != 0:
            raise RuntimeError("mode must be 0")
        base = abs(float(gamma0) - 1.0)
        sub = np.asarray([base + 0.2], dtype=np.float64)
        inf = np.asarray([0.0], dtype=np.float64)
        return {"rel_subopt_err": sub, "rel_infeas_err": inf}

    best_gamma, info = tune_mod.tune_gamma_xy_family(
        runner=_runner,
        input_data={"dummy": True},
        optval=0.0,
        x0=np.zeros(3, dtype=np.float64),
        y0=np.zeros(2, dtype=np.float64),
        epsilon=1e-8,
        max_iter=20,
        gamma_grid=np.asarray([0.3, 1.0, 3.0], dtype=np.float64),
        tune_iter=10,
        device="cpu",
    )

    assert best_gamma == pytest.approx(1.0)
    assert info["best_score"] == pytest.approx(0.2)
    assert info["best_subopt_mode0"] == pytest.approx(0.2)


def test_tune_gamma_xy_family_prints_debug_and_raises_on_failed_candidate(capsys):
    tune_mod = importlib.import_module("apdb_python.solvers.apdb_tune_gamma_xy")

    def _runner(
        input_data, optval, x0, y0, stop_criteria, max_iter, non_mono, device="cpu", gamma0=1.0, ls_debug=False
    ):
        if float(gamma0) == 1.0 and int(non_mono) == 0:
            raise RuntimeError("debug-failure")
        return {
            "rel_subopt_err": np.asarray([1.0], dtype=np.float64),
            "rel_infeas_err": np.asarray([0.0], dtype=np.float64),
        }

    with pytest.raises(RuntimeError, match="failed during XY gamma tuning"):
        tune_mod.tune_gamma_xy_family(
            runner=_runner,
            input_data={"dummy": True},
            optval=0.0,
            x0=np.zeros(3, dtype=np.float64),
            y0=np.zeros(2, dtype=np.float64),
            epsilon=1e-8,
            max_iter=20,
            gamma_grid=np.asarray([0.3, 1.0, 3.0], dtype=np.float64),
            tune_iter=10,
            device="cpu",
            sim_idx=0,
        )
    captured = capsys.readouterr()
    assert "XY gamma tuning failed" in captured.out
    assert "gamma=1.0" in captured.out
    assert "mode=0" in captured.out
    assert "debug-failure" in captured.out


def test_tune_gamma_yx_family_uses_terminal_subopt_mode0_only():
    tune_mod = importlib.import_module("apdb_python.solvers.apdb_tune_gamma_yx")

    def _runner(input_data, optval, x0, y0, stop_criteria, max_iter, non_mono, device="cpu", gamma0=1.0):
        if int(non_mono) != 0:
            raise RuntimeError("mode must be 0")
        base = abs(float(gamma0) - 3.0)
        sub = np.asarray([base + 0.4], dtype=np.float64)
        inf = np.asarray([0.0], dtype=np.float64)
        return {"rel_subopt_err": sub, "rel_infeas_err": inf}

    best_gamma, info = tune_mod.tune_gamma_yx_family(
        runner=_runner,
        input_data={"dummy": True},
        optval=0.0,
        x0=np.zeros(3, dtype=np.float64),
        y0=np.zeros(2, dtype=np.float64),
        epsilon=1e-8,
        max_iter=20,
        gamma_grid=np.asarray([1.0, 3.0, 6.0], dtype=np.float64),
        tune_iter=10,
        device="cpu",
    )

    assert best_gamma == pytest.approx(3.0)
    assert info["best_score"] == pytest.approx(0.4)
    assert info["best_subopt_mode0"] == pytest.approx(0.4)


def test_tune_gamma_yx_family_prints_debug_and_raises_on_failed_candidate(capsys):
    tune_mod = importlib.import_module("apdb_python.solvers.apdb_tune_gamma_yx")

    def _runner(input_data, optval, x0, y0, stop_criteria, max_iter, non_mono, device="cpu", gamma0=1.0):
        if float(gamma0) == 1.0 and int(non_mono) == 0:
            raise RuntimeError("debug-failure")
        return {
            "rel_subopt_err": np.asarray([1.0], dtype=np.float64),
            "rel_infeas_err": np.asarray([0.0], dtype=np.float64),
        }

    with pytest.raises(RuntimeError, match="failed during YX gamma tuning"):
        tune_mod.tune_gamma_yx_family(
            runner=_runner,
            input_data={"dummy": True},
            optval=0.0,
            x0=np.zeros(3, dtype=np.float64),
            y0=np.zeros(2, dtype=np.float64),
            epsilon=1e-8,
            max_iter=20,
            gamma_grid=np.asarray([0.3, 1.0, 3.0], dtype=np.float64),
            tune_iter=10,
            device="cpu",
            sim_idx=0,
        )
    captured = capsys.readouterr()
    assert "YX gamma tuning failed" in captured.out
    assert "gamma=1.0" in captured.out
    assert "mode=0" in captured.out
    assert "debug-failure" in captured.out


def test_yx_family_accepts_gamma0_parameter():
    yx = importlib.import_module("apdb_python.solvers.apdb_c_lr")
    yx_restart = importlib.import_module("apdb_python.solvers.apdb_c_lr_restart")
    yx_ada = importlib.import_module("apdb_python.solvers.apdb_c_lr_ada_restart")
    problem = _toy_problem(m=4, n=2)
    x0 = np.zeros(4, dtype=np.float64)
    y0 = np.zeros(2, dtype=np.float64)

    out_yx = yx.run_apdb_c_lr(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=1e-8,
        max_iter=5,
        non_mono=0,
        device="cpu",
        gamma0=3.0,
    )
    out_yx_restart = yx_restart.run_apdb_c_lr_restart(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=1e-8,
        max_iter=5,
        non_mono=0,
        k_restart=2,
        device="cpu",
        gamma0=3.0,
    )
    out_yx_ada = yx_ada.run_apdb_c_lr_ada_restart(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=1e-8,
        max_iter=5,
        non_mono=0,
        device="cpu",
        gamma0=3.0,
    )

    for out in [out_yx, out_yx_restart, out_yx_ada]:
        assert "rel_infeas_err" in out
        assert "rel_subopt_err" in out
        assert "time_period" in out
        assert "iter_epoch" in out
        assert "oracle" in out


def test_ada_restart_solvers_return_gxi_time_total():
    lr_ada = importlib.import_module("apdb_python.solvers.apdb_c_lr_ada_restart")
    xy_ada = importlib.import_module("apdb_python.solvers.apdb_c_xy_ada_restart")
    problem = _toy_problem(m=4, n=2)
    x0 = np.zeros(4, dtype=np.float64)
    y0 = np.zeros(2, dtype=np.float64)

    out_lr = lr_ada.run_apdb_c_lr_ada_restart(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=1e-8,
        max_iter=10,
        non_mono=0,
        device="cpu",
    )
    out_xy = xy_ada.run_apdb_c_xy_ada_restart(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=1e-8,
        max_iter=10,
        non_mono=0,
        device="cpu",
    )

    assert "gxi_check_restart_time_total" in out_lr
    assert "gxi_check_restart_time_total" in out_xy
    assert "restart_iter_epoch" in out_lr
    assert "restart_oracle" in out_lr
    assert "restart_iter_epoch" in out_xy
    assert "restart_oracle" in out_xy
    assert float(out_lr["gxi_check_restart_time_total"]) >= 0.0
    assert float(out_xy["gxi_check_restart_time_total"]) >= 0.0
    assert len(out_lr["restart_iter_epoch"]) == len(out_lr["restart_oracle"])
    assert len(out_xy["restart_iter_epoch"]) == len(out_xy["restart_oracle"])


def test_yx_ada_mono_checks_gxi_every_500_iterations(monkeypatch):
    lr_ada = importlib.import_module("apdb_python.solvers.apdb_c_lr_ada_restart")
    problem = _toy_problem(m=4, n=2)
    x0 = np.zeros(4, dtype=np.float64)
    y0 = np.zeros(2, dtype=np.float64)
    calls = {"mono": 0, "nonmono": 0}

    def _fake_gap(*_args, **_kwargs):
        calls[mode] += 1
        return 1.0

    monkeypatch.setattr(lr_ada, "smoothed_duality_gap", _fake_gap)

    mode = "mono"
    lr_ada.run_apdb_c_lr_ada_restart(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=-1.0,
        max_iter=210,
        non_mono=0,
        device="cpu",
    )

    mode = "nonmono"
    lr_ada.run_apdb_c_lr_ada_restart(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=-1.0,
        max_iter=210,
        non_mono=1,
        device="cpu",
    )

    assert calls == {"mono": 1, "nonmono": 2}


def test_xy_family_accepts_independent_gamma_0_parameter():
    xy = importlib.import_module("apdb_python.solvers.apdb_c_xy")
    xy_restart = importlib.import_module("apdb_python.solvers.apdb_c_xy_restart")
    xy_ada = importlib.import_module("apdb_python.solvers.apdb_c_xy_ada_restart")
    problem = _toy_problem(m=4, n=2)
    x0 = np.zeros(4, dtype=np.float64)
    y0 = np.zeros(2, dtype=np.float64)

    out_xy = xy.run_apdb_c_xy(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=1e-8,
        max_iter=5,
        non_mono=0,
        device="cpu",
        gamma0=3.0,
        gamma_0=1.0,
    )
    out_xy_restart = xy_restart.run_apdb_c_xy_restart(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=1e-8,
        max_iter=5,
        non_mono=0,
        k_restart=2,
        device="cpu",
        gamma0=3.0,
        gamma_0=1.0,
    )
    out_xy_ada = xy_ada.run_apdb_c_xy_ada_restart(
        input_data=problem,
        optval=0.0,
        x0=x0,
        y0=y0,
        stop_criteria=1e-8,
        max_iter=5,
        non_mono=0,
        device="cpu",
        gamma0=3.0,
        gamma_0=1.0,
    )

    for out in [out_xy, out_xy_restart, out_xy_ada]:
        assert "rel_infeas_err" in out
        assert "rel_subopt_err" in out
        assert "time_period" in out
        assert "iter_epoch" in out
        assert "oracle" in out


def test_run_for_device_summary_contains_gxitime_for_ada(monkeypatch, tmp_path):
    main_mod = importlib.import_module("apdb_python.main_qcqp")

    def _hist():
        return {
            "rel_infeas_err": np.array([1.0, 1e-1], dtype=np.float64),
            "rel_subopt_err": np.array([1.0, 1e-1], dtype=np.float64),
            "time_period": np.array([1e-2, 2e-2], dtype=np.float64),
            "iter_epoch": np.array([1.0, 2.0], dtype=np.float64),
            "oracle": np.array([2.0, 4.0], dtype=np.float64),
        }

    def _fake_tune_tau_egm(**kwargs):
        base = _hist()
        return 1e-3, base, {"tau": [1e-3]}

    def _fake_lr(*args, **kwargs):
        assert kwargs.get("gamma0") == pytest.approx(6.0)
        return _hist()

    def _fake_lr_restart(*args, **kwargs):
        assert kwargs.get("gamma0") == pytest.approx(6.0)
        return _hist()

    def _fake_xy(*args, **kwargs):
        assert kwargs.get("gamma0") == pytest.approx(3.0)
        return _hist()

    def _fake_xy_restart(*args, **kwargs):
        assert kwargs.get("gamma0") == pytest.approx(3.0)
        return _hist()

    def _fake_lr_ada(*args, **kwargs):
        assert kwargs.get("gamma0") == pytest.approx(6.0)
        non_mono = int(args[6])
        out = _hist()
        out["gxi_check_restart_time_total"] = 1.0 if non_mono == 0 else 2.0
        return out

    def _fake_xy_ada(*args, **kwargs):
        assert kwargs.get("gamma0") == pytest.approx(3.0)
        non_mono = int(args[6])
        out = _hist()
        out["gxi_check_restart_time_total"] = 3.0 if non_mono == 0 else 4.0
        return out

    monkeypatch.setattr(main_mod, "tune_tau_egm", _fake_tune_tau_egm)
    monkeypatch.setattr(main_mod, "run_apdb_c_lr", _fake_lr)
    monkeypatch.setattr(main_mod, "run_apdb_c_lr_restart", _fake_lr_restart)
    monkeypatch.setattr(main_mod, "run_apdb_c_xy", _fake_xy)
    monkeypatch.setattr(main_mod, "run_apdb_c_xy_restart", _fake_xy_restart)
    monkeypatch.setattr(main_mod, "run_apdb_c_lr_ada_restart", _fake_lr_ada)
    monkeypatch.setattr(main_mod, "run_apdb_c_xy_ada_restart", _fake_xy_ada)
    monkeypatch.setattr(main_mod, "plot_results", lambda *args, **kwargs: [])

    args = main_mod.build_arg_parser().parse_args(["--numsim", "1", "--max-iter", "10", "--outdir", str(tmp_path)])
    context = {
        "n_run": 2,
        "m_run": 3,
        "x0_base": np.zeros(3, dtype=np.float64),
        "y0_base": np.zeros(2, dtype=np.float64),
        "problems": [{"dummy": True}],
        "optvals": np.array([0.0], dtype=np.float64),
        "mosek_time": np.array([0.0], dtype=np.float64),
        "Total_time_Mosek": 0.0,
        "best_gamma_xy_sim": np.array([3.0], dtype=np.float64),
        "best_gamma_yx_sim": np.array([6.0], dtype=np.float64),
    }
    out = main_mod._run_for_device(args=args, context=context, device="cpu", outdir=tmp_path, save_legacy=False)
    df = out["summary"]

    assert "GxiTime" in df.columns

    row_mono_lr_ada = df[(df["Group"] == "Monotone") & (df["Algorithm"] == "rAPDB-yx-ada")].iloc[0]
    row_nonmono_lr_ada = df[(df["Group"] == "Nonmonotone") & (df["Algorithm"] == "rAPDB-yx-ada")].iloc[0]
    row_mono_xy_ada = df[(df["Group"] == "Monotone") & (df["Algorithm"] == "rAPDB-xy-ada")].iloc[0]
    row_nonmono_xy_ada = df[(df["Group"] == "Nonmonotone") & (df["Algorithm"] == "rAPDB-xy-ada")].iloc[0]
    row_mono_lr = df[(df["Group"] == "Monotone") & (df["Algorithm"] == "APDB-yx")].iloc[0]

    assert row_mono_lr_ada["GxiTime"] == pytest.approx(1.0)
    assert row_nonmono_lr_ada["GxiTime"] == pytest.approx(2.0)
    assert row_mono_xy_ada["GxiTime"] == pytest.approx(3.0)
    assert row_nonmono_xy_ada["GxiTime"] == pytest.approx(4.0)
    assert np.isnan(row_mono_lr["GxiTime"])


def test_plot_results_both_style_writes_strict_pdf(tmp_path):
    plotting = importlib.import_module("apdb_python.plotting")

    one = [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)]
    two = [[np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)] for _ in range(2)]
    vals = [np.array([1e0, 1e-1, 1e-2, 1e-3], dtype=np.float64)]
    vals2 = [[np.array([1e0, 1e-1, 1e-2, 1e-3], dtype=np.float64)] for _ in range(2)]
    restart_epochs = [[np.array([400.0, 800.0, 1000.0, 2000.0], dtype=np.float64)] for _ in range(2)]
    results = {
        "oracle_egm": one,
        "rel_subopt_egm": vals,
        "rel_infeas_err_egm": vals,
        "oracle_1": two,
        "rel_subopt_1": vals2,
        "rel_infeas_1": vals2,
        "oracle_1_ada": two,
        "rel_subopt_1_ada": vals2,
        "rel_infeas_1_ada": vals2,
        "oracle_r1": two,
        "iter_epoch_r1": restart_epochs,
        "rel_subopt_r1": vals2,
        "rel_infeas_r1": vals2,
        "oracle_xy_1": two,
        "rel_subopt_xy_1": vals2,
        "rel_infeas_xy_1": vals2,
        "oracle_xy_ada1": two,
        "rel_subopt_xy_ada1": vals2,
        "rel_infeas_xy_ada1": vals2,
        "oracle_xy_r1": two,
        "iter_epoch_xy_r1": restart_epochs,
        "rel_subopt_xy_r1": vals2,
        "rel_infeas_xy_r1": vals2,
    }

    out = plotting.plot_results(results, numsim=1, n=10, m=90, device="cpu", outdir=tmp_path, plot_style="both")
    assert len(out) == 3
    assert out[0].name.endswith("_strict.png")
    assert out[1].name.endswith("_strict.pdf")
    assert out[2].name.endswith("_custom.png")
    assert all(p.exists() and p.stat().st_size > 0 for p in out)


def test_plot_style_strict_config_matches_matlab():
    plotting = importlib.import_module("apdb_python.plotting")
    restart_color = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    cfg = plotting._style_config("strict")
    assert cfg["linewidth"] == pytest.approx(2.2)
    assert cfg["markersize"] == pytest.approx(12.0)
    assert cfg["nonmono_markers"] == [("o", 500, 0), ("^", 700, 0), ("x", 800, 0)]
    assert cfg["infeas_legend_last"] == "rAPDB-xy"
    assert cfg["restart_color"] == restart_color
    assert cfg["axis_fontsize"] == 16
    assert cfg["axis_linewidth"] == pytest.approx(1.2)
    assert cfg["label_fontsize"] == 16
    assert cfg["top_legend_fontsize"] == 14
    assert cfg["bottom_right_legend_fontsize"] == 14
    assert cfg["legend_fontsize"] == 11
    assert cfg["legend_loc"] == "upper right"


def test_plot_style_strict_uses_matlab_like_labels_and_axes():
    plotting = importlib.import_module("apdb_python.plotting")
    restart_color = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    one = [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)]
    two = [[np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)] for _ in range(2)]
    vals = [np.array([1e0, 1e-1, 1e-2, 1e-3], dtype=np.float64)]
    vals2 = [[np.array([1e0, 1e-1, 1e-2, 1e-3], dtype=np.float64)] for _ in range(2)]
    infeas_vals2 = [
        [np.array([1e0, 1e-1, 1e-2, 1e-3], dtype=np.float64)],
        [np.array([1e4, 1e0, 1e-4, 1e-8], dtype=np.float64)],
    ]
    restart_epochs = [[np.array([400.0, 800.0, 1000.0, 2000.0], dtype=np.float64)] for _ in range(2)]
    results = {
        "oracle_egm": one,
        "rel_subopt_egm": vals,
        "rel_infeas_err_egm": vals,
        "oracle_1": two,
        "rel_subopt_1": vals2,
        "rel_infeas_1": infeas_vals2,
        "oracle_1_ada": two,
        "rel_subopt_1_ada": vals2,
        "rel_infeas_1_ada": infeas_vals2,
        "oracle_r1": two,
        "iter_epoch_r1": restart_epochs,
        "rel_subopt_r1": vals2,
        "rel_infeas_r1": infeas_vals2,
        "oracle_xy_1": two,
        "rel_subopt_xy_1": vals2,
        "rel_infeas_xy_1": infeas_vals2,
        "oracle_xy_ada1": two,
        "rel_subopt_xy_ada1": vals2,
        "rel_infeas_xy_ada1": infeas_vals2,
        "oracle_xy_r1": two,
        "iter_epoch_xy_r1": restart_epochs,
        "rel_subopt_xy_r1": vals2,
        "rel_infeas_xy_r1": infeas_vals2,
    }

    fig = plotting._plot_one_style(results, numsim=1, style_cfg=plotting._style_config("strict"))
    ax00 = fig.axes[0]
    ax01 = fig.axes[1]
    ax10 = fig.axes[2]
    ax11 = fig.axes[3]
    lg0 = ax00.get_legend()
    lg1 = ax01.get_legend()
    lg2 = ax10.get_legend()
    lg3 = ax11.get_legend()

    assert ax00.get_xlabel() == "Number of gradient computations"
    assert ax00.get_ylabel() == r"$|f(x^k)-f^\star|/(1+|f^\star|)$"
    assert ax01.get_ylabel() == r"$\frac{1}{m}\sum_{i=1}^m \max\{g_i(x^k),0\}$"
    assert lg0 is not None and lg1 is not None and lg2 is not None and lg3 is not None
    assert lg0.get_texts()[-1].get_text() == "rAPDB-xy"
    assert lg1.get_texts()[-1].get_text() == "rAPDB-xy"
    lg0_handles = lg0.legend_handles if hasattr(lg0, "legend_handles") else lg0.legendHandles
    assert lg0_handles[3].get_marker() == "x"
    assert lg0_handles[6].get_marker() == "x"
    assert lg0_handles[1].get_marker() in {None, "None", ""}
    lg1_handles = lg1.legend_handles if hasattr(lg1, "legend_handles") else lg1.legendHandles
    assert lg1_handles[3].get_marker() == "x"
    assert lg1_handles[6].get_marker() == "x"
    assert lg1_handles[1].get_marker() in {None, "None", ""}
    for ax, yx_idx, xy_idx in [(fig.axes[0], 3, 6), (fig.axes[1], 3, 6), (fig.axes[2], 2, 5), (fig.axes[3], 2, 5)]:
        assert ax.lines[yx_idx].get_color() == restart_color
        assert ax.lines[xy_idx].get_color() == restart_color
    assert fig.axes[2].lines[6].get_marker() == "x"
    assert fig.axes[2].lines[7].get_marker() == "x"
    assert fig.axes[3].lines[6].get_marker() == "x"
    assert fig.axes[3].lines[7].get_marker() == "x"
    assert lg0.get_texts()[0].get_fontsize() == pytest.approx(14.0)
    assert lg1.get_texts()[0].get_fontsize() == pytest.approx(14.0)
    assert lg2.get_texts()[0].get_fontsize() == pytest.approx(11.0)
    assert lg3.get_texts()[0].get_fontsize() == pytest.approx(14.0)
    assert ax11.get_ylim() == pytest.approx(ax10.get_ylim())
    assert ax11.get_yticks() == pytest.approx(ax10.get_yticks())
    assert ax00.spines["left"].get_linewidth() == pytest.approx(1.2)
    assert ax00.xaxis.get_ticklabels()[0].get_fontsize() == pytest.approx(16.0)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_main_masks_nonpositive_values_for_infeasibility():
    plotting = importlib.import_module("apdb_python.plotting")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    line = plotting._plot_main(
        ax=ax,
        oracle_list=[np.array([1.0, 2.0, 3.0], dtype=np.float64)],
        rel_list=[np.array([1.0, 0.0, -1.0], dtype=np.float64)],
        color=(0.0, 0.0, 1.0),
        linestyle="-",
        numsim=1,
        mask_nonpositive=True,
    )
    assert line is not None
    y = np.asarray(line.get_ydata(), dtype=np.float64)
    assert np.isnan(y).any()
    plt.close(fig)


def test_build_device_compare_table_cpu_only_has_nan_gpu():
    main_mod = importlib.import_module("apdb_python.main_qcqp")
    cpu_df = importlib.import_module("pandas").DataFrame(
        [
            {"Group": "EGM", "Algorithm": "EGM", "Time": 1.0, "Iter": 10.0, "SuboptRes": 1e-2, "InfeasRes": 2e-2},
            {"Group": "Monotone", "Algorithm": "APDB-yx", "Time": 2.0, "Iter": 20.0, "SuboptRes": 1e-3, "InfeasRes": 2e-3},
        ]
    )
    compare = main_mod._build_device_compare_table(cpu_df, None)
    assert compare.shape[0] == 2
    assert np.all(np.isnan(compare["Time_gpu"].to_numpy(dtype=np.float64)))
    assert np.all(np.isnan(compare["Time_abs_diff"].to_numpy(dtype=np.float64)))
    assert np.all(np.isnan(compare["Time_rel_diff"].to_numpy(dtype=np.float64)))


def test_run_experiment_run_all_cpu_only(monkeypatch, tmp_path):
    main_mod = importlib.import_module("apdb_python.main_qcqp")
    pd = importlib.import_module("pandas")

    def _fake_prepare(_args):
        return {
            "n_run": 10,
            "m_run": 90,
            "x0_base": np.zeros(90, dtype=np.float64),
            "y0_base": np.zeros(10, dtype=np.float64),
            "problems": [],
            "optvals": np.zeros(1, dtype=np.float64),
            "mosek_time": np.zeros(1, dtype=np.float64),
            "Total_time_Mosek": 0.0,
        }

    calls = []

    def _fake_run_for_device(*, args, context, device, outdir, save_legacy):
        calls.append((device, save_legacy))
        df = pd.DataFrame(
            [
                {"Group": "EGM", "Algorithm": "EGM", "Time": 1.0, "Iter": 10.0, "SuboptRes": 1e-2, "InfeasRes": 2e-2},
                {"Group": "Monotone", "Algorithm": "APDB-yx", "Time": 2.0, "Iter": 20.0, "SuboptRes": 1e-3, "InfeasRes": 2e-3},
            ]
        )
        return {
            "device": device,
            "results": {},
            "summary": df,
            "plot_paths": [],
            "results_mat_path": outdir / f"dummy_{device}.mat",
            "summary_csv_path": outdir / f"dummy_{device}.csv",
            "summary_mat_path": outdir / f"dummy_{device}.mat",
        }

    monkeypatch.setattr(main_mod, "_prepare_run_context", _fake_prepare)
    monkeypatch.setattr(main_mod, "_run_for_device", _fake_run_for_device)
    monkeypatch.setattr(main_mod.torch.cuda, "is_available", lambda: False)

    args = main_mod.build_arg_parser().parse_args(["--run-all", "--numsim", "1", "--outdir", str(tmp_path)])
    out = main_mod.run_experiment(args)

    assert calls == [("cpu", True)]
    assert out["gpu"] is None
    assert out["overview"]["compare_csv_path"].exists()
    assert out["overview"]["compare_mat_path"].exists()
    assert out["overview"]["manifest_path"].exists()

    manifest = json.loads(out["overview"]["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["devices_ran"] == ["cpu"]
    assert manifest["gpu_skipped_reason"] == "CUDA not available"
