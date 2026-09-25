import inspect

from apdb_real_data_python.main_real_data import build_arg_parser
from apdb_real_data_python.solvers import run_apdb_yx, run_rapdb_yx, run_rapdb_yx_ada


def test_cli_defaults_and_run_all_flag():
    parser = build_arg_parser()
    args = parser.parse_args([])

    assert args.device == "cpu"
    assert args.run_all is False
    assert args.max_iter == 8000
    assert args.stop_criteria == 1e-8
    assert args.seed == 17
    assert args.n_samples == 10000
    assert args.alignment_mode == "legacy_matlab"
    assert args.progress == "auto"
    assert args.progress_refresh == 0.5
    assert args.progress_every == 20
    assert args.egm_tune is False
    assert args.egm_tau_scale is None
    assert args.egm_tune_max_iter == 1000
    assert args.egm_tune_grid == "0.05,0.1,0.2,0.5"
    assert args.mu_values == "2"

    args2 = parser.parse_args(["--run-all"])
    assert args2.run_all is True


def test_cli_progress_flags():
    parser = build_arg_parser()
    args = parser.parse_args(["--progress", "on", "--progress-refresh", "0.2", "--progress-every", "50"])

    assert args.progress == "on"
    assert args.progress_refresh == 0.2
    assert args.progress_every == 50


def test_cli_egm_tuning_flags():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--egm-tune",
            "--egm-tau-scale",
            "0.2",
            "--egm-tune-max-iter",
            "777",
            "--egm-tune-grid",
            "0.05,0.1,0.3",
        ]
    )

    assert args.egm_tune is True
    assert args.egm_tau_scale == 0.2
    assert args.egm_tune_max_iter == 777
    assert args.egm_tune_grid == "0.05,0.1,0.3"


def test_cli_mu_values_flag():
    parser = build_arg_parser()
    args = parser.parse_args(["--mu-values", "0,1,2.5"])

    assert args.mu_values == "0,1,2.5"


def test_apdb_solver_mu_defaults_match_matlab():
    for solver in (run_apdb_yx, run_rapdb_yx, run_rapdb_yx_ada):
        assert inspect.signature(solver).parameters["mu"].default == 2.0
