from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "run_apdb_real_data_n10000_runall_mu2.sh"


def test_n10000_server_launcher_records_formal_experiment_settings():
    text = LAUNCHER.read_text(encoding="utf-8")

    required_fragments = (
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "$SCRIPT_DIR"',
        "python -m apdb_real_data_python.main_real_data",
        "--run-all",
        "--n-samples 12500",
        "--max-iter 14999",
        "--stop-criteria 1e-6",
        "--reg-param 1",
        "--strict-mosek",
        "--mosek-tol 1e-9",
        "--progress on",
        "--progress-every 200",
        "--mu-values 2",
        "--outdir apdb_real_data_python/outputs_n10000_runall_mu0_mu2_strict",
    )

    for fragment in required_fragments:
        assert fragment in text

    assert "--alignment-mode" not in text
    assert "--egm-tau-scale" not in text
    assert "--seed" not in text
