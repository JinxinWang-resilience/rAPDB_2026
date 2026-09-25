from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "run_apdb_real_data_n5000_runall_mu2_k1000.sh"


def test_n5000_k1000_server_launcher_records_formal_experiment_settings():
    text = LAUNCHER.read_text(encoding="utf-8")

    required_fragments = (
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "$SCRIPT_DIR"',
        "python -m apdb_real_data_python.main_real_data",
        "--run-all",
        "--n-samples 5000",
        "--max-iter 9999",
        "--stop-criteria 1e-7",
        "--reg-param 1",
        "--strict-mosek",
        "--mosek-tol 1e-9",
        "--progress on",
        "--progress-every 200",
        "--mu-values 2",
        "--outdir apdb_real_data_python/outputs_n5000_runall_mu2_k1000",
    )

    for fragment in required_fragments:
        assert fragment in text

    assert "--alignment-mode" not in text
    assert "--egm-tau-scale" not in text
    assert "--seed" not in text


def test_n5000_k1000_server_launcher_does_not_advertise_mu0():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "mu0" not in text.lower()
