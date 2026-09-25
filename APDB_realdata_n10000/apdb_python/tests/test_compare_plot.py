from __future__ import annotations

import importlib
from pathlib import Path


def test_load_results_mat_has_expected_keys():
    mod = importlib.import_module("apdb_python.compare_plot")
    base = Path("/Users/wangjinxin/Library/CloudStorage/Dropbox/合作paper/QCQP_APD/APDB")
    data = mod.load_results_mat(base / "apdb_matlab" / "results_apdb_both_modes.mat")

    required = {
        "oracle_egm",
        "rel_subopt_egm",
        "rel_infeas_err_egm",
        "oracle_1",
        "rel_subopt_1",
        "rel_infeas_1",
        "oracle_xy_r1",
        "rel_subopt_xy_r1",
        "rel_infeas_xy_r1",
    }
    assert required.issubset(set(data.keys()))
    assert len(data["oracle_egm"]) >= 1


def test_generate_compare_plots_writes_two_pngs(tmp_path):
    mod = importlib.import_module("apdb_python.compare_plot")
    base = Path("/Users/wangjinxin/Library/CloudStorage/Dropbox/合作paper/QCQP_APD/APDB")

    mat_png, py_png = mod.generate_compare_plots(
        matlab_mat=base / "apdb_matlab" / "results_apdb_both_modes.mat",
        python_mat=base / "apdb_python" / "outputs_compare_m90" / "results_apdb_both_modes.mat",
        outdir=tmp_path,
        n=10,
        m=90,
        numsim=2,
    )

    assert mat_png.exists() and mat_png.stat().st_size > 0
    assert py_png.exists() and py_png.stat().st_size > 0
