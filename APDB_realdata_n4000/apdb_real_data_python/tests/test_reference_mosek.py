import pytest

from apdb_real_data_python.reference_mosek import mosek_high_accuracy_params


def test_reference_mosek_uses_tol_1e9():
    params = mosek_high_accuracy_params(1e-9)
    assert params["MSK_DPAR_INTPNT_CO_TOL_PFEAS"] == pytest.approx(1e-9)
    assert params["MSK_DPAR_INTPNT_CO_TOL_DFEAS"] == pytest.approx(1e-9)
    assert params["MSK_DPAR_INTPNT_CO_TOL_REL_GAP"] == pytest.approx(1e-9)
