import pytest
import pandas as pd

from apdb_real_data_python.reporting import build_cpu_gpu_summary


def test_build_cpu_gpu_summary_has_expected_columns_and_speedup():
    cpu_df = pd.DataFrame(
        [
            {"Algorithm": "EGM", "Mode": "-", "Mu": "-", "Iterations": 100.0, "FinalResidual": 1e-3, "TimeSeconds": 2.0},
            {
                "Algorithm": "APDB-yx",
                "Mode": "mono",
                "Mu": 0.0,
                "Iterations": 120.0,
                "FinalResidual": 2e-3,
                "TimeSeconds": 5.0,
            },
            {
                "Algorithm": "APDB-yx",
                "Mode": "mono",
                "Mu": 1.0,
                "Iterations": 130.0,
                "FinalResidual": 3e-3,
                "TimeSeconds": 6.0,
            },
        ]
    )
    gpu_df = pd.DataFrame(
        [
            {"Algorithm": "EGM", "Mode": "-", "Mu": "-", "Iterations": 100.0, "FinalResidual": 1e-3, "TimeSeconds": 1.0},
            {
                "Algorithm": "APDB-yx",
                "Mode": "mono",
                "Mu": 0.0,
                "Iterations": 120.0,
                "FinalResidual": 2e-3,
                "TimeSeconds": 2.5,
            },
            {
                "Algorithm": "APDB-yx",
                "Mode": "mono",
                "Mu": 1.0,
                "Iterations": 130.0,
                "FinalResidual": 3e-3,
                "TimeSeconds": 3.0,
            },
        ]
    )

    out = build_cpu_gpu_summary(cpu_df, gpu_df)

    expected_cols = {
        "Algorithm",
        "Mode",
        "Mu",
        "Iter_CPU",
        "Residual_CPU",
        "Time_CPU",
        "Iter_GPU",
        "Residual_GPU",
        "Time_GPU",
        "Speedup_CPU_over_GPU",
    }
    assert expected_cols.issubset(set(out.columns))

    row = out[(out["Algorithm"] == "APDB-yx") & (out["Mode"] == "mono") & (out["Mu"] == 0.0)].iloc[0]
    assert row["Speedup_CPU_over_GPU"] == pytest.approx(2.0)
