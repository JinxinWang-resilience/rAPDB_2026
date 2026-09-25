from __future__ import annotations

import numpy as np
import pandas as pd


def build_single_device_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def build_cpu_gpu_summary(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame | None) -> pd.DataFrame:
    key = ["Algorithm", "Mode", "Mu"]
    if "Mu" not in cpu_df.columns:
        cpu_df = cpu_df.copy()
        cpu_df["Mu"] = "-"
    if gpu_df is not None and "Mu" not in gpu_df.columns:
        gpu_df = gpu_df.copy()
        gpu_df["Mu"] = "-"

    cpu = cpu_df[["Algorithm", "Mode", "Mu", "Iterations", "FinalResidual", "TimeSeconds"]].copy()
    cpu = cpu.rename(
        columns={
            "Iterations": "Iter_CPU",
            "FinalResidual": "Residual_CPU",
            "TimeSeconds": "Time_CPU",
        }
    )

    if gpu_df is None:
        out = cpu.copy()
        out["Iter_GPU"] = np.nan
        out["Residual_GPU"] = np.nan
        out["Time_GPU"] = np.nan
        out["Speedup_CPU_over_GPU"] = np.nan
        return out

    gpu = gpu_df[["Algorithm", "Mode", "Mu", "Iterations", "FinalResidual", "TimeSeconds"]].copy()
    gpu = gpu.rename(
        columns={
            "Iterations": "Iter_GPU",
            "FinalResidual": "Residual_GPU",
            "TimeSeconds": "Time_GPU",
        }
    )

    out = cpu.merge(gpu, on=key, how="outer")
    cpu_t = out["Time_CPU"].to_numpy(dtype=np.float64)
    gpu_t = out["Time_GPU"].to_numpy(dtype=np.float64)

    speedup = np.full_like(cpu_t, np.nan, dtype=np.float64)
    good = (~np.isnan(cpu_t)) & (~np.isnan(gpu_t)) & (gpu_t > 0)
    speedup[good] = cpu_t[good] / gpu_t[good]
    out["Speedup_CPU_over_GPU"] = speedup
    return out
