from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import savemat


def _to_row_cell(series_list: list[np.ndarray]) -> np.ndarray:
    arr = np.empty((1, len(series_list)), dtype=object)
    for i, v in enumerate(series_list):
        arr[0, i] = np.asarray(v, dtype=np.float64)
    return arr


def save_curves_mat(path: str | Path, payload: dict[str, Any]) -> None:
    out = {}
    for k, v in payload.items():
        if isinstance(v, list):
            out[k] = _to_row_cell([np.asarray(x, dtype=np.float64) for x in v])
        else:
            out[k] = v
    savemat(path, out)


def save_table_csv(path: str | Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def save_table_mat(path: str | Path, df: pd.DataFrame) -> None:
    cols = np.asarray(df.columns.tolist(), dtype=object).reshape(1, -1)
    rows_cell = np.empty(df.shape, dtype=object)
    for i in range(df.shape[0]):
        for j, col in enumerate(df.columns):
            rows_cell[i, j] = df.iloc[i, j]

    savemat(path, {"ResultTable": df.to_records(index=False), "ResultTable_columns": cols, "ResultTable_rows_cell": rows_cell})
