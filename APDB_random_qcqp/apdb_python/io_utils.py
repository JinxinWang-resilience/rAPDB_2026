from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def _to_object_cell(two_d_list: list[list[Any]]) -> np.ndarray:
    rows = len(two_d_list)
    cols = len(two_d_list[0]) if rows else 0
    arr = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            arr[i, j] = np.asarray(two_d_list[i][j], dtype=np.float64)
    return arr


def _to_row_cell(one_d_list: list[Any]) -> np.ndarray:
    arr = np.empty((1, len(one_d_list)), dtype=object)
    for i, v in enumerate(one_d_list):
        arr[0, i] = np.asarray(v, dtype=np.float64)
    return arr


def save_results_mat(path: str | Path, results: dict[str, Any]) -> None:
    path = Path(path)
    payload = {}
    for key, value in results.items():
        if isinstance(value, list) and value and isinstance(value[0], list):
            payload[key] = _to_object_cell(value)
        elif isinstance(value, list):
            payload[key] = _to_row_cell(value)
        elif isinstance(value, (float, int)):
            payload[key] = np.array([[value]], dtype=np.float64)
        else:
            payload[key] = value
    savemat(path, payload)


def save_summary_table(path_csv: str | Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path_csv, index=False)
    return df


def save_summary_table_mat(path_mat: str | Path, df: pd.DataFrame) -> None:
    columns = np.asarray(df.columns.tolist(), dtype=object).reshape(1, -1)
    rows_cell = np.empty(df.shape, dtype=object)
    for i in range(df.shape[0]):
        for j, col in enumerate(df.columns):
            rows_cell[i, j] = df.iloc[i][col]
    savemat(
        path_mat,
        {
            "ResultTable": df.to_records(index=False),
            "ResultTable_columns": columns,
            "ResultTable_rows_cell": rows_cell,
        },
    )


def _extract_q_list(q_cell: np.ndarray) -> list[np.ndarray]:
    q_list = []
    if q_cell.dtype == object and q_cell.ndim >= 4 and q_cell.shape[1] == 1:
        # Case from scipy savemat roundtrip: (n,1,m,m) object.
        q_iter = [q_cell[i, 0] for i in range(q_cell.shape[0])]
    else:
        q_iter = np.ravel(q_cell, order="F")

    for q in q_iter:
        q_arr = np.asarray(q)
        if q_arr.dtype == object:
            # scipy.loadmat may return object matrices with scalar cells.
            q_arr = np.vectorize(lambda z: float(np.asarray(z).reshape(-1)[0]))(q_arr)
        q_list.append(np.asarray(q_arr, dtype=np.float64))
    return q_list


def load_problem_from_mat(path: str | Path) -> dict[str, Any]:
    mat = loadmat(path)
    required = ["A", "Q", "b", "d", "c"]
    missing = [k for k in required if k not in mat]
    if missing:
        raise KeyError(f"Missing keys in mat problem file: {missing}")

    problem = {
        "A": np.asarray(mat["A"], dtype=np.float64),
        "Q": _extract_q_list(mat["Q"]),
        "b": np.asarray(mat["b"], dtype=np.float64).reshape(-1),
        "d": np.asarray(mat["d"], dtype=np.float64),
        "c": np.asarray(mat["c"], dtype=np.float64).reshape(-1),
        "sc": float(np.asarray(mat.get("sc", [[0.0]]), dtype=np.float64).reshape(-1)[0]),
    }

    out = {"problem": problem}
    if "x0" in mat:
        out["x0"] = np.asarray(mat["x0"], dtype=np.float64).reshape(-1)
    if "y0" in mat:
        out["y0"] = np.asarray(mat["y0"], dtype=np.float64).reshape(-1)
    if "optval" in mat:
        out["optval"] = float(np.asarray(mat["optval"], dtype=np.float64).reshape(-1)[0])
    return out
