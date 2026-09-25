from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    base = Path('/Users/wangjinxin/Library/CloudStorage/Dropbox/合作paper/QCQP_APD/APDB')
    matlab_csv = base / 'apdb_matlab' / 'results_summary_table.csv'
    py_csv = base / 'apdb_python' / 'outputs_numsim1' / 'results_summary_table_n10_m90_numsim1_cpu.csv'

    if not matlab_csv.exists():
        raise FileNotFoundError(f'Missing MATLAB csv: {matlab_csv}')
    if not py_csv.exists():
        raise FileNotFoundError(f'Missing Python csv: {py_csv}')

    m = pd.read_csv(matlab_csv)
    p = pd.read_csv(py_csv)

    keys = ['Group', 'Algorithm']
    merged = m.merge(p, on=keys, suffixes=('_matlab', '_python'))

    for col in ['Time', 'Iter', 'SuboptRes', 'InfeasRes']:
        num = (merged[f'{col}_python'] - merged[f'{col}_matlab']).abs()
        den = merged[f'{col}_matlab'].abs() + 1e-16
        merged[f'{col}_rel_err'] = num / den

    out = base / 'apdb_python' / 'outputs_numsim1' / 'matlab_python_numsim1_diff.csv'
    merged.to_csv(out, index=False)

    view_cols = keys + [
        'Time_rel_err', 'Iter_rel_err', 'SuboptRes_rel_err', 'InfeasRes_rel_err'
    ]
    print(merged[view_cols].to_string(index=False))
    print(f'\nSaved diff: {out}')


if __name__ == '__main__':
    main()
