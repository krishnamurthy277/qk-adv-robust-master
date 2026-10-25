"""Compare independently computed smoke tables, excluding timing/provenance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    keys = ['dataset', 'seed', 'reps', 'noise_p', 'bandwidth', 'eps_rel']
    a = pd.read_csv('results/tables/smoke_test_results.csv').sort_values(keys).reset_index(drop=True)
    b = pd.read_csv('results/tables/repro_smoke_results.csv').sort_values(keys).reset_index(drop=True)
    if not a[keys].equals(b[keys]):
        raise ValueError('Smoke grids differ')
    differences = {}
    for name in a.select_dtypes(include='number'):
        if name.startswith('t_'):
            continue
        if not a[name].isna().equals(b[name].isna()):
            raise ValueError(f'Missingness differs: {name}')
        if a[name].isna().all():
            continue
        differences[name] = float(np.nanmax(np.abs(a[name] - b[name])))
    flags = list(a.select_dtypes(include='bool').columns)
    if not a[flags].equals(b[flags]):
        raise ValueError('Diagnostic flags differ')
    result = dict(rows=len(b), numeric_columns_compared=len(differences),
                  maximum_difference=max(differences.values()), differences=differences,
                  diagnostic_flags_equal=True,
                  excluded='Timing columns t_gram_s and t_total_s; hashes/provenance differ by config and environment')
    Path('results/clean_smoke_comparison.json').write_text(json.dumps(result, indent=2))
    if result['maximum_difference'] != 0:
        raise ValueError('Smoke scientific values are not exactly equal; inspect saved comparison')
    print(f'{len(b)} rows match exactly across environments')


if __name__ == '__main__':
    main()
