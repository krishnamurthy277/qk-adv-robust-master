"""Read-only experiment progress report; no inference from partial metrics."""
import json
from pathlib import Path
import pandas as pd


def main():
    root=Path(__file__).resolve().parent.parent
    path=root/'results/tables/tier_a_results.csv'
    df=pd.read_csv(path)
    df=df[df.version=='v2_per_gate']
    cells=df.groupby(['cell_hash','dataset','reps','noise_p'],as_index=False).t_total_s.max()
    print(f'Tier A: {len(cells)}/144 cells; {len(df)}/576 rows')
    print('Completed-cell wall seconds (maximum epsilon-row time per cell):')
    print(cells.groupby(['dataset','reps','noise_p']).t_total_s.agg(['count','mean','max']).round(1).to_string())
    run=root/'results/runs/tier_a_main/24d365adc00fcd58'
    failures=list((run/'failures').glob('*.json'))
    print(f'Recorded cell failure attempts: {len(failures)}')
    for name in ('workflow_status.json','validation_checkpoint.json'):
        path=root/'results'/name
        if path.exists():
            print(name+':')
            print(json.dumps(json.loads(path.read_text(encoding='utf-8-sig')),indent=2))


if __name__=='__main__':main()
