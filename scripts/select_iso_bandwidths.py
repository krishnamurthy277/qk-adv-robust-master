"""Reproduce the post-review bandwidth choice using training variance only."""
from itertools import combinations
from pathlib import Path
import json
import pandas as pd


def main():
    root = Path('results/tables/analysis')
    scan = pd.read_csv(root / 'bandwidth_concentration_scan.csv')
    models = pd.read_csv('results/tables/tier_a_results.csv').drop_duplicates('cell_hash')
    existing = [.1, .25, .5, 1., 2.]
    candidates = sorted(set(scan.bandwidth) - set(existing))
    rows = []
    for c1, c2 in combinations(candidates, 2):
        widths = []
        for (dataset, reps), group in models.groupby(['dataset', 'reps']):
            lows, highs = [], []
            for seed, part in group.groupby('seed'):
                noise = part[part.bandwidth == 1.].log10_var_K
                band = scan[(scan.dataset == dataset) & (scan.reps == reps)
                            & (scan.seed == seed) & scan.bandwidth.isin(existing + [c1, c2])].log10_var_K
                lows.append(max(noise.min(), band.min()))
                highs.append(min(noise.max(), band.max()))
            widths.append(min(highs) - max(lows))
        rows.append(dict(c1=c1, c2=c2, groups_with_overlap=sum(w > 1e-3 for w in widths),
                         total_common_width=sum(max(w, 0) for w in widths)))
    scores = pd.DataFrame(rows).sort_values(
        ['groups_with_overlap', 'total_common_width', 'c1', 'c2'], ascending=[False, False, True, True])
    scores.to_csv(root / 'bandwidth_pair_selection.csv', index=False)
    best = scores.iloc[0]
    selection = dict(rule='Choose two scanned bandwidths maximizing number of common cross-seed overlap groups, then total width; ties ascending bandwidth',
                     selection_data='Training-kernel variance only; no test accuracy or attack outcomes',
                     selected_bandwidths=[best.c1, best.c2],
                     predicted_groups_with_overlap=int(best.groups_with_overlap),
                     predicted_total_common_width=float(best.total_common_width))
    (root / 'bandwidth_selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
