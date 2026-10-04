"""Read published screen_002 data only; no forecast of round 004 or model calls."""
import hashlib
import json
from pathlib import Path
import statistics


def recount():
    source = Path(__file__).resolve().parents[2] / 'screen_002/results/run_001/cases.jsonl'
    raw = source.read_bytes()
    rows = [r for r in map(json.loads, raw.decode().splitlines()) if r['stratum'] == 'accepted']
    if len(rows) != 32 or any(len(r['cells']) != 8 for r in rows):
        raise ValueError('Expected the published 32 accepted eight-observation families')
    variants = {name: [] for name in ('B_reuse', 'C_reuse', 'B_replica0_to_1',
                                     'B_replica1_to_0', 'C_single_prefix_to_other')}
    selected, gaps = [], []
    sd = {f'{s}_{p}': [] for s in ('neg', 'pos') for p in (20, 28)}
    for row in rows:
        y = {c: v['float64']['patched_margin'] for c, v in row['cells'].items()}
        a = {c: (y[c+'_0']+y[c+'_1'])/2 for c in sd}
        b = {s: (a[s+'_20']+a[s+'_28'])/2 for s in ('neg', 'pos')}
        pos = {str(p): (a['neg_'+str(p)]+a['pos_'+str(p)])/2 for p in (20, 28)}
        selected.append(max(abs(b[s]-pos[p]) for s, p in (c.split('_') for c in sd)) > .202)
        variants['B_reuse'].append(max(abs(y[c+'_'+str(r)]-b[c.split('_')[0]]) for c in sd for r in (0, 1)))
        variants['C_reuse'].append(max(abs(y[c+'_'+str(r)]-a[c]) for c in sd for r in (0, 1)))
        for calibration, target in ((0, 1), (1, 0)):
            predictions = {s: (y[f'{s}_20_{calibration}']+y[f'{s}_28_{calibration}'])/2 for s in b}
            variants[f'B_replica{calibration}_to_{target}'].append(max(
                abs(y[f'{c}_{target}']-predictions[c.split('_')[0]]) for c in sd))
        pair_gap = max(abs(y[c+'_1']-y[c+'_0']) for c in sd)
        variants['C_single_prefix_to_other'].append(pair_gap)
        gaps.append(pair_gap)
        for c in sd:
            sd[c].append(statistics.stdev((y[c+'_0'], y[c+'_1'])))
    return {
        'status': 'old_data_sensitivity_not_a_new_run_forecast',
        'source': 'screen_002/results/run_001/cases.jsonl',
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'families': 32, 'new_formula_reused_separating': sum(selected),
        'variants': {name: {
            'all_definite_hits_le_0_099': sum(v <= .099 for v in values),
            'separating_definite_hits_le_0_099': sum(v <= .099 for v, s in zip(values, selected) if s),
            'maximum_absolute_error_nat': max(values)} for name, values in variants.items()},
        'maximum_within_cell_pair_gap_nat': max(gaps),
        'same_cell_witness_families_gt_0_202': sum(v > .202 for v in gaps),
        'two_prefix_sample_sd_nat': {c: {'mean': statistics.mean(v), 'maximum': max(v)} for c, v in sd.items()},
        'limitations': [
            'Only two old prefixes per cell; cannot instantiate two calibration PLUS two independent targets.',
            'Reuse scores are in-sample; replica splits use one calibration prefix per cell and only four targets.',
            'Separating subset is computed from these same old observations, not independent calibration.',
            'Old sampling partition differs; no estimate is adopted as the new round accuracy or yield.']}


if __name__ == '__main__':
    print(json.dumps(recount(), indent=2, sort_keys=True))
