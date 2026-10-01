"""Descriptive companion to the frozen profile test; no new significance tests."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    cases = json.loads((directory / 'cases.json').read_text())
    raw = [json.loads(line) for line in (directory / 'records.jsonl').read_text().splitlines()]
    contrasts = [c for r in summary['records']
                 for c in r['precisions']['float64']['contrasts']]
    metrics = {}
    for name in ('T', 'R', 'S', 'interaction_T_minus_R_minus_S'):
        values = [c[name] for c in contrasts]
        metrics[name] = {'mean': statistics.mean(values), 'median': statistics.median(values),
                         'minimum': min(values), 'maximum': max(values),
                         'mean_absolute': statistics.mean(map(abs, values))}
    discrepancies = [d['resolution']['discrepancies'][k]
                     for r in summary['records'] for d in r['directions'] for k in ('T', 'R', 'S')]
    reset_only = {'removal': 0, 'preservation': 0}
    for record in summary['records']:
        all_contrasts = [c for p in ('float32', 'float64')
                         for c in record['precisions'][p]['contrasts']]
        reset_only['removal'] += record['resolved'] and all(
            abs(c['R']) <= .25*abs(c['T']) for c in all_contrasts)
        reset_only['preservation'] += record['resolved'] and all(
            abs(c['R']-c['T']) <= .25*abs(c['T']) for c in all_contrasts)
    return {
        'status': 'post-hoc descriptive companion; frozen primary profile decisions unchanged',
        'n_independent_base_pairs': len(raw), 'n_paired_directions_not_independent': len(contrasts),
        'primary_profiles': summary['profiles'], 'metrics_float64': metrics,
        'post_hoc_reset_only_ablation_success_counts': reset_only,
        'direction_counts_descriptive': {
            'T_negative': sum(c['T'] < 0 for c in contrasts),
            'R_same_sign_as_T': sum(c['R']*c['T'] > 0 for c in contrasts),
            'S_same_sign_as_T': sum(c['S']*c['T'] > 0 for c in contrasts),
            'R_smaller_absolute_than_T': sum(abs(c['R']) < abs(c['T']) for c in contrasts),
        },
        'maximum_cross_precision_contrast_discrepancy': max(discrepancies),
        'identity_checks_bit_exact': all(all(p['controls']['identity_exact'].values())
                                        for r in raw for p in r['precisions'].values()),
        'worked_example_selection': 'first case in the frozen manifest, not selected for fit',
        'worked_example': {
            'pair_id': cases[0]['case_id'], 'prompts': cases[0]['prompts'],
            'correct_IO': cases[0]['io'], 'subject': cases[0]['subject'],
            'cells_float64': raw[0]['precisions']['float64']['cells'],
            'contrasts_float64': summary['records'][0]['precisions']['float64']['contrasts'],
        },
        'scope': 'No confidence claim for these descriptive means/counts; no mechanism share, '
                 'new candidate confirmation, or identified natural semantic computation.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), indent=2, allow_nan=False))
