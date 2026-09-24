"""Compare two runs of the read-source pilot and write the comparison as an artifact.

    python3 scripts/compare_read_source_runs.py <original run dir> <re-run dir> <output.json>

Records whether the two runs used identical prompts and directions, and how far every
directed margin and every summary statistic moved. Copies no prompts, weights or vectors.
"""
import hashlib
import json
import sys
from pathlib import Path

STATS = ('A_visible_read_MAE', 'B_null_read_MAE', 'paired_MAE_A_minus_B_positive_favors_B',
         'conditional_rival_separation_mean_absolute_full_effect')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original, rerun, output = (Path(a) for a in sys.argv[1:4])
    a = json.loads((original / 'summary.json').read_text())
    b = json.loads((rerun / 'summary.json').read_text())
    rows_a = {(r['case_id'], r['receiver_pattern']): r['margins'] for r in
              map(json.loads, (original / 'records.jsonl').read_text().splitlines())}
    rows_b = {(r['case_id'], r['receiver_pattern']): r['margins'] for r in
              map(json.loads, (rerun / 'records.jsonl').read_text().splitlines())}
    if set(rows_a) != set(rows_b):
        sys.exit('the two runs cover different directed pairs')
    margin_diff = max(abs(rows_a[k][c] - rows_b[k][c]) for k in rows_a for c in rows_a[k])
    manifest_b = json.loads((rerun / 'manifest.json').read_text())
    result = {
        'original': {'records_sha256': sha(original / 'records.jsonl'),
                     'device': json.loads((original / 'manifest.json').read_text())['working_device']},
        'rerun': {'records_sha256': sha(rerun / 'records.jsonl'),
                  'device': manifest_b['working_device'], 'platform': manifest_b['platform']},
        'identical_cases': sha(original / 'cases.json') == sha(rerun / 'cases.json'),
        'identical_directions': sha(original / 'directions.npz') == sha(rerun / 'directions.npz'),
        'directed_pairs': len(rows_a),
        'max_abs_margin_difference_nats': margin_diff,
        'summary': {k: {'original': a[k]['mean'], 'rerun': b[k]['mean'],
                        'abs_difference': abs(a[k]['mean'] - b[k]['mean'])} for k in STATS},
        'controls_passed_in_rerun': bool(b['identity_controls_passed']
                                         and b['fidelity_controls_passed']),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
