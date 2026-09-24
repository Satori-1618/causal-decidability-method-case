"""Independent arithmetic/provenance audit, including precision at the final estimand."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def score(row):
    m = row['margins']
    a = (abs(m['read_row']-m['full'])+abs(m['read_null']-m['baseline']))/2
    b = (abs(m['read_null']-m['full'])+abs(m['read_row']-m['baseline']))/2
    return a, b, a-b


def verify(root):
    summary = json.loads((root/'summary.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    for name, digest in summary['input_hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'run input changed: {name}')
    rows = [json.loads(x) for x in (root/'records.jsonl').read_text().splitlines()]
    cases = json.loads((root/'cases.json').read_text())
    expected_ids = {c['case_id'] for c in cases}
    if len(cases) != manifest['n_base_pairs'] or len(expected_ids) != len(cases):
        raise ValueError('invalid independent units')
    pairs = []
    for c in cases:
        group = [r for r in rows if r['case_id'] == c['case_id']]
        if len(group) != 2 or {r['receiver_pattern'] for r in group} != {'ABB', 'BAB'}:
            raise ValueError('missing or duplicated directed pair')
        for r in group:
            if r['position'] != c['position']:
                raise ValueError('position mismatch')
            for arm, logits in r['answer_logits'].items():
                if not math.isclose(logits[0]-logits[1], r['margins'][arm], abs_tol=1e-12):
                    raise ValueError('margin arithmetic mismatch')
            if r['answer_logits']['identity'] != r['answer_logits']['baseline']:
                raise ValueError('identity mismatch')
            for f in r['fidelity'].values():
                if not (f['calls'] == 1 and f['passed'] and f['other_positions_unchanged']
                        and f['insertion_error_max_per_item'] <= f['rounding_budget_per_item']):
                    raise ValueError('fidelity mismatch')
        pairs.append([statistics.mean(score(r)[i] for r in group) for i in range(3)])
    if len(rows) != 2*len(cases):
        raise ValueError('unexpected additional records')
    names = ['A_visible_read_MAE', 'B_null_read_MAE', 'paired_MAE_A_minus_B_positive_favors_B']
    means = {name: statistics.mean(x[i] for x in pairs) for i, name in enumerate(names)}
    for name, value in means.items():
        if not math.isclose(value, summary[name]['mean'], abs_tol=1e-12):
            raise ValueError(f'summary mismatch: {name}')
    precision = {}
    refs = {'MPS32': rows, 'CPU32': json.loads((root/'reference_cpu32.json').read_text()),
            'CPU64': json.loads((root/'reference_cpu64.json').read_text())}
    for left, right in [('MPS32', 'CPU64'), ('MPS32', 'CPU32'), ('CPU32', 'CPU64')]:
        a = {(r['case_id'], r['receiver_pattern']): score(r) for r in refs[left]}
        expected = set(manifest['reference_case_ids'])
        if {r['case_id'] for r in refs[right]} != expected:
            raise ValueError('reference sample differs from frozen manifest')
        diffs = [(r['case_id'], [x-y for x, y in zip(
            a[(r['case_id'], r['receiver_pattern'])], score(r))]) for r in refs[right]]
        precision[left+'_vs_'+right] = {
            'max_directed_difference_in_paired_MAE_advantage': max(abs(d[2]) for _, d in diffs),
            'mean_difference_in_MAE_advantage': statistics.mean(d[2] for _, d in diffs),
            'n_base_pairs': len(expected)}
    return {'passed': True, 'n_base_pairs': len(cases), 'n_directed_pairs': len(rows),
            'independently_recomputed_means': means, 'final_estimand_precision': precision,
            'scope': 'Arithmetic, provenance and controls; not a fresh confirmation or a universal error bound.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = verify(args.run)
    result['verifier_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    content = json.dumps(result, indent=2)+'\n'
    if args.output:
        with args.output.open('x') as f:
            f.write(content)
    print(content)
