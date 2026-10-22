"""Post-run independent raw recount; no producer or production analysis imports.

Requires SciPy for an independent beta-quantile implementation of CP bounds.
This is saved-record verification, not a neural rerun or a pre-run prediction.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import scipy
from scipy.stats import beta

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parents[3]


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cp(k):
    alpha, n = .05 / 24, 64
    return (float(beta.ppf(alpha, k, n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha, k+1, n-k)) if k < n else 1.)


def recount(run):
    manifest, plan = read(run / 'manifest.json'), read(BASE / 'plan.json')
    release = read(BASE / 'EXECUTION_RELEASE.json')
    assert manifest['status'] == 'completed'
    assert release['final_review']['status'] == 'PASS'
    assert release['final_review']['reviewed_commit'] == manifest['reviewed_commit']
    assert sha(BASE / 'plan.json') == manifest['plan_sha256']
    assert sha(BASE / 'EXECUTION_RELEASE.json') == manifest['execution_release_sha256']
    assert sha(run / 'normalization.json') == manifest['normalization_sha256']
    for name, expected in manifest['source_hashes'].items():
        assert sha(ROOT / name) == expected, name
    expected_heads = [h['model'] + '_head' + str(h['head_one_based']) for h in plan['cohort']]
    assert [h['head_key'] for h in manifest['heads']] == expected_heads
    output = []
    for entry in manifest['heads']:
        key = entry['head_key']
        directory = run / 'heads' / key
        status = read(directory / 'status.json')
        assert sha(directory / 'status.json') == entry['status_sha256']
        assert status['status'] == entry['status'] == 'completed'
        for name, expected in status['output_hashes'].items():
            assert sha(directory / name) == expected, (key, name)
        norm = status['normalization']
        w = [a-b for a, b in zip(norm['w_false'], norm['w_true'])]
        center = math.fsum(a*b for a, b in zip(w, norm['ln_beta']))
        u = [a*b for a, b in zip(w, norm['ln_gamma'])]
        ubar = math.fsum(u) / 64
        radius = 8 * math.sqrt(math.fsum((a-ubar)**2 for a in u))
        cutoff = center + radius * plan['reference']['normalized_cutoff']
        assert abs(center-norm['center']) < 1e-12 and abs(radius-norm['radius']) < 1e-12
        native = records(directory / 'screening.jsonl')
        inputs = records(BASE / 'inputs' / key / 'candidates.jsonl')
        assert len(native) == len(inputs) == 1024
        indices = {'accepted': [], 'rejected': []}
        native_error = 0.
        for i, (row, source) in enumerate(zip(native, inputs)):
            assert row['candidate_index'] == i
            assert row['recipient'] == source['recipient'] and row['candidate_id'] == source['candidate_id']
            accepts = row['margin_float64'] < cutoff
            assert accepts == row['screen_accept_float64'] == row['screen_accept_float32']
            assert accepts == (row['margin_float32'] < cutoff)
            stratum = 'accepted' if accepts else 'rejected'
            assert row['stratum'] == stratum
            indices[stratum].append(row['candidate_id'])
            native_error = max(native_error, abs(row['margin_float32']-row['margin_float64']))
        selected = indices['accepted'][:64] + indices['rejected'][:64]
        assert len(selected) == 128
        assert all(row['selected'] == (row['candidate_id'] in selected) for row in native)
        families = records(directory / 'cases.jsonl')
        assert [row['candidate_id'] for row in families] == selected
        hits = {'accepted': 0, 'rejected': 0}
        anchor_error = 0.
        for i, row in enumerate(families):
            assert row['stratum'] == ('accepted' if i < 64 else 'rejected')
            gaps = [row['cells']['pos_28_0'][t]['patched_margin'] -
                    row['cells']['neg_20_0'][t]['patched_margin'] for t in ('float32', 'float64')]
            assert (abs(gaps[0]) > .202) == (abs(gaps[1]) > .202) == row['anchor_separating']
            hits[row['stratum']] += int(abs(gaps[1]) > .202)
            anchor_error = max(anchor_error, abs(gaps[0]-gaps[1]))
        assert native_error <= .001 and anchor_error <= .001
        lower_a, upper_a = cp(hits['accepted'])
        lower_r, upper_r = cp(hits['rejected'])
        lower, upper = lower_a-upper_r, upper_a-lower_r
        q = len(indices['accepted']) / 1024
        pa, pr = hits['accepted'] / 64, hits['rejected'] / 64
        random_yield = q*pa + (1-q)*pr
        events = records(directory / 'forward_events.jsonl')
        attempted = sum(e['sequences'] for e in events if e['event'] == 'started')
        completed = sum(e['sequences'] for e in events if e['event'] == 'completed')
        batches = sum(e['event'] == 'completed' for e in events)
        assert attempted == completed == 4096
        assert completed == status['cost']['sequence_forwards_completed']
        assert attempted == status['cost']['sequence_forwards_attempted']
        assert batches == status['cost']['forward_batches_completed']
        assert all(e['at'] <= manifest['all_screening_completed_at'] for e in events if e['stage'] == 'native')
        assert all(e['at'] >= status['anchors_measurement_started_at'] > manifest['all_screening_completed_at']
                   for e in events if e['stage'] == 'anchors')
        output.append({'head': key, 'status': 'completed', 'accepted_pool': len(indices['accepted']),
            'q': q, 'counts': hits, 'delta': pa-pr, 'delta_interval': [lower, upper],
            'strong': lower > .25, 'positive': lower > 0, 'exclude_strong': upper < .25,
            'accepted_yield': pa, 'estimated_random_yield': random_yield,
            'improvement_vs_estimated_random': pa-random_yield,
            'fixed_cost': 3072/hits['accepted'] if pa else 'infinite',
            'estimated_random_cost': 18/random_yield if random_yield else 'infinite',
            'ideal_streaming_cost_not_executed': (2/q+16)/pa if pa else 'infinite',
            'native_precision_max': native_error, 'anchor_precision_max': anchor_error,
            'sequence_forwards_attempted': attempted, 'sequence_forwards_completed': completed,
            'completed_batches': batches})
    strong = sum(h['strong'] for h in output)
    return {'audit_kind': 'Post-run independent recount; no production analysis imported or neural forward run',
        'provenance': {'script_sha256': sha(Path(__file__)), 'scipy_version': scipy.__version__,
            'manifest_sha256': sha(run / 'manifest.json'), 'run_git_head': manifest['git_head'],
            'reviewed_commit': manifest['reviewed_commit'], 'verified_source_hashes': len(manifest['source_hashes'])},
        'heads': output, 'strong_heads': strong, 'positive_heads': sum(h['positive'] for h in output),
        'primary_success': strong >= 4, 'total_sequence_forwards': sum(h['sequence_forwards_completed'] for h in output),
        'recorded_elapsed_seconds': manifest['elapsed_seconds'],
        'limitation': 'Random-policy yield/cost are descriptive estimates; fixed-pool cost is not lower in these six heads.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE / 'results' / 'run_001')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(recount(args.run), indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as handle:
            handle.write(text)
    else:
        print(text, end='')
