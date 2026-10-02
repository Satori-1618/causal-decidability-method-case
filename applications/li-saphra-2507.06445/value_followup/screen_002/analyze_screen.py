"""Frozen prospective screen evaluation; no fitting or threshold selection.

Uses exact binomial bounds on two independently sampled conditional populations.
The earlier six-cell candidate test is secondary and remains developmental.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyze import analyze_family

N = 32
GAP = .202
MARGIN_CUTOFF = 8.
NUMERICAL = .001
MINIMUM_UPLIFT = .25
TAIL_ALPHA = .05 / 4
ANCHORS = ('neg_20_0', 'pos_28_0')


def binomial_tail(n, p, k, upper):
    indices = range(k, n + 1) if upper else range(k + 1)
    return math.fsum(math.comb(n, j) * p**j * (1-p)**(n-j) for j in indices)


def exact_bounds(k, n, alpha=TAIL_ALPHA):
    """Clopper--Pearson bounds, each tail alpha; no bootstrap at 0/n or n/n."""
    if not isinstance(k, int) or not isinstance(n, int) or not 0 <= k <= n or n < 1:
        raise ValueError('Invalid binomial counts')
    if not 0 < alpha < .5:
        raise ValueError('Invalid tail probability')
    lower = 0.
    if k:
        lo, hi = 0., 1.
        for _ in range(70):
            mid = (lo + hi) / 2
            if binomial_tail(n, mid, k, True) < alpha:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    upper = 1.
    if k < n:
        lo, hi = 0., 1.
        for _ in range(70):
            mid = (lo + hi) / 2
            if binomial_tail(n, mid, k, False) > alpha:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return [lower, upper]


def primary_counts(accepted_hits, rejected_hits):
    groups = {}
    for name, k in [('accepted', accepted_hits), ('rejected', rejected_hits)]:
        groups[name] = {'separating': k, 'families': N, 'rate': k/N,
                        'simultaneous_interval': exact_bounds(k, N)}
    a, r = (groups[name]['simultaneous_interval'] for name in ['accepted', 'rejected'])
    interval = [a[0] - r[1], a[1] - r[0]]
    if interval[0] > MINIMUM_UPLIFT:
        decision = 'supports_at_least_25pp_enrichment'
    elif interval[1] < MINIMUM_UPLIFT:
        decision = 'excludes_25pp_enrichment'
    else:
        decision = 'unresolved_at_25pp'
    enrichment_status = ('positive_enrichment' if interval[0] > 0 else
                         'negative_enrichment' if interval[1] < 0 else 'unresolved')
    return {'strata': groups, 'enrichment': (accepted_hits-rejected_hits)/N,
            'simultaneous_interval': interval, 'minimum_uplift': MINIMUM_UPLIFT,
            'decision': decision, 'directional_status': enrichment_status,
            'uncertainty': 'Clopper-Pearson; four one-sided tails of 0.0125; joint coverage >=95%'}


def summarize(rows):
    if len(rows) != 2*N or len({row['family_id'] for row in rows}) != 2*N:
        raise ValueError('Expected 64 unique family identifiers')
    strata = {name: [r for r in rows if r['stratum'] == name] for name in ['accepted', 'rejected']}
    if any(len(group) != N for group in strata.values()):
        raise ValueError('Expected 32 families in each frozen stratum')
    family_results = []
    hit_counts = {}
    for name, group in strata.items():
        hits = 0
        for row in group:
            cells = row['cells']
            if name == 'rejected' and set(cells) != set(ANCHORS):
                raise ValueError('Rejected families must have exactly the two anchors')
            gaps = {}
            for dtype in ['float32', 'float64']:
                x, y = [float(cells[c][dtype]['patched_margin']) for c in ANCHORS]
                if not all(math.isfinite(v) for v in (x, y)):
                    raise ValueError('Nonfinite anchor margin')
                gaps[dtype] = abs(y-x)
            if (abs(gaps['float32'] - gaps['float64']) > NUMERICAL
                    or (gaps['float32'] > GAP) != (gaps['float64'] > GAP)):
                raise ValueError('Numerically unresolved anchor-separation label')
            eligible = gaps['float64'] > GAP
            native = cells[ANCHORS[0]]['float64']['native_margin']
            if not math.isfinite(native) or (native < MARGIN_CUTOFF) != (name == 'accepted'):
                raise ValueError('Screen stratum differs from measured native margin')
            hits += eligible
            record = {'family_id': row['family_id'], 'stratum': name,
                      'native_margin': native, 'anchor_gap': gaps['float64'],
                      'separating': eligible, 'anchor_gap_dtype_difference': abs(gaps['float32']-gaps['float64'])}
            if name == 'accepted':
                record['secondary'] = analyze_family(row)
            family_results.append(record)
        hit_counts[name] = hits
    eligible = [f['secondary'] for f in family_results if f['stratum'] == 'accepted' and f['separating']]
    candidates = {}
    for candidate in ['H_state', 'H_position']:
        errors = [r['max_prediction_error'][candidate] for r in eligible]
        candidates[candidate] = {'eligible_families': len(errors),
            'definite_hits': sum(error <= .099 for error in errors),
            'possible_hits': sum(error <= .101 for error in errors)}
    gate = len(eligible) >= 16 and any(c['definite_hits'] >= .9*len(eligible) for c in candidates.values())
    return {'primary': primary_counts(hit_counts['accepted'], hit_counts['rejected']),
            'secondary': {'status': 'development_only', 'accepted_families': N,
                          'eligible_families': len(eligible), 'candidates': candidates,
                          'confirmation_start_gate': gate,
                          'rule': 'At least 16/32 separating, and >=90% definite max-six-cell hits for one candidate'},
            'family_results': family_results,
            'limits': ['Local validation of a post-pilot frozen screen on one previously selected head.',
                      'Anchor separation is a sufficient design criterion, not mechanism identification.',
                      'Rejected means margin >=8, not proven absence of interpretable causal effects.',
                      'No generalization to other heads, tasks or intervention operators is tested.',
                      'Primary screen inference and secondary candidate-development gate are separate.']}


def analyze_directory(run):
    manifest = json.loads((run/'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Incomplete or technically failed run cannot support a result')
    for name, expected in manifest['output_hashes'].items():
        if hashlib.sha256((run/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Altered artifact: '+name)
    rows = [json.loads(line) for line in (run/'cases.jsonl').read_text().splitlines()]
    result = summarize(rows)
    result['provenance'] = {'git_head': manifest['git_head'],
        'manifest_sha256': hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
        'cases_sha256': manifest['output_hashes']['cases.jsonl'],
        'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing overwrite')
    result = analyze_directory(args.run)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'family_results'}, indent=2))


if __name__ == '__main__':
    main()
