"""Score the Q1 confirmation of PREREG_READ_SOURCE_Q1.md. Runs once, after the run.

    python scripts/score_read_source_q1.py

Reads results/makelov_read_source_q1/ (written by run_makelov_read_source.py) and writes
results/makelov_read_source_q1/score.json. Only the exact sign test is judged; everything
else is reported. Pure Python, no model.
"""
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / 'PREREG_READ_SOURCE_Q1.md'
PREREG_SHA = 'f5bb62caaea253e7789f919b20a9bb78c68fc4e14e6219e260110f0f415f8a72'
RUN = ROOT / 'results' / 'makelov_read_source_q1'
PILOT_CASES = ROOT / 'results' / 'makelov_read_source_001' / 'cases.json'
N_PAIRS, SEED, ALPHA = 64, 20260922, 0.05
PREDICTIONS = {'A_visible_read': {'read_row': 'full', 'read_null': 'baseline'},
               'B_null_read': {'read_row': 'baseline', 'read_null': 'full'}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_sign_p(wins_a, wins_b):
    total = wins_a + wins_b
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(wins_a, wins_b) + 1))
    return min(1.0, 2 * tail / (1 << total))


def pair_losses(records):
    """Pilot loss: per directed pair, mean |observed - predicted| over the two read
    conditions; then the mean over the pair's two directions."""
    by_case = {}
    for r in records:
        by_case.setdefault(r['case_id'], []).append(r['margins'])
    out = {}
    for case, directions in by_case.items():
        if len(directions) != 2:
            raise ValueError(f'base pair {case} has {len(directions)} directions, not 2')
        out[case] = {name: fmean(fmean(abs(m[c] - m[anchor]) for c, anchor in pred.items())
                                 for m in directions)
                     for name, pred in PREDICTIONS.items()}
    return out


def main():
    if sha(PREREG) != PREREG_SHA:
        sys.exit('refused: PREREG_READ_SOURCE_Q1.md differs from the frozen version')
    output = RUN / 'score.json'
    if output.exists():
        sys.exit('refused: score.json exists; the confirmation is scored once')
    summary = json.loads((RUN / 'summary.json').read_text())
    manifest = json.loads((RUN / 'manifest.json').read_text())
    cases = json.loads((RUN / 'cases.json').read_text())
    records = [json.loads(x) for x in (RUN / 'records.jsonl').read_text().splitlines()]
    pilot = json.loads(PILOT_CASES.read_text())

    problems = []
    if not (summary.get('identity_controls_passed') and summary.get('fidelity_controls_passed')):
        problems.append('identity or fidelity controls failed')
    if manifest.get('seed') != SEED or manifest.get('n_base_pairs') != N_PAIRS:
        problems.append('seed or pair count differs from the preregistration')
    if len(cases) != N_PAIRS:
        problems.append(f'{len(cases)} base pairs, not {N_PAIRS}')
    overlap = ({c['case_id'] for c in cases} & {c['case_id'] for c in pilot}) | (
        {p for c in cases for p in c['prompts']} & {p for c in pilot for p in c['prompts']})
    if overlap:
        problems.append(f'{len(overlap)} case ids or prompts overlap the pilot')

    result = {'prereg_sha256': PREREG_SHA, 'scorer_sha256': sha(__file__),
              'run': {'git_head': manifest.get('git_head'), 'seed': manifest.get('seed'),
                      'device': manifest.get('working_device'),
                      'records_sha256': sha(RUN / 'records.jsonl')}}
    if problems:
        result.update({'outcome': 'invalid', 'problems': problems})
    else:
        losses = pair_losses(records)
        diffs = [v['A_visible_read'] - v['B_null_read'] for v in losses.values()]
        wins_b, wins_a = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        p = exact_sign_p(wins_a, wins_b)
        outcome = ('no difference shown' if p > ALPHA else
                   'B predicts better' if wins_b > wins_a else 'A predicts better')
        rng = random.Random(0)
        n = len(diffs)
        boot = sorted(fmean(rng.choices(diffs, k=n)) for _ in range(10_000))
        observed = abs(fmean(diffs))
        flips = sum(abs(fmean(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12
                    for _ in range(20_000))
        result.update({
            'outcome': outcome,
            'primary': {'test': 'exact two-sided sign test on per-pair loss differences',
                        'wins': {'A_visible_read': wins_a, 'B_null_read': wins_b},
                        'ties': n - wins_a - wins_b, 'p': p, 'alpha': ALPHA},
            'reported_not_judged': {
                'mean_loss': {k: fmean(v[k] for v in losses.values()) for k in PREDICTIONS},
                'mean_difference_A_minus_B': fmean(diffs),
                'nominal_95_interval': [boot[249], boot[9749]],
                'sign_flip_p': (flips + 1) / 20_001,
                'sign_flip_assumption': 'per-pair differences symmetric about zero under H0'},
            'base_pairs': n})
    output.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
