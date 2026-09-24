"""Recheck the Q1 confirmation from the stored files, in one command. Writes nothing.

    python3 scripts/check_read_source_q1.py

With the frozen scorer's own functions, recomputes the judged result (wins, ties, exact
sign-test p, outcome) and the reported mean losses from records.jsonl, and compares them
with score.json. Checks the hashes pinned in score.json (preregistration, scorer, records),
the run's commit, seed and pair count, the disjointness from the pilot, and the rejection
threshold stated in the preregistration. Then runs the existing verifier: controls,
directed pairs, ids, summary means and input hashes. The verifier needs directions.npz,
which scripts/rebuild_read_source_directions.py rebuilds.
"""
import hashlib
import json
import sys
from pathlib import Path
from statistics import fmean

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_read_source_q1 as scorer  # noqa: E402  frozen; only its functions are used
from verify_makelov_read_source import verify  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(run=scorer.RUN):
    run = Path(run)
    score = json.loads((run / 'score.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    cases = json.loads((run / 'cases.json').read_text())
    pilot = json.loads(scorer.PILOT_CASES.read_text())
    records = [json.loads(x) for x in (run / 'records.jsonl').read_text().splitlines()]
    problems = []

    def expect(what, found, wanted):
        if found != wanted:
            problems.append(f'{what}: found {found!r}, expected {wanted!r}')

    expect('preregistration sha256', sha(scorer.PREREG), score['prereg_sha256'])
    expect('preregistration pinned by the scorer', score['prereg_sha256'], scorer.PREREG_SHA)
    expect('scorer sha256', sha(scorer.__file__), score['scorer_sha256'])
    expect('records sha256', sha(run / 'records.jsonl'), score['run']['records_sha256'])
    expect('run commit', manifest.get('git_head'), score['run']['git_head'])
    expect('seed', manifest.get('seed'), scorer.SEED)
    expect('base pairs', len(cases), scorer.N_PAIRS)
    expect('distinct base pairs', len({c['case_id'] for c in cases}), scorer.N_PAIRS)
    expect('directed pairs', len(records), 2 * scorer.N_PAIRS)
    expect('controls passed', bool(summary.get('identity_controls_passed')
                                   and summary.get('fidelity_controls_passed')), True)
    shared = ({c['case_id'] for c in cases} & {c['case_id'] for c in pilot}) | (
        {p for c in cases for p in c['prompts']} & {p for c in pilot for p in c['prompts']})
    expect('case ids or prompts shared with the pilot', len(shared), 0)
    expect('rejection threshold of the preregistration (41 of 64 wins)',
           (scorer.exact_sign_p(23, 41) <= scorer.ALPHA, scorer.exact_sign_p(24, 40) <= scorer.ALPHA),
           (True, False))

    losses = scorer.pair_losses(records)
    diffs = [v['A_visible_read'] - v['B_null_read'] for v in losses.values()]
    wins_b, wins_a = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    p = scorer.exact_sign_p(wins_a, wins_b)
    outcome = ('no difference shown' if p > scorer.ALPHA else
               'B predicts better' if wins_b > wins_a else 'A predicts better')
    recomputed = {
        'outcome': outcome,
        'wins': {'A_visible_read': wins_a, 'B_null_read': wins_b},
        'ties': len(diffs) - wins_a - wins_b,
        'p': p,
        'mean_loss': {k: fmean(v[k] for v in losses.values()) for k in scorer.PREDICTIONS},
        'mean_difference_A_minus_B': fmean(diffs)}
    primary, reported = score['primary'], score['reported_not_judged']
    expect('outcome', outcome, score['outcome'])
    expect('wins', recomputed['wins'], primary['wins'])
    expect('ties', recomputed['ties'], primary['ties'])
    expect('sign-test p', p, primary['p'])
    expect('mean losses', recomputed['mean_loss'], reported['mean_loss'])
    expect('mean difference', recomputed['mean_difference_A_minus_B'],
           reported['mean_difference_A_minus_B'])

    try:
        verified = verify(run)
    except FileNotFoundError as err:
        problems.append(f'verifier: {err.filename} is missing; run '
                        'scripts/rebuild_read_source_directions.py first')
        verified = None
    except (ValueError, KeyError) as err:
        problems.append(f'verifier: {err}')
        verified = None
    return recomputed, verified, problems


def main():
    recomputed, verified, problems = check()
    wins = recomputed['wins']
    print(f"outcome: {recomputed['outcome']}; wins A {wins['A_visible_read']}, "
          f"B {wins['B_null_read']}, ties {recomputed['ties']}; "
          f"exact sign test p = {recomputed['p']:.4g}")
    print('mean loss (reported, not judged): ' + ', '.join(
        f'{k} {v:.4f}' for k, v in recomputed['mean_loss'].items()))
    if verified:
        print(f"verifier: passed on {verified['n_base_pairs']} base pairs, "
              f"{verified['n_directed_pairs']} directed pairs")
    if problems:
        print('\nFAILED:')
        for problem in problems:
            print(f'  - {problem}')
        sys.exit(1)
    print('\nEvery recomputed value matches score.json and the pinned hashes.')


if __name__ == '__main__':
    main()
