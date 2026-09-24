"""From your own per-unit data to the explanations still compatible, under a declared contract.

    python3 examples/from_data.py
    python3 examples/from_data.py --data my.csv --candidates my.json --tested c1 c2 \\
        --loss absolute --scope pooled --tolerance 0.3

Input:

  --data         CSV with columns unit, repeat, condition, value. A repeat is a measurement
                 of the whole unit that belongs together (one swap direction, one seed);
                 give a constant if there is only one.
  --candidates   JSON: candidate -> {tested condition: number or "same_as:<condition>"}.
                 "same_as" anchors a prediction to the unit's own measurement in another
                 condition, in the same repeat.
  --tested       the conditions on which the candidates are compared.

The contract, declared before the data are seen:

  --loss         absolute (how well each case is predicted; errors do not cancel) or
                 signed (whether the candidate is right on average; errors cancel).
  --scope        per_condition or pooled over the tested conditions.
  --tolerance    the largest loss that still counts as adequate, in readout units; or
  --tolerance-fraction f --scale c1 c2: a fraction of the mean |c1 - c2| gap, resampled
                 together with the loss.
  --compare-only which candidate predicts better, on the absolute loss; no tolerance.

Without arguments it runs the stored Makelov read-source pilot (32 base pairs of GPT-2
Small, MLP 8) with the pilot's own loss: absolute error, pooled over the two read
conditions. That is development data, so it shows two tolerances as an illustration, not
a declared result.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from causal_decidability.evaluate import compare, evaluate, load_rows  # noqa: E402

DEFAULT_DATA = os.path.join(HERE, 'data', 'makelov_read_source.csv')
DEFAULT_CANDIDATES = os.path.join(HERE, 'data', 'makelov_read_source_candidates.json')


def load(path):
    with open(path, newline='') as handle:
        return load_rows((r['unit'], r.get('repeat', '0'), r['condition'], r['value'])
                         for r in csv.DictReader(handle))


def report(result, label):
    print(f'\n{label}')
    print('Nominal percentile-bootstrap adequacy analysis: coverage is not guaranteed by '
          'the minimum sample size. Calibrate this rule for your data before confirmation.')
    print(f"{'candidate':<16}{'scope':<12}{'loss':>8}{'tolerance':>11}"
          f"{'loss - tolerance [nominal interval]':>38}  status")
    for r in result['rows']:
        lo, hi = r['interval']
        print(f"{r['candidate']:<16}{r['scope']:<12}{r['loss']:>8.3f}{r['tolerance']:>11.3f}"
              f"{r['margin']:>12.3f}  [{lo:>7.3f}, {hi:>7.3f}]{'':>7}  {r['status']}")
    adequate = [n for n, s in result['status'].items() if s == 'adequate']
    undecided = [n for n, s in result['status'].items() if s == 'undecided']
    print(f"compatible set: {', '.join(result['retained']) or 'none'} ({result['outcome']}); "
          f"shown adequate: {', '.join(adequate) or 'none'}; "
          f"undecided: {', '.join(undecided) or 'none'}")
    for name, scopes in result['excluded_by'].items():
        print(f"  {name} excluded in: {', '.join(scopes)}")


def print_comparisons(comparisons):
    if len(comparisons) > 1:
        print('Pairwise p-values below are unadjusted. Predeclare a primary comparison '
              'or apply a justified multiple-testing procedure across the inspected family.')
    for c in comparisons:
        a, b = c['pair']
        lo, hi = c['interval']
        print(f"\ncomparison {a} vs {b} ({c['scope']}, {c['units']} units)")
        print(f"  units won: {a} {c['units_favouring'][a]}, {b} {c['units_favouring'][b]}, "
              f"ties {c['ties']}; exact sign test p = {c['sign_test_p']:.3g} "
              f"(assumes {c['assumptions']['sign_test_p']})")
        print(f"  mean loss difference {c['mean_difference']:.3f}, nominal interval "
              f"[{lo:.3f}, {hi:.3f}]; sign-flip p = {c['sign_flip_p']:.2g} "
              f"({c['sign_flip_method']}; assumes {c['assumptions']['sign_flip_p']})")
        if c.get('sign_test_status') == 'no_untied_units':
            print('  All units tie: no evidence of a difference, not evidence of equivalence.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--candidates', default=DEFAULT_CANDIDATES)
    ap.add_argument('--tested', nargs='+', default=['read_row', 'read_null'])
    ap.add_argument('--loss', choices=['absolute', 'signed'], default=None)
    ap.add_argument('--scope', choices=['per_condition', 'pooled'], default=None)
    ap.add_argument('--tolerance', type=float)
    ap.add_argument('--tolerance-fraction', type=float, nargs='+')
    ap.add_argument('--scale', nargs=2, default=['full', 'baseline'])
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--min-units', type=int, default=10)
    ap.add_argument('--compare-only', action='store_true')
    args = ap.parse_args()
    if args.compare_only:
        with open(args.candidates) as handle:
            candidates = json.load(handle)
        comparisons = compare(load(args.data), candidates, args.tested,
                              scope=args.scope or 'pooled', alpha=args.alpha,
                              min_units=args.min_units)
        print_comparisons(comparisons)
        return

    stored = args.data == DEFAULT_DATA
    if not stored and (args.loss is None or args.scope is None
                       or (args.tolerance is None and not args.tolerance_fraction)):
        ap.error('declare --loss, --scope and a tolerance: they are the contract')
    loss = args.loss or 'absolute'
    scope = args.scope or 'pooled'
    data = load(args.data)
    with open(args.candidates) as handle:
        candidates = json.load(handle)
    if args.tolerance is not None:
        runs = [dict(tolerance=args.tolerance)]
    else:
        runs = [dict(tolerance_fraction=f, scale=tuple(args.scale))
                for f in (args.tolerance_fraction or [0.10, 0.25])]
    if stored:
        print('Stored example: Makelov read-source pilot, development data. The tolerances '
              'below illustrate the rule; none was declared before these data were seen.')
    result = None
    for kwargs in runs:
        result = evaluate(data, candidates, args.tested, loss, scope=scope, alpha=args.alpha,
                          min_units=args.min_units, **kwargs)
        tol = result['contract']['tolerance']
        label = (f"tolerance {tol['absolute']:g}" if 'absolute' in tol else
                 f"tolerance {tol['fraction']:.0%} of the mean |{tol['of_gap_between'][0]} - "
                 f"{tol['of_gap_between'][1]}| gap, resampled with the loss")
        report(result, f'loss {loss}, scope {scope}, {label}')
    print_comparisons(result['comparisons'])
    contract = result['contract']
    print(f"\n{contract['units']} units; alpha {contract['alpha']}; {contract['intervals']}; "
          f"{contract['resamples']} resamples, seed {contract['seed']}.")


if __name__ == '__main__':
    main()
