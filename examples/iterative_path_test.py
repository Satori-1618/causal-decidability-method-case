"""A constructed second round of explanation testing. NOT LLM RESULTS.

    python3 -I -S examples/iterative_path_test.py
    python3 -I -S examples/iterative_path_test.py --json

The four outcomes are supplied numbers, not model measurements. ``controls_passed``
and an absolute error bound per cell are declared assumptions of this teaching example;
this script neither validates neural hooks nor estimates empirical error or power.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from causal_decidability.design import gains, restrict, signatures  # noqa: E402


CELLS = ('y00', 'y10', 'y01', 'y11')
CANDIDATES = ('selected_queries', 'bypass')
NOTICE = 'CONSTRUCTED TEACHING EXAMPLES — NOT LLM RESULTS'


def _finite(value, label, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a finite number')
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (nonnegative and value < 0):
        raise ValueError(f'{label} must be finite' + (' and non-negative' if nonnegative else ''))
    return float(value)


def _interval(center, radius):
    return [_finite(center - radius, 'interval lower endpoint'),
            _finite(center + radius, 'interval upper endpoint')]


def _null_zone_status(interval, tolerance):
    lo, hi = interval
    if lo >= -tolerance and hi <= tolerance:
        return 'adequate'
    if hi < -tolerance or lo > tolerance:
        return 'excluded'
    return 'undecided'


def analyze_world(cells, *, per_cell_error=0.02, tolerance=0.2, controls_passed=True):
    """Check two narrow endpoint contracts under declared deterministic bounds.

    y00 = patch off / queries free; y10 = patch on / queries free;
    y01 = patch off / queries clamped; y11 = patch on / queries clamped.
    T = y10-y00; R = y11-y01; K = T-R.

    selected_queries predicts R=0; bypass predicts R=T, or equivalently K=0.
    The uncertainty of the measured T anchor is included in K's four-cell bound.
    These are predictions about this particular clamp, not exhaustive neural stories.
    """
    if set(cells) != set(CELLS):
        raise ValueError('provide exactly y00, y10, y01, y11')
    cells = {name: _finite(cells[name], name) for name in CELLS}
    error = _finite(per_cell_error, 'per_cell_error', nonnegative=True)
    tolerance = _finite(tolerance, 'tolerance', nonnegative=True)
    if not isinstance(controls_passed, bool):
        raise ValueError('controls_passed must be a boolean declaration')

    total = _finite(cells['y10'] - cells['y00'], 'T')
    residual = _finite(cells['y11'] - cells['y01'], 'R')
    interaction = _finite(total - residual, 'K')
    contrasts = {'T': total, 'R': residual, 'K': interaction}
    radii = {'T': _finite(2 * error, 'T radius'),
             'R': _finite(2 * error, 'R radius'),
             'K': _finite(4 * error, 'K radius')}
    intervals = {name: _interval(value, radii[name]) for name, value in contrasts.items()}

    # Both candidates reproduce the measured full effect by construction. Only the
    # additional residual condition tests them. These are the shipped design functions.
    predictions = {'selected_queries': (total, 0.0), 'bypass': (total, total)}
    design = {'conditions': ['T_anchor_not_a_test', 'R_after_clamp'],
              'predictions': predictions,
              'before_groups': signatures(restrict(predictions, [0])),
              'after_groups': signatures(predictions),
              'newly_separated_pairs': gains(predictions, [0], [1])}

    # Candidate residuals are R and K. Retaining both requires |R| <= tol+2e
    # and |K| <= tol+4e, hence |T| <= 2tol+6e by the triangle inequality.
    # Strictly exceeding that sum is a conservative sufficient resolution gate.
    # Its failure is this illustration's declared stopping rule, not a proof that
    # every possible observation or different experiment would be uninformative.
    threshold = _finite(2 * tolerance + radii['R'] + radii['K'], 'separation threshold')
    resolution_passed = abs(total) > threshold
    status = {name: 'not_assessed' for name in CANDIDATES}
    retained = list(CANDIDATES)
    if not controls_passed:
        outcome, reason = 'invalid_control', 'declared intervention controls did not pass'
    elif total == 0:
        outcome, reason = 'insufficient_resolution', 'zero T gives identical endpoint predictions'
    elif not resolution_passed:
        outcome, reason = ('insufficient_resolution',
                           'the declared tolerance and error bounds do not clear the separation gate')
    else:
        status = {'selected_queries': _null_zone_status(intervals['R'], tolerance),
                  'bypass': _null_zone_status(intervals['K'], tolerance)}
        retained = [name for name in CANDIDATES if status[name] != 'excluded']
        adequate = [name for name in CANDIDATES if status[name] == 'adequate']
        if not retained:
            outcome, reason = 'no_candidate_fits', 'both narrow endpoint contracts are excluded'
        elif len(retained) == 1 and adequate == retained:
            outcome = retained[0] + '_endpoint_adequate'
            reason = 'one endpoint meets the illustrative tolerance; the rival is excluded'
        else:
            outcome, reason = ('insufficient_resolution',
                               'a retained candidate has not been shown adequate')

    return {'source': NOTICE, 'cells': cells, 'contrasts': contrasts,
            'declared_per_cell_absolute_error_bound': error,
            'illustration_only_tolerance': tolerance, 'deterministic_radii': radii,
            'deterministic_intervals_not_confidence_intervals': intervals,
            'controls_passed_declaration': controls_passed,
            'design': design,
            'resolution_gate': {'criterion': '|T| > 2*tolerance + R_radius + K_radius',
                                'threshold': threshold, 'passed': resolution_passed},
            'candidate_status': status, 'retained_candidates': retained,
            'outcome': outcome, 'reason': reason,
            'illustrative_absolute_losses': {'selected_queries': abs(residual),
                                             'bypass': abs(interaction)}}


def demonstration():
    """Four explicitly constructed outcomes and two blocked-interpretation examples."""
    specs = [
        ('route', {'y00': 1, 'y10': 3, 'y01': 1, 'y11': 1}, 0.02),
        ('bypass', {'y00': 1, 'y10': 3, 'y01': 1, 'y11': 3}, 0.02),
        ('intermediate', {'y00': 1, 'y10': 3, 'y01': 1, 'y11': 2}, 0.02),
        ('insufficient_resolution', {'y00': 1, 'y10': 1.5, 'y01': 1, 'y11': 1.25}, 0.1),
    ]
    worlds = [{'name': name, **analyze_world(cells, per_cell_error=error)}
              for name, cells, error in specs]
    guards = [
        {'name': 'invalid_control', **analyze_world(specs[0][1], controls_passed=False)},
        {'name': 'zero_total_effect', **analyze_world(
            {'y00': 1, 'y10': 1, 'y01': 1, 'y11': 1})},
    ]
    return {'notice': NOTICE,
            'bounds': 'Mathematically declared absolute bounds per cell; no empirical coverage claim.',
            'scope': 'Tests two endpoint predictions for a specified clamp; not a unique mechanism '
                     'or a mechanism share. World names label outcome tables, not known neural '
                     'mechanisms. No model inference, power estimate or measured data.',
            'empirical_follow_up': 'On fresh paired units, compare absolute losses |R| versus '
                                   '|R-T| under a frozen rule, keeping adequacy separate.',
            'worlds': worlds, 'guard_examples': guards}


def render(result):
    lines = [result['notice'], result['bounds'], result['scope']]
    for world in result['worlds'] + result['guard_examples']:
        lines.append('\nCONSTRUCTED: ' + world['name'])
        lines.append('  cells: ' + ', '.join(f'{k}={v:g}' for k, v in world['cells'].items()))
        lines.append('  contrasts: ' + ', '.join(f'{k}={v:g}' for k, v in world['contrasts'].items()))
        for name, interval in world['deterministic_intervals_not_confidence_intervals'].items():
            lines.append(f'  {name} deterministic interval: [{interval[0]:g}, {interval[1]:g}]')
        design = world['design']
        lines.append(f'  prediction groups: {design["before_groups"]} -> {design["after_groups"]}')
        lines.append(f'  separating addition: {design["newly_separated_pairs"]}')
        lines.append(f'  statuses: {world["candidate_status"]}')
        lines.append(f'  outcome: {world["outcome"]}; {world["reason"]}')
    lines.append('\nEmpirical follow-up (not executed): ' + result['empirical_follow_up'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = demonstration()
    print(json.dumps(result, indent=2, allow_nan=False) if args.json else render(result))


if __name__ == '__main__':
    main()
