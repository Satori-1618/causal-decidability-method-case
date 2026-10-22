"""Pure planning arithmetic: no model, outcome-file or network access.

Counts passed to these functions are hypothetical unless a caller explicitly
uses this code in a later, separately authorized analysis. The command line
only generates/checks the frozen hypothetical planning table.
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

SCREEN_BUDGET = 1024
N = 64
HEADS = 6
REQUIRED_HEADS = 4
ALPHA = .05
TAIL_ALPHA = ALPHA / (4 * HEADS)
MINIMUM_UPLIFT = .25
BASELINE_SEQUENCE_FORWARDS = 2
FAMILY_SEQUENCE_FORWARDS = 16
HYPOTHETICAL_RATES = ((.90, .05), (.80, .10), (.75, .10), (.60, .10), (.50, .10))


def _count(value, maximum, name):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(name + ' must be an integer in [0, ' + str(maximum) + ']')
    return value


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a finite probability')
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + ' must be a finite probability')
    return value


def _binomial_mass(n, k, p):
    return math.comb(n, k) * p**k * (1-p)**(n-k)


def _tail(n, p, k, upper):
    indices = range(k, n+1) if upper else range(k+1)
    return math.fsum(_binomial_mass(n, j, p) for j in indices)


@lru_cache(maxsize=None)
def exact_bounds(k, n=N, alpha=TAIL_ALPHA):
    """Two CP bounds, each with failure probability alpha (not alpha/2)."""
    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')
    _count(k, n, 'k')
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < .5:
        raise ValueError('alpha must lie strictly between 0 and 0.5')
    lower, upper = 0., 1.
    if k:
        lo, hi = 0., 1.
        for _ in range(70):
            mid = (lo+hi)/2
            if _tail(n, mid, k, True) < alpha:
                lo = mid
            else:
                hi = mid
        lower = (lo+hi)/2
    if k < n:
        lo, hi = 0., 1.
        for _ in range(70):
            mid = (lo+hi)/2
            if _tail(n, mid, k, False) > alpha:
                lo = mid
            else:
                hi = mid
        upper = (lo+hi)/2
    return lower, upper


def head_result(accepted_hits, rejected_hits):
    """One head, exactly 64 measured families in each stratum."""
    _count(accepted_hits, N, 'accepted_hits')
    _count(rejected_hits, N, 'rejected_hits')
    a = exact_bounds(accepted_hits)
    r = exact_bounds(rejected_hits)
    interval = (a[0]-r[1], a[1]-r[0])
    return {
        'accepted_hits': accepted_hits, 'rejected_hits': rejected_hits,
        'families_per_stratum': N,
        'accepted_interval': list(a), 'rejected_interval': list(r),
        'delta': (accepted_hits-rejected_hits)/N,
        'delta_interval': list(interval),
        'substantial_enrichment': interval[0] > MINIMUM_UPLIFT,
        'practical_status': ('supported' if interval[0] > MINIMUM_UPLIFT else
                             'excluded' if interval[1] < MINIMUM_UPLIFT else 'unresolved'),
        'directional_status': ('positive' if interval[0] > 0 else
                               'negative' if interval[1] < 0 else 'unresolved')}


def cohort_result(head_counts):
    """Fixed six-head cohort; never replace an unavailable head with a new one.

    Each entry is an (accepted_hits, rejected_hits) pair or None for a head that
    cannot contribute a scientific decision. None is retained as unavailable,
    not imputed as a zero effect. Individual failure reasons belong in the later
    run manifest, not this arithmetic helper.
    """
    if len(head_counts) != HEADS:
        raise ValueError('The fixed cohort must contain exactly six entries')
    results = [None if item is None else head_result(*item) for item in head_counts]
    successes = sum(r is not None and r['substantial_enrichment'] for r in results)
    return {'head_results': results, 'substantial_heads': successes,
            'unavailable_heads': sum(r is None for r in results),
            'positive_direction_heads': sum(r is not None and r['directional_status']=='positive' for r in results),
            'primary_success': successes >= REQUIRED_HEADS,
            'criterion': 'At least four of the fixed six lower Delta bounds exceed 0.25'}


def hypothetical_power(p_accepted, p_rejected):
    """Exact count enumeration; six-head binomial power is illustrative only."""
    _probability(p_accepted, 'p_accepted')
    _probability(p_rejected, 'p_rejected')
    masses_a = [_binomial_mass(N, k, p_accepted) for k in range(N+1)]
    masses_r = [_binomial_mass(N, k, p_rejected) for k in range(N+1)]
    per_head = math.fsum(masses_a[a]*masses_r[r] for a in range(N+1) for r in range(N+1)
                         if head_result(a, r)['substantial_enrichment'])
    independent = math.fsum(_binomial_mass(HEADS, k, per_head)
                            for k in range(REQUIRED_HEADS, HEADS+1))
    return {'assumed_accepted_rate': p_accepted, 'assumed_rejected_rate': p_rejected,
            'per_head_power': per_head,
            'at_least_four_if_equal_and_independent': independent,
            'at_least_four_without_independence_bounds': [max(0., 1-2*(1-per_head)),
                                                          min(1., HEADS*per_head/REQUIRED_HEADS)],
            'assumptions': 'Filled strata and valid gates; equal per-head rates. Binomial six-head power additionally assumes independent decisions.'}


def ratio_record(numerator, denominator):
    """Explicit JSON-safe finite, infinite or undefined cost ratio."""
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v < 0 for v in (numerator, denominator)):
        raise ValueError('Ratio arguments must be finite and nonnegative')
    if denominator == 0:
        return {'status': 'undefined' if numerator == 0 else 'infinite', 'value': None,
                'reason': 'zero_expected_separating_families'}
    return {'status': 'finite', 'value': numerator/denominator}


def cost_estimates(acceptance_count, accepted_hits, rejected_hits,
                   b=BASELINE_SEQUENCE_FORWARDS, f=FAMILY_SEQUENCE_FORWARDS):
    """Descriptive policy estimates, not a measured random-control comparison.

    b counts both dtype baseline forwards. f=16 counts the unchanged two-anchor
    runtime, including its repeated recipient native passes. No cache reuse or
    subtraction is assumed. Even the random-policy comparator pays b outside f.
    """
    _count(acceptance_count, SCREEN_BUDGET, 'acceptance_count')
    if acceptance_count < N or SCREEN_BUDGET-acceptance_count < N:
        raise ValueError('insufficient_yield: both strata need at least 64 of the fixed 1024 candidates')
    _count(accepted_hits, N, 'accepted_hits')
    _count(rejected_hits, N, 'rejected_hits')
    for name, value in [('b', b), ('f', f)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(name + ' must be finite and nonnegative')
    q, pa, pr = acceptance_count/SCREEN_BUDGET, accepted_hits/N, rejected_hits/N
    random_yield = q*pa+(1-q)*pr
    fixed_total = SCREEN_BUDGET*b+N*f
    validation_total = SCREEN_BUDGET*b+2*N*f
    ratios = {
        'fixed_screened_policy': ratio_record(fixed_total, N*pa),
        'random_selection_policy': ratio_record(b+f, random_yield),
        'ideal_stream_NOT_SCHEDULED': ratio_record(b/q+f, pa),
        'validation_run_NOT_DEPLOYMENT': ratio_record(validation_total, N*(pa+pr))}
    fixed, random = ratios['fixed_screened_policy'], ratios['random_selection_policy']
    cheaper = (fixed['value'] < random['value']
               if fixed['status'] == random['status'] == 'finite' else None)
    return {'status': 'descriptive_point_estimates_only',
            'counts': {'screened': SCREEN_BUDGET, 'accepted': acceptance_count,
                       'accepted_separating': accepted_hits, 'rejected_separating': rejected_hits},
            'acceptance_rate': q, 'accepted_yield': pa, 'rejected_yield': pr,
            'random_selection_yield_estimate': random_yield,
            'accepted_minus_random_yield_estimate': pa-random_yield,
            'cost_unit': 'sequence forwards, including both dtypes; not walltime',
            'baseline_cost_b': b, 'two_anchor_family_cost_f': f,
            'fixed_screened_policy_total_cost': fixed_total,
            'validation_total_cost': validation_total,
            'rejected_arm_measurement_overhead': N*f,
            'estimated_cost_per_separating_family': ratios,
            'fixed_policy_cheaper_under_point_estimates': cheaper,
            'limits': ['Random yield is a stratified plug-in estimate, not an observed random arm.',
                       'No uncertainty or verified net savings is claimed for these ratios.',
                       'Ideal stream amortization is not the fixed-budget scheduled validation.',
                       'Family cost includes repeated native passes; no unimplemented caching is credited.',
                       'Validation pays for rejected families; deployment estimates omit that validation-only overhead.']}


def planning_report():
    return {'status': 'hypothetical_planning_only_no_measured_outcomes',
            'constants': {'screen_budget_per_head': SCREEN_BUDGET, 'families_per_stratum': N,
                          'heads': HEADS, 'required_substantial_heads': REQUIRED_HEADS,
                          'tail_alpha': TAIL_ALPHA, 'minimum_uplift': MINIMUM_UPLIFT,
                          'baseline_sequence_forwards': BASELINE_SEQUENCE_FORWARDS,
                          'two_anchor_family_sequence_forwards': FAMILY_SEQUENCE_FORWARDS},
            'primary': 'At least four of six simultaneous lower Delta bounds strictly exceed 0.25; direction reported separately.',
            'scenarios': [hypothetical_power(*rates) for rates in HYPOTHETICAL_RATES],
            'minimum_accepted_hits_by_rejected_hits': [
                {'rejected_hits': r, 'minimum_accepted_hits': next((a for a in range(N+1)
                 if head_result(a, r)['substantial_enrichment']), None)} for r in range(N+1)],
            'cost_formulas': {'random_yield': 'q*p_A + (1-q)*p_R',
                              'fixed_screened_policy': '(1024*b + 64*f)/(64*p_A)',
                              'random_selection_policy': '(b+f)/p_random',
                              'ideal_stream_NOT_SCHEDULED': '(b/q+f)/p_A',
                              'validation_total': '1024*b + 128*f'},
            'shortfall': 'Fewer than 64 accepted OR 64 rejected means insufficient yield; no imputation or additional screening.',
            'limitations': ['Hypothetical rates are assumptions, not predictions from measured results.',
                            'Power is conditional on quotas and technical gates passing.',
                            'Shared inputs may invalidate independent six-head power; dependence-free bounds are also reported.',
                            'The fixed six-head cohort does not identify a population success rate.',
                            'Cost estimates are descriptive and use no measured outcomes.']}


def _assert_same(actual, expected, path='root'):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('Planning keys changed at '+path)
        for key in actual:
            _assert_same(actual[key], expected[key], path+'.'+key)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Planning length changed at '+path)
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_same(a, e, path+'.'+str(i))
    elif type(actual) is float and isinstance(expected, (int, float)):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Planning number changed at '+path)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError('Planning value changed at '+path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check sibling planning.json without modifying it')
    args = parser.parse_args()
    report = planning_report()
    if args.check:
        _assert_same(report, json.loads(Path(__file__).with_name('planning.json').read_text()))
        print('PASS: frozen hypothetical planning table matches the calculator')
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
