"""Post hoc descriptive companion for Q1, round 2 and round 3A; no new decisions.

Everything here was computed after the frozen analyses, from the stored records, with
the standard library and without loading a model. It describes measured effect ratios
and how strict the frozen tolerances were. Tolerance rows other than the frozen one are
not tests: no interval, status or claim is derived from them. The script refuses to
report if the frozen tolerance does not reproduce the frozen profile counts.
"""
import argparse
import json
import statistics
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
RESULTS = APP / 'results'
Q1 = RESULTS / 'makelov_read_source_q1'
ROUND2 = RESULTS / 'query_route_confirmation'
ROUND3A = RESULTS / 'donor_factor_confirmation_512'

PRECISIONS = ('float32', 'float64')
CELLS = ('00', '01', '10', '11')

ROUND2_FROZEN_KAPPA = .25
ROUND2_NUMERICAL_FRACTION = .025
# Predicted (R, S) in units of the direction's own T, as frozen in round 2.
ROUND2_PROFILES = {'transfer': (0, 1), 'joint_dependence': (0, 0), 'preservation': (1, 0)}
# Endpoint profiles differ by |T| in at least one contrast, so kappa >= 0.5 lets two
# profiles fit one case: mutual exclusivity is no longer guaranteed, although other cases
# may still distinguish them. The frozen analyzer refuses such kappas for that reason.
ROUND2_KAPPAS = (.10, .15, .20, .25, .30, .35, .40, .45)

ROUND3A_FROZEN_EPSILON = .25
ROUND3A_NUMERICAL_BUDGET = .01
ROUND3A_PROFILE_RESIDUALS = {'position_only': ('identity_at_p0', 'identity_at_p1'),
                             'identity_only': ('position_at_i0', 'position_at_i1')}
ROUND3A_EPSILONS = (.10, .15, .20, .25, .30, .35, .40, .50)


def _records(directory):
    with open(directory / 'records.jsonl', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _spread(values):
    return {'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values)}


def read_source_q1(records):
    """Measured read effects as fractions of the same directed case's full effect."""
    null_fraction, row_fraction, inside, order_agrees, magnitude_agrees = [], [], 0, 0, 0
    for record in records:
        m, fidelity = record['margins'], record['fidelity']
        y0, full, row, null = (m[k] for k in ('baseline', 'full', 'read_row', 'read_null'))
        if full == y0:
            raise ValueError(f"{record['case_id']}: zero full-patch effect")
        r_row, r_null = (row - y0) / (full - y0), (null - y0) / (full - y0)
        row_fraction.append(r_row)
        null_fraction.append(r_null)
        inside += 0 <= r_row <= 1 and 0 <= r_null <= 1
        loss_a = (abs(row - full) + abs(null - y0)) / 2
        loss_b = (abs(row - y0) + abs(null - full)) / 2
        # All read conditions write along the same v, so inserted-change L2 norms order
        # the read-coefficient magnitudes; effects are compared as magnitudes too.
        coefficient_row = fidelity['read_row']['actual_delta_l2_per_item']
        coefficient_null = fidelity['read_null']['actual_delta_l2_per_item']
        order_agrees += (loss_b < loss_a) == (coefficient_null > coefficient_row)
        magnitude_agrees += ((coefficient_null > coefficient_row)
                             == (abs(null - y0) > abs(row - y0)))
    return {
        'n_directed_cases_not_independent': len(records),
        'null_read_effect_over_full_effect': _spread(null_fraction),
        'visible_read_effect_over_full_effect': _spread(row_fraction),
        'directed_cases_with_both_fractions_in_0_1': inside,
        'directed_cases_where_coefficient_magnitude_order_matches_B_versus_A': order_agrees,
        'directed_cases_where_larger_coefficient_magnitude_gives_larger_absolute_effect':
            magnitude_agrees,
        'reading': 'Ratios of measured effects, not identified mechanism shares. Coefficient '
                   'and effect magnitudes were ordered alike in the measured conditions; '
                   'this is not a response curve established beyond them.',
    }


def round2_profile_counts(records, kappa, numerical_fraction=ROUND2_NUMERICAL_FRACTION):
    """Case-wise round-2 profile successes under the frozen rule with another kappa."""
    if not 0 <= kappa < .5:
        raise ValueError('kappa must lie in [0, 0.5): larger values let two profiles fit one case')
    counts = dict.fromkeys(ROUND2_PROFILES, 0)
    for record in records:
        contrasts = {}
        for p in PRECISIONS:
            cells = record['precisions'][p]['cells']
            contrasts[p] = [{'T': cells['B'][i] - cells['A'][i],
                             'R': cells['D'][i] - cells['C'][i],
                             'S': cells['E'][i] - cells['C'][i]} for i in (0, 1)]
        resolved = True
        for i in (0, 1):
            a, b = contrasts['float32'][i], contrasts['float64'][i]
            t = min(abs(a['T']), abs(b['T']))
            same_sign = (a['T'] > 0 and b['T'] > 0) or (a['T'] < 0 and b['T'] < 0)
            if (t == 0 or not same_sign
                    or max(abs(a[c] - b[c]) for c in 'TRS') > numerical_fraction * t):
                resolved = False
        for profile, (r_units, s_units) in ROUND2_PROFILES.items():
            counts[profile] += resolved and all(
                abs(c['R'] - r_units * c['T']) <= kappa * abs(c['T'])
                and abs(c['S'] - s_units * c['T']) <= kappa * abs(c['T'])
                for p in PRECISIONS for c in contrasts[p])
    return counts


def _round3a_panels(precision_record):
    panels = []
    for panel in precision_record['panels']:
        base = panel['baseline_margin']
        d = {c: panel['patched_margins'][c] - base for c in CELLS}
        simple = {'identity_at_p0': d['10'] - d['00'], 'identity_at_p1': d['11'] - d['01'],
                  'position_at_i0': d['01'] - d['00'], 'position_at_i1': d['11'] - d['10']}
        contrasts = {'I': simple['identity_at_p0'] / 2 + simple['identity_at_p1'] / 2,
                     'P': simple['position_at_i0'] / 2 + simple['position_at_i1'] / 2,
                     'J': simple['identity_at_p1'] - simple['identity_at_p0']}
        panels.append({'deltas': d, 'simple': simple, 'contrasts': contrasts,
                       'alphas': panel['alphas']})
    return panels


def round3a_profile_counts(records, epsilon, budget=ROUND3A_NUMERICAL_BUDGET):
    """Complete-family round-3A profile successes under the frozen rule with another epsilon."""
    if epsilon < 0:
        raise ValueError('epsilon must be nonnegative')
    counts = dict.fromkeys(ROUND3A_PROFILE_RESIDUALS, 0)
    for record in records:
        panels = {p: _round3a_panels(record['precisions'][p]) for p in PRECISIONS}
        differences = [abs(panels['float32'][i][field][name] - panels['float64'][i][field][name])
                       for i in (0, 1) for field in ('deltas', 'simple', 'contrasts')
                       for name in panels['float32'][i][field]]
        units = {p: {c: panels[p][0]['contrasts'][c] / 2 + panels[p][1]['contrasts'][c] / 2
                     for c in 'IPJ'} for p in PRECISIONS}
        differences += [abs(units['float32'][c] - units['float64'][c]) for c in 'IPJ']
        resolved = max(differences) <= budget
        for profile, residuals in ROUND3A_PROFILE_RESIDUALS.items():
            counts[profile] += resolved and all(
                abs(panel['simple'][name]) <= epsilon
                for p in PRECISIONS for panel in panels[p] for name in residuals)
    return counts


def round3a_mean_contrast_diagnostics(records):
    """Why panel-averaged name contrasts are weak evidence of name independence (float64)."""
    abs_name, abs_interaction = ([], []), ([], [])
    opposite_name_signs = opposite_slopes = scalar_name_positive = 0
    same_interaction_signs = scalar_interaction_positive = 0
    positive_interaction = [0, 0]
    residuals, scalar_name_panel_gap = [], 0.
    for record in records:
        panels = _round3a_panels(record['precisions']['float64'])
        slopes = []
        for i, panel in enumerate(panels):
            abs_name[i].append(abs(panel['contrasts']['I']))
            abs_interaction[i].append(abs(panel['contrasts']['J']))
            positive_interaction[i] += panel['contrasts']['J'] > 0
            a, d = panel['alphas'], panel['deltas']
            others = ('01', '10', '11')
            slope = sum(d[c] * a[c] for c in others) / sum(a[c] ** 2 for c in others)
            slopes.append(slope)
            residuals.extend(abs(d[c] - slope * a[c]) for c in others)
        opposite_name_signs += panels[0]['contrasts']['I'] * panels[1]['contrasts']['I'] < 0
        opposite_slopes += slopes[0] * slopes[1] < 0
        same_interaction_signs += panels[0]['contrasts']['J'] * panels[1]['contrasts']['J'] > 0
        a0 = panels[0]['alphas']
        scalar_interaction_positive += a0['11'] - a0['10'] - a0['01'] + a0['00'] > 0
        scalar_name = [(p['alphas']['10'] + p['alphas']['11']
                        - p['alphas']['00'] - p['alphas']['01']) / 2 for p in panels]
        scalar_name_panel_gap = max(scalar_name_panel_gap, abs(scalar_name[0] - scalar_name[1]))
        scalar_name_positive += scalar_name[0] > 0
    return {
        'n_families': len(records),
        'mean_absolute_name_contrast_I_by_recipient_order': [statistics.mean(v) for v in abs_name],
        'families_with_opposite_sign_I_across_recipient_orders': opposite_name_signs,
        'families_with_opposite_sign_response_slopes_across_recipient_orders': opposite_slopes,
        'median_absolute_residual_of_linear_response_in_read_scalar': statistics.median(residuals),
        'maximum_absolute_residual_of_linear_response_in_read_scalar': max(residuals),
        'families_with_positive_scalar_level_name_contrast': scalar_name_positive,
        'mean_absolute_interaction_contrast_J_by_recipient_order':
            [statistics.mean(v) for v in abs_interaction],
        'families_with_same_sign_J_across_recipient_orders': same_interaction_signs,
        'families_with_positive_J_by_recipient_order': positive_interaction,
        'families_with_positive_scalar_level_interaction_contrast': scalar_interaction_positive,
        'maximum_scalar_level_name_contrast_gap_between_recipient_orders': scalar_name_panel_gap,
        'reading': 'By design, name exchange reverses the scalar-level name and interaction '
                   'contrasts, so these are symmetric about zero. The model response need not '
                   'preserve that symmetry: output signs, opposite slopes and near-linearity '
                   'are measured here, not guaranteed. In these data averaged I largely '
                   'cancels between recipient orders, whereas J adds there and varies in sign '
                   'across families. Use the case-wise invariance profiles for invariance claims.',
    }


def describe(q1=Q1, round2=ROUND2, round3a=ROUND3A):
    q1_records, r2_records, r3_records = _records(q1), _records(round2), _records(round3a)
    r2_summary = json.loads((round2 / 'summary.json').read_text())
    r3_summary = json.loads((round3a / 'summary.json').read_text())
    r2_rows = {k: round2_profile_counts(r2_records, k) for k in ROUND2_KAPPAS}
    r3_rows = {e: round3a_profile_counts(r3_records, e) for e in ROUND3A_EPSILONS}
    for frozen, rows, summary in ((ROUND2_FROZEN_KAPPA, r2_rows, r2_summary),
                                  (ROUND3A_FROZEN_EPSILON, r3_rows, r3_summary)):
        stored = {name: profile['successes'] for name, profile in summary['profiles'].items()}
        if rows[frozen] != stored:
            raise ValueError(f'frozen tolerance does not reproduce stored counts: {stored}')
    return {
        'status': 'post hoc descriptive; every frozen decision is unchanged',
        'read_source_q1': read_source_q1(q1_records),
        'round2_tolerance_sensitivity': {
            'n_base_pairs': len(r2_records), 'frozen_kappa': ROUND2_FROZEN_KAPPA,
            'rule': 'both predicted contrasts within kappa*|T|, both directions, both precisions; '
                    'numerical gate unchanged',
            'successes_by_kappa': {f'{k:.2f}': v for k, v in r2_rows.items()},
            'upper_limit': 'kappa >= 0.5 lets two endpoint profiles fit the same case, so '
                           'mutual exclusivity is no longer guaranteed',
        },
        'round3a_tolerance_sensitivity': {
            'n_families': len(r3_records), 'frozen_epsilon_nat': ROUND3A_FROZEN_EPSILON,
            'rule': 'both relevant simple effects within epsilon, both recipient orders, both '
                    'precisions; numerical gate unchanged',
            'successes_by_epsilon_nat': {f'{e:.2f}': v for e, v in r3_rows.items()},
        },
        'round3a_mean_contrast_diagnostics': round3a_mean_contrast_diagnostics(r3_records),
        'scope': 'No model was loaded. Non-frozen tolerance rows are descriptions of '
                 'strictness, not tests; they receive no interval, status or claim.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(describe(), indent=2, allow_nan=False))
