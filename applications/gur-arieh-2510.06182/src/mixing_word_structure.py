"""Round 2 development: paired word/target-site exchange. No model imports.

This tests narrow invariances of a baseline-adjusted patch effect, not a complete
positional/lexical/reflexive mechanism. The unit is a pair of binding matrices.
"""
import math
import statistics

from mixing_round1_design import TARGET, design_indices, target_rebind
from mixing_round1_analysis import softmax


def swap_pair(matrix, cell):
    """Swap only the genres at i_P and i_L; rebuild both conflict donors.

Both native correct answers and both query strings are unchanged. The token
multiset and all four candidate indices are unchanged. Swapping twice is identity.
"""
    original = [tuple(row) for row in matrix]
    swapped = [list(row) for row in matrix]
    p, l = cell['i_P'], cell['i_L']
    swapped[p][TARGET], swapped[l][TARGET] = original[l][TARGET], original[p][TARGET]
    swapped = [tuple(row) for row in swapped]
    cases = [target_rebind(g, cell, len(g)) for g in (original, swapped)]
    for case in cases:
        if design_indices(**case) != cell:
            raise ValueError('swap changed the candidate indices')
    for field in ('recipient_query', 'donor_query'):
        if cases[0][field] != cases[1][field]:
            raise ValueError('swap changed a query')
    for key, index in (('recipient', cell['i_N']), ('donor', p)):
        if cases[0][key][index][TARGET] != cases[1][key][index][TARGET]:
            raise ValueError('swap changed a native correct answer')
    return cases


def arm_measures(arm, cell):
    p, l = cell['i_P'], cell['i_L']
    patch = arm['patch']
    base = arm['native']['recipient']['readout']
    values = patch['answer_logits'] + base['answer_logits']
    if not all(math.isfinite(v) for v in values):
        raise ValueError('non-finite logits')
    raw = patch['answer_logits'][p] - patch['answer_logits'][l]
    baseline = base['answer_logits'][p] - base['answer_logits'][l]
    probs = softmax(patch['answer_logits'])
    return {'patch_log_odds_P_over_L': raw, 'native_log_odds_P_over_L': baseline,
            'delta': raw - baseline, 'pair_mass_within_answers': probs[p] + probs[l],
            'answer_mass': patch['answer_mass_full_vocab']}


def paired_measures(record, rule):
    a, b = [arm_measures(arm, record['cell']) for arm in record['arms']]
    d0, d1 = a['delta'], b['delta']
    scale = (abs(d0) + abs(d1)) / 2
    signal = scale >= rule['minimum_mean_abs_delta_nats']
    support = all(x['pair_mass_within_answers'] >= rule['pair_mass_floor']
                  and x['answer_mass'] >= rule['answer_mass_floor'] for x in (a, b))
    # Symmetric losses avoid choosing a winner in the original and using it as truth.
    loss_structure = abs(d1 - d0) / (2 * scale) if scale else None
    loss_word = abs(d1 + d0) / (2 * scale) if scale else None
    resolved = signal and support
    if not support:
        label = 'outside_pair_support'
    elif not signal:
        label = 'weak_patch_contrast'
    elif loss_structure <= rule['relative_residual_tolerance']:
        label = 'structure_like'
    elif loss_word <= rule['relative_residual_tolerance']:
        label = 'word_like'
    else:
        label = 'neither_narrow_profile'
    return {'case_id': record['case_id'], 'arms': [a, b], 'scale': scale,
            'support': support, 'signal': signal, 'resolved': resolved,
            'loss_structure': loss_structure, 'loss_word': loss_word,
            'loss_word_minus_structure': (loss_word - loss_structure) if scale else None,
            'label': label}


def summarize(records, rule):
    if not records or len({r['case_id'] for r in records}) != len(records):
        raise ValueError('empty or duplicated families')
    qualified = [r for r in records if r['qualifies']]
    rows = [paired_measures(r, rule) for r in qualified]
    labels = ('structure_like', 'word_like', 'neither_narrow_profile',
              'weak_patch_contrast', 'outside_pair_support')
    counts = {label: sum(x['label'] == label for x in rows) for label in labels}
    return {'stage': 'DEVELOPMENT_ONLY', 'generated': len(records),
            'qualified': len(qualified), 'nonqualifying': len(records)-len(qualified),
            'yield': len(qualified)/len(records), 'counts': counts,
            'resolved': sum(x['resolved'] for x in rows),
            'median_mean_abs_delta_nats': statistics.median(x['scale'] for x in rows) if rows else None,
            'confirmatory_claim': None, 'cases': rows}
