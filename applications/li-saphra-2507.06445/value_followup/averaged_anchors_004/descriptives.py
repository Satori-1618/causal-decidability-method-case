"""Separate descriptive diagnostics; never alter the frozen primary decisions.

C predicts a fresh target from the two CALIBRATION prefixes in that same cell.
The 2x2 tables are calculated only after targets: four observed prefixes/cell,
two from each role. Their spread is prefix variation, not measurement noise.
Sum-of-squares terms describe this finite table, not causal shares or tests.
"""
import math

import analysis

CELL_KEYS = tuple(f'{sign}_{position}' for sign in ('neg', 'pos') for position in (20, 28))
TOLERANCE = .10


def _grids(values, role):
    analysis.require(isinstance(values, dict) and set(values) == set(analysis.DTYPES),
                     'Both complete dtypes are required')
    return {dtype: analysis.grid(values[dtype], role) for dtype in analysis.DTYPES}


def forecasts(calibration_by_dtype):
    """No target input is accepted by this prospective forecast function."""
    calibration = _grids(calibration_by_dtype, 'calibration')
    result = {}
    for dtype, values in calibration.items():
        means = {cell: (values['calibration_' + cell + '_0'] + values['calibration_' + cell + '_1']) / 2
                 for cell in CELL_KEYS}
        result[dtype] = {target: means[target.removeprefix('target_').rsplit('_', 1)[0]]
                         for target in analysis.TARGET_CELLS}
    return result


def _table(calibration, targets):
    observations = {cell: [values[role + '_' + cell + '_' + str(replica)]
                           for role, values in (('calibration', calibration), ('target', targets))
                           for replica in (0, 1)] for cell in CELL_KEYS}
    means = {cell: math.fsum(values) / 4 for cell, values in observations.items()}
    grand = math.fsum(value for values in observations.values() for value in values) / 16
    balance_simple = {str(p): means[f'pos_{p}'] - means[f'neg_{p}'] for p in (20, 28)}
    position_simple = {s: means[f'{s}_28'] - means[f'{s}_20'] for s in ('neg', 'pos')}
    balance = math.fsum(balance_simple.values()) / 2
    position = math.fsum(position_simple.values()) / 2
    interaction = balance_simple['28'] - balance_simple['20']
    cell_ss = {cell: math.fsum((x - means[cell]) ** 2 for x in values)
               for cell, values in observations.items()}
    within = math.fsum(cell_ss.values())
    total = math.fsum((x - grand) ** 2 for values in observations.values() for x in values)
    ss = {'balance': 4 * balance ** 2, 'position': 4 * position ** 2,
          'interaction': interaction ** 2, 'within': within, 'total': total}
    decomposed = math.fsum(ss[key] for key in ('balance', 'position', 'interaction', 'within'))
    identity_error = abs(decomposed - total)
    identity_tolerance = 1e-10 * max(1., abs(total), abs(decomposed))
    role_means = {role: {cell: math.fsum(values[role + '_' + cell + '_' + str(r)] for r in (0, 1)) / 2
                         for cell in CELL_KEYS} for role, values in (('calibration', calibration), ('target', targets))}
    role_differences = {cell: role_means['target'][cell] - role_means['calibration'][cell] for cell in CELL_KEYS}
    sd = {cell: math.sqrt(value / 3) for cell, value in cell_ss.items()}
    ranges = {cell: max(values) - min(values) for cell, values in observations.items()}
    pooled = math.sqrt(within / 12)
    ratio_values = {'balance_delta': balance, 'position_delta': position, 'interaction_delta': interaction,
                    'pooled_within_sd': pooled}
    for name, values in (('cell_sample_sd', sd), ('cell_range', ranges),
                         ('balance_simple', balance_simple), ('position_simple', position_simple),
                         ('target_minus_calibration', role_differences)):
        ratio_values.update({name + '.' + key: value for key, value in values.items()})
    ratios = {key: abs(value) / TOLERANCE for key, value in ratio_values.items()}
    analysis.require(all(math.isfinite(v) for v in [grand, pooled, *ss.values(), *ratios.values()]),
                     'Non-finite derived descriptive statistic')
    return {'prefixes_per_cell': 4, 'grand_mean': grand, 'cell_means': means,
            'balance_delta': balance, 'position_delta': position, 'interaction_delta': interaction,
            'balance_simple_effects': balance_simple, 'position_simple_effects': position_simple,
            'cell_sample_sd': sd, 'cell_ranges': ranges, 'pooled_within_sd': pooled,
            'calibration_cell_means': role_means['calibration'], 'target_cell_means': role_means['target'],
            'target_minus_calibration_cell_mean': role_differences,
            'sum_squares': ss, 'sum_squares_identity': {'decomposed_total': decomposed,
                'absolute_error': identity_error, 'floating_tolerance': identity_tolerance,
                'holds_within_tolerance': identity_error <= identity_tolerance},
            'absolute_ratios_to_0_10_nat': ratios}


def _nat_metrics(table):
    values = {key: table[key] for key in ('grand_mean', 'balance_delta', 'position_delta',
                                         'interaction_delta', 'pooled_within_sd')}
    for name in ('cell_means', 'balance_simple_effects', 'position_simple_effects', 'cell_sample_sd',
                 'cell_ranges', 'calibration_cell_means', 'target_cell_means', 'target_minus_calibration_cell_mean'):
        values.update({name + '.' + key: value for key, value in table[name].items()})
    return values


def _score(predictions, targets):
    signed = {dtype: {cell: targets[dtype][cell] - predictions[dtype][cell] for cell in analysis.TARGET_CELLS}
              for dtype in analysis.DTYPES}
    error = max(abs(value) for value in signed['float64'].values())
    discrepancy = max(abs(signed['float32'][cell] - signed['float64'][cell]) for cell in analysis.TARGET_CELLS)
    analysis.require(all(math.isfinite(value) for values in signed.values() for value in values.values())
                     and math.isfinite(discrepancy), 'Non-finite derived forecast error')
    supported = discrepancy <= analysis.PRECISION
    return {'signed_errors': signed, 'max_abs_error_float64': error,
            'prediction_error_dtype_discrepancy': discrepancy,
            'numerical_status': 'supported' if supported else 'unresolved',
            'definite_hit': error <= analysis.DEFINITE if supported else None,
            'possible_hit': error <= analysis.POSSIBLE if supported else None}


def family_result(calibration_by_dtype, targets_by_dtype):
    calibration = _grids(calibration_by_dtype, 'calibration')
    targets = _grids(targets_by_dtype, 'target')
    predicted = forecasts(calibration)
    score = _score(predicted, targets)
    b_predicted = {dtype: analysis.forecasts(values)['B_avg'] for dtype, values in calibration.items()}
    b_score = _score(b_predicted, targets)
    tables = {dtype: _table(calibration[dtype], targets[dtype]) for dtype in analysis.DTYPES}
    nat = {dtype: _nat_metrics(table) for dtype, table in tables.items()}
    nat_discrepancy = {key: abs(nat['float32'][key] - nat['float64'][key]) for key in nat['float64']}
    ss_discrepancy = {key: abs(tables['float32']['sum_squares'][key] - tables['float64']['sum_squares'][key])
                      for key in tables['float64']['sum_squares']}
    joint = {}
    for kind in ('definite_hit', 'possible_hit'):
        b, c = b_score[kind], score[kind]
        joint[kind] = ('numerically_unresolved' if b is None or c is None else
                       'both' if b and c else 'B_only' if b else 'C_only' if c else 'neither')
    return {'forecasts': predicted, **score, 'B_avg': b_score, 'B_avg_C_joint': joint,
            'tables': tables, 'dtype_discrepancies': {'nat_metrics': nat_discrepancy,
                'max_nat_metric_difference': max(nat_discrepancy.values()),
                'squared_nat_metrics': ss_discrepancy,
                'max_squared_nat_metric_difference': max(ss_discrepancy.values())},
            'scope': 'Descriptive only; no primary gate, adequacy decision or causal proportion.'}


def _summary(values):
    values = sorted(analysis.number(x) for x in values)
    if not values:
        return {'n': 0, 'median': None, 'q1': None, 'q3': None, 'iqr': None, 'min': None, 'max': None, 'range': None}
    def quantile(q):
        index = (len(values) - 1) * q
        low = int(index)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)
    q1, q3 = quantile(.25), quantile(.75)
    return {'n': len(values), 'median': quantile(.5), 'q1': q1, 'q3': q3,
            'iqr': q3 - q1, 'min': values[0], 'max': values[-1], 'range': values[-1] - values[0]}


def _group(rows):
    counts = {'families': len(rows), 'C_numerically_unresolved': sum(r['numerical_status'] == 'unresolved' for r in rows),
              'C_definite_hits': sum(r['definite_hit'] is True for r in rows),
              'C_possible_hits': sum(r['possible_hit'] is True for r in rows),
              'table_identity_failures': {tag: sum(not r['tables'][tag]['sum_squares_identity']['holds_within_tolerance']
                                                 for r in rows) for tag in analysis.DTYPES}}
    joint = {kind: {state: sum(r['B_avg_C_joint'][kind] == state for r in rows)
                    for state in ('both', 'B_only', 'C_only', 'neither', 'numerically_unresolved')}
             for kind in ('definite_hit', 'possible_hit')}
    metrics = {'C_max_abs_error_nat': [r['max_abs_error_float64'] for r in rows],
               'B_avg_max_abs_error_nat': [r['B_avg']['max_abs_error_float64'] for r in rows],
               'C_prediction_error_dtype_discrepancy_nat': [r['prediction_error_dtype_discrepancy'] for r in rows],
               'max_table_dtype_discrepancy_nat': [r['dtype_discrepancies']['max_nat_metric_difference'] for r in rows]}
    # Distribution summaries use fp64 as a numerical reference, not exact truth.
    if rows:
        for key in _nat_metrics(rows[0]['tables']['float64']):
            metrics[key] = [_nat_metrics(r['tables']['float64'])[key] for r in rows]
        for key in rows[0]['tables']['float64']['absolute_ratios_to_0_10_nat']:
            metrics['absolute_ratio_to_0_10.' + key] = [r['tables']['float64']['absolute_ratios_to_0_10_nat'][key] for r in rows]
        for key in rows[0]['tables']['float64']['sum_squares']:
            metrics['sum_squares.' + key] = [r['tables']['float64']['sum_squares'][key] for r in rows]
    return {'counts': counts, 'B_avg_C_joint_counts': joint,
            'metric_summaries_fp64': {key: _summary(values) for key, values in metrics.items()}}


def cohort_result(rows):
    analysis.require(isinstance(rows, list) and len(rows) == analysis.N,
                     'Exactly256 accepted-family descriptive results are required')
    analysis.require(all(type(row.get('separating')) is bool for row in rows),
                     'Every result needs the frozen separating flag')
    return {'all_families': _group(rows), 'separating_subset': _group([r for r in rows if r['separating']]),
            'quantiles': 'Linear interpolation at (n-1)*q; IQR=Q3-Q1; range=max-min.',
            'scope': 'Descriptive distributions and counts only. No p-values, confidence intervals, adequacy labels or causal shares.'}
