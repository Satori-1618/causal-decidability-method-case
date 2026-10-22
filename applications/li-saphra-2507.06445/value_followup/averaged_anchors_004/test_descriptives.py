"""Known finite tables and leakage/numerical cases, without model inference."""
import copy
import math
import unittest

import analysis
import descriptives as desc


def inputs(cell_means=None, residuals=(0., 0., 0., 0.)):
    means = {cell: 0. for cell in desc.CELL_KEYS} if cell_means is None else cell_means
    calibration, targets = {}, {}
    for cell in desc.CELL_KEYS:
        for replica in (0, 1):
            calibration[f'calibration_{cell}_{replica}'] = means[cell] + residuals[replica]
            targets[f'target_{cell}_{replica}'] = means[cell] + residuals[replica + 2]
    return ({tag: copy.deepcopy(calibration) for tag in analysis.DTYPES},
            {tag: copy.deepcopy(targets) for tag in analysis.DTYPES})


class DescriptiveTests(unittest.TestCase):
    def test_target_changes_never_change_calibration_forecasts(self):
        cal, targets = inputs({'neg_20': 1., 'neg_28': 2., 'pos_20': 3., 'pos_28': 4.}, (-1., 1., 0., 0.))
        frozen = desc.forecasts(cal)
        before = desc.family_result(cal, targets)
        for tag in analysis.DTYPES:
            targets[tag]['target_neg_20_0'] += 100.
        after = desc.family_result(cal, targets)
        self.assertEqual(frozen, before['forecasts'])
        self.assertEqual(before['forecasts'], after['forecasts'])
        self.assertEqual(after['forecasts']['float64']['target_neg_20_0'], 1.)
        self.assertNotEqual(before['max_abs_error_float64'], after['max_abs_error_float64'])

    def test_pure_balance(self):
        cal, targets = inputs({'neg_20': 1., 'neg_28': 1., 'pos_20': 3., 'pos_28': 3.})
        table = desc.family_result(cal, targets)['tables']['float64']
        self.assertEqual(table['grand_mean'], 2.)
        self.assertEqual(table['balance_delta'], 2.)
        self.assertEqual(table['position_delta'], 0.)
        self.assertEqual(table['interaction_delta'], 0.)
        self.assertEqual(table['balance_simple_effects'], {'20': 2., '28': 2.})
        self.assertEqual(table['sum_squares'], {'balance': 16., 'position': 0., 'interaction': 0., 'within': 0., 'total': 16.})
        self.assertEqual(table['absolute_ratios_to_0_10_nat']['balance_delta'], 20.)

    def test_pure_position(self):
        cal, targets = inputs({'neg_20': 1., 'neg_28': 4., 'pos_20': 1., 'pos_28': 4.})
        result = desc.family_result(cal, targets)
        table = result['tables']['float64']
        self.assertEqual(table['position_delta'], 3.)
        self.assertEqual(table['balance_delta'], 0.)
        self.assertEqual(table['sum_squares']['position'], 36.)
        self.assertEqual(table['position_simple_effects'], {'neg': 3., 'pos': 3.})
        self.assertEqual(result['B_avg_C_joint'], {'definite_hit': 'C_only', 'possible_hit': 'C_only'})

    def test_interaction_exposes_cancelled_main_effects(self):
        cal, targets = inputs({'neg_20': 1., 'neg_28': -1., 'pos_20': -1., 'pos_28': 1.})
        table = desc.family_result(cal, targets)['tables']['float64']
        self.assertEqual(table['balance_delta'], 0.)
        self.assertEqual(table['position_delta'], 0.)
        self.assertEqual(table['interaction_delta'], 4.)
        self.assertEqual(table['balance_simple_effects'], {'20': -2., '28': 2.})
        self.assertEqual(table['sum_squares']['interaction'], 16.)
        self.assertEqual(table['sum_squares']['total'], 16.)

    def test_within_cell_sd_uses_four_prefixes_and_denominator_three(self):
        cal, targets = inputs(residuals=(-3., -1., 1., 3.))
        result = desc.family_result(cal, targets)
        table = result['tables']['float64']
        expected = math.sqrt(20 / 3)
        self.assertEqual(table['sum_squares']['within'], 80.)
        self.assertEqual(table['sum_squares']['total'], 80.)
        self.assertAlmostEqual(table['pooled_within_sd'], expected)
        for cell in desc.CELL_KEYS:
            self.assertAlmostEqual(table['cell_sample_sd'][cell], expected)
            self.assertEqual(table['cell_ranges'][cell], 6.)
            self.assertEqual(table['target_minus_calibration_cell_mean'][cell], 4.)
            self.assertEqual(result['forecasts']['float64']['target_' + cell + '_0'], -2.)
        self.assertEqual(table['grand_mean'], 0.)

    def test_mixed_table_sum_of_squares_identity_in_both_dtypes(self):
        cal, targets = inputs({'neg_20': 10.1, 'neg_28': 11.2, 'pos_20': 15.3, 'pos_28': 13.4}, (-.03, -.01, .01, .03))
        result = desc.family_result(cal, targets)
        for table in result['tables'].values():
            ss = table['sum_squares']
            self.assertAlmostEqual(sum(ss[k] for k in ('balance', 'position', 'interaction', 'within')), ss['total'])
            self.assertTrue(table['sum_squares_identity']['holds_within_tolerance'])
        self.assertEqual(result['dtype_discrepancies']['max_nat_metric_difference'], 0.)

    def test_precision_failure_is_unresolved_descriptive_only(self):
        cal, targets = inputs()
        targets['float32']['target_neg_20_0'] = .002
        result = desc.family_result(cal, targets)
        self.assertEqual(result['numerical_status'], 'unresolved')
        self.assertIsNone(result['definite_hit'])
        self.assertIsNone(result['possible_hit'])
        self.assertEqual(result['B_avg_C_joint']['definite_hit'], 'numerically_unresolved')
        self.assertTrue(result['tables']['float64']['sum_squares_identity']['holds_within_tolerance'])
        with self.assertRaisesRegex(ValueError, 'Prediction-error precision'):
            analysis.family_result(cal, targets)

    def test_definite_and_possible_keep_the_original_tolerance_band(self):
        cal, targets = inputs()
        for tag in analysis.DTYPES:
            targets[tag]['target_neg_20_0'] = .1
        result = desc.family_result(cal, targets)
        self.assertEqual(result['numerical_status'], 'supported')
        self.assertFalse(result['definite_hit'])
        self.assertTrue(result['possible_hit'])
        self.assertEqual(result['B_avg_C_joint']['definite_hit'], 'neither')
        self.assertEqual(result['B_avg_C_joint']['possible_hit'], 'both')

    def test_dtype_discrepancies_separate_nat_and_squared_nat_metrics(self):
        cal, targets = inputs({'neg_20': -1., 'neg_28': -1., 'pos_20': 1., 'pos_28': 1.})
        for values in (cal['float32'], targets['float32']):
            for key in values:
                values[key] *= 1.0001
        result = desc.family_result(cal, targets)
        self.assertAlmostEqual(result['dtype_discrepancies']['nat_metrics']['balance_delta'], .0002)
        self.assertGreater(result['dtype_discrepancies']['squared_nat_metrics']['balance'], 0.)
        self.assertEqual(result['max_abs_error_float64'], 0.)
        self.assertTrue(result['definite_hit'])

    def test_missing_extra_nonfinite_and_boolean_values_are_rejected(self):
        cal, targets = inputs()
        for bad in (math.nan, math.inf, -math.inf, True, '0'):
            edited = copy.deepcopy(cal)
            edited['float64']['calibration_neg_20_0'] = bad
            with self.assertRaises(ValueError):
                desc.forecasts(edited)
        edited = copy.deepcopy(targets)
        del edited['float64']['target_neg_20_0']
        with self.assertRaises(ValueError):
            desc.family_result(cal, edited)
        with self.assertRaises(ValueError):
            desc.forecasts({'float64': cal['float64']})
        edited = copy.deepcopy(cal)
        edited['float32']['target_neg_20_0'] = 0.
        with self.assertRaises(ValueError):
            desc.forecasts(edited)

    def test_cohort_counts_and_quantiles_are_descriptive(self):
        cal, targets = inputs()
        result = desc.family_result(cal, targets)
        rows = [{**copy.deepcopy(result), 'separating': i % 2 == 0} for i in range(256)]
        targets['float32']['target_neg_20_0'] = .002
        rows[0] = {**desc.family_result(cal, targets), 'separating': True}
        summary = desc.cohort_result(rows)
        self.assertEqual(summary['all_families']['counts']['families'], 256)
        self.assertEqual(summary['separating_subset']['counts']['families'], 128)
        self.assertEqual(summary['all_families']['counts']['C_definite_hits'], 255)
        self.assertEqual(summary['all_families']['counts']['C_numerically_unresolved'], 1)
        self.assertEqual(summary['separating_subset']['B_avg_C_joint_counts']['definite_hit']['numerically_unresolved'], 1)
        quartiles = desc._summary([0., 1., 2., 3.])
        self.assertEqual((quartiles['median'], quartiles['q1'], quartiles['q3'], quartiles['iqr'], quartiles['range']),
                         (1.5, .75, 2.25, 1.5, 3.))
        self.assertNotIn('status', summary['all_families'])
        self.assertNotIn('p_value', summary)

    def test_cohort_requires_full_accepted_set_and_separation_flags(self):
        cal, targets = inputs()
        result = desc.family_result(cal, targets)
        with self.assertRaises(ValueError):
            desc.cohort_result([{**result, 'separating': True}] * 255)
        with self.assertRaises(ValueError):
            desc.cohort_result([result] * 256)
        summary = desc.cohort_result([{**result, 'separating': False}] * 256)
        self.assertEqual(summary['separating_subset']['counts']['families'], 0)
        self.assertIsNone(summary['separating_subset']['metric_summaries_fp64']['C_max_abs_error_nat']['median'])


if __name__ == '__main__':
    unittest.main()
