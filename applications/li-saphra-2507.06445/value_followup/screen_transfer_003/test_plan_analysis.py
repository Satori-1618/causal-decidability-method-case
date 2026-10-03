"""Arithmetic/decision tests only; no model or measured-outcome dependencies."""
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_analysis as p


class PrecisionTests(unittest.TestCase):
    def test_endpoint_intervals_are_not_degenerate(self):
        low0, high0 = p.exact_bounds(0)
        low32, high32 = p.exact_bounds(32)
        self.assertEqual(low0, 0)
        self.assertEqual(high32, 1)
        self.assertAlmostEqual(high0, 1-p.TAIL_ALPHA**(1/32), places=14)
        self.assertAlmostEqual(low32, p.TAIL_ALPHA**(1/32), places=14)
        self.assertGreater(high0, 0)
        self.assertLess(low32, 1)

    def test_all_counts_have_symmetric_monotone_intervals(self):
        intervals = [p.exact_bounds(k) for k in range(33)]
        for k, (lo, hi) in enumerate(intervals):
            self.assertLessEqual(lo, k/32)
            self.assertGreaterEqual(hi, k/32)
            self.assertAlmostEqual(lo, 1-intervals[32-k][1], places=13)
            if k:
                self.assertGreaterEqual(lo, intervals[k-1][0])
                self.assertGreaterEqual(hi, intervals[k-1][1])

    def test_interior_bounds_invert_binomial_tails(self):
        for k in range(1, 32):
            lo, hi = p.exact_bounds(k)
            self.assertAlmostEqual(sum(math.comb(32, j)*lo**j*(1-lo)**(32-j)
                                       for j in range(k, 33)), p.TAIL_ALPHA, places=13)
            self.assertAlmostEqual(sum(math.comb(32, j)*hi**j*(1-hi)**(32-j)
                                       for j in range(k+1)), p.TAIL_ALPHA, places=13)

    def test_perfect_opposition_and_identical_outcomes(self):
        perfect = p.head_result(32, 0)
        self.assertTrue(perfect['substantial_enrichment'])
        self.assertAlmostEqual(perfect['delta_interval'][0], .6490780345637499, places=13)
        self.assertEqual(p.head_result(32, 32)['directional_status'], 'unresolved')
        self.assertEqual(p.head_result(0, 32)['directional_status'], 'negative')

    def test_marginal_decision_counts(self):
        for rejected, minimum in [(0, 23), (1, 24), (2, 26), (3, 27), (4, 28), (5, 29), (8, 31)]:
            self.assertFalse(p.head_result(minimum-1, rejected)['substantial_enrichment'])
            self.assertTrue(p.head_result(minimum, rejected)['substantial_enrichment'])

    def test_direction_does_not_override_primary(self):
        result = p.head_result(20, 0)
        self.assertEqual(result['directional_status'], 'positive')
        self.assertFalse(result['substantial_enrichment'])

    def test_four_of_six_rule_and_unavailable_heads(self):
        self.assertTrue(p.cohort_result([(32, 0)]*6)['primary_success'])
        self.assertTrue(p.cohort_result([(32, 0)]*4+[None]*2)['primary_success'])
        result = p.cohort_result([(32, 0)]*3+[None]*3)
        self.assertFalse(result['primary_success'])
        self.assertEqual(result['unavailable_heads'], 3)
        with self.assertRaises(ValueError):
            p.cohort_result([(32, 0)]*5)

    def test_hypothetical_power_matches_independent_reference_enumeration(self):
        # Reference values were independently computed with SciPy beta quantiles;
        # these tests themselves require only the standard library.
        expected = [(.9, .05, .9553035834196212, .998387393662827),
                    (.8, .1, .38021444266364623, .15298688822418552),
                    (.75, .1, .20843020816323662, .019688592120762398),
                    (.5, .1, .0008431831198496462, 7.571689223737824e-12)]
        for pa, pr, head, cohort in expected:
            result = p.hypothetical_power(pa, pr)
            self.assertAlmostEqual(result['per_head_power'], head, places=13)
            self.assertAlmostEqual(result['at_least_four_if_equal_and_independent'], cohort, places=13)
            lo, hi = result['at_least_four_without_independence_bounds']
            self.assertAlmostEqual(lo, max(0, 2*head-1), places=13)
            self.assertAlmostEqual(hi, min(1, 1.5*head), places=13)
            self.assertLessEqual(lo, cohort)
            self.assertGreaterEqual(hi, cohort)

    def test_probability_endpoints(self):
        self.assertEqual(p.hypothetical_power(1, 0)['per_head_power'], 1)
        self.assertEqual(p.hypothetical_power(0, 1)['per_head_power'], 0)

    def test_invalid_counts_and_rates(self):
        for k in [-1, 33, 2.5, True]:
            with self.assertRaises(ValueError):
                p.head_result(k, 0)
        for probability in [-.1, 1.1, float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                p.hypothetical_power(probability, .1)


class CostTests(unittest.TestCase):
    def test_full_validation_and_random_yield(self):
        result = p.cost_estimates(512, 24, 8)
        self.assertEqual(result['random_selection_yield_estimate'], .5)
        self.assertEqual(result['accepted_minus_random_yield_estimate'], .25)
        self.assertEqual(result['validation_total_cost'], 3072)
        self.assertEqual(result['rejected_arm_measurement_overhead'], 512)
        self.assertEqual(result['fixed_screened_policy_total_cost'], 2560)
        ratios = result['estimated_cost_per_separating_family']
        self.assertAlmostEqual(ratios['fixed_screened_policy']['value'], 2560/24)
        self.assertEqual(ratios['random_selection_policy']['value'], 36)
        self.assertAlmostEqual(ratios['ideal_stream_NOT_SCHEDULED']['value'], 20/.75)

    def test_higher_conditional_yield_does_not_guarantee_saving(self):
        result = p.cost_estimates(512, 32, 0)
        self.assertGreater(result['accepted_yield'], result['random_selection_yield_estimate'])
        self.assertFalse(result['fixed_policy_cheaper_under_point_estimates'])
        ratios = result['estimated_cost_per_separating_family']
        self.assertEqual(ratios['fixed_screened_policy']['value'], 80)
        self.assertEqual(ratios['random_selection_policy']['value'], 36)
        self.assertEqual(ratios['ideal_stream_NOT_SCHEDULED']['value'], 20)

    def test_fixed_batch_cost_increases_with_native_cost(self):
        cheap = p.cost_estimates(512, 32, 0, b=0)
        costly = p.cost_estimates(512, 32, 0, b=2)
        self.assertTrue(cheap['fixed_policy_cheaper_under_point_estimates'])
        self.assertFalse(costly['fixed_policy_cheaper_under_point_estimates'])
        self.assertGreater(costly['fixed_screened_policy_total_cost'], cheap['fixed_screened_policy_total_cost'])

    def test_rejected_yield_contributes_to_random_comparator(self):
        low = p.cost_estimates(256, 24, 0)
        high = p.cost_estimates(256, 24, 16)
        self.assertGreater(high['random_selection_yield_estimate'], low['random_selection_yield_estimate'])
        self.assertEqual(high['fixed_screened_policy_total_cost'], low['fixed_screened_policy_total_cost'])

    def test_shortfalls_stop_instead_of_imputing(self):
        for acceptance_count in [0, 31, 993, 1024]:
            with self.assertRaisesRegex(ValueError, 'insufficient_yield'):
                p.cost_estimates(acceptance_count, 0, 0)
        for acceptance_count in [32, 992]:
            self.assertEqual(p.cost_estimates(acceptance_count, 1, 1)['status'], 'descriptive_point_estimates_only')

    def test_zero_yield_is_json_safe_infinity_not_zero(self):
        result = p.cost_estimates(512, 0, 0)
        for ratio in result['estimated_cost_per_separating_family'].values():
            self.assertEqual(ratio['status'], 'infinite')
            self.assertIsNone(ratio['value'])
        self.assertIsNone(result['fixed_policy_cheaper_under_point_estimates'])
        json.dumps(result, allow_nan=False)

    def test_zero_over_zero_is_undefined(self):
        result = p.cost_estimates(512, 0, 0, b=0, f=0)
        self.assertEqual(result['estimated_cost_per_separating_family']['fixed_screened_policy']['status'], 'undefined')
        self.assertEqual(p.ratio_record(0, 1), {'status': 'finite', 'value': 0})

    def test_invalid_costs_and_counts(self):
        for b in [-1, float('inf'), float('nan'), True]:
            with self.assertRaises(ValueError):
                p.cost_estimates(512, 1, 1, b=b)
        for count in [-1, 1025, 512.0, True]:
            with self.assertRaises(ValueError):
                p.cost_estimates(count, 1, 1)

    def test_frozen_table_and_json_safe_report(self):
        generated = p.planning_report()
        stored = json.loads(Path(p.__file__).with_name('planning.json').read_text())
        p._assert_same(generated, stored)
        json.dumps(generated, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
