"""Coverage arithmetic, simulation gates, and exact resampling rule checks."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "applications/makelov-2311.17030/src"
sys.path.insert(0, str(SRC))
import donor_factor_analysis as analysis
import donor_factor_planning as planning
from test_donor_factor_analysis import world


class DonorFactorPlanningTests(unittest.TestCase):
    def test_exact_coverage_matches_frozen_table(self):
        expected = {128: (114, .7020), 192: (168, .8960),
                    256: (221, .9765), 384: (327, .9989)}
        for n, (threshold, power) in expected.items():
            observed = planning.exact_coverage_plan(n)
            self.assertEqual(observed["required_successes"], threshold)
            self.assertAlmostEqual(observed["power_at_90_percent_success"], power, places=4)
            lo = planning.clopper_pearson(threshold, n, .025, 2)[0]
            prev = planning.clopper_pearson(threshold - 1, n, .025, 2)[0]
            self.assertGreater(lo, .8)
            self.assertLessEqual(prev, .8)

    def test_bootstrap_matrix_compression_matches_actual_analyzer(self):
        values = np.random.default_rng(5).normal(size=(32, 2, 3))
        actual = analysis.bootstrap_intervals(values, 512, analysis.PRIMARY_SEED)
        weights = planning._bootstrap_weights(32, 512)
        means = (weights @ values.reshape(32, 6)).reshape(512, 2, 3)
        expected = np.quantile(means, [.025 / 6, 1 - .025 / 6], axis=0)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)

    def test_each_contrast_each_sign_and_both_precisions_required(self):
        intervals = np.zeros((2, 100, 2, 3))
        intervals[0, :, 1, 2] = -.11
        counts = planning._counts_from_intervals(intervals)
        np.testing.assert_array_equal(counts, [[100, 100, 0], [100, 100, 100]])
        report = planning._power_report(counts, 100, True)
        self.assertFalse(report["passes"])
        self.assertFalse(report["directions"]["positive"]["J"]["passes"])

    def test_mc_lower_bound_not_point_estimate_controls_bootstrap_gate(self):
        counts = np.full((2, 3), 82)
        self.assertTrue(planning._power_report(counts, 100, False)["passes"])
        self.assertFalse(planning._power_report(counts, 100, True)["passes"])

    def test_zero_variance_selects_192_but_reduced_simulations_are_test_only(self):
        records = [world(lambda i, p: .01*i, str(j)) for j in range(2)]
        result = planning.plan_confirmation(records, gaussian_simulations=16,
                                             bootstrap_simulations=32, bootstrap_draws=32)
        self.assertEqual(result["selected_n"], 192)
        self.assertEqual(result["status"], "TEST_ONLY")
        self.assertFalse(result["confirmation_authorized"])
        self.assertEqual([c["n"] for c in result["candidates"]], [128, 192])
        self.assertIsNone(result["candidates"][0]["bootstrap_checks"])
        json.dumps(result, allow_nan=False)

    def test_mechanical_selection_tries_next_size_after_bootstrap_failure(self):
        records = [world(lambda i, p: 0., str(j)) for j in range(32)]
        calls = []
        def check(n, *args):
            calls.append(n)
            return {"passes": n == 256}
        with patch.object(planning, "_normal_power", return_value={"passes": True}), \
                patch.object(planning, "_bootstrap_power", side_effect=check):
            result = planning.plan_confirmation(records)
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected_n"], 256)
        self.assertEqual(calls, [192, 192, 256, 256])

    def test_stop_if_no_size_passes_and_never_smuggle_normal_only_selection(self):
        records = [world(lambda i, p: 0., str(j)) for j in range(2)]
        with patch.object(planning, "_normal_power", return_value={"passes": True}), \
                patch.object(planning, "_bootstrap_power", return_value={"passes": False}):
            result = planning.plan_confirmation(records)
        self.assertEqual(result["status"], "STOP")
        self.assertIsNone(result["selected_n"])
        self.assertEqual(len(result["candidates"]), 4)

    def test_stop_on_unresolved_or_single_development_unit_without_dropping(self):
        one = planning.plan_confirmation([world(lambda i, p: p)])
        self.assertEqual(one["status"], "STOP")
        self.assertEqual(one["candidates"], [])
        records = [world(lambda i, p: p, str(i)) for i in range(2)]
        records[1]["precisions"]["float32"]["panels"][0]["patched_margins"]["11"] += .1
        result = planning.plan_confirmation(records)
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(result["n_development_units"], 2)

    def test_empirical_rescaling_preserves_inflated_paired_covariance(self):
        records = [world(lambda i, p, j=j: (j - 2) * (i + 2*p), str(j)) for j in range(5)]
        with patch.object(planning, "_normal_power", return_value={"passes": False}):
            result = planning.plan_confirmation(records)
        covariance = np.array(result["inflated_development_covariance"])
        self.assertAlmostEqual(covariance[0, 0], np.var(np.arange(5), ddof=1) * 1.5**2)
        self.assertAlmostEqual(covariance[0, 3], covariance[0, 0])
        self.assertAlmostEqual(covariance[1, 1], 4 * covariance[0, 0])


if __name__ == "__main__":
    unittest.main()
