import json
from pathlib import Path
import unittest

import analysis as a
import planning as p


class PlanningTests(unittest.TestCase):
    def test_exact_adequacy_cutoffs_match_intervals(self):
        for n, expected in ((128, 123), (192, 183), (256, 242)):
            cutoff = p.adequacy_cutoffs(n)
            self.assertEqual(cutoff["minimum_definite_hits_for_lower_above_90pct"], expected)
            self.assertGreater(a.exact_bounds(expected, n)[0], .90)
            self.assertLessEqual(a.exact_bounds(expected-1, n)[0], .90)
            k = cutoff["maximum_possible_hits_for_upper_below_90pct"]
            self.assertLess(a.exact_bounds(k, n)[1], .90)
            self.assertGreaterEqual(a.exact_bounds(k+1, n)[1], .90)

    def test_independent_scipy_reference_values(self):
        self.assertAlmostEqual(p.paired_power(.15, .05)["one_sided_exact_paired_test_power"], .8929161483200385, places=11)
        self.assertAlmostEqual(p.paired_power(.20, .05)["one_sided_exact_paired_test_power"], .9958570970310225, places=11)
        self.assertAlmostEqual(p.paired_power(.30, .05)["one_sided_exact_paired_test_power"], .9999997874949547, places=11)
        self.assertAlmostEqual(p.adequacy_power(128, .95)["adequacy_probability"], .3783883473153392, places=11)

    def test_null_power_is_controlled(self):
        self.assertLessEqual(p.paired_power(.10, .10)["one_sided_exact_paired_test_power"], .01)
        self.assertEqual(p.paired_power(0, 0)["one_sided_exact_paired_test_power"], 0.)
        with self.assertRaises(ValueError):
            p.paired_power(.8, .3)

    def test_start_cost_and_error_budget(self):
        report = p.planning_report()
        self.assertEqual(report["cost"]["maximum"], 36864)
        self.assertEqual(report["cost"]["if_calibration_start_fails"], 20480)
        self.assertEqual(report["constants"]["global_error_budget"], .05)
        rows = report["new_forecast_start_probability"]
        self.assertAlmostEqual(next(r for r in rows if r["assumed_new_forecast_separation_rate"] == .6)["probability_at_least_128_of_256"], .9995141063922002, places=11)

    def test_probability_roundoff_does_not_hide_integer_or_material_drift(self):
        p.compare_planning({"p": .5+1e-14, "n": 256}, {"p": .5, "n": 256})
        with self.assertRaises(ValueError):
            p.compare_planning({"p": .51, "n": 256}, {"p": .5, "n": 256})
        with self.assertRaises(ValueError):
            p.compare_planning({"p": .5, "n": 257}, {"p": .5, "n": 256})

    def test_committed_planning_table(self):
        frozen = json.loads(Path(p.__file__).with_name("planning.json").read_text())
        p.compare_planning(p.planning_report(), frozen)


if __name__ == "__main__":
    unittest.main()
