import copy
import json
from pathlib import Path
import unittest

import analysis as a
import planning as p


class PlanningTests(unittest.TestCase):
    def test_independent_scipy_reference_values(self):
        self.assertAlmostEqual(p.paired_power(.15,.05)["favorable_direction_power"], .8798106689755036, places=11)
        self.assertAlmostEqual(p.paired_power(.20,.05)["favorable_direction_power"], .9949187011877709, places=11)
        self.assertAlmostEqual(p.paired_power(.30,.05)["favorable_direction_power"], .9999996879096789, places=11)
        self.assertAlmostEqual(p.paired_power(.15,.05,128)["favorable_direction_power"], .4948418982341076, places=11)

    def test_two_sided_null_error_is_bounded(self):
        report=p.paired_power(.10,.10)
        self.assertLessEqual(report["either_direction_probability"], a.PAIR_ALPHA)
        self.assertAlmostEqual(report["favorable_direction_power"],report["opposite_direction_probability"])
        self.assertEqual(p.paired_power(0,0)["either_direction_probability"],0.)

    def test_cost_and_frozen_table(self):
        r=p.planning_report()
        self.assertEqual(r["cost"]["maximum"],36864)
        self.assertEqual(r["cost"]["calibration_stop"],20480)
        self.assertEqual(r["constants"]["minimum_potentially_separating"],64)
        self.assertEqual(r["constants"]["familywise_alpha_upper_bound"],.05)
        p.assert_table_matches(json.loads(Path(p.__file__).with_name("planning.json").read_text()),r)

    def test_table_portability_only_tolerates_tiny_calculated_float_differences(self):
        frozen=json.loads(Path(p.__file__).with_name("planning.json").read_text())
        tiny=copy.deepcopy(frozen)
        tiny["paired_scenarios"][0]["favorable_direction_power"]+=1e-13
        p.assert_table_matches(tiny,frozen)
        large=copy.deepcopy(frozen)
        large["paired_scenarios"][0]["favorable_direction_power"]+=1e-7
        with self.assertRaises(ValueError):
            p.assert_table_matches(large,frozen)
        altered_rule=copy.deepcopy(frozen)
        altered_rule["constants"]["robust_advantage_nat"]+=1e-13
        with self.assertRaises(ValueError):
            p.assert_table_matches(altered_rule,frozen)
        wrong_type=copy.deepcopy(frozen)
        wrong_type["constants"]["families"]=256.0
        with self.assertRaises(ValueError):
            p.assert_table_matches(wrong_type,frozen)
        extra=copy.deepcopy(frozen)
        extra["undeclared"]=True
        with self.assertRaises(ValueError):
            p.assert_table_matches(extra,frozen)

    def test_invalid_hypothetical_probabilities(self):
        with self.assertRaises(ValueError):
            p.paired_power(.9,.2)
        with self.assertRaises(ValueError):
            p.paired_power(.1,.1,0)


if __name__=="__main__":
    unittest.main()
