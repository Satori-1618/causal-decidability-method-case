import copy
import math
import unittest

import analysis as a


def cal_table(function):
    return {"calibration_"+cell: function(*cell.split("_")) for cell in a.CELLS}


def target_table(function):
    return {"target_"+cell: function(*cell.split("_")) for cell in a.CELLS}


def both(values):
    return {dtype: dict(values) for dtype in a.DTYPES}


class ForecastTests(unittest.TestCase):
    def test_both_candidates_use_all_four_two_prefix_means(self):
        c = cal_table(lambda s, p, r: (0 if s == "neg" else 4)+(0 if p == "20" else 2)+2*int(r))
        f = a.forecasts(c)
        self.assertEqual(f["B_avg"]["target_neg_28_1"], 2.)
        self.assertEqual(f["B_avg"]["target_pos_20_0"], 6.)
        self.assertEqual(f["P_avg"]["target_neg_20_1"], 3.)
        self.assertEqual(f["P_avg"]["target_pos_28_0"], 5.)
        self.assertEqual(f["B_single4"]["target_neg_28_1"], 1.)
        self.assertEqual(f["B_legacy2"]["target_pos_20_1"], 6.)
        self.assertEqual(set(f["B_avg"]), set(a.TARGET_CELLS))

    def test_averaging_comparator_same_cells_no_target_fitting(self):
        c = cal_table(lambda s, p, r: (-1 if s == "neg" else 1) + (-.16 if r == "0" else .16))
        t = target_table(lambda s, p, r: -1 if s == "neg" else 1)
        result = a.family_result(both(c), both(t))
        self.assertTrue(result["definite_hits"]["B_avg"])
        self.assertFalse(result["possible_hits"]["B_single4"])
        self.assertFalse(result["possible_hits"]["P_avg"])
        self.assertFalse(result["extra_prefix_dependence_witness"])

    def test_averaged_separation_is_half_balanced_anchor_gap(self):
        c = cal_table(lambda s, p, r: 0 if s == "neg" else .4)
        result = a.calibration_result(both(c))
        self.assertAlmostEqual(result["forecast_gap"], .2)
        self.assertFalse(result["separating"])
        c = cal_table(lambda s, p, r: 0 if s == "neg" else .406)
        self.assertTrue(a.calibration_result(both(c))["separating"])

    def test_missing_unknown_nonfinite_cells_rejected(self):
        c = cal_table(lambda s, p, r: 0.)
        for bad in ({}, dict(c, unknown=0), dict(c, calibration_neg_20_0=float("nan")),
                    dict(c, calibration_neg_20_0=True)):
            with self.assertRaises(ValueError):
                a.forecasts(bad)

    def test_max_error_does_not_hide_one_bad_target(self):
        c = cal_table(lambda s, p, r: -1 if s == "neg" else 1)
        t = target_table(lambda s, p, r: -1 if s == "neg" else 1)
        t["target_neg_20_1"] += .12
        result = a.family_result(both(c), both(t))
        self.assertAlmostEqual(result["max_error"]["B_avg"], .12)
        self.assertFalse(result["possible_hits"]["B_avg"])
        self.assertFalse(result["extra_prefix_dependence_witness"])

    def test_within_cell_witness_is_stronger_than_bad_calibrated_fit(self):
        c = cal_table(lambda s, p, r: -1 if s == "neg" else 1)
        t = target_table(lambda s, p, r: (-1 if s == "neg" else 1)+(.11 if r == "1" else -.11))
        result = a.family_result(both(c), both(t))
        self.assertTrue(result["extra_prefix_dependence_witness"])
        self.assertTrue(all(result["within_cell_witnesses"].values()))
        # Even the best shared scalar, the midpoint, misses the possible band.
        self.assertGreater(.22/2, a.POSSIBLE)

    def test_dtype_precision_and_separation_disagreement_stop(self):
        c = cal_table(lambda s, p, r: 0 if s == "neg" else .404)
        d = both(c)
        for cell in d["float32"]:
            if "pos" in cell:
                d["float32"][cell] += .0001
        with self.assertRaisesRegex(ValueError, "classification"):
            a.calibration_result(d)
        c = cal_table(lambda s, p, r: -1 if s == "neg" else 1)
        t = both(target_table(lambda s, p, r: -1 if s == "neg" else 1))
        t["float32"]["target_neg_20_0"] += .002
        with self.assertRaisesRegex(ValueError, "precision"):
            a.family_result(both(c), t)

    def test_descriptive_witness_straddle_is_reported_not_a_stop(self):
        c = cal_table(lambda s, p, r: 0 if s == "neg" else 1)
        t = both(target_table(lambda s, p, r: 0 if s == "neg" else 1))
        t["float64"]["target_neg_20_1"] = .202
        t["float32"]["target_neg_20_1"] = .20201
        result = a.family_result(both(c), t)
        self.assertFalse(result["extra_prefix_dependence_witness"])
        self.assertTrue(result["extra_prefix_dependence_possible_witness"])
        self.assertEqual(result["within_cell_numerically_unresolved"], ["neg_20"])

    def test_start_requires_fixed_count_and_128_new_separations(self):
        rows = [{"separating": i < 128} for i in range(256)]
        self.assertTrue(a.start_rule(rows)["start_targets"])
        rows[127]["separating"] = False
        self.assertFalse(a.start_rule(rows)["start_targets"])
        with self.assertRaises(ValueError):
            a.start_rule(rows[:-1])


class StatisticalTests(unittest.TestCase):
    def test_cp_extremes_non_degenerate(self):
        self.assertAlmostEqual(a.exact_bounds(0, 16)[1], 1-.01**(1/16), places=13)
        self.assertEqual(a.exact_bounds(0, 16)[0], 0.)
        self.assertAlmostEqual(a.exact_bounds(16, 16)[0], .01**(1/16), places=13)
        self.assertEqual(a.exact_bounds(16, 16)[1], 1.)

    def test_possible_definite_ambiguity_cannot_help_candidate(self):
        clean = a.candidate_result(123, 123, 128)
        uncertain = a.candidate_result(121, 124, 128)
        self.assertEqual(clean["status"], "adequate")
        self.assertEqual(uncertain["status"], "unresolved")
        self.assertLess(uncertain["interval"][0], clean["interval"][0])
        self.assertGreater(uncertain["interval"][1], clean["interval"][1])

    def test_exact_paired_test_uses_discordants(self):
        self.assertAlmostEqual(a.paired_result(7, 0)["one_sided_exact_mcnemar_p"], 1/128)
        self.assertTrue(a.paired_result(7, 0)["improvement_supported"])
        self.assertFalse(a.paired_result(6, 0)["improvement_supported"])
        self.assertEqual(a.paired_result(0, 0)["one_sided_exact_mcnemar_p"], 1.)
        self.assertFalse(a.paired_result(0, 7)["improvement_supported"])

    def test_cohort_uses_all_families_for_paired_comparison(self):
        c = cal_table(lambda s, p, r: (-1 if s == "neg" else 1) + (-.16 if r == "0" else .16))
        t = target_table(lambda s, p, r: -1 if s == "neg" else 1)
        good = a.family_result(both(c), both(t))
        rows = [copy.deepcopy(good) for _ in range(256)]
        for r in rows[128:]:
            r["separating"] = False
        report = a.cohort_result(rows)
        self.assertEqual(report["candidates_on_separating_families"]["B_avg"]["families"], 128)
        self.assertEqual(report["paired_averaging_on_all_families"]["robust_gains"], 256)
        self.assertEqual(report["future_confirmation_planning_candidates"], ["B_avg"])
        self.assertFalse(report["future_confirmation_permission"])

    def test_point_gate_is_not_population_adequacy(self):
        c = cal_table(lambda s, p, r: -1 if s == "neg" else 1)
        t = target_table(lambda s, p, r: -1 if s == "neg" else 1)
        rows = [a.family_result(both(c), both(t)) for _ in range(256)]
        for row in rows[128:]:
            row["separating"] = False
        for row in rows[116:128]:
            row["definite_hits"]["B_avg"] = False
            row["possible_hits"]["B_avg"] = False
        report = a.cohort_result(rows)
        self.assertEqual(report["candidates_on_separating_families"]["B_avg"]["status"], "unresolved")
        self.assertEqual(report["future_confirmation_planning_candidates"], ["B_avg"])


if __name__ == "__main__":
    unittest.main()
