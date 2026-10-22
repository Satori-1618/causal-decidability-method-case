import copy
import unittest

import analysis as a


def table(role, recency=0., suffix=0., interaction=0., offsets=(0., 0.)):
    return {f"{role}_{t}_{e}_{r}": offsets[r]
            + (-1 if t == "t6" else 1)*recency
            + (-1 if e == "alt" else 1)*suffix
            + (-1 if t == "t6" else 1)*(-1 if e == "alt" else 1)*interaction
            for t in a.RECENCIES for e in a.ENDINGS for r in (0,1)}


def both(values):
    return {dtype: dict(values) for dtype in a.DTYPES}


class CenteredPredictionTests(unittest.TestCase):
    def test_shared_offsets_removed_separately_not_fitted_per_candidate(self):
        cal = both(table("calibration", recency=.04, offsets=(2.,7.)))
        target = both(table("target", recency=.04, offsets=(-3.,12.)))
        r = a.family_result(cal, target)
        self.assertLess(r["max_error"]["H_recency"], 1e-14)
        self.assertAlmostEqual(r["max_error"]["H_constant"], .04)
        self.assertGreater(r["raw_max_errors_descriptive"]["H_recency"], 7.)
        self.assertEqual(r["paired_comparisons"][a.pair_id("H_recency","H_constant")]["outcome"], "win")
        self.assertEqual(r["target_stem_means"]["float64"], {"0": -3., "1": 12.})

    def test_suffix_story_gives_opposite_direction(self):
        r = a.family_result(both(table("calibration", suffix=.04)), both(table("target", suffix=.04)))
        self.assertEqual(r["paired_comparisons"][a.pair_id("H_recency","H_suffix")]["outcome"], "loss")
        self.assertEqual(r["max_error"]["H_suffix"], 0.)

    def test_pure_interaction_is_preserved_but_not_a_main_candidate(self):
        cal = both(table("calibration", interaction=.05))
        t = both(table("target", interaction=.05))
        r = a.family_result(cal, t)
        self.assertFalse(r["separating"])
        self.assertEqual(r["max_error"]["H_cell"], 0.)
        self.assertEqual(r["max_error"]["H_recency"], .05)
        self.assertAlmostEqual(r["factorial_contrasts_descriptive"]["target"]["float64"]["0"]["interaction"], .2)
        self.assertFalse(a.start_rule([r]*256)["start_targets"])

    def test_opposite_target_stem_effects_cannot_average_away(self):
        cal = both(table("calibration"))
        t = table("target", recency=.05)
        for key in t:
            if key.endswith("_1"):
                t[key] *= -1
        r = a.family_result(cal, both(t))
        self.assertEqual(r["max_error"]["H_constant"], .05)
        self.assertAlmostEqual(sum(r["centered_targets"]["float64"].values()), 0.)

    def test_bad_grid_and_nonfinite_values_fail_closed(self):
        c = table("calibration")
        for malformed in ({}, dict(c, extra=0.), dict(c, calibration_t6_alt_0=float("nan")),
                          dict(c, calibration_t6_alt_0=True)):
            with self.assertRaises(ValueError):
                a.forecasts(malformed)

    def test_calibration_boundary_straddle_is_ambiguous_not_failure(self):
        cal = {"float64": table("calibration", recency=.0221),
               "float32": table("calibration", recency=.0219)}
        r = a.calibration_result(cal)
        self.assertFalse(r["separating"])
        self.assertEqual(len(r["pair_numerically_unresolved_separation"]), 2)
        self.assertLess(r["forecast_contrast_dtype_error"], .001)

    def test_guarded_win_needs_scientific_not_guard_boundary_dtype_agreement(self):
        cal = both(table("calibration", recency=.03))
        t = {"float64": table("target", recency=.02605),
             "float32": table("target", recency=.02595)}
        r = a.family_result(cal, t)
        pair = r["paired_comparisons"][a.pair_id("H_recency","H_constant")]
        self.assertEqual(pair["outcome"], "win")
        self.assertGreater(pair["error_advantage_by_dtype"]["float64"], .022)
        self.assertLess(pair["error_advantage_by_dtype"]["float32"], .022)
        self.assertGreater(pair["error_advantage_by_dtype"]["float32"], .020)

    def test_main_signed_error_precision_failure_is_not_a_scientific_null(self):
        cal = both(table("calibration", recency=.04))
        t = both(table("target", recency=.04))
        t["float32"]["target_t6_alt_0"] += .01
        with self.assertRaisesRegex(ValueError, "precision"):
            a.family_result(cal,t)

    def test_start_counts_any_main_pair_and_requires64_not128(self):
        active = a.calibration_result(both(table("calibration", recency=.03)))
        neutral = a.calibration_result(both(table("calibration")))
        self.assertTrue(a.start_rule([active]*64+[neutral]*192)["start_targets"])
        self.assertFalse(a.start_rule([active]*63+[neutral]*193)["start_targets"])
        with self.assertRaises(ValueError):
            a.start_rule([active]*64)

    def test_supnorm_triangle_preflight_holds_for_arbitrary_targets(self):
        cal = both(table("calibration", recency=.008, suffix=.006, interaction=.07))
        r = a.family_result(cal, both(table("target", recency=.10, suffix=-.25, interaction=.40)))
        self.assertFalse(r["separating"])
        for first, second in a.PAIRS:
            key = a.pair_id(first,second)
            observed = abs(r["paired_comparisons"][key]["error_advantage_by_dtype"]["float64"])
            self.assertLessEqual(observed, r["pair_forecast_gaps_by_dtype"]["float64"][key]+1e-14)
            self.assertEqual(r["paired_comparisons"][key]["outcome"], "neutral")


class ComparisonTests(unittest.TestCase):
    def test_two_sided_exact_test_and_neutral_denominator(self):
        r = a.pair_result(7,0,"R","S")
        self.assertAlmostEqual(r["two_sided_exact_sign_p"], 1/64)
        self.assertEqual(r["direction_supported"], "R")
        self.assertEqual(r["neutral"], 249)
        self.assertIsNone(a.pair_result(6,0,"R","S")["direction_supported"])
        self.assertEqual(a.pair_result(0,7,"R","S")["direction_supported"], "S")
        self.assertEqual(a.pair_result(0,0,"R","S")["two_sided_exact_sign_p"], 1.)

    def test_all256_families_remain_in_test_even_if64_can_separate(self):
        active = a.family_result(both(table("calibration",recency=.04)), both(table("target",recency=.04)))
        neutral = a.family_result(both(table("calibration")),both(table("target")))
        report = a.cohort_result([active]*64+[neutral]*192)
        key=a.pair_id("H_recency","H_constant")
        self.assertEqual(report["pair_results"][key]["families"],256)
        self.assertEqual(report["pair_results"][key]["robust_wins"],64)
        self.assertEqual(report["pair_results"][key]["neutral"],192)
        self.assertFalse(report["absolute_adequacy_claim"])
        self.assertFalse(report["unique_mechanism_identification"])
        self.assertFalse(report["automatic_continuation_authorized"])

    def test_statistic_does_not_imply_mean_loss_superiority(self):
        # Many small wins and a few enormous losses can pass the frequency test.
        r=a.pair_result(60,10,"R","S")
        self.assertEqual(r["direction_supported"],"R")
        self.assertLess(60*.03-10*1.,0)
        self.assertIn("not mean-loss",r["interpretation"])


if __name__=="__main__":
    unittest.main()
