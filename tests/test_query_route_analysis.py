"""Synthetic response contracts and independently checked binomial arithmetic."""

import copy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "applications/makelov-2311.17030/src/query_route_analysis.py"
SPEC = importlib.util.spec_from_file_location("query_route_analysis", MODULE)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def world(equation, pair_id="draw-0", scales=(2.0, -3.0), offsets=(1.0, 7.0)):
    """CONSTRUCTED cells from Y=f(X,M), with natural mediator M=X."""
    interventions = {"A": (0, 0), "B": (1, 1), "C": (0, 0),
                     "D": (1, 0), "E": (0, 1), "F": (1, 1), "G": (0, 0)}
    cells = {cell: [offset + scale * equation(*setting)
                    for offset, scale in zip(offsets, scales)]
             for cell, setting in interventions.items()}
    precision = {"cells": cells,
                 "controls": {"passed": True, "identity_tolerance": 0.0}}
    return {"pair_id": pair_id, "case_id": "content-may-repeat",
            "precisions": {p: copy.deepcopy(precision) for p in analysis.PRECISIONS}}


class QueryRouteAnalysisTests(unittest.TestCase):
    def test_reverse_transfer_separates_the_original_countermodels(self):
        route = world(lambda x, m: m)
        joint = world(lambda x, m: x * m)
        for p in analysis.PRECISIONS:
            for cell in "ABCD":
                self.assertEqual(route["precisions"][p]["cells"][cell],
                                 joint["precisions"][p]["cells"][cell])
            self.assertNotEqual(route["precisions"][p]["cells"]["E"],
                                joint["precisions"][p]["cells"]["E"])
        for record, expected in ((route, "transfer"), (joint, "joint_dependence"),
                                 (world(lambda x, m: x), "preservation")):
            with self.subTest(expected=expected):
                result = analysis.analyze_records([record])
                successes = result["records"][0]["profile_successes"]
                self.assertEqual([p for p, fit in successes.items() if fit], [expected])
                self.assertEqual(result["outcome"], "unresolved")  # one case is not coverage

    def test_intermediate_and_redundant_cases_fit_none(self):
        for equation in (lambda x, m: .5 * x + .5 * m,
                         lambda x, m: x + m - x * m):
            with self.subTest(equation=equation):
                result = analysis.analyze_records([world(equation)])
                record = result["records"][0]
                self.assertTrue(record["resolved"])
                self.assertFalse(any(record["profile_successes"].values()))

    def test_cancelling_paths_do_not_identify_a_unique_graph(self):
        direct = analysis.analyze_records([world(lambda x, m: m)])
        extra_paths = analysis.analyze_records([world(lambda x, m: m + 2*x - 2*x)])
        self.assertEqual(direct, extra_paths)
        self.assertIn("not unique graphs", " ".join(direct["assumptions_and_scope"]))

    def test_nonadditive_joint_effect_is_reported_not_divided_into_a_share(self):
        result = analysis.analyze_records([world(lambda x, m: x*m)])
        for p in analysis.PRECISIONS:
            for c in result["records"][0]["precisions"][p]["contrasts"]:
                self.assertEqual(c["interaction_T_minus_R_minus_S"], c["T"])
        self.assertNotIn("mechanism_share", json.dumps(result))

    def test_common_cell_offset_cancels_without_breaking_identities(self):
        original = world(lambda x, m: m)
        shifted = copy.deepcopy(original)
        for precision in shifted["precisions"].values():
            precision["cells"] = {c: [v+32 for v in values]
                                   for c, values in precision["cells"].items()}
        a, b = (analysis.analyze_records([r]) for r in (original, shifted))
        self.assertEqual(a["profiles"], b["profiles"])
        for p in analysis.PRECISIONS:
            self.assertEqual(a["records"][0]["precisions"][p]["contrasts"],
                             b["records"][0]["precisions"][p]["contrasts"])

    def test_both_reciprocal_directions_are_required(self):
        mixed = world(lambda x, m: m)
        preserved = world(lambda x, m: x)
        for p in analysis.PRECISIONS:
            for cell in analysis.CELLS:
                mixed["precisions"][p]["cells"][cell][1] = (
                    preserved["precisions"][p]["cells"][cell][1])
        result = analysis.analyze_records([mixed])
        self.assertTrue(result["records"][0]["resolved"])
        self.assertFalse(any(result["records"][0]["profile_successes"].values()))

    def test_both_precisions_must_fit_even_when_resolution_passes(self):
        record = world(lambda x, m: m, scales=(4, 4), offsets=(0, 0))
        record["precisions"]["float32"]["cells"]["D"] = [.99, .99]
        record["precisions"]["float64"]["cells"]["D"] = [1.01, 1.01]
        result = analysis.analyze_records([record])["records"][0]
        self.assertTrue(result["resolved"])
        self.assertFalse(result["profile_successes"]["transfer"])
        profiles = result["directions"][0]["profiles_by_precision"]
        self.assertTrue(profiles["float32"]["transfer"]["fits"])
        self.assertFalse(profiles["float64"]["transfer"]["fits"])

    def test_relative_scientific_boundary_is_inclusive(self):
        record = world(lambda x, m: m, scales=(4, 4), offsets=(0, 0))
        for precision in record["precisions"].values():
            precision["cells"]["D"] = [1, 1]
            precision["cells"]["E"] = [3, 3]
        self.assertTrue(analysis.analyze_records([record])["records"][0]
                        ["profile_successes"]["transfer"])
        for precision in record["precisions"].values():
            precision["cells"]["D"][0] = 1.00001
        self.assertFalse(analysis.analyze_records([record])["records"][0]
                         ["profile_successes"]["transfer"])

    def test_numerical_budget_is_inclusive_and_checks_all_contrasts(self):
        for changed_cell in ("B", "D", "E"):
            with self.subTest(changed_cell=changed_cell):
                record = world(lambda x, m: m, scales=(40, 40), offsets=(0, 0))
                record["precisions"]["float64"]["cells"][changed_cell][0] += 1
                if changed_cell == "B":  # preserve technical self-insertion
                    record["precisions"]["float64"]["cells"]["F"][0] += 1
                result = analysis.analyze_records([record])
                self.assertTrue(result["records"][0]["resolved"])
                record["precisions"]["float64"]["cells"][changed_cell][0] += .01
                if changed_cell == "B":
                    record["precisions"]["float64"]["cells"]["F"][0] += .01
                result = analysis.analyze_records([record])
                self.assertFalse(result["records"][0]["resolved"])
                self.assertEqual(result["direction_resolution_failure_counts"],
                                 {"cross_precision_disagreement": 1})

    def test_zero_and_opposite_sign_anchors_fail_and_stay_in_denominator(self):
        good = world(lambda x, m: m, "good")
        zero = world(lambda x, m: 0, "zero")
        flipped = world(lambda x, m: m, "flipped", offsets=(0, 0))
        for c in analysis.CELLS:
            flipped["precisions"]["float64"]["cells"][c] = [
                -v for v in flipped["precisions"]["float64"]["cells"][c]]
        result = analysis.analyze_records([good, zero, flipped])
        self.assertEqual(result["n_base_pairs"], 3)
        self.assertEqual(result["n_resolved_pairs"], 1)
        self.assertEqual(result["n_unresolved_pairs"], 2)
        self.assertEqual(result["profiles"]["transfer"]["fraction"], 1/3)
        self.assertEqual(result["direction_resolution_failure_counts"],
                         {"zero_anchor": 2, "anchor_sign_disagreement": 2})
        self.assertFalse(any(result["records"][1]["profile_successes"].values()))

    def test_no_undeclared_absolute_effect_minimum(self):
        tiny = world(lambda x, m: m, scales=(2.0**-900, -(2.0**-900)), offsets=(0, 0))
        result = analysis.analyze_records([tiny])
        self.assertTrue(result["records"][0]["resolved"])
        self.assertTrue(result["records"][0]["profile_successes"]["transfer"])

    def test_technical_failures_block_run_instead_of_counting_a_failure(self):
        for cell, field in (("C", None), ("F", None), ("G", None), (None, "passed")):
            with self.subTest(cell=cell, field=field):
                broken = world(lambda x, m: m, "broken")
                p = broken["precisions"]["float64"]
                if cell:
                    p["cells"][cell][1] += .01
                else:
                    p["controls"][field] = False
                with self.assertRaises(analysis.InvalidControlsError):
                    analysis.analyze_records([world(lambda x, m: m), broken])

    def test_recorded_identity_tolerance_is_checked_at_its_boundary(self):
        record = world(lambda x, m: m, offsets=(0, 0))
        for p in record["precisions"].values():
            p["controls"]["identity_tolerance"] = .125
            p["cells"]["G"][0] = .125
        analysis.analyze_records([record])
        record["precisions"]["float32"]["cells"]["G"][0] += .001
        with self.assertRaises(analysis.InvalidControlsError):
            analysis.analyze_records([record])

    def test_duplicate_content_allowed_but_duplicate_draw_id_refused(self):
        records = [world(lambda x, m: m, f"draw-{i}") for i in range(2)]
        self.assertEqual(analysis.analyze_records(records)["n_base_pairs"], 2)
        records[1]["pair_id"] = records[0]["pair_id"]
        with self.assertRaises(ValueError):
            analysis.analyze_records(records)

    def test_invalid_schema_and_nonfinite_values_fail_closed(self):
        records = []
        for value in (True, math.nan, math.inf, -math.inf, "1"):
            r = world(lambda x, m: m)
            r["precisions"]["float32"]["cells"]["E"][0] = value
            records.append(r)
        missing = world(lambda x, m: m)
        del missing["precisions"]["float64"]["cells"]["G"]
        records.append(missing)
        short = world(lambda x, m: m)
        short["precisions"]["float32"]["cells"]["A"] = [1]
        records.append(short)
        for r in records:
            with self.subTest(record=r), self.assertRaises(ValueError):
                analysis.analyze_records([r])
        for kwargs in ({"kappa": .5}, {"kappa": -.1}, {"coverage": 1},
                       {"alpha": 0}, {"numerical_fraction": -.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analysis.analyze_records([world(lambda x, m: m)], **kwargs)
        for value in ([], None, "records", {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                analysis.analyze_records(value)

    def test_finite_inputs_with_overflowing_contrasts_are_refused(self):
        record = world(lambda x, m: m)
        for precision in record["precisions"].values():
            for c in ("A", "C", "G"):
                precision["cells"][c] = [-1e308, -1e308]
            for c in ("B", "F"):
                precision["cells"][c] = [1e308, 1e308]
        with self.assertRaisesRegex(ValueError, "finite"):
            analysis.analyze_records([record])

    def test_192_case_decision_requires_167_complete_successes(self):
        for count in (166, 167):
            with self.subTest(count=count):
                records = [world(lambda x, m: m if i < count else .5*(x+m), str(i))
                           for i in range(192)]
                result = analysis.analyze_records(records)
                self.assertEqual(result["profiles"]["transfer"]["successes"], count)
                expected = "adequate" if count == 167 else "unresolved"
                self.assertEqual(result["profiles"]["transfer"]["status"], expected)
                self.assertEqual(result["n_base_pairs"], 192)

    def test_all_profiles_can_fail_coverage_without_no_causality_claim(self):
        records = [world(lambda x, m: .5*(x+m), str(i)) for i in range(64)]
        result = analysis.analyze_records(records)
        self.assertEqual(result["outcome"], "none_meets_coverage")
        self.assertTrue(all(v["status"] == "excluded" for v in result["profiles"].values()))
        self.assertTrue(all(v["interval"][1] > 0 for v in result["profiles"].values()))
        json.dumps(result, allow_nan=False)


class ExactBinomialTests(unittest.TestCase):
    def test_tails_match_independent_rational_event_summation(self):
        probability = Fraction(1, 4)
        for n in (1, 3, 8, 19):
            for k in range(n+1):
                for side in ("lower", "upper"):
                    with self.subTest(n=n, k=k, side=side):
                        indices = range(k+1) if side == "lower" else range(k, n+1)
                        exact = sum(math.comb(n, j) * probability**j
                                    * (1-probability)**(n-j) for j in indices)
                        self.assertAlmostEqual(analysis.binomial_tail(k, n, .25, side),
                                               float(exact), places=13)

    def test_cp_matches_independently_computed_scipy_reference(self):
        # scipy.stats.beta.ppf(.05/6, 167, 26), computed outside this stdlib test.
        lower, upper = analysis.clopper_pearson(167, 192)
        self.assertAlmostEqual(lower, .800918427554962, places=12)
        self.assertLess(lower, 167/192)
        self.assertGreater(upper, 167/192)
        lower166, _ = analysis.clopper_pearson(166, 192)
        self.assertLessEqual(lower166, .8)
        self.assertAlmostEqual(analysis.binomial_tail(167, 192, .9, "upper"),
                               .9306992744779211, places=12)
        self.assertAlmostEqual(analysis.binomial_tail(167, 192, .8, "upper"),
                               .007667458097390943, places=12)

    def test_zero_and_full_success_boundaries_are_not_degenerate(self):
        tail = .05/6
        for n in (1, 16, 192):
            with self.subTest(n=n):
                lo, hi = analysis.clopper_pearson(0, n)
                self.assertEqual(lo, 0)
                self.assertAlmostEqual(hi, 1-tail**(1/n))
                lo, hi = analysis.clopper_pearson(n, n)
                self.assertAlmostEqual(lo, tail**(1/n))
                self.assertEqual(hi, 1)

    def test_cp_is_symmetric_and_monotone_across_success_counts(self):
        previous = (0, 0)
        for k in range(18):
            bounds = analysis.clopper_pearson(k, 17)
            complement = analysis.clopper_pearson(17-k, 17)
            self.assertAlmostEqual(bounds[0], 1-complement[1], places=13)
            self.assertAlmostEqual(bounds[1], 1-complement[0], places=13)
            self.assertGreaterEqual(bounds[0], previous[0])
            self.assertGreaterEqual(bounds[1], previous[1])
            previous = bounds

    def test_invalid_counts_and_parameters_are_refused(self):
        for args in ((1, 0), (3, 2), (True, 2), (1, 2, 0), (1, 2, .05, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                analysis.clopper_pearson(*args)


if __name__ == "__main__":
    unittest.main()
