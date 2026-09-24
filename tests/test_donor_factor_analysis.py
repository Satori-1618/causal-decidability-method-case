"""Synthetic scientific counterexamples for the actual Round-3A analyzer."""

import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "applications/makelov-2311.17030/src"
sys.path.insert(0, str(SRC))
import donor_factor_analysis as analysis


def world(equation, case_id="unit-0", panel_equations=None):
    equations = panel_equations or (equation, equation)
    panels = []
    for index, function in enumerate(equations):
        baseline = float(index * 8)
        panels.append({"recipient_index": index, "baseline_margin": baseline,
                       "source_indices": [0, 1, 2, 3] if index == 0 else [1, 0, 3, 2],
                       "donor_indices": dict(zip(analysis.CELLS, [0, 1, 2, 3] if index == 0 else [1, 0, 3, 2])),
                       "patched_margins": {f"{i}{p}": baseline + function(i, p)
                                           for i in (0, 1) for p in (0, 1)},
                       "alphas": {"00": 0., "01": 1., "10": 2., "11": 3.},
                       "controls": {"passed": True}})
    precision = {"baseline_margins": [0., 8., 1., 2.], "panels": panels,
                 "controls": {"passed": True, "identity_tolerance": 0.}}
    return {"case_id": case_id, "provenance": {"preserved": "all raw fields"},
            "precisions": {p: copy.deepcopy(precision) for p in analysis.PRECISIONS}}


def analyze(records, **kwargs):
    return analysis.analyze_records(records, bootstrap_draws=512, **kwargs)


class DonorFactorAnalysisTests(unittest.TestCase):
    def test_pure_position_identity_mixed_interaction_and_all_small_worlds(self):
        cases = ((lambda i, p: 2*p, (0., 2., 0.), (True, False)),
                 (lambda i, p: 2*i, (2., 0., 0.), (False, True)),
                 (lambda i, p: i+p, (1., 1., 0.), (False, False)),
                 (lambda i, p: 2*i*p, (1., 1., 2.), (False, False)),
                 (lambda i, p: .025*(i+p), (.025, .025, 0.), (True, True)))
        for equation, expected, fits in cases:
            with self.subTest(expected=expected):
                result = analyze([world(equation)])
                unit = result["records"][0]
                self.assertEqual(tuple(unit["profile_successes"].values()), fits)
                for precision in analysis.PRECISIONS:
                    observed = unit["precisions"][precision]["unit_means"]
                    for contrast, value in zip(analysis.CONTRASTS, expected):
                        self.assertAlmostEqual(observed[contrast], value)
                self.assertTrue(unit["resolved"])
                self.assertEqual(result["n_units"], 1)
                self.assertEqual(result["records"][0]["provenance"],
                                 {"preserved": "all raw fields"})

    def test_mirrored_affine_scalar_patches_cancel_mean_without_no_effect(self):
        # Scores [0,1,0,1] and maps [0,1,2,3]/[1,0,3,2] give +/-p.
        records = [world(None, str(i), (lambda i, p: p, lambda i, p: -p)) for i in range(2)]
        for record in records:
            for raw in record["precisions"].values():
                for panel in raw["panels"]:
                    panel["alphas"] = {cell: value - panel["baseline_margin"]
                                       for cell, value in panel["patched_margins"].items()}
        result = analyze(records)
        self.assertTrue(all(v["status"] == "equivalent" for v in result["mean_effects"].values()))
        unit = result["records"][0]
        self.assertTrue(unit["profile_successes"]["position_only"])
        self.assertFalse(unit["profile_successes"]["identity_only"])
        self.assertEqual([p["contrasts"]["P"] for p in unit["precisions"]["float64"]["panels"]], [1., -1.])

    def test_coarse_profile_adequacy_can_coexist_with_relevant_mean_identity(self):
        records = [world(lambda i, p: .20*i, str(j)) for j in range(64)]
        result = analyze(records)
        self.assertEqual(result["profiles"]["position_only"]["status"], "adequate")
        self.assertEqual(result["mean_effects"]["I"]["status"], "positive_relevant")
        self.assertEqual(result["n_units"], 64)

    def test_both_panels_and_both_precisions_are_required(self):
        record = world(lambda i, p: p)
        record["precisions"]["float64"]["panels"][1]["patched_margins"]["10"] += .251
        result = analyze([record])
        self.assertFalse(result["records"][0]["profile_successes"]["position_only"])
        self.assertEqual(result["n_unresolved_units"], 1)

    def test_numerical_failures_stay_in_denominator_and_block_all_mean_claims(self):
        records = [world(lambda i, p: p, str(i)) for i in range(2)]
        records[1]["precisions"]["float64"]["panels"][0]["patched_margins"]["11"] += .02
        result = analyze(records)
        self.assertEqual(result["profiles"]["position_only"]["successes"], 1)
        self.assertEqual(result["profiles"]["position_only"]["n"], 2)
        for contrast in analysis.CONTRASTS:
            self.assertEqual(result["mean_effects"][contrast]["status"], "unresolved")
            self.assertIn("cross_precision_disagreement", result["mean_effects"][contrast]["blocked_reasons"])

    def test_precision_decisions_must_agree_even_below_numeric_budget(self):
        records = [world(lambda i, p: .099*i, str(i)) for i in range(2)]
        for record in records:
            for panel in record["precisions"]["float64"]["panels"]:
                for cell in ("10", "11"):
                    panel["patched_margins"][cell] += .002
        result = analyze(records)
        self.assertEqual(result["n_unresolved_units"], 0)
        effect = result["mean_effects"]["I"]
        self.assertEqual(effect["status"], "unresolved")
        self.assertIn("precision_decisions_disagree", effect["blocked_reasons"])

    def test_boundaries_and_zero_covering_interval(self):
        for interval in ((-.1, .05), (-.05, .1), (.1, .2), (-.2, -.1), (-.2, .2)):
            self.assertEqual(analysis.interval_decision(interval), "unresolved")
        self.assertEqual(analysis.interval_decision((-.09, .09)), "equivalent")
        record = world(lambda i, p: .25*i)
        self.assertTrue(analyze([record])["records"][0]["profile_successes"]["position_only"])

    def test_secondary_seed_mc_shift_or_decision_disagreement_blocks(self):
        records = [world(lambda i, p: i, str(j)) for j in range(2)]
        primary = np.full((2, 2, 3), .2)
        repeat = primary + .011
        with patch.object(analysis, "bootstrap_intervals", side_effect=[primary, repeat]):
            result = analyze(records)
        self.assertTrue(all(v["status"] == "unresolved" for v in result["mean_effects"].values()))
        repeat = np.full((2, 2, 3), .101)
        primary = np.full((2, 2, 3), .099)
        with patch.object(analysis, "bootstrap_intervals", side_effect=[primary, repeat]):
            result = analyze(records)
        self.assertIn("bootstrap_monte_carlo_instability", result["mean_effects"]["I"]["blocked_reasons"])

    def test_whole_unit_bootstrap_uses_identical_draws_across_precisions(self):
        values = np.array([[[i, -i, .5*i], [i, -i, .5*i]] for i in range(8)])
        observed = analysis.bootstrap_intervals(values, 512, 123)
        indices = np.random.default_rng(123).integers(0, 8, size=(512, 8))
        tail = .025 / 6
        expected = np.quantile(values[indices].mean(axis=1), [tail, 1-tail], axis=0)
        np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(observed[:, 0], observed[:, 1])

    def test_technical_and_baseline_failures_raise(self):
        for mutation in (lambda p: p["controls"].update(passed=False),
                         lambda p: p["panels"][0]["controls"].update(passed=False),
                         lambda p: p["panels"][0]["patched_margins"].update({"00": .001}),
                         lambda p: p["panels"][0]["alphas"].update({"00": .001}),
                         lambda p: p["baseline_margins"].__setitem__(0, .001)):
            record = world(lambda i, p: p)
            mutation(record["precisions"]["float32"])
            with self.assertRaises(analysis.InvalidControlsError):
                analyze([record])

    def test_malformed_and_nonfinite_data_fail_closed(self):
        for value in (True, math.nan, math.inf, -math.inf, "1"):
            record = world(lambda i, p: p)
            record["precisions"]["float32"]["panels"][0]["alphas"]["01"] = value
            with self.assertRaises(ValueError):
                analyze([record])
        for records in ([], None, {}, "records", [world(lambda i, p: p)] * 2):
            with self.assertRaises(ValueError):
                analyze(records)
        record = world(lambda i, p: p)
        record["precisions"]["float64"]["panels"][1]["recipient_index"] = 0
        with self.assertRaises(ValueError):
            analyze([record])
        with self.assertRaises(ValueError):
            analyze([world(lambda i, p: p)], secondary_seed=analysis.PRIMARY_SEED)
        record = world(lambda i, p: p)
        record["precisions"]["float64"]["panels"][1]["donor_indices"]["01"] = 1
        with self.assertRaisesRegex(ValueError, "donor_indices"):
            analyze([record])

    def test_finite_input_overflow_rejected_and_summary_json_serializable(self):
        record = world(lambda i, p: p)
        raw = record["precisions"]["float32"]
        raw["baseline_margins"][0] = -1e308
        raw["panels"][0]["baseline_margin"] = -1e308
        raw["panels"][0]["patched_margins"]["00"] = -1e308
        raw["panels"][0]["patched_margins"]["01"] = 1e308
        with self.assertRaisesRegex(ValueError, "finite"):
            analyze([record])
        json.dumps(analyze([world(lambda i, p: p)]), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
