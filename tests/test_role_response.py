"""Constructed positive identification and competing failure modes; no model runs."""

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
import role_response as analysis


GAINS = {q: .5 for q in analysis.QUERIES}
ROLE_DIRECTIONS = ((-8., -4., 4.), (-4., -8., -4.))


def design(groups=1):
    ids = [f"group{g}/{q}" for g in range(groups) for q in ("receiver", "observer")]
    diagnostics = {"name": [ids[0]], "position": [ids[0]],
                   "switch": ids[:2], "no_op": ids[:2]}
    structure = {cell_id: {"group": f"group{i//2}", "recipient_query": "giver",
                           "donor_query": ("receiver", "observer")[i % 2]}
                 for i, cell_id in enumerate(ids)}
    return {"expected_cell_ids": ids, "expected_diagnostics": diagnostics,
            "expected_structure": structure}


def world(case_id="unit-0", groups=1, observed_gain=.5, gains=None):
    specification = design(groups)
    rows = []
    for i, cell_id in enumerate(specification["expected_cell_ids"]):
        direction = ROLE_DIRECTIONS[i % 2]
        rows.append({"cell_id": cell_id, "group": f"group{i//2}",
                     "recipient_query": "giver", "donor_query": ("receiver", "observer")[i % 2],
                     "diagnostics": [r for r, ids in specification["expected_diagnostics"].items()
                                     if cell_id in ids],
                     "role_direction": list(direction),
                     "name_direction": list(ROLE_DIRECTIONS[1-i % 2]),
                     "position_direction": list(ROLE_DIRECTIONS[1-i % 2]),
                     "observed": [observed_gain*x for x in direction]})
    precision = {"controls": {"passed": True},
                 "wordings": {w: {"cells": copy.deepcopy(rows), "controls": {"passed": True}}
                              for w in analysis.DEFAULT_WORDINGS}}
    return {"case_id": case_id, "source_note": "synthetic mathematical example",
            "precisions": {p: copy.deepcopy(precision) for p in analysis.PRECISIONS}}


def analyze(records, groups=1, gains=None, **kwargs):
    kwargs.setdefault("bootstrap_draws", 256)
    kwargs.setdefault("evidence_kind", "synthetic")
    return analysis.analyze_records(records, gains or GAINS, **design(groups), **kwargs)


def all_cells(record):
    return [cell for precision in record["precisions"].values()
            for wording in precision["wordings"].values() for cell in wording["cells"]]


class RoleResponseTests(unittest.TestCase):
    def test_frozen_true_role_prediction_can_pass_absolute_and_all_rival_gates(self):
        records = [world(str(i)) for i in range(32)]
        result = analyze(records, bootstrap_draws=20000)
        self.assertEqual(result["status"], "supports_relational_response_within_declared_menu")
        self.assertEqual(result["joint_coverage"]["successes"], 32)
        self.assertGreater(result["joint_coverage"]["interval"][0], .8)
        self.assertEqual(result["contract"]["advantage_family_size"], 16)
        self.assertTrue(result["protocol_matches_frozen_statistics"])
        self.assertFalse(result["model_evidence_supported"])
        self.assertEqual(result["evidence_kind"], "synthetic")
        for wording in analysis.DEFAULT_WORDINGS:
            for rival in analysis.RIVALS:
                self.assertEqual(result["paired_advantages"][wording][rival]["status"], "role_advantage")
        json.dumps(result, allow_nan=False)

    def test_no_op_and_near_zero_gain_cannot_pass_through_low_absolute_error(self):
        for gain in (0., .01):
            records = [world(str(i), observed_gain=0.) for i in range(32)]
            result = analyze(records, gains={q: gain for q in analysis.QUERIES})
            self.assertEqual(result["descriptive_counts"]["adequate"], 32)
            self.assertEqual(result["descriptive_counts"]["informative"], 0)
            self.assertEqual(result["joint_coverage"]["successes"], 0)
            self.assertNotEqual(result["status"], "supports_relational_response_within_declared_menu")

    def test_declared_model_kind_does_not_certify_external_gates(self):
        result = analyze([world(str(i)) for i in range(32)],
                         bootstrap_draws=20000, evidence_kind="model_measurements")
        self.assertTrue(result["statistical_support_given_external_gates"])
        self.assertFalse(result["model_evidence_supported"])
        self.assertEqual(len(result["external_gates_unchecked"]), 6)

    def test_arbitrary_signed_positional_gain_is_not_a_strawman(self):
        records = [world(str(i)) for i in range(32)]
        for record in records:
            for cell in all_cells(record):
                gain = 2. if cell["donor_query"] == "receiver" else -3.
                cell["observed"] = [gain*x for x in cell["position_direction"]]
        result = analyze(records)
        for wording in analysis.DEFAULT_WORDINGS:
            comparison = result["paired_advantages"][wording]["position"]
            self.assertEqual(comparison["status"], "rival_advantage")
            cells = result["records"][0]["precisions"]["float64"]["wordings"][wording]["cells"]
            self.assertEqual([c["oracle_fits"]["position"]["gain"] for c in cells], [2., -3.])
            self.assertTrue(all(c["oracle_fits"]["position"]["distance"] < 1e-14 for c in cells))
        self.assertEqual(result["joint_coverage"]["successes"], 0)

    def test_diagnostic_error_cannot_hide_inside_small_family_average(self):
        record = world(groups=4)
        for cell in all_cells(record):
            if cell["cell_id"] == "group0/receiver":
                cell["observed"] = [x+y for x, y in zip(cell["observed"], (.8, .4, -.4))]
        result = analyze([record], groups=4)
        section = result["records"][0]["precisions"]["float64"]["wordings"]["heldout"]
        self.assertLess(section["losses"]["role"], .25)
        self.assertGreater(section["cells"][0]["role_error"], .25)
        self.assertFalse(section["adequate"])
        self.assertFalse(result["records"][0]["joint_success"])

    def test_relative_win_without_absolute_adequacy_is_reported_separately(self):
        records = [world(str(i), observed_gain=.56) for i in range(32)]
        result = analyze(records)
        self.assertEqual(result["status"], "relative_winner_without_joint_adequacy")
        self.assertEqual(result["descriptive_counts"]["adequate"], 0)
        self.assertEqual(result["joint_coverage"]["status"], "excluded")
        self.assertFalse(result["model_evidence_supported"])

    def test_switch_oracle_uses_best_shared_response_not_role_prediction(self):
        record = world()
        for cell in all_cells(record):
            cell["observed"] = [3., 1., -2.]
        result = analyze([record])
        section = result["records"][0]["precisions"]["float64"]["wordings"]["fit"]
        self.assertEqual(section["losses"]["switch"], 0.)
        self.assertTrue(all(c["oracle_fits"]["switch"]["projection"] == [3., 1., -2.]
                            for c in section["cells"]))

    def test_heldout_wording_failure_is_not_averaged_away(self):
        records = [world(str(i)) for i in range(32)]
        for record in records:
            for precision in record["precisions"].values():
                for cell in precision["wordings"]["heldout"]["cells"]:
                    cell["observed"] = list(cell["position_direction"])
        result = analyze(records)
        self.assertEqual(result["paired_advantages"]["fit"]["position"]["status"], "role_advantage")
        self.assertEqual(result["paired_advantages"]["heldout"]["position"]["status"], "rival_advantage")
        self.assertNotEqual(result["status"], "supports_relational_response_within_declared_menu")

    def test_numerical_failure_keeps_family_and_blocks_all_mean_claims(self):
        records = [world(str(i)) for i in range(32)]
        cell = records[-1]["precisions"]["float32"]["wordings"]["heldout"]["cells"][0]
        cell["observed"] = [v+d for v, d in zip(cell["observed"], (.02, .02, 0))]
        result = analyze(records)
        self.assertEqual(result["n_units"], 32)
        self.assertEqual(result["n_unresolved_units"], 1)
        self.assertEqual(result["joint_coverage"]["successes"], 31)
        for wording in analysis.DEFAULT_WORDINGS:
            for rival in analysis.RIVALS:
                item = result["paired_advantages"][wording][rival]
                self.assertEqual(item["status"], "unresolved")
                self.assertIn("cross_precision_disagreement", item["blocked_reasons"])

    def test_bootstrap_precision_disagreement_and_seed_instability_block(self):
        records = [world(str(i)) for i in range(32)]
        interval = np.ones((2, 2, 2, 4))
        with patch.object(analysis, "_bootstrap", side_effect=[interval, interval+.011]):
            result = analyze(records)
        self.assertIn("bootstrap_monte_carlo_instability",
                      result["paired_advantages"]["fit"]["name"]["blocked_reasons"])
        interval[:, 0] = .099
        interval[:, 1] = .101
        with patch.object(analysis, "_bootstrap", return_value=interval):
            result = analyze(records)
        self.assertEqual(result["n_unresolved_units"], 0)
        self.assertIn("precision_decisions_disagree",
                      result["paired_advantages"]["fit"]["name"]["blocked_reasons"])

    def test_unit_bootstrap_shares_indices_and_has_declared_family_tail(self):
        values = np.arange(8*2*2*4).reshape(8, 2, 2, 4).astype(float)
        observed = analysis._bootstrap(values, 256, 17)
        indices = np.random.default_rng(17).integers(0, 8, size=(256, 8))
        expected = np.quantile(values[indices].mean(axis=1), [.025/32, 1-.025/32], axis=0)
        np.testing.assert_array_equal(observed, expected)

    def test_missing_wording_duplicate_family_and_repeated_cell_rejected(self):
        record = world()
        del record["precisions"]["float32"]["wordings"]["heldout"]
        with self.assertRaisesRegex(ValueError, "wordings"):
            analyze([record])
        with self.assertRaisesRegex(ValueError, "case_id"):
            analyze([world(), world()])
        record = world()
        cells = record["precisions"]["float64"]["wordings"]["fit"]["cells"]
        cells[1]["cell_id"] = cells[0]["cell_id"]
        with self.assertRaisesRegex(ValueError, "cell_id"):
            analyze([record])

    def test_deterministic_reanalysis_allowed_and_preserves_metadata(self):
        records = [world(str(i)) for i in range(2)]
        first, second = analyze(records), analyze(records)
        self.assertEqual(first, second)
        self.assertEqual(first["records"][0]["source_note"], records[0]["source_note"])
        self.assertNotIn("role_prediction", records[0]["precisions"]["float32"]["wordings"]["fit"]["cells"][0])

    def test_invalid_values_margins_controls_and_structural_zero_rows_fail_closed(self):
        for value in (math.nan, math.inf, True, "1"):
            record = world()
            record["precisions"]["float32"]["wordings"]["fit"]["cells"][0]["observed"][0] = value
            with self.assertRaises(ValueError):
                analyze([record])
        record = world()
        record["precisions"]["float32"]["wordings"]["fit"]["cells"][0]["observed"][0] += .001
        with self.assertRaisesRegex(ValueError, "A-B"):
            analyze([record])
        record = world()
        record["precisions"]["float32"]["controls"]["passed"] = False
        with self.assertRaises(analysis.InvalidControlsError):
            analyze([record])
        record = world()
        record["precisions"]["float32"]["wordings"]["fit"]["cells"][0]["donor_query"] = "giver"
        with self.assertRaisesRegex(ValueError, "same-query"):
            analyze([record])

    def test_diagnostic_ids_and_query_group_cannot_be_changed_after_observation(self):
        record = world()
        record["precisions"]["float32"]["wordings"]["fit"]["cells"][0]["diagnostics"].remove("name")
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            analyze([record])
        record = world()
        record["precisions"]["float32"]["wordings"]["fit"]["cells"][0]["group"] = "changed"
        with self.assertRaisesRegex(ValueError, "expected_structure"):
            analyze([record])
        record = world(groups=2)
        specification = design(2)
        specification["expected_diagnostics"]["switch"] = ["group0/receiver", "group1/observer"]
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            analysis.analyze_records([record], GAINS, **specification, bootstrap_draws=32)

    def test_consistent_relabeling_of_every_family_cannot_redefine_frozen_structure(self):
        records = [world(str(i), groups=2) for i in range(2)]
        permutation = {"giver": "receiver", "receiver": "observer", "observer": "giver"}
        for record in records:
            for cell in all_cells(record):
                cell["recipient_query"] = permutation[cell["recipient_query"]]
                cell["donor_query"] = permutation[cell["donor_query"]]
        with self.assertRaisesRegex(ValueError, "expected_structure"):
            analyze(records, groups=2)
        records = [world(str(i), groups=2) for i in range(2)]
        for record in records:
            for cell in all_cells(record):
                cell["group"] = "group1" if cell["group"] == "group0" else "group0"
        with self.assertRaisesRegex(ValueError, "expected_structure"):
            analyze(records, groups=2)

    def test_missing_or_malformed_frozen_structure_rejected(self):
        specification = design()
        specification["expected_structure"].pop(specification["expected_cell_ids"][0])
        with self.assertRaisesRegex(ValueError, "expected_structure"):
            analysis.analyze_records([world()], GAINS, **specification, bootstrap_draws=32)

    def test_full_72_cell_design_uses_absolute_response_and_fixed_diagnostics(self):
        import role_design
        endpoints = {rid: {q: [4. if name == context["binding"][role_design.ROLES.index(q)] else 0.
                              for name in range(3)] for q in role_design.ROLES}
                     for rid, context in role_design.RECIPIENTS.items()}
        rows = role_design.directions(endpoints)
        for row in rows:
            row["observed"] = [.5*x for x in row["role_direction"]]
        raw_p = {"controls": {"passed": True}, "wordings": {
            w: {"controls": {"passed": True}, "cells": rows} for w in role_design.WORDINGS}}
        records = [{"case_id": str(i), "precisions": {p: raw_p for p in analysis.PRECISIONS}}
                   for i in range(32)]
        result = analysis.analyze_records(records, GAINS,
            expected_cell_ids=role_design.expected_cell_ids(), expected_diagnostics=role_design.DIAGNOSTICS,
            expected_structure=role_design.expected_structure(),
            bootstrap_draws=128, evidence_kind="synthetic")
        self.assertEqual(len(result["records"][0]["precisions"]["float64"]["wordings"]["fit"]["cells"]), 72)
        self.assertEqual(result["status"], "supports_relational_response_within_declared_menu")
        self.assertFalse(result["model_evidence_supported"])
        self.assertFalse(result["protocol_matches_frozen_statistics"])


if __name__ == "__main__":
    unittest.main()
