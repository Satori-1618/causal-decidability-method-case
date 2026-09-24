"""Native-only Stage A gate tests on constructed logits; no model inference."""

import copy
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "applications/makelov-2311.17030/src"
sys.path.insert(0, str(SRC))
import role_baseline_analysis as analysis


def world(scale=4., endpoint_logits=None):
    cases, records = [], []
    for index in range(32):
        rows, measurements = [], []
        for wording in analysis.WORDINGS:
            for context_id, context in analysis.CONTEXTS.items():
                for query in analysis.ROLES:
                    correct = context["binding"][analysis.ROLES.index(query)]
                    row = {"row_id": f"{wording}/{context_id}/{query}", "wording": wording,
                           "context_id": context_id, "context": copy.deepcopy(context),
                           "form": context["form"], "query": query, "correct_name_index": correct,
                           "prompt": f"synthetic {wording}/{context_id}/{query}"}
                    rows.append(row)
                    logits = ([scale if i == correct else 0. for i in range(3)]
                              if endpoint_logits is None else list(endpoint_logits[correct]))
                    measurements.append({**copy.deepcopy(row), "name_logits": logits,
                        "name_probability_mass": .8, "full_vocab_argmax_id": 10+correct,
                        "full_vocab_argmax_token": ("Alice", "Bob", "Carol")[correct], "vocab_size": 50257})
        case_id = f"family-{index}"
        cases.append({"case_id": case_id, "names": ["Alice", "Bob", "Carol"],
                      "answer_token_ids": [10, 11, 12], "rows": rows})
        records.append({"case_id": case_id, "provenance_note": "constructed logits, not a model",
                        "precisions": {p: {"rows": copy.deepcopy(measurements), "controls": {"passed": True}}
                                       for p in analysis.PRECISIONS}})
    return cases, records


def run(cases, records, **kwargs):
    kwargs.setdefault("bootstrap_draws", 256)
    return analysis.analyze_baselines(cases, records, **kwargs)


def row(record, precision="float64", row_id="fit/d1/observer"):
    return next(r for r in record["precisions"][precision]["rows"] if r["row_id"] == row_id)


def weaken_geometry(records, n):
    for record in records[:n]:
        for precision in record["precisions"].values():
            for r in precision["rows"]:
                r["name_logits"] = [.1 if i == r["correct_name_index"] else 0. for i in range(3)]


class RoleBaselineAnalysisTests(unittest.TestCase):
    def test_actual_sampler_schema_flows_through_native_analysis_without_a_model(self):
        from role_baseline_sampling import sample_families

        class ToyTokenizer:
            bos_token_id = 0
            name_ids = {"Ada": 10, "Bea": 11, "Cal": 12}

            def encode(self, text, add_special_tokens=False):
                return [self.name_ids.get(word, 100+sum(map(ord, word)))
                        for word in re.findall(r"\w+|[^\w\s]", text)]

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source/"data").mkdir()
            (source/"data/names.json").write_text(json.dumps(["old0", "old1", "old2", "Ada", "Bea", "Cal"]))
            (source/"data/objects.json").write_text(json.dumps(["old", "cup"]))
            cases, _ = sample_families(source, ToyTokenizer(), n=32, tokenizer_fingerprint="stage-a-analysis-fixture")
        records = []
        for case in cases:
            native_rows = []
            for frozen in case["rows"]:
                self.assertEqual(frozen["context"], frozen["context_id"])
                correct = frozen["correct_name_index"]
                native_rows.append({**copy.deepcopy(frozen),
                    "name_logits": [4. if i == correct else 0. for i in range(3)],
                    "name_probability_mass": .9,
                    "full_vocab_argmax_id": case["answer_token_ids"][correct],
                    "full_vocab_argmax_token": case["names"][correct]})
            records.append({"case_id": case["case_id"], "precisions": {
                p: {"rows": copy.deepcopy(native_rows), "controls": {"passed": True}}
                for p in analysis.PRECISIONS}})
        result = run(cases, records)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["potential_geometry"]["successes"], 32)
        self.assertTrue(all(r["full_vocab_argmax_is_correct_name"]
                            for r in result["records"][0]["precisions"]["float64"]["rows"]))
        # String metadata must match exactly, rather than merely being accepted.
        records[0]["precisions"]["float64"]["rows"][0]["context"] = "d1"
        with self.assertRaisesRegex(ValueError, "context metadata"):
            run(cases, records)

    def test_complete_competent_separated_native_world_passes_only_stage_a(self):
        cases, records = world()
        result = run(cases, records)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["potential_geometry"]["successes"], 32)
        self.assertEqual(len(result["competence_groups"]), 18)
        self.assertEqual(result["n_rows_per_precision_per_family"], 48)
        self.assertFalse(result["relational_response_supported"])
        self.assertFalse(result["confirmation_authorized"])
        self.assertEqual(result["records"][0]["provenance_note"], records[0]["provenance_note"])
        self.assertNotIn("pairwise_margins", records[0]["precisions"]["float32"]["rows"][0])
        json.dumps(result, allow_nan=False)

    def test_argmax_ties_fail_even_if_indexed_argmax_would_choose_correct_name(self):
        cases, records = world()
        for record in records[:4]:
            for p in analysis.PRECISIONS:
                target = row(record, p, "fit/d1/receiver")  # Correct name is index zero.
                self.assertEqual(target["correct_name_index"], 0)
                target["name_logits"] = [4., 4., 0.]
        result = run(cases, records)
        self.assertEqual(result["status"], "STOP_COMPETENCE")
        self.assertTrue(result["potential_geometry"]["passes"])
        group = result["competence_groups"]["receiver|fit|received"]
        self.assertEqual(group["precisions"]["float64"]["mean_family_accuracy"], 28/32)
        tied = row(result["records"][0], "float64", "fit/d1/receiver")
        self.assertTrue(tied["three_name_top_tie"])
        self.assertIsNone(tied["three_name_prediction"])
        self.assertFalse(tied["three_name_correct"])

    def test_full_vocabulary_argmax_is_diagnostic_not_the_declared_accuracy_gate(self):
        cases, records = world()
        for record in records:
            for precision in record["precisions"].values():
                for r in precision["rows"]:
                    r["full_vocab_argmax_id"] = 999
                    r["full_vocab_argmax_token"] = "other-token"
                    r["name_probability_mass"] = .01
        result = run(cases, records)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["records"][0]["precisions"]["float64"]["rows"][0]["full_vocab_argmax_is_correct_name"])

    def test_every_query_wording_form_and_both_precisions_are_required(self):
        cases, records = world()
        for record in records[:4]:
            target = row(record, "float64", "heldout/d1/observer")
            target["name_logits"] = [4., 0., 0.]
            # Keep both precisions numerically consistent; another test checks precision failure.
            row(record, "float32", "heldout/d1/observer")["name_logits"] = [4., 0., 0.]
        result = run(cases, records)
        self.assertEqual(result["failed_competence_groups"], ["observer|heldout|received"])
        self.assertEqual(result["n_units"], 32)
        self.assertEqual(len(result["records"]), 32)

    def test_29_geometry_families_pass_but_28_fail_without_replacement(self):
        for failures in (3, 4):
            cases, records = world()
            weaken_geometry(records, failures)
            result = run(cases, records)
            self.assertEqual(result["potential_geometry"]["successes"], 32-failures)
            self.assertEqual(result["potential_geometry"]["minimum_required_families"], 29)
            self.assertEqual(result["status"], "PASS" if failures == 3 else "STOP_GEOMETRY")
            self.assertEqual(result["n_units"], 32)
            self.assertFalse(result["failed_competence_groups"])

    def test_correct_labels_do_not_guarantee_noncollinear_geometry(self):
        cases, records = world(endpoint_logits=((2., 1., -2.), (0., 1., 0.), (-2., 1., 2.)))
        result = run(cases, records)
        self.assertEqual(result["status"], "STOP_GEOMETRY")
        self.assertFalse(result["failed_competence_groups"])
        self.assertEqual(result["potential_geometry"]["successes"], 0)
        geometry = result["records"][0]["precisions"]["float64"]["potential_geometry"]["fit"]
        self.assertLess(geometry["diagnostics"]["position"][0]["gap"], 1e-14)

    def test_gain_one_gate_is_only_potential_not_a_learned_half_gain(self):
        cases, records = world(scale=.8)
        result = run(cases, records)
        self.assertEqual(result["status"], "PASS")
        geometry = result["records"][0]["precisions"]["float64"]["potential_geometry"]["fit"]
        self.assertGreater(geometry["minimum_gap"], .52)
        self.assertLess(geometry["minimum_gap"]*.5, .52)
        self.assertTrue(result["potential_geometry"]["maximum_gain_only"])

    def test_accuracy_bootstrap_resamples_32_family_scores_not_context_rows(self):
        cases, records = world()
        for record in records[:4]:
            for precision in record["precisions"].values():
                for r in precision["rows"]:
                    if r["wording"] == "fit" and r["query"] == "giver" and r["form"] == "gave":
                        r["name_logits"] = [0., 0., 0.]
        result = run(cases, records, seed=123)
        group = result["competence_groups"]["giver|fit|gave"]["precisions"]["float64"]
        self.assertEqual(group["rows_per_family"], 4)
        self.assertEqual(group["n_families"], 32)
        self.assertEqual(group["correct_rows"], 28*4)
        scores = np.array([0.]*4 + [1.]*28)
        indices = np.random.default_rng(123).integers(0, 32, size=(256, 32))
        expected = np.quantile(scores[indices].mean(axis=1), [.025, .975])
        np.testing.assert_array_equal(group["descriptive_95_percent_interval"], expected)

    def test_native_margin_precision_failure_invalidates_without_dropping(self):
        cases, records = world()
        row(records[0], "float32", "fit/d1/giver")["name_logits"][0] += .02
        result = run(cases, records)
        self.assertEqual(result["status"], "INVALIDNUMERICS")
        self.assertEqual(result["numerical_resolution"]["unresolved_case_ids"], ["family-0"])
        self.assertEqual(result["potential_geometry"]["successes"], 31)
        self.assertEqual(len(result["records"]), 32)

    def test_final_endpoint_predictions_checked_even_if_native_margins_are_within_budget(self):
        cases, records = world()
        row(records[0], "float32", "fit/b1_gave/observer")["name_logits"][0] += .006
        row(records[0], "float32", "fit/b1_gave/receiver")["name_logits"][2] += .006
        result = run(cases, records)
        resolution = result["records"][0]["numerical_resolution"]
        native_max = max(v for k, v in resolution["discrepancies_nat"].items() if k.startswith("row."))
        self.assertLess(native_max, .01)
        self.assertGreater(resolution["maximum_discrepancy_nat"], .01)
        self.assertEqual(result["status"], "INVALIDNUMERICS")

    def test_common_native_logit_offset_does_not_create_margin_disagreement(self):
        cases, records = world()
        for r in records[0]["precisions"]["float32"]["rows"]:
            r["name_logits"] = [v+8 for v in r["name_logits"]]
        self.assertEqual(run(cases, records)["status"], "PASS")

    def test_all_stops_reported_together_and_geometry_is_not_a_patch_effect(self):
        cases, records = world(scale=0.)
        result = run(cases, records)
        self.assertEqual(result["failure_reasons"], ["STOP_COMPETENCE", "STOP_GEOMETRY"])
        self.assertEqual(result["potential_geometry"]["successes"], 0)
        self.assertFalse(result["relational_response_supported"])

    def test_false_metadata_even_consistent_across_precisions_is_rejected(self):
        cases, records = world()
        for p in analysis.PRECISIONS:
            target = row(records[0], p)
            target["correct_name_index"] = 0
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            run(cases, records)
        cases, records = world()
        cases[0]["rows"][0]["correct_name_index"] = 2
        with self.assertRaisesRegex(ValueError, "role_design"):
            run(cases, records)

    def test_missing_repeated_rows_duplicate_families_and_missing_precision_fail(self):
        cases, records = world()
        with self.assertRaisesRegex(ValueError, "32"):
            run(cases[:-1], records[:-1])
        altered = copy.deepcopy(records)
        altered[1]["case_id"] = altered[0]["case_id"]
        with self.assertRaisesRegex(ValueError, "case_id"):
            run(cases, altered)
        altered = copy.deepcopy(records)
        altered[0]["precisions"]["float64"]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "48"):
            run(cases, altered)
        altered = copy.deepcopy(records)
        rows = altered[0]["precisions"]["float64"]["rows"]
        rows[1]["row_id"] = rows[0]["row_id"]
        with self.assertRaisesRegex(ValueError, "row_id"):
            run(cases, altered)
        altered = copy.deepcopy(records)
        del altered[0]["precisions"]["float64"]
        with self.assertRaisesRegex(ValueError, "float32 and float64"):
            run(cases, altered)

    def test_nonfinite_probabilities_invalid_argmax_and_failed_controls_are_technical(self):
        for field, value in (("name_probability_mass", math.nan), ("name_probability_mass", 1.01),
                             ("full_vocab_argmax_id", True), ("full_vocab_argmax_id", -1)):
            cases, records = world()
            row(records[0])[field] = value
            with self.assertRaises(ValueError):
                run(cases, records)
        cases, records = world()
        row(records[0])["name_logits"][0] = math.inf
        with self.assertRaises(ValueError):
            run(cases, records)
        cases, records = world()
        records[0]["precisions"]["float64"]["controls"]["passed"] = False
        with self.assertRaises(analysis.InvalidControlsError):
            run(cases, records)

    def test_saved_replay_controls_are_recomputed_and_checked_as_one_control(self):
        cases, records = world()
        precision = records[0]["precisions"]["float64"]
        first = precision["rows"][0]
        controls = {"passed": True, "batch_vs_single_max_name_logit_error": 0.,
                    "batch_vs_single_tolerance": 1e-10, "token_replay_passed": True,
                    "no_padding": True, "replay_row_id": first["row_id"],
                    "replay_name_logits": list(first["name_logits"])}
        precision["controls"] = copy.deepcopy(controls)
        self.assertEqual(run(cases, records)["status"], "PASS")
        corruptions = [
            {"batch_vs_single_max_name_logit_error": 1e-5},
            {"batch_vs_single_max_name_logit_error": -1.},
            {"batch_vs_single_tolerance": -1.},
            {"batch_vs_single_tolerance": math.nan},
            {"token_replay_passed": False}, {"no_padding": False},
            {"replay_row_id": precision["rows"][1]["row_id"]},
            {"replay_name_logits": [v+1e-5 for v in first["name_logits"]]},
            {"replay_name_logits": [math.inf, 0., 0.]},
        ]
        for corruption in corruptions:
            with self.subTest(corruption=corruption):
                precision["controls"] = {**copy.deepcopy(controls), **corruption}
                with self.assertRaises(ValueError):
                    run(cases, records)
        for field in controls.keys() - {"passed"}:
            with self.subTest(missing=field):
                precision["controls"] = copy.deepcopy(controls)
                del precision["controls"][field]
                with self.assertRaises(analysis.InvalidControlsError):
                    run(cases, records)


if __name__ == "__main__":
    unittest.main()
