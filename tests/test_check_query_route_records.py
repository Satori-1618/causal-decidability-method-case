"""Constructed artifact bundles test records-only verification, never model evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "applications/makelov-2311.17030/scripts/check_query_route_records.py"
SPEC = importlib.util.spec_from_file_location("check_query_route_records", CHECKER)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def write_lines(path, values):
    path.write_text("".join(json.dumps(value, allow_nan=False)+"\n" for value in values))


def measured(precision):
    margins = {"A": [1., 4.], "B": [3., 1.], "C": [1., 4.],
               "D": [1., 4.], "E": [3., 1.], "F": [3., 1.], "G": [1., 4.]}
    logits = {c: [[v, 0.] for v in values] for c, values in margins.items()}
    tolerance = 64 * 2.0**(-23 if precision == "float32" else -52) * 4
    events = {"B": [checker.MLP], "C": list(checker.QUERIES),
              "D": [checker.MLP, *checker.QUERIES], "E": list(checker.QUERIES),
              "F": [checker.MLP, *checker.QUERIES], "G": [checker.MLP]}
    sources = {q: {site: letter*64 for site in checker.QUERIES}
               for q, letter in (("q0", "a"), ("q1", "b"))}
    audits = {}
    for arm, sites in events.items():
        audits[arm] = {}
        for site in sites:
            if site == checker.MLP:
                audit = {"calls": 1, "passed": True, "other_positions_unchanged": True,
                         "insertion_error_max_per_item": [0., 0.],
                         "rounding_budget_per_item": [tolerance, tolerance],
                         "actual_delta_l2_per_item": [0., 0.] if arm == "G" else [1., 1.]}
            else:
                source = "q0" if arm in ("C", "D") else "q1"
                audit = {"calls": 1, "passed": True, "position": 2,
                         "heads": checker.QUERIES[site], "other_slices_unchanged": True,
                         "inserted_equals_cast_source": True,
                         "source_sha256": sources[source][site],
                         "inserted_sha256": sources[source][site],
                         "insertion_error_per_item": [0., 0.],
                         "dtype_budget_per_item": [tolerance, tolerance],
                         "actual_change_l2_per_item": [0., 0.],
                         "actual_change_l2_per_item_head": [[0.]*len(checker.QUERIES[site])]*2}
            audits[arm][site] = audit
    return {"cells": margins, "answer_logits": logits,
            "controls": {"passed": True, "identity_tolerance": 2*tolerance,
                         "answer_logit_identity_tolerance": tolerance,
                         "identity_errors": {name: 0. for name in ("C_vs_A", "F_vs_B", "G_vs_A")},
                         "identity_exact": {name: True for name in ("C_vs_A", "F_vs_B", "G_vs_A")},
                         "fidelity": audits, "hook_events": events},
            "query_source_hashes": sources,
            "natural_query_change_l2": {site: [[1.]*len(heads)]*2 for site, heads in checker.QUERIES.items()},
            "model_forward_calls": 7}


class QueryRouteRecordCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        self.application = self.repo / "applications/makelov-2311.17030"
        self.results = self.application / "results/constructed"
        self.results.mkdir(parents=True)
        self.cases = [{"case_id": f"draw-{i}", "unique_draw_id": f"draw-{i}", "draw_index": i,
                       "patterns": ["ABB", "BAB"], "prompts": ["first", "second"],
                       "token_ids": [[0, 1, 2], [0, 2, 1]], "position": 2} for i in range(2)]
        self.records = [{"pair_id": c["case_id"], "precisions": {
            p: measured(p) for p in checker.PRECISIONS}} for c in self.cases]
        code = {}
        for name in checker.CODE_PATHS:
            path = self.application / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Constructed code hash fixture, not executable experiment code.\n"+name)
            code[str(path.relative_to(self.repo))] = checker.sha256(path)
        save(self.results/"cases.json", self.cases)
        definition = {"excluded_prompts": ["old excluded prompt"]}
        self.manifest = {
            "n_base_pairs": 2, "stage": "development", "code_files_sha256": code,
            "cases_sha256": checker.sha256(self.results/"cases.json"), "directions_sha256": "d"*64,
            "source": {"files": {"missing_original.joblib": "e"*64}},
            "model_snapshot_revision": "constructed-not-a-real-model",
            "model_files_sha256": {"missing_model.safetensors": "f"*64},
            "scientific_contract": {"kappa": .25, "numerical_fraction": .025,
                                    "coverage": .8, "alpha": .05, "profiles": 3},
            "query_heads": checker.QUERIES, "mlp_site": checker.MLP,
            "precisions": list(checker.PRECISIONS),
            "sampling": {"n_base_pair_draws": 2, "draw_cases_sha256": checker.canonical_hash(self.cases),
                         "population_definition_sha256": checker.canonical_hash(definition),
                         "sampling_definition": definition}}
        save(self.results/"manifest.json", self.manifest)
        self.manifest_hash = checker.sha256(self.results/"manifest.json")
        save(self.results/"RUN_STARTED.json", {"manifest_sha256": self.manifest_hash})
        self.write_measurements()
        self.summary = checker.analyze_records(self.records)
        self.summary.update(manifest_sha256=self.manifest_hash,
                            records_sha256=checker.sha256(self.results/"records.jsonl"),
                            model_forward_calls=28, stage="development", confirmation=False)
        save(self.results/"summary.json", self.summary)
        self.rehash()

    def tearDown(self):
        self.temporary.cleanup()

    def rehash(self):
        save(self.results/"artifact_hashes.json", {p.name: checker.sha256(p)
             for p in self.results.iterdir() if p.name != "artifact_hashes.json"})

    def write_measurements(self):
        write_lines(self.results/"records.jsonl", self.records)
        for precision in checker.PRECISIONS:
            write_lines(self.results/f"measurements_{precision}.jsonl",
                        [{"pair_id": r["pair_id"], **r["precisions"][precision]} for r in self.records])

    def update_records_hash(self):
        self.summary["records_sha256"] = checker.sha256(self.results/"records.jsonl")
        save(self.results/"summary.json", self.summary)
        self.rehash()

    def verify(self):
        return checker.verify(self.results, repository_root=self.repo,
                              model_snapshot=self.repo/"absent-model")

    def test_complete_constructed_bundle_passes_without_original_model(self):
        result = self.verify()
        self.assertTrue(result["verified"])
        self.assertEqual(result["independent_profile_counts"],
                         {"transfer": 2, "joint_dependence": 0, "preservation": 0})
        self.assertIn("directions.npz", result["optional_original_inputs_unavailable"])
        self.assertIn("model:missing_model.safetensors", result["optional_original_inputs_unavailable"])
        self.assertEqual(result["mode"], "records_only")

    def test_raw_cell_mutation_rejected_even_with_updated_hashes(self):
        self.records[0]["precisions"]["float32"]["cells"]["D"][0] += .2
        self.write_measurements()
        self.update_records_hash()
        with self.assertRaisesRegex(checker.VerificationError, "margin disagrees with logits"):
            self.verify()

    def test_summary_mutation_rejected_even_with_updated_artifact_hash(self):
        self.summary["profiles"]["transfer"]["successes"] = 1
        save(self.results/"summary.json", self.summary)
        self.rehash()
        with self.assertRaisesRegex(checker.VerificationError, "independent count"):
            self.verify()

    def test_cp_reproduction_allows_only_tiny_cross_runtime_roundoff(self):
        self.summary["profiles"]["transfer"]["interval"][0] += 1e-15
        save(self.results/"summary.json", self.summary)
        self.rehash()
        checked = self.verify()
        self.assertGreater(checked["cp_interval_reproduction_max_abs_error"], 0)
        self.assertEqual(checked["cp_interval_reproduction_tolerance"], 1e-12)
        self.summary["profiles"]["transfer"]["interval"][0] += 1e-8
        save(self.results/"summary.json", self.summary)
        self.rehash()
        with self.assertRaisesRegex(checker.VerificationError, "CP interval differs"):
            self.verify()

    def test_summary_status_must_match_exactly(self):
        self.summary["profiles"]["transfer"]["status"] = "adequate"
        save(self.results/"summary.json", self.summary)
        self.rehash()
        with self.assertRaisesRegex(checker.VerificationError, "summary profile"):
            self.verify()

    def test_missing_record_is_not_silently_excluded(self):
        self.records.pop()
        self.write_measurements()
        self.update_records_hash()
        with self.assertRaisesRegex(checker.VerificationError, "omit/reorder/duplicate"):
            self.verify()

    def test_reordered_measurements_are_rejected(self):
        path = self.results/"measurements_float64.jsonl"
        write_lines(path, list(reversed(checker.jsonl(path))))
        self.rehash()
        with self.assertRaisesRegex(checker.VerificationError, "measurements differ"):
            self.verify()

    def test_wrong_cached_source_rejected_despite_passed_flag(self):
        self.records[0]["precisions"]["float32"]["controls"]["fidelity"]["D"][
            "blocks.9.attn.hook_q"]["source_sha256"] = "b"*64
        self.write_measurements()
        self.update_records_hash()
        with self.assertRaisesRegex(checker.VerificationError, "cached query provenance"):
            self.verify()

    def test_wrong_hook_order_and_wrong_position_are_rejected(self):
        original = copy.deepcopy(self.records)
        for kind in ("order", "position"):
            with self.subTest(kind=kind):
                self.records = copy.deepcopy(original)
                controls = self.records[0]["precisions"]["float64"]["controls"]
                if kind == "order":
                    controls["hook_events"]["D"].reverse()
                else:
                    controls["fidelity"]["D"]["blocks.9.attn.hook_q"]["position"] = 1
                self.write_measurements()
                self.update_records_hash()
                with self.assertRaises(checker.VerificationError):
                    self.verify()

    def test_frozen_code_change_is_rejected(self):
        path = self.application/"src/query_route.py"
        path.write_text(path.read_text()+"\n# changed")
        with self.assertRaisesRegex(checker.VerificationError, "hash mismatch"):
            self.verify()

    def test_unrehased_change_is_caught_before_semantic_checks(self):
        with (self.results/"records.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(checker.VerificationError, "hash mismatch"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
