"""Records-only audit tests; standard library, no model or upstream artifacts."""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
SCRIPT = APP / "scripts/check_read_source_q1_records.py"
spec = importlib.util.spec_from_file_location("q1_records_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RecordsOnlyAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((APP / audit.RUN / "cases.json").read_text())
        cls.records = [json.loads(s) for s in
                       (APP / audit.RUN / "records.jsonl").read_text().splitlines()]

    def minimal_bundle(self, target):
        for relative in audit.PINNED:
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(APP / relative, output)

    def test_clean_bundle_needs_neither_tensor_archive_nor_model_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            self.minimal_bundle(target)
            before = {str(p.relative_to(target)): p.read_bytes()
                      for p in target.rglob("*") if p.is_file()}
            # -S omits site-packages; -I omits user modules and PYTHONPATH.
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(SCRIPT),
                 "--application-root", str(target), "--json"],
                text=True, capture_output=True, check=True)
            report = json.loads(completed.stdout)
            self.assertEqual(report["primary"]["wins"]["B_null_read"], 64)
            self.assertEqual(report["primary"]["p"], 2 ** -63)
            self.assertEqual(report["controls"]["recorded_arm_checks"], 512)
            self.assertIn("live model or hook execution", report["scope"]["not_verified"])
            self.assertEqual(before, {str(p.relative_to(target)): p.read_bytes()
                                     for p in target.rglob("*") if p.is_file()})
            self.assertFalse(list(target.rglob("*.npz")))

    def test_tampered_raw_record_is_rejected_by_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            self.minimal_bundle(target)
            raw = target / audit.RUN / "records.jsonl"
            raw.write_bytes(raw.read_bytes() + b"\n")
            with self.assertRaisesRegex(audit.RecordCheckError, "hash mismatch: .*records.jsonl"):
                audit.check(target)

    def test_rewritten_score_or_scorer_cannot_replace_frozen_evidence(self):
        for relative in (audit.RUN + "score.json", "scripts/score_read_source_q1.py"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                self.minimal_bundle(target)
                (target / relative).write_text("changed")
                with self.assertRaisesRegex(audit.RecordCheckError, "frozen hash mismatch"):
                    audit.check(target)

    def test_missing_reference_fails_instead_of_silently_skipping_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            self.minimal_bundle(target)
            (target / audit.RUN / "reference_cpu64.json").unlink()
            with self.assertRaisesRegex(audit.RecordCheckError, "missing.*reference_cpu64.json"):
                audit.check(target)

    def test_duplicate_direction_cannot_count_as_a_second_independent_case(self):
        records = copy.deepcopy(self.records)
        records[1]["receiver_pattern"] = records[0]["receiver_pattern"]
        with self.assertRaisesRegex(audit.RecordCheckError, "reciprocal directions"):
            audit.audit_records(records, self.cases, 64)

    def test_unknown_case_cannot_leave_a_silently_incomplete_cluster(self):
        records = copy.deepcopy(self.records)
        records[0]["case_id"] = "not-a-declared-case"
        with self.assertRaisesRegex(audit.RecordCheckError, "unknown record case"):
            audit.audit_records(records, self.cases, 64)

    def test_prompt_edit_is_detected_without_relying_on_file_hashes(self):
        cases = copy.deepcopy(self.cases)
        cases[0]["prompts"][0] += " changed"
        with self.assertRaisesRegex(audit.RecordCheckError, "does not hash"):
            audit.audit_records(self.records, cases, 64)

    def test_recorded_control_boolean_cannot_hide_an_excess_insertion_error(self):
        records = copy.deepcopy(self.records)
        control = records[0]["fidelity"]["read_null"]
        control["insertion_error_max_per_item"] = 2 * control["rounding_budget_per_item"]
        self.assertTrue(control["passed"])
        with self.assertRaisesRegex(audit.RecordCheckError, "fidelity control failed"):
            audit.audit_records(records, self.cases, 64)

    def test_wrong_margin_is_detected_independently_of_frozen_hashes(self):
        records = copy.deepcopy(self.records)
        records[0]["margins"]["full"] += .1
        with self.assertRaisesRegex(audit.RecordCheckError, "logit-margin arithmetic"):
            audit.audit_records(records, self.cases, 64)

    def test_nonfinite_measurement_is_not_a_tie(self):
        records = copy.deepcopy(self.records)
        records[0]["margins"]["full"] = float("nan")
        with self.assertRaisesRegex(audit.RecordCheckError, "logit-margin arithmetic"):
            audit.audit_records(records, self.cases, 64)

    def test_absolute_position_is_checked(self):
        records = copy.deepcopy(self.records)
        records[0]["position"] -= 1
        with self.assertRaisesRegex(audit.RecordCheckError, "position mismatch"):
            audit.audit_records(records, self.cases, 64)

    def test_primary_test_drops_ties_but_reports_them(self):
        primary, _ = audit.primary_result({
            "one": {"A_visible_read": 2, "B_null_read": 1},
            "two": {"A_visible_read": 1, "B_null_read": 2},
            "three": {"A_visible_read": 1, "B_null_read": 1},
        })
        self.assertEqual(primary["ties"], 1)
        self.assertEqual(primary["wins"], {"A_visible_read": 1, "B_null_read": 1})
        self.assertEqual(primary["p"], 1)
        self.assertEqual(primary["outcome"], "no difference shown")


if __name__ == "__main__":
    unittest.main()
