"""The stored Round 1 pilot (development data) reproduces from its own files. No model.

Checks the artifact hashes, that the gate table in summary.json recomputes from the
stored records, and that the qualifying records pass the records-only checker's
per-record technical checks. The pilot is descriptive; nothing here freezes a value.
"""
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
PILOT = APP / "results/pilot"
if str(APP / "src") not in sys.path:
    sys.path.insert(0, str(APP / "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PilotResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((PILOT / "summary.json").read_text())
        cls.records = [json.loads(line) for line in (PILOT / "records.jsonl").read_text().splitlines()]

    def test_artifact_hashes_match(self):
        hashes = json.loads((PILOT / "artifact_hashes.json").read_text())
        self.assertLessEqual({"manifest.json", "records.jsonl", "summary.json", "fp32_reference.jsonl",
                              "diagnostic_answer_form.json", "diagnostic_fp32_execution.json"}, set(hashes))
        for name, digest in hashes.items():
            self.assertEqual(hashlib.sha256((PILOT / name).read_bytes()).hexdigest(), digest, name)
        started = json.loads((PILOT / "RUN_STARTED.json").read_text())
        self.assertEqual(started["manifest_sha256"], hashes["manifest.json"])

    def test_summary_recomputes_from_the_stored_files(self):
        try:
            import numpy  # noqa: F401  (the audit of full logits needs it)
        except ImportError:
            self.skipTest("numpy is needed for the audit part of the summary")
        import mixing_pilot_summary
        script = load("run_mixing_pilot", APP / "scripts/run_mixing_pilot.py")
        fresh = json.loads(json.dumps(mixing_pilot_summary.summarize(PILOT, script.PILOT)))
        self.assertEqual(fresh, self.summary)

    def test_recorded_outcome(self):
        gates = self.summary["gates"]
        self.assertFalse(gates["7_dtype"]["passed"])
        self.assertEqual(self.summary["n_rule"]["N"], 400)
        self.assertEqual(self.summary["gates"]["8_support"]["pooled_unresolved"]["k"], 3)
        self.assertTrue(all(gates[g]["passed"] for g in ("1_model_hashes", "2_native_and_yield", "3_tokens",
                                                          "4_hooks", "5_identity", "6_agreement", "8_support")))

    def test_qualifying_records_pass_the_checkers_record_checks(self):
        checker = load("check_mixing_round1_records", APP / "scripts/check_mixing_round1_records.py")
        self.assertEqual([r["draw_index"] for r in self.records], list(range(80)))
        for record in self.records:
            if record["qualifies"]:
                self.assertIsNone(checker.technical_failure(record, record["cell"], 7), record["case_id"])


if __name__ == "__main__":
    unittest.main()
