"""The stored Round 1 pilot 2 (protocol v2, development data) reproduces. No model.

Checks that the generated artifact index covers exactly the files of results/pilot2 and
the producer scripts as they were at the recorded commit, that the gate table in
summary.json recomputes from the stored files with that commit's code (read from git
history, run in a subprocess), that the qualifying records pass the records-only
checker's per-record checks, and the recorded outcome. The pilot is descriptive.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
PILOT2 = APP / "results/pilot2"
SUMMARY_CODE = [
    "applications/gur-arieh-2510.06182/src/mixing_pilot_summary.py",
    "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py",
    "applications/gur-arieh-2510.06182/scripts/run_mixing_pilot.py",
    "applications/makelov-2311.17030/src/query_route_analysis.py",
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


class PilotTwoResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = json.loads((PILOT2 / "artifact_hashes.json").read_text())
        cls.summary = json.loads((PILOT2 / "summary.json").read_text())
        cls.manifest = json.loads((PILOT2 / "manifest.json").read_text())

    def test_index_covers_every_file_and_matches(self):
        files = {p.name for p in PILOT2.iterdir() if p.is_file()} - {"artifact_hashes.json"}
        self.assertEqual(set(self.index["data"]), files)
        self.assertIn("PILOT2_REPORT.md", files)
        for name, digest in self.index["data"].items():
            self.assertEqual(sha256((PILOT2 / name).read_bytes()), digest, name)
        started = json.loads((PILOT2 / "RUN_STARTED.json").read_text())
        self.assertEqual(started["manifest_sha256"], self.index["data"]["manifest.json"])

    def test_producers_match_the_recorded_commit(self):
        head = self.manifest["git_head"]
        self.assertFalse(self.manifest["git_dirty"])
        self.assertEqual(self.index["producers"], self.manifest["producers_sha256"])
        try:
            for path, digest in self.index["producers"].items():
                self.assertEqual(sha256(git_show(head, path)), digest, path)
        except (OSError, subprocess.CalledProcessError) as error:
            self.skipTest(f"needs the git history ({error})")

    def test_summary_recomputes_with_the_recorded_code(self):
        head = self.manifest["git_head"]
        try:
            import numpy  # noqa: F401  (the audit of full logits needs it)
            sources = {path: git_show(head, path) for path in SUMMARY_CODE}
        except (ImportError, OSError, subprocess.CalledProcessError) as error:
            self.skipTest(f"needs numpy and the git history ({error})")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path, content in sources.items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_bytes(content)
            program = (
                "import importlib.util, json, sys\n"
                f"sys.path.insert(0, {str(root / 'applications/gur-arieh-2510.06182/src')!r})\n"
                "spec = importlib.util.spec_from_file_location('pilot2_script', "
                f"{str(root / 'applications/gur-arieh-2510.06182/scripts/run_mixing_pilot.py')!r})\n"
                "script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)\n"
                "import mixing_pilot_summary\n"
                f"print(json.dumps(mixing_pilot_summary.summarize({str(PILOT2)!r}, script.PILOT)))\n")
            completed = subprocess.run([sys.executable, "-B", "-c", program], capture_output=True, text=True,
                                       check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertEqual(json.loads(completed.stdout), self.summary)

    def test_recorded_outcome(self):
        gates = self.summary["gates"]
        for gate in ("1_model_hashes", "2_native_and_yield", "3_tokens", "4_hooks", "5_identity",
                     "6_agreement", "7_dtype_device", "8_support"):
            self.assertTrue(gates[gate]["passed"], gate)
        self.assertEqual(gates["7_dtype_device"]["per_cell"], {"c1": 8, "c2": 8, "c3": 8, "c4": 8})
        self.assertEqual(gates["8_support"]["pooled_unresolved"]["k"], 1)
        self.assertEqual(self.summary["planning_N"]["N"], 200)
        self.assertEqual(self.manifest["seed_base_used"], 1100000)
        self.assertEqual(self.manifest["stage"], "pilot2")

    def test_records_use_the_v2_schema_and_pass_the_checkers_record_checks(self):
        spec = importlib.util.spec_from_file_location("check_mixing_round1_records",
                                                      APP / "scripts/check_mixing_round1_records.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        records = [json.loads(line) for line in (PILOT2 / "records.jsonl").read_text().splitlines()]
        self.assertEqual([r["draw_index"] for r in records], list(range(80)))
        for record in records:
            self.assertNotIn("entity_logits", record)
            self.assertIn("paper_readout", record["descriptive"])
            if record["qualifies"]:
                self.assertIsNone(checker.technical_failure(record, record["cell"], 7), record["case_id"])


if __name__ == "__main__":
    unittest.main()
