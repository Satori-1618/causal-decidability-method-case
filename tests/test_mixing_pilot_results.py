"""The stored Round 1 pilot 1 (protocol v1, development data) reproduces. No model.

Checks the artifact hashes (the directory stays byte for byte as committed), that the
gate table in summary.json recomputes from the stored records with the code that made
it (commit 3cdaffd, read from git history into a temporary directory and run in a
subprocess, so the protocol-v2 modules in the tree are not involved), and that the
recorded outcome is what the report states. The pilot is descriptive.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
PILOT = APP / "results/pilot"
V1_COMMIT = "3cdaffd"
V1_FILES = [
    "applications/gur-arieh-2510.06182/src/mixing_pilot_summary.py",
    "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py",
    "applications/gur-arieh-2510.06182/src/mixing_round1_design.py",
    "applications/gur-arieh-2510.06182/scripts/run_mixing_pilot.py",
    "applications/makelov-2311.17030/src/query_route_analysis.py",
]


def git_show(path):
    return subprocess.run(["git", "show", f"{V1_COMMIT}:{path}"], cwd=ROOT, capture_output=True,
                          check=True).stdout


class PilotOneResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((PILOT / "summary.json").read_text())

    def test_artifact_hashes_match(self):
        hashes = json.loads((PILOT / "artifact_hashes.json").read_text())
        self.assertLessEqual({"manifest.json", "records.jsonl", "summary.json", "fp32_reference.jsonl",
                              "diagnostic_answer_form.json", "diagnostic_fp32_execution.json"}, set(hashes))
        for name, digest in hashes.items():
            self.assertEqual(hashlib.sha256((PILOT / name).read_bytes()).hexdigest(), digest, name)
        started = json.loads((PILOT / "RUN_STARTED.json").read_text())
        self.assertEqual(started["manifest_sha256"], hashes["manifest.json"])

    def test_summary_recomputes_with_the_protocol_v1_code(self):
        try:
            import numpy  # noqa: F401  (the audit of full logits needs it)
            sources = {path: git_show(path) for path in V1_FILES}
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
                "spec = importlib.util.spec_from_file_location('v1_pilot', "
                f"{str(root / 'applications/gur-arieh-2510.06182/scripts/run_mixing_pilot.py')!r})\n"
                "script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)\n"
                "import mixing_pilot_summary\n"
                f"print(json.dumps(mixing_pilot_summary.summarize({str(PILOT)!r}, script.PILOT)))\n")
            completed = subprocess.run([sys.executable, "-B", "-c", program], capture_output=True, text=True,
                                       check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertEqual(json.loads(completed.stdout), self.summary)

    def test_recorded_outcome(self):
        gates = self.summary["gates"]
        self.assertFalse(gates["7_dtype"]["passed"])
        self.assertEqual(self.summary["n_rule"]["N"], 400)
        self.assertEqual(gates["8_support"]["pooled_unresolved"]["k"], 3)
        self.assertTrue(all(gates[g]["passed"] for g in ("1_model_hashes", "2_native_and_yield", "3_tokens",
                                                          "4_hooks", "5_identity", "6_agreement", "8_support")))


if __name__ == "__main__":
    unittest.main()
