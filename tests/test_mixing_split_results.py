"""Stored Round 1 splits A and B (protocol v2, development data) reproduce. No model.

For each split directory present: the generated index covers exactly its files, the
producers match the recorded commit, the decision recomputes from the stored files with
that commit's code (read from git history, run in a subprocess), and the qualifying
records pass the records-only checker's per-record checks. Skipped for a split that has
not run.
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
SPLITS = {"A": APP / "results/split_A", "B": APP / "results/split_B"}
DECISION_CODE = [
    "applications/gur-arieh-2510.06182/src/mixing_splits.py",
    "applications/gur-arieh-2510.06182/src/mixing_pilot_summary.py",
    "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py",
    "applications/makelov-2311.17030/src/query_route_analysis.py",
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


class SplitResultTests(unittest.TestCase):
    def each(self):
        present = [(name, path) for name, path in SPLITS.items() if (path / "manifest.json").exists()]
        if not present:
            self.skipTest("no split has run")
        return present

    def test_index_covers_every_file_and_matches(self):
        for name, path in self.each():
            index = json.loads((path / "artifact_hashes.json").read_text())
            files = {p.name for p in path.iterdir() if p.is_file()} - {"artifact_hashes.json"}
            with self.subTest(split=name):
                self.assertEqual(set(index["data"]), files)
                self.assertIn(f"split_{name}_decision.json", files)
                for file, digest in index["data"].items():
                    self.assertEqual(sha256((path / file).read_bytes()), digest, file)
                started = json.loads((path / "RUN_STARTED.json").read_text())
                self.assertEqual(started["manifest_sha256"], index["data"]["manifest.json"])

    def test_producers_match_the_recorded_commit(self):
        for name, path in self.each():
            manifest = json.loads((path / "manifest.json").read_text())
            index = json.loads((path / "artifact_hashes.json").read_text())
            with self.subTest(split=name):
                self.assertFalse(manifest["git_dirty"])
                self.assertEqual(index["producers"], manifest["producers_sha256"])
                try:
                    for file, digest in index["producers"].items():
                        self.assertEqual(sha256(git_show(manifest["git_head"], file)), digest, file)
                except (OSError, subprocess.CalledProcessError) as error:
                    self.skipTest(f"needs the git history ({error})")

    def test_decision_recomputes_with_the_recorded_code(self):
        for name, path in self.each():
            manifest = json.loads((path / "manifest.json").read_text())
            try:
                import numpy  # noqa: F401
                sources = {file: git_show(manifest["git_head"], file) for file in DECISION_CODE}
            except (ImportError, OSError, subprocess.CalledProcessError) as error:
                self.skipTest(f"needs numpy and the git history ({error})")
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for file, content in sources.items():
                    (root / file).parent.mkdir(parents=True, exist_ok=True)
                    (root / file).write_bytes(content)
                function = "split_a_decision" if name == "A" else "split_b_decision"
                program = (
                    "import json, sys\n"
                    f"sys.path.insert(0, {str(root / 'applications/gur-arieh-2510.06182/src')!r})\n"
                    "import mixing_splits\n"
                    f"spec = json.load(open({str(path / 'manifest.json')!r}))['spec']\n"
                    f"print(json.dumps(mixing_splits.{function}({str(path)!r}, spec)))\n")
                completed = subprocess.run([sys.executable, "-B", "-c", program], capture_output=True, text=True,
                                           check=False)
            with self.subTest(split=name):
                self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
                stored = json.loads((path / f"split_{name}_decision.json").read_text())
                self.assertEqual(json.loads(completed.stdout), stored)

    def test_records_pass_the_checkers_record_checks(self):
        spec = importlib.util.spec_from_file_location("checker", APP / "scripts/check_mixing_round1_records.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        for name, path in self.each():
            records = [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()]
            with self.subTest(split=name):
                self.assertTrue(records)
                for record in records:
                    self.assertNotIn("entity_logits", record)
                    if record["qualifies"]:
                        self.assertIsNone(checker.technical_failure(record, record["cell"], 7))


if __name__ == "__main__":
    unittest.main()
