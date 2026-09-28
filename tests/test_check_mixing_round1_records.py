"""Tamper tests for the Round 1 records-only checker. Constructed bundles, not evidence."""
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
if str(APP / "src") not in sys.path:
    sys.path.insert(0, str(APP / "src"))
import mixing_round1_analysis as ra  # noqa: E402
import mixing_synthetic_worlds as worlds  # noqa: E402

CHECKER = APP / "scripts/check_mixing_round1_records.py"
SPEC = importlib.util.spec_from_file_location("check_mixing_round1_records", CHECKER)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

WORLD = {name: worlds.run_world(name) for name in ("2_heterogeneous_concentrated", "5_stale_anchor")}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_lines(path, values):
    path.write_text("".join(json.dumps(v, allow_nan=False) + "\n" for v in values))


class Bundle:
    """A results directory in a temporary repository with fixture code files."""

    def __init__(self, repo, world, real_code=False):
        self.repo = repo
        self.results = repo / "applications/gur-arieh-2510.06182/results/constructed"
        self.results.mkdir(parents=True)
        code = {}
        for name in sorted(checker.REQUIRED_CODE):
            source = ROOT / name
            if real_code:
                code[name] = sha256(source)
                continue
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == checker.HELPER:
                shutil.copyfile(source, target)
            else:
                target.write_text("# constructed code-hash fixture\n" + name)
            code[name] = sha256(target)
        self.manifest = copy.deepcopy(WORLD[world]["manifest"])
        self.manifest["code_files_sha256"] = code
        self.records = copy.deepcopy(WORLD[world]["records"])
        self.write_all()

    def write_all(self, summary=None):
        save(self.results / "manifest.json", self.manifest)
        manifest_hash = sha256(self.results / "manifest.json")
        save(self.results / "RUN_STARTED.json", {"manifest_sha256": manifest_hash})
        write_lines(self.results / "records.jsonl", self.records)
        if summary is None:
            summary = ra.analyze_confirmation(self.manifest, self.records)
        summary.update(manifest_sha256=manifest_hash,
                       records_sha256=sha256(self.results / "records.jsonl"))
        self.summary = summary
        save(self.results / "summary.json", summary)
        self.rehash()

    def save_summary(self, summary):
        self.summary = summary
        save(self.results / "summary.json", summary)
        self.rehash()

    def rehash(self):
        save(self.results / "artifact_hashes.json",
             {p.name: sha256(p) for p in self.results.iterdir() if p.name != "artifact_hashes.json"})

    def verify(self, repository_root=None):
        return checker.verify(self.results, repository_root=repository_root or self.repo)


class RoundOneCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        self.bundle = Bundle(self.repo, "2_heterogeneous_concentrated")

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_bundle_reproduces_statuses_level_and_sentence(self):
        report = self.bundle.verify()
        self.assertTrue(report["verified"])
        self.assertEqual(report["level"], "S3 (W_T excluded only)")
        self.assertEqual(report["statuses"], {"W_T": "excluded", "A_T": "adequate"})
        self.assertTrue(report["between_case_earned"])

    def test_stale_anchor_bundle_reproduces_invalid(self):
        with tempfile.TemporaryDirectory() as other:
            report = Bundle(Path(other), "5_stale_anchor").verify()
        self.assertEqual((report["run_status"], report["level"]), ("INVALID", "S1"))
        self.assertFalse(report["between_case_earned"])

    def test_altered_manifest_contract_is_rejected_even_when_rehashed(self):
        self.bundle.manifest["rule"]["kappa"] = 0.3
        self.bundle.manifest["rule"]["coverage"] = 0.8
        summary = copy.deepcopy(self.bundle.summary)
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "declared contract"):
            self.bundle.verify()

    def test_altered_anchor_is_rejected_even_when_rehashed(self):
        anchors = self.bundle.manifest["anchors"]
        anchors["T_A"] -= 0.3
        anchors["d"] = anchors["T_A"] - anchors["T_W"]
        summary = copy.deepcopy(self.bundle.summary)
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "differ"):
            self.bundle.verify()

    def test_frozen_agreement_resolution_below_floor_is_rejected(self):
        self.bundle.manifest["development_gates"]["agreement_resolution_rate_B"] = 0.85
        summary = copy.deepcopy(self.bundle.summary)
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "agreement-control resolution"):
            self.bundle.verify()

    def test_manifest_changed_after_start_is_rejected(self):
        self.bundle.manifest["N"] = 199
        save(self.bundle.results / "manifest.json", self.bundle.manifest)
        self.bundle.rehash()
        with self.assertRaisesRegex(checker.VerificationError, "freeze hash"):
            self.bundle.verify()

    def test_missing_record_is_rejected(self):
        summary = copy.deepcopy(self.bundle.summary)
        del self.bundle.records[10]
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "omit, duplicate or reorder"):
            self.bundle.verify()

    def test_missing_record_with_renumbered_draws_is_rejected(self):
        summary = copy.deepcopy(self.bundle.summary)
        del self.bundle.records[10]
        for i, r in enumerate(self.bundle.records):
            r["draw_index"] = i
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "frozen N"):
            self.bundle.verify()

    def test_duplicated_record_is_rejected(self):
        summary = copy.deepcopy(self.bundle.summary)
        self.bundle.records.insert(11, copy.deepcopy(self.bundle.records[10]))
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "duplicate"):
            self.bundle.verify()

    def test_edited_status_is_rejected_even_when_rehashed(self):
        summary = copy.deepcopy(self.bundle.summary)
        summary["statuses"]["W_T"] = "undecided"
        self.bundle.save_summary(summary)
        with self.assertRaisesRegex(checker.VerificationError, "statuses differ"):
            self.bundle.verify()

    def test_edited_between_case_decision_is_rejected(self):
        summary = copy.deepcopy(self.bundle.summary)
        summary["between_case"]["earned"] = False
        self.bundle.save_summary(summary)
        with self.assertRaisesRegex(checker.VerificationError, "between-case"):
            self.bundle.verify()

    def test_edited_level_is_rejected(self):
        summary = copy.deepcopy(self.bundle.summary)
        summary["level"] = "S2"
        self.bundle.save_summary(summary)
        with self.assertRaisesRegex(checker.VerificationError, "level"):
            self.bundle.verify()

    def test_edited_raw_readout_is_rejected_even_when_rehashed(self):
        summary = copy.deepcopy(self.bundle.summary)
        first = next(r for r in self.bundle.records if r["qualifies"])
        first["entity_logits"] = worlds.logits(worlds.W_LIKE, 0.8)
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "differ"):
            self.bundle.verify()

    def test_unrehashed_change_is_caught_first(self):
        with (self.bundle.results / "records.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(checker.VerificationError, "hash mismatch"):
            self.bundle.verify()

    def test_changed_frozen_code_is_rejected(self):
        path = self.repo / "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py"
        path.write_text(path.read_text() + "\n# changed")
        with self.assertRaisesRegex(checker.VerificationError, "hash mismatch"):
            self.bundle.verify()

    def test_technical_failure_must_be_reported_as_invalid(self):
        record = next(r for r in self.bundle.records if r["qualifies"])
        record["technical"] = {"passed": False}
        self.bundle.write_all()
        report = self.bundle.verify()
        self.assertEqual((report["run_status"], report["level"]), ("INVALID", "S1"))
        claimed_valid = copy.deepcopy(WORLD["2_heterogeneous_concentrated"]["summary"])
        claimed_valid.update(manifest_sha256=self.bundle.summary["manifest_sha256"],
                             records_sha256=self.bundle.summary["records_sha256"])
        self.bundle.save_summary(claimed_valid)
        with self.assertRaisesRegex(checker.VerificationError, "run status"):
            self.bundle.verify()

    def test_command_line_runs_with_the_standard_library_only(self):
        with tempfile.TemporaryDirectory() as other:
            bundle = Bundle(Path(other), "2_heterogeneous_concentrated", real_code=True)
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(CHECKER), "--results", str(bundle.results)],
                capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
