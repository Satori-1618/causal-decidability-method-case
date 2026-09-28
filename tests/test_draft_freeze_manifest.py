"""End-to-end checks for the Round 1 freeze-draft generator. No model is loaded."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APPLICATION_RELATIVE = Path("applications/gur-arieh-2510.06182")
GENERATOR_RELATIVE = APPLICATION_RELATIVE / "scripts/draft_freeze_manifest.py"
CHECKER_RELATIVE = (
    "applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py")
DIRECT_DATA = {
    "results/split_A/manifest.json",
    "results/split_A/records.jsonl",
    "results/split_A/split_A_decision.json",
    "results/split_A/artifact_hashes.json",
    "results/split_B/manifest.json",
    "results/split_B/records.jsonl",
    "results/split_B/split_B_decision.json",
    "results/split_B/artifact_hashes.json",
}


def run(command, *, cwd, check=True):
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=check)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DraftFreezeManifestTests(unittest.TestCase):
    def test_real_artifacts_generate_a_bound_finalizable_draft_and_reject_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "repo"
            run(["git", "clone", "--quiet", "--shared", str(ROOT), str(clone)], cwd=ROOT)

            # Before this test itself is committed, copy the working-tree generator into the
            # clone and commit it. In a clean checkout this is a no-op.
            shutil.copyfile(ROOT / GENERATOR_RELATIVE, clone / GENERATOR_RELATIVE)
            if run(["git", "status", "--porcelain"], cwd=clone).stdout.strip():
                run(["git", "add", str(GENERATOR_RELATIVE)], cwd=clone)
                run(["git", "-c", "user.name=Draft Test", "-c",
                     "user.email=draft-test@example.invalid", "commit", "--quiet", "-m",
                     "Test the working-tree draft generator"], cwd=clone)

            generated = run([sys.executable, str(clone / GENERATOR_RELATIVE)], cwd=clone)
            self.assertIn("DRAFT, NOT FROZEN", generated.stdout)
            application = clone / APPLICATION_RELATIVE
            manifest = json.loads((application / "FREEZE_DRAFT.json").read_text())
            split_b = json.loads((application / "results/split_B/manifest.json").read_text())

            self.assertEqual(manifest["based_on_commit"],
                             run(["git", "rev-parse", "HEAD"], cwd=clone).stdout.strip())
            self.assertEqual(manifest["task_spec"], split_b["task_spec"])
            self.assertEqual(manifest["entity_pools"], split_b["entity_pools"])
            self.assertEqual(manifest["upstream"], split_b["upstream"]["check"])
            self.assertEqual(manifest["development_environment"], split_b["environment"])
            self.assertTrue(DIRECT_DATA <= set(manifest["data_sha256"]))
            for relative, digest in manifest["data_sha256"].items():
                self.assertEqual(sha256(application / relative), digest, relative)
            for relative, digest in manifest["code_files_sha256"].items():
                self.assertEqual(sha256(clone / relative), digest, relative)

            shared = set(split_b["producers_sha256"]) - {CHECKER_RELATIVE}
            for relative in shared:
                self.assertEqual(manifest["code_files_sha256"][relative],
                                 split_b["producers_sha256"][relative], relative)
            split_model = split_b["model"]
            self.assertEqual(manifest["model"], {
                "id": split_model["id"],
                "revision": split_model["revision"],
                "files_sha256": {name: row["sha256"]
                                 for name, row in split_model["hashes"]["files"].items()},
            })
            for stage in ("split_A", "split_B"):
                provenance = manifest["development_provenance"][stage]
                results = application / "results" / stage
                self.assertEqual(provenance["manifest_sha256"],
                                 sha256(results / "manifest.json"))
                self.assertEqual(provenance["records_sha256"],
                                 sha256(results / "records.jsonl"))

            # The reviewed finalization changes only the three authorization markers. The
            # current runner must accept the resulting structure and split-A/B binding.
            validator = r"""
import importlib.util
import json
import sys
from pathlib import Path
repo = Path(sys.argv[1])
app = repo / "applications/gur-arieh-2510.06182"
spec = importlib.util.spec_from_file_location(
    "confirmation_runner", app / "scripts/run_mixing_confirmation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
manifest = json.loads((app / "FREEZE_DRAFT.json").read_text())
manifest["stage"] = "confirmation"
manifest["freeze_status"] = "FROZEN"
manifest["confirmation"]["authorized"] = True
runner.validate_frozen_manifest(manifest)
runner.validate_development_binding(manifest, app)
"""
            run([sys.executable, "-c", validator, str(clone)], cwd=clone)

            # A later clean commit cannot silently substitute another source lock/model.
            (application / "FREEZE_DRAFT.md").unlink()
            (application / "FREEZE_DRAFT.json").unlink()
            source_lock_path = application / "SOURCE_LOCK.json"
            source_lock = json.loads(source_lock_path.read_text())
            source_lock["model"]["id"] += "-drift"
            source_lock_path.write_text(json.dumps(source_lock, indent=2) + "\n")
            run(["git", "add", str(APPLICATION_RELATIVE / "SOURCE_LOCK.json")], cwd=clone)
            run(["git", "-c", "user.name=Draft Test", "-c",
                 "user.email=draft-test@example.invalid", "commit", "--quiet", "-m",
                 "Introduce source-lock drift"], cwd=clone)
            rejected = run([sys.executable, str(clone / GENERATOR_RELATIVE)],
                           cwd=clone, check=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("SOURCE_LOCK.json differs from the verified split-B manifest",
                          rejected.stderr)

            # Restoring the lock does not permit drift in a shared development producer.
            run(["git", "checkout", "HEAD^", "--",
                 str(APPLICATION_RELATIVE / "SOURCE_LOCK.json")], cwd=clone)
            shared_runner = application / "src/mixing_runner.py"
            shared_runner.write_text(shared_runner.read_text() + "\n# drift fixture\n")
            run(["git", "add", str(APPLICATION_RELATIVE / "SOURCE_LOCK.json"),
                 str(APPLICATION_RELATIVE / "src/mixing_runner.py")], cwd=clone)
            run(["git", "-c", "user.name=Draft Test", "-c",
                 "user.email=draft-test@example.invalid", "commit", "--quiet", "-m",
                 "Introduce shared-producer drift"], cwd=clone)
            rejected = run([sys.executable, str(clone / GENERATOR_RELATIVE)],
                           cwd=clone, check=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("shared development producer differs from split B", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
