"""No-model tests for the frozen Round 1 confirmation runner."""
import copy
import importlib.util
import json
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

RUNNER_PATH = APP / "scripts/run_mixing_confirmation.py"
SPEC = importlib.util.spec_from_file_location("run_mixing_confirmation", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value, allow_nan=False) + "\n" for value in values))


class FrozenFixture:
    """A tiny fake repository/cache whose hashes satisfy the real preflight."""

    def __init__(self, root):
        self.root = Path(root)
        self.repo = self.root / "repository"
        self.app = self.repo / "applications/gur-arieh-2510.06182"
        self.cache = self.root / "cache"
        self.output = self.root / "output"

        code = {}
        for name in sorted(runner.REQUIRED_CODE):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"frozen fixture for {name}\n")
            code[name] = runner.sha256(path)

        model_id, revision = "example/tiny", "frozen-revision"
        snapshot = self.cache / "models--example--tiny/snapshots" / revision
        snapshot.mkdir(parents=True)
        model_files = {}
        content_hashes = {}
        for name in sorted(runner.REQUIRED_MODEL_FILES):
            path = snapshot / name
            path.write_bytes(("model fixture " + name).encode())
            digest = runner.sha256(path)
            model_files[name] = digest
            content_hashes[name] = {"sha256": digest, "size": path.stat().st_size}

        source_lock = {
            "model": {"id": model_id, "revision": revision,
                      "content_hashes_gate1": content_hashes},
            "round1_entity_pools": {},
        }
        save(self.app / "SOURCE_LOCK.json", source_lock)
        sizing = ra.n_rule(0.0)
        self.manifest = {
            "schema_version": 2,
            "application": "gur-arieh-2510.06182",
            "round": 1,
            "stage": "confirmation",
            "freeze_status": "FROZEN",
            "n_groups": 7,
            "task": "music_performance",
            "task_spec": {"name": "music_performance"},
            "t_entity": 2,
            "layer": 18,
            "patch_positions": [-1],
            "cell_key": "c4",
            "cell": {"i_P": 3, "i_L": 5, "i_R": 1, "i_N": 6},
            "N": sizing["N"],
            "N_rule": sizing,
            "rule": {**ra.CONTRACT, "s_min": 0.1, "d_min": 0.2,
                     "agreement_transfer_floor": 0.9},
            "anchors": {"T_W": 0.4, "T_A": 0.9, "d": 0.5,
                        "q_bar_B": [0.4, 0.35, 0.25], "m_B": 200},
            "mean_gate": {"delta": 0.1},
            "development_gates": {"resolution_rate_B": 1.0,
                                  "agreement_resolution_rate_B": 1.0,
                                  "agreement_transfer_rate_B": 1.0},
            "execution_contract": {
                "device": "mps", "dtype": "float32", "attention": "eager",
                "layer": 18, "patch_position": "last token",
                "identity_tolerance": runner.IDENTITY_TOLERANCE,
                "gate7": {"reference_device": "cpu", "reference_dtype": "float32",
                          "tolerance_T": runner.GATE7_T_TOLERANCE,
                          "same_resolution_required": True, "same_labels_required": True},
            },
            "confirmation": {
                "authorized": True,
                "seed_base": runner.SEED_BASE,
                "seed_formula": "seed_base + draw_index",
                "case_id_prefix": runner.CASE_ID_PREFIX,
                "cap": 2 * sizing["N"],
                "gate7_draw_indices": list(runner.GATE7_DRAW_INDICES),
                "audit_full_logit_draw_indices": list(runner.AUDIT_DRAW_INDICES),
            },
            "code_files_sha256": code,
            "data_sha256": {"SOURCE_LOCK.json": runner.sha256(self.app / "SOURCE_LOCK.json")},
            "model": {"id": model_id, "revision": revision, "files_sha256": model_files},
            "entity_pools": {},
            "upstream": {},
            "development_environment": {},
        }
        self.manifest_path = self.root / "FREEZE.json"
        self.write_manifest()

    def write_manifest(self):
        save(self.manifest_path, self.manifest)

    def preflight(self):
        return runner.preflight(
            self.manifest_path, self.output, repository_root=self.repo,
            application_root=self.app, cache_root=self.cache)


class ConfirmationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = FrozenFixture(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_draft_or_unauthorized_manifest_is_refused_without_output(self):
        for field, value, message in (
                ("stage", "confirmation (DRAFT, NOT FROZEN)", "stage"),
                ("freeze_status", "DRAFT", "freeze_status")):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.fixture.manifest)
                manifest[field] = value
                save(self.fixture.manifest_path, manifest)
                with self.assertRaisesRegex(runner.ConfirmationError, message):
                    self.fixture.preflight()
                self.assertFalse(self.fixture.output.exists())
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["confirmation"]["authorized"] = False
        save(self.fixture.manifest_path, manifest)
        with self.assertRaisesRegex(runner.ConfirmationError, "authorized"):
            self.fixture.preflight()
        self.assertFalse(self.fixture.output.exists())

    def test_any_frozen_code_data_or_model_hash_mismatch_is_refused_without_output(self):
        cases = []
        changed = copy.deepcopy(self.fixture.manifest)
        changed["code_files_sha256"][runner.RUNNER_RELATIVE] = "0" * 64
        cases.append(("code", changed, None))
        changed = copy.deepcopy(self.fixture.manifest)
        changed["data_sha256"]["SOURCE_LOCK.json"] = "0" * 64
        cases.append(("data", changed, None))
        changed = copy.deepcopy(self.fixture.manifest)
        model_path = (self.fixture.cache / "models--example--tiny/snapshots/frozen-revision/config.json")
        cases.append(("model", changed, model_path))

        for label, manifest, tamper_path in cases:
            with self.subTest(label=label):
                original = tamper_path.read_bytes() if tamper_path else None
                if tamper_path:
                    tamper_path.write_bytes(original + b"tampered")
                save(self.fixture.manifest_path, manifest)
                with self.assertRaisesRegex(runner.ConfirmationError, "hash mismatch"):
                    self.fixture.preflight()
                self.assertFalse(self.fixture.output.exists())
                if tamper_path:
                    tamper_path.write_bytes(original)

    def test_seed_and_case_id_formulas_are_exact_and_seed_base_must_be_numeric(self):
        self.assertEqual(runner.case_id(0), "confirmation-0000")
        self.assertEqual(runner.case_id(37), "confirmation-0037")
        self.assertEqual(runner.seed_for(0), 4_000_000)
        self.assertEqual(runner.seed_for(37), 4_000_037)
        with self.assertRaises(runner.ConfirmationError):
            runner.seed_for(True)
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["confirmation"]["seed_base"] = True
        with self.assertRaisesRegex(runner.ConfirmationError, "numeric"):
            runner.validate_frozen_manifest(manifest)

    def test_analyzed_output_is_compatible_with_the_records_only_checker(self):
        result = worlds.run_world("2_heterogeneous_concentrated")
        manifest = copy.deepcopy(result["manifest"])
        records = copy.deepcopy(result["records"])
        manifest.update(stage="confirmation", freeze_status="FROZEN", cell_key="c4")
        manifest["confirmation"] = {
            "authorized": True, "seed_base": runner.SEED_BASE,
            "case_id_prefix": runner.CASE_ID_PREFIX, "cap": 2 * manifest["N"],
        }
        code_paths = {
            "applications/makelov-2311.17030/src/query_route_analysis.py",
            "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py",
            "applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py",
            runner.RUNNER_RELATIVE,
        }
        manifest["code_files_sha256"] = {
            name: runner.sha256(ROOT / name) for name in sorted(code_paths)}
        for index, record in enumerate(records):
            record.update(case_id=runner.case_id(index), seed=runner.seed_for(index), cell_key="c4")

        results = Path(self.temporary.name) / "checker-compatible"
        results.mkdir()
        save(results / "manifest.json", manifest)
        save(results / "RUN_STARTED.json", {"manifest_sha256": runner.sha256(results / "manifest.json")})
        write_jsonl(results / "records.jsonl", records)
        save(results / "timings.json", {"fixture": True})
        save(results / "confirmation_gates.json", {"status": "PROCEED", "fixture": True})
        write_jsonl(results / "gate7_cpu_reference.jsonl", [])
        (results / "audit_full_logits.npz").write_bytes(b"no-model fixture")

        summary, report = runner.analyze_and_check(manifest, records, results,
                                                   repository_root=ROOT)
        self.assertTrue(report["verified"])
        self.assertEqual(summary["statuses"], {"W_T": "excluded", "A_T": "adequate"})
        hashes = json.loads((results / "artifact_hashes.json").read_text())
        self.assertIsInstance(hashes, dict)
        self.assertTrue({"manifest.json", "records.jsonl", "summary.json",
                         "confirmation_gates.json", "checker_report.json"} <= set(hashes))


if __name__ == "__main__":
    unittest.main()
