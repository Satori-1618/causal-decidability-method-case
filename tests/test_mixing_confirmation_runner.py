"""No-model tests for the frozen Round 1 confirmation runner."""
import copy
import importlib.util
import json
import random
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

RUNNER_PATH = APP / "scripts/run_mixing_confirmation.py"
SPEC = importlib.util.spec_from_file_location("run_mixing_confirmation", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
checker = runner._load_checker(ROOT)


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
            "development_environment": {
                "python": "3.11.15",
                "packages": {"torch": "2.5.1", "transformers": "4.57.3",
                             "tokenizers": "0.22.2", "numpy": "1.26.4"},
            },
        }
        freeze = {
            "cell": copy.deepcopy(self.manifest["cell"]),
            "N": self.manifest["N"], "N_rule": copy.deepcopy(self.manifest["N_rule"]),
            "s_min": self.manifest["rule"]["s_min"],
            "d_min": self.manifest["rule"]["d_min"],
            "agreement_transfer_floor": self.manifest["rule"]["agreement_transfer_floor"],
            "anchors": copy.deepcopy(self.manifest["anchors"]),
            "mean_gate": copy.deepcopy(self.manifest["mean_gate"]),
            "development_gates": copy.deepcopy(self.manifest["development_gates"]),
        }
        split_a_path = self.app / runner.SPLIT_A_DECISION_RELATIVE
        split_b_path = self.app / runner.SPLIT_B_DECISION_RELATIVE
        save(split_a_path, {"status": "PROCEED", "stops": [], "selected": "c4"})
        save(split_b_path, {"status": "PROCEED", "stops": [], "cell_key": "c4",
                            "split_B": {"freeze": freeze}})
        self.manifest["data_sha256"].update({
            runner.SPLIT_A_DECISION_RELATIVE: runner.sha256(split_a_path),
            runner.SPLIT_B_DECISION_RELATIVE: runner.sha256(split_b_path),
        })
        self.manifest_path = self.repo / "FREEZE.json"
        self.write_manifest()
        self.git("init", "-q")
        self.git("config", "user.name", "Confirmation Test")
        self.git("config", "user.email", "confirmation-test@example.invalid")
        self.git("add", "--all")
        self.git("commit", "-q", "-m", "frozen fixture")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, check=True,
                              capture_output=True, text=True).stdout.strip()

    def write_manifest(self, manifest=None, *, commit=False):
        if manifest is not None:
            save(self.manifest_path, manifest)
        else:
            save(self.manifest_path, self.manifest)
        if commit:
            self.git("add", self.manifest_path.relative_to(self.repo).as_posix())
            self.git("commit", "-q", "-m", "update freeze fixture")

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
                self.fixture.write_manifest(manifest, commit=True)
                with self.assertRaisesRegex(runner.ConfirmationError, message):
                    self.fixture.preflight()
                self.assertFalse(self.fixture.output.exists())
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["confirmation"]["authorized"] = False
        self.fixture.write_manifest(manifest, commit=True)
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
                self.fixture.write_manifest(manifest, commit=True)
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

    def test_clean_committed_freeze_reports_git_binding(self):
        checked = self.fixture.preflight()
        binding = checked["git_freeze"]
        self.assertEqual(binding["git_head"], self.fixture.git("rev-parse", "HEAD"))
        self.assertEqual(binding["manifest_path"], "FREEZE.json")
        self.assertEqual(binding["manifest_sha256"], runner.sha256(self.fixture.manifest_path))
        self.assertFalse(binding["git_dirty"])

    def test_untracked_freeze_is_refused_without_output(self):
        untracked = self.fixture.repo / "UNTRACKED.json"
        untracked.write_bytes(self.fixture.manifest_path.read_bytes())
        with self.assertRaisesRegex(runner.ConfirmationError, "tracked by Git"):
            runner.preflight(untracked, self.fixture.output,
                             repository_root=self.fixture.repo,
                             application_root=self.fixture.app, cache_root=self.fixture.cache)
        self.assertFalse(self.fixture.output.exists())

    def test_uncommitted_freeze_is_refused_without_output(self):
        self.fixture.manifest_path.write_bytes(self.fixture.manifest_path.read_bytes() + b" ")
        with self.assertRaisesRegex(runner.ConfirmationError, "byte-identical"):
            self.fixture.preflight()
        self.assertFalse(self.fixture.output.exists())

    def test_dirty_worktree_is_refused_without_output(self):
        path = self.fixture.repo / runner.RUNNER_RELATIVE
        path.write_text(path.read_text() + "dirty\n")
        with self.assertRaisesRegex(runner.ConfirmationError, "clean Git worktree"):
            self.fixture.preflight()
        self.assertFalse(self.fixture.output.exists())

    def test_runtime_versions_must_equal_the_frozen_development_versions(self):
        expected = copy.deepcopy(self.fixture.manifest["development_environment"])
        report = runner.validate_runtime_environment(self.fixture.manifest, actual=expected)
        self.assertTrue(report["passed"])
        for field, package in (("python", None), ("packages", "torch"),
                               ("packages", "transformers"), ("packages", "tokenizers"),
                               ("packages", "numpy")):
            with self.subTest(field=field, package=package):
                actual = copy.deepcopy(expected)
                if package is None:
                    actual[field] = "0.0.0"
                else:
                    actual[field][package] = "0.0.0"
                with self.assertRaisesRegex(runner.ConfirmationError, "runtime versions differ"):
                    runner.validate_runtime_environment(self.fixture.manifest, actual=actual)

    def test_consistent_anchor_tamper_is_rejected_by_split_b_binding(self):
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["anchors"]["T_A"] += 0.01
        manifest["anchors"]["d"] += 0.01
        self.fixture.write_manifest(manifest, commit=True)
        with self.assertRaisesRegex(runner.ConfirmationError, "split-B freeze"):
            self.fixture.preflight()
        self.assertFalse(self.fixture.output.exists())

    def test_full_logit_audit_converts_paper_entity_index_to_python_index(self):
        import numpy as np

        genres = [f"genre-{i}" for i in range(7)]
        answer_ids = {genre: i + 1 for i, genre in enumerate(genres)}
        matrix = [[f"musician-{i}", genre, f"instrument-{i}"]
                  for i, genre in enumerate(genres)]
        logits = np.linspace(-2.0, 2.0, 16, dtype="float32")
        logits64 = logits.astype("float64")
        top = float(logits64.max())
        lse = top + float(np.log(np.exp(logits64 - top).sum()))
        selected = [float(logits64[answer_ids[genre]]) for genre in genres]
        mass = sum(float(np.exp(value - lse)) for value in selected)
        records, arrays = [], {}
        for draw_index in runner.AUDIT_DRAW_INDICES:
            name = runner.case_id(draw_index)
            records.append({"case_id": name, "matrix": matrix,
                            "answer_logits": selected,
                            "answer_mass_full_vocab": mass,
                            "readout": {"logsumexp_full": lse}})
            arrays[name] = logits
        path = Path(self.temporary.name) / "audit.npz"
        np.savez_compressed(path, **arrays)

        audit = runner._full_logit_audit(
            path, records, answer_ids, self.fixture.manifest["t_entity"] - 1)
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["cases"], [runner.case_id(i) for i in range(4)])

    def test_analyzed_output_is_compatible_with_the_records_only_checker(self):
        freeze = json.loads((APP / runner.SPLIT_B_DECISION_RELATIVE).read_text())["split_B"]["freeze"]
        manifest = {
            "schema_version": 2, "application": "gur-arieh-2510.06182", "round": 1,
            "stage": "confirmation", "freeze_status": "FROZEN", "n_groups": 7,
            "t_entity": 2, "cell_key": "c4", "cell": copy.deepcopy(freeze["cell"]),
            "N": freeze["N"], "N_rule": copy.deepcopy(freeze["N_rule"]),
            "rule": {**ra.CONTRACT, "s_min": freeze["s_min"], "d_min": freeze["d_min"],
                     "agreement_transfer_floor": freeze["agreement_transfer_floor"]},
            "anchors": copy.deepcopy(freeze["anchors"]),
            "mean_gate": copy.deepcopy(freeze["mean_gate"]),
            "development_gates": copy.deepcopy(freeze["development_gates"]),
            "confirmation": {"authorized": True, "seed_base": runner.SEED_BASE,
                             "case_id_prefix": runner.CASE_ID_PREFIX,
                             "cap": 2 * freeze["N"]},
        }
        manifest["execution_contract"] = {
            "identity_tolerance": runner.IDENTITY_TOLERANCE,
            "gate7": {"tolerance_T": runner.GATE7_T_TOLERANCE},
        }
        manifest["entity_pools"] = {"answer_form_ids": {"target": 1}}
        code_paths = checker.REQUIRED_CODE
        manifest["code_files_sha256"] = {
            name: runner.sha256(ROOT / name) for name in sorted(code_paths)}
        manifest["data_sha256"] = {
            name: runner.sha256(APP / name) for name in (
                "SOURCE_LOCK.json", runner.SPLIT_A_DECISION_RELATIVE,
                runner.SPLIT_B_DECISION_RELATIVE)}
        records = worlds.confirmation_records(
            "heterogeneous", manifest["N"], random.Random(20260928), "fixture",
            cell=manifest["cell"])
        for index, record in enumerate(records):
            correct = record["qualifies"]
            record.update(
                case_id=runner.case_id(index), seed=runner.seed_for(index), cell_key="c4",
                native={role: {"answer": "target",
                               "first_word": "target" if correct else "other",
                               "generation_ids": [1],
                               "readout_argmax_entity": "target" if correct else "other",
                               "first_token_is_answer_form": True, "correct": correct,
                               "readout_matches_generation": True}
                        for role in ("recipient", "donor")})
            if "answer_logits" not in record:
                record.update(
                    design_indices=copy.deepcopy(manifest["cell"]),
                    technical={"passed": True},
                    answer_logits=worlds.logits(worlds.W_LIKE, 0.8, manifest["cell"]),
                    answer_mass_full_vocab=worlds.MASS,
                )
            record["identity"] = {"same_answer_argmax": True, "same_generation": True,
                                  "max_abs_answer_logit_difference": 0.0}

        references = []
        gate7_rows = []
        for record in records[:32]:
            references.append({
                "case_id": record["case_id"], "device": "cpu", "dtype": "torch.float32",
                "hook_ok": True,
                "conflict": {"answer_logits": copy.deepcopy(record["answer_logits"]),
                             "answer_mass_full_vocab": record["answer_mass_full_vocab"]},
            })
            gate7_rows.append({"case_id": record["case_id"], "abs_T_difference": 0.0,
                               "same_resolution": True, "same_labels": True,
                               "hook_ok_cpu": True, "cpu_device": "cpu",
                               "cpu_dtype": "torch.float32"})
        gates = {name: {"passed": True} for name in checker.BINDING_GATES}
        gates["4_hooks"]["full_logit_audit"] = {
            "checked": True, "passed": True,
            "cases": [runner.case_id(i) for i in runner.AUDIT_DRAW_INDICES],
        }
        gates["5_identity"].update(
            families=len(records), measured_families=len(records),
            same_argmax=len(records), same_generation=len(records),
            max_abs_answer_logit_difference=0.0, tolerance=runner.IDENTITY_TOLERANCE)
        gates["7_dtype_device"].update(
            declared_draw_indices=list(runner.GATE7_DRAW_INDICES), compared=32, missing=[],
            tolerance=runner.GATE7_T_TOLERANCE, max_abs_T_difference=0.0,
            rows=gate7_rows)
        confirmation_gates = {
            "status": "PROCEED", "stops": [], "cell_key": "c4",
            "families": len(records), "qualifying": manifest["N"], "gates": gates,
            "quota": {"passed": True, "N": manifest["N"],
                      "cap": manifest["confirmation"]["cap"],
                      "counts": {"c4": {"generated": len(records),
                                          "qualifying": manifest["N"]}},
                      "generator_reported_met": True},
        }

        results = Path(self.temporary.name) / "checker-compatible"
        results.mkdir()
        save(results / "manifest.json", manifest)
        save(results / "RUN_STARTED.json", {"manifest_sha256": runner.sha256(results / "manifest.json")})
        write_jsonl(results / "records.jsonl", records)
        save(results / "timings.json", {"fixture": True})
        save(results / "confirmation_gates.json", confirmation_gates)
        write_jsonl(results / "gate7_cpu_reference.jsonl", references)
        (results / "audit_full_logits.npz").write_bytes(b"no-model fixture")

        summary, report = runner.analyze_and_check(manifest, records, results,
                                                   repository_root=ROOT)
        self.assertTrue(report["verified"])
        self.assertEqual(checker.verify(results, repository_root=ROOT), report)
        self.assertEqual(summary["statuses"], report["statuses"])
        hashes = json.loads((results / "artifact_hashes.json").read_text())
        self.assertIsInstance(hashes, dict)
        self.assertTrue({"manifest.json", "records.jsonl", "summary.json",
                         "confirmation_gates.json", "checker_report.json"} <= set(hashes))


if __name__ == "__main__":
    unittest.main()
