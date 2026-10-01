"""Tamper tests for the Round 1 records-only checker. Constructed bundles, not evidence."""
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_lines(path, values):
    path.write_text("".join(json.dumps(v, allow_nan=False) + "\n" for v in values))


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def npy_float32(values):
    values = [float32(value) for value in values]
    header = repr({"descr": "<f4", "fortran_order": False, "shape": (len(values),)})
    padding = (-((10 + len(header) + 1) % 64)) % 64
    header_bytes = (header + " " * padding + "\n").encode("latin1")
    return (b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes))
            + header_bytes + struct.pack(f"<{len(values)}f", *values))


def write_audit_npz(path, arrays):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, values in arrays:
            archive.writestr(f"{name}.npy", npy_float32(values))


def rate(k, n):
    return {"k": k, "n": n, "rate": k / n if n else None}


def tally(pairs):
    values = {}
    for key, passed in pairs:
        row = values.setdefault(str(key), [0, 0])
        row[0] += bool(passed)
        row[1] += 1
    return {key: rate(*row) for key, row in sorted(values.items())}


class Bundle:
    """A results directory in a temporary repository with fixture code files."""

    def __init__(self, repo, world, real_code=False):
        self.repo = repo
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "fixture@example.invalid"],
                       cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Checker Fixture"],
                       cwd=repo, check=True)
        self.freeze_relative = Path(
            "applications/gur-arieh-2510.06182/FROZEN_CONFIRMATION.json")
        (repo / ".gitignore").write_text(
            "applications/gur-arieh-2510.06182/results/constructed/\n")
        self.results = repo / "applications/gur-arieh-2510.06182/results/constructed"
        self.results.mkdir(parents=True)
        code = {}
        for name in sorted(checker.REQUIRED_CODE):
            source = ROOT / name
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if real_code or name == checker.HELPER:
                shutil.copyfile(source, target)
            else:
                target.write_text("# constructed code-hash fixture\n" + name)
            code[name] = sha256(target)
        source_lock = repo / "applications/gur-arieh-2510.06182/SOURCE_LOCK.json"
        source_lock.parent.mkdir(parents=True, exist_ok=True)
        save(source_lock, {"fixture": True})
        self.manifest = copy.deepcopy(WORLD[world]["manifest"])
        self.manifest.update(stage="confirmation", freeze_status="FROZEN")
        self.manifest["code_files_sha256"] = code
        self.records = copy.deepcopy(WORLD[world]["records"])
        self.manifest["cell_key"] = "c4"
        self.manifest["confirmation"] = {
            "authorized": True,
            "seed_base": checker.CONFIRMATION_SEED_BASE,
            "case_id_prefix": "confirmation",
            "cap": 2 * self.manifest["N"],
        }
        self.manifest["execution_contract"] = {
            "identity_tolerance": 0.001,
            "gate7": {"tolerance_T": 0.01},
        }
        answer_ids = {f"target{i}": i for i in range(self.manifest["n_groups"])}
        self.manifest["entity_pools"] = {
            "answer_form_ids": answer_ids,
            "pools": {"Fixture": [f"item{i}" for i in range(self.manifest["n_groups"] + 2)]},
            "dropped": {},
        }
        self.manifest["model"] = {"files_sha256": {"fixture.safetensors": "0" * 64}}
        application_root = repo / "applications/gur-arieh-2510.06182"
        decision_a = application_root / checker.SPLIT_A_DECISION
        decision_b = application_root / checker.SPLIT_B_DECISION
        save(decision_a, {"status": "PROCEED", "stops": [], "selected": "c4"})
        save(decision_b, {"status": "PROCEED", "stops": [], "cell_key": "c4",
                          "split_B": {"freeze": {
                              "cell": copy.deepcopy(self.manifest["cell"]),
                              "N": self.manifest["N"], "N_rule": self.manifest["N_rule"],
                              "s_min": self.manifest["rule"]["s_min"],
                              "d_min": self.manifest["rule"]["d_min"],
                              "agreement_transfer_floor":
                                  self.manifest["rule"]["agreement_transfer_floor"],
                              "anchors": copy.deepcopy(self.manifest["anchors"]),
                              "mean_gate": copy.deepcopy(self.manifest["mean_gate"]),
                              "development_gates":
                                  copy.deepcopy(self.manifest["development_gates"]),
                          }}})
        self.manifest["data_sha256"] = {
            "SOURCE_LOCK.json": sha256(source_lock),
            checker.SPLIT_A_DECISION: sha256(decision_a),
            checker.SPLIT_B_DECISION: sha256(decision_b),
        }
        for i, record in enumerate(self.records):
            correct = record["qualifies"]
            matrix = [[f"person{j}", f"target{j}", f"object{j}"]
                      for j in range(self.manifest["n_groups"])]
            record.update(case_id=f"confirmation-{i:04d}",
                          seed=checker.CONFIRMATION_SEED_BASE + i,
                          cell_key="c4",
                          cell=copy.deepcopy(self.manifest["cell"]),
                          matrix=matrix,
                          agreement=[],
                          native={role: {"answer": "target0",
                                         "first_word": "target0" if correct else "other",
                                         "generation_ids": [0],
                                         "readout_argmax_entity":
                                             "target0" if correct else "other",
                                         "first_token_is_answer_form": True,
                                         "correct": correct,
                                         "readout_matches_generation": True}
                                  for role in ("recipient", "donor")},
                          technical={
                              "checks": {name: True for name in checker.REQUIRED_TECHNICAL_CHECKS},
                              "failures": [], "passed": True},
                          identity={"max_abs_answer_logit_difference": 0.0,
                                    "same_answer_argmax": True,
                                    "same_generation": True})
        # In a real run native failures may be nonqualifying while their patch readout remains
        # measurable.  The synthetic worlds omit that unused payload, so add a neutral fixture
        # payload for the frozen first-32 gate-7 cases without changing qualification.
        exemplar = next(record for record in self.records if record["qualifies"])
        for record in self.records:
            if "answer_logits" not in record:
                record.update(design_indices=copy.deepcopy(self.manifest["cell"]),
                              answer_logits=copy.deepcopy(exemplar["answer_logits"]),
                              answer_mass_full_vocab=exemplar["answer_mass_full_vocab"])
            record["design_indices"] = copy.deepcopy(self.manifest["cell"])
        self.audit_arrays = []
        for record in self.records[:len(checker.AUDIT_DRAW_INDICES)]:
            answer = [float32(value) for value in record["answer_logits"]]
            mass = record["answer_mass_full_vocab"]
            top = max(answer)
            answer_lse = top + math.log(math.fsum(math.exp(value - top) for value in answer))
            complement = float32(answer_lse + math.log((1 - mass) / mass))
            full = answer + [complement]
            full_top = max(full)
            full_lse = full_top + math.log(
                math.fsum(math.exp(value - full_top) for value in full))
            audited_mass = math.fsum(math.exp(value - full_lse) for value in answer)
            record["answer_logits"] = answer
            record["answer_mass_full_vocab"] = audited_mass
            record["readout"] = {"answer_logits": copy.deepcopy(answer),
                                 "answer_mass_full_vocab": audited_mass,
                                 "logsumexp_full": full_lse}
            self.audit_arrays.append((record["case_id"], full))
        self.write_all()

    def commit_manifest(self):
        frozen = self.repo / self.freeze_relative
        save(frozen, self.manifest)
        has_head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=self.repo,
            capture_output=True, check=False).returncode == 0
        subprocess.run(["git", "add", str(self.freeze_relative)] if has_head
                       else ["git", "add", "-A"], cwd=self.repo, check=True)
        changed = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=self.repo,
            check=False).returncode == 1
        if changed:
            subprocess.run(["git", "commit", "-q", "-m", "freeze fixture manifest"],
                           cwd=self.repo, check=True)
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True,
            text=True, check=True).stdout.strip()

    def write_all(self, summary=None):
        save(self.results / "manifest.json", self.manifest)
        git_head = self.commit_manifest()
        manifest_hash = sha256(self.results / "manifest.json")
        save(self.results / "RUN_STARTED.json", {
            "manifest_sha256": manifest_hash,
            "git_head": git_head,
            "manifest_path": self.freeze_relative.as_posix(),
            "git_dirty": False,
            "seed_base": checker.CONFIRMATION_SEED_BASE,
            "case_id_prefix": "confirmation",
        })
        write_lines(self.results / "records.jsonl", self.records)
        save(self.results / "timings.json", {"fixture": True})
        write_audit_npz(self.results / "audit_full_logits.npz", self.audit_arrays)
        references = [{
            "case_id": record["case_id"], "device": "cpu",
            "dtype": "torch.float32", "hook_ok": True,
            "conflict": {"answer_logits": record["answer_logits"],
                         "answer_mass_full_vocab": record["answer_mass_full_vocab"]},
        } for record in self.records[:32]]
        write_lines(self.results / "gate7_cpu_reference.jsonl", references)

        n_records = len(self.records)
        qualifying = [record for record in self.records if record["qualifies"]]
        native = [(record, role, record["native"][role]) for record in self.records
                  for role in ("recipient", "donor")]
        overall = rate(len(qualifying), n_records)
        gate2 = {
            "passed": overall["rate"] >= checker.YIELD_FLOOR,
            "yield": overall, "floor": checker.YIELD_FLOOR,
            "recipient_correct_by_i_N": tally(
                (record["cell"]["i_N"], row["correct"])
                for record, role, row in native if role == "recipient"),
            "donor_correct_by_queried_position": tally(
                (record["cell"]["i_P"], row["correct"])
                for record, role, row in native if role == "donor"),
            "first_token_is_answer_form": tally(
                (role, row["first_token_is_answer_form"]) for _, role, row in native),
            "readout_matches_generation": tally(
                (role, row["readout_matches_generation"]) for _, role, row in native),
            "agreement_donor_correct_by_target": {},
        }
        hook_counts = {name: n_records for name in checker.REQUIRED_TECHNICAL_CHECKS
                       if name.endswith("hook")}
        audit = {
            "checked": True,
            "expected_cases": [f"confirmation-{i:04d}" for i in checker.AUDIT_DRAW_INDICES],
            "tolerances": checker.AUDIT_TOLERANCES,
            "cases": [f"confirmation-{i:04d}" for i in checker.AUDIT_DRAW_INDICES],
            "max_abs_logsumexp_difference": 0.0,
            "max_abs_answer_mass_difference": 0.0,
            "max_abs_answer_logit_difference": 0.0,
            "passed": True,
        }
        gate7_rows = [{
            "draw_index": i, "case_id": record["case_id"],
            "abs_T_difference": 0.0, "same_resolution": True, "same_labels": True,
            "max_abs_answer_logit_difference": 0.0, "hook_ok_cpu": True,
            "cpu_device": "cpu", "cpu_dtype": "torch.float32",
        } for i, record in enumerate(self.records[:32])]
        pools = self.manifest["entity_pools"]
        save(self.results / "confirmation_gates.json", {
            "status": "PROCEED", "stops": [], "cell_key": "c4",
            "families": n_records, "qualifying": self.manifest["N"],
            "gates": {
                "1_model_hashes": {"passed": True, "verified_counts": {
                    "code": len(self.manifest["code_files_sha256"]),
                    "data": len(self.manifest["data_sha256"]),
                    "model": len(self.manifest["model"]["files_sha256"])}},
                "2_native_and_yield": gate2,
                "3_tokens": {"pools_equal_the_manifest": True,
                    "pools_equal_the_lock": True,
                    "pools_kept": {key: len(value) for key, value in pools["pools"].items()},
                    "pools_dropped": pools["dropped"], "dropped_prefix": "<bos>",
                    "passed": True, "alignment_failures": []},
                "4_hooks": {"passed": True, "checks": hook_counts, "failed": [],
                    "design_index_failures": [], "technical_failures": [],
                    "full_logit_audit": audit},
                "5_identity": {"passed": True, "families": n_records,
                    "measured_families": n_records, "same_argmax": n_records,
                    "same_generation": n_records,
                    "max_abs_answer_logit_difference": 0.0, "tolerance": 0.001},
                "7_dtype_device": {"passed": True,
                    "comparison": "MPS float32/eager against CPU float32/eager, conflict patch",
                    "declared_draw_indices": list(range(32)), "compared": 32,
                    "missing": [], "max_abs_T_difference": 0.0, "tolerance": 0.01,
                    "execution": {"main_device": "mps", "main_dtype": "torch.float32",
                                  "attention": "eager"},
                    "rows": gate7_rows},
            },
            "quota": {"passed": True, "N": self.manifest["N"],
                      "cap": 2 * self.manifest["N"],
                      "counts": {"c4": {"generated": n_records,
                                          "qualifying": self.manifest["N"]}},
                      "generator_reported_met": True},
        })
        if summary is None:
            summary = ra.analyze_confirmation(self.manifest, self.records)
        summary.update(manifest_sha256=manifest_hash,
                       records_sha256=sha256(self.results / "records.jsonl"),
                       confirmation_gates_sha256=sha256(
                           self.results / "confirmation_gates.json"),
                       timings_sha256=sha256(self.results / "timings.json"))
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

    def test_consistent_posthoc_anchor_change_is_rejected_by_split_b_binding(self):
        self.bundle.manifest["anchors"]["T_A"] += 0.1
        self.bundle.manifest["anchors"]["d"] += 0.1
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError, "frozen split-B values"):
            self.bundle.verify()

    def test_frozen_agreement_resolution_below_floor_is_rejected(self):
        self.bundle.manifest["development_gates"]["agreement_resolution_rate_B"] = 0.85
        summary = copy.deepcopy(self.bundle.summary)
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError,
                                    "frozen split-B values|agreement-control resolution"):
            self.bundle.verify()

    def test_manifest_changed_after_start_is_rejected(self):
        self.bundle.manifest["N"] = 199
        save(self.bundle.results / "manifest.json", self.bundle.manifest)
        self.bundle.rehash()
        with self.assertRaisesRegex(checker.VerificationError, "freeze hash"):
            self.bundle.verify()

    def test_run_started_rejects_false_git_claims(self):
        path = self.bundle.results / "RUN_STARTED.json"
        original = json.loads(path.read_text())
        attacks = (
            ({**original, "git_dirty": True}, "clean preflight"),
            ({**original, "git_head": "0" * 40}, "git_head"),
            ({**original, "manifest_path": "../../fabricated.json"}, "manifest_path"),
        )
        for started, message in attacks:
            with self.subTest(message=message):
                save(path, started)
                self.bundle.rehash()
                with self.assertRaisesRegex(checker.VerificationError, message):
                    self.bundle.verify()
        save(path, original)
        self.bundle.rehash()

    def test_run_started_commit_must_contain_the_result_manifest(self):
        frozen = self.repo / self.bundle.freeze_relative
        frozen.write_text(frozen.read_text() + "\n")
        with self.assertRaisesRegex(checker.VerificationError, "current repository manifest"):
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
            r.update(draw_index=i, case_id=f"confirmation-{i:04d}",
                     seed=checker.CONFIRMATION_SEED_BASE + i)
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "frozen N"):
            self.bundle.verify()

    def test_changed_seed_is_rejected_even_when_rehashed(self):
        self.bundle.records[10]["seed"] += 1
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError, "seed differs"):
            self.bundle.verify()

    def test_qualification_bit_is_recomputed_from_native_observations(self):
        self.bundle.records[0]["qualifies"] = False
        self.bundle.write_all(summary=copy.deepcopy(self.bundle.summary))
        with self.assertRaisesRegex(checker.VerificationError, "qualifies differs"):
            self.bundle.verify()

    def test_native_correctness_must_match_first_word_and_token(self):
        self.bundle.records[0]["native"]["recipient"]["first_token_is_answer_form"] = False
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError,
                                    "first-token claim|correctness is inconsistent"):
            self.bundle.verify()

    def test_boolean_seed_is_rejected(self):
        self.bundle.records[0]["seed"] = True
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError, "seed differs"):
            self.bundle.verify()

    def test_float_draw_index_is_rejected(self):
        self.bundle.records[0]["draw_index"] = 0.0
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError, "omit, duplicate or reorder"):
            self.bundle.verify()

    def test_changed_case_id_formula_is_rejected_even_when_unique(self):
        self.bundle.records[10]["case_id"] = "confirmation-x010"
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError, "case_id differs"):
            self.bundle.verify()

    def test_changed_confirmation_seed_base_is_rejected(self):
        self.bundle.manifest["confirmation"]["seed_base"] = 4_100_000
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError, "seed base"):
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
        first["answer_logits"] = worlds.logits(worlds.W_LIKE, 0.8)
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "differ"):
            self.bundle.verify()

    def test_edited_answer_mass_is_rejected_even_when_rehashed(self):
        summary = copy.deepcopy(self.bundle.summary)
        for record in [r for r in self.bundle.records if r["qualifies"]][:20]:
            record["answer_mass_full_vocab"] = 0.1
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "differ"):
            self.bundle.verify()

    def test_missing_answer_mass_is_a_technical_failure(self):
        record = [r for r in self.bundle.records if r["qualifies"]][40]
        del record["answer_mass_full_vocab"]
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError, "raw technical|run status"):
            self.bundle.verify()

    def test_paper_readout_changes_no_verified_status(self):
        before = self.bundle.verify()
        for record in self.bundle.records:
            if record["qualifies"]:
                record["descriptive"]["paper_readout"]["entity_logits"] = [0.0] * 7
                record["descriptive"]["paper_readout"]["entity_mass_full_vocab"] = 0.99
                record["entity_logits"] = [5.0, -5.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        self.bundle.write_all(summary=copy.deepcopy(self.bundle.summary))
        after = self.bundle.verify()
        self.assertEqual(after, before)

    def test_frozen_N_off_the_rule_is_rejected_even_when_rehashed(self):
        self.bundle.manifest["development_gates"]["resolution_rate_B"] = 0.95
        summary = copy.deepcopy(self.bundle.summary)
        self.bundle.write_all(summary=summary)
        with self.assertRaisesRegex(checker.VerificationError, "frozen split-B values|N rule"):
            self.bundle.verify()

    def test_checker_n_rule_matches_the_analyzer(self):
        for u in (0.0, 0.01, 0.02, 0.025, 0.038, 0.05, 0.08, 0.2):
            expected = ra.n_rule(u)
            N, powered, status = checker.rule_for(u)
            with self.subTest(u=u):
                self.assertEqual((N, powered, status),
                                 (expected["N"], expected["adequacy_powered"], expected["status"]))

    def test_unrehashed_change_is_caught_first(self):
        with (self.bundle.results / "records.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(checker.VerificationError, "hash mismatch"):
            self.bundle.verify()

    def test_unindexed_extra_artifact_is_rejected(self):
        (self.bundle.results / "untracked.txt").write_text("not in artifact_hashes.json")
        with self.assertRaisesRegex(checker.VerificationError, "exactly cover"):
            self.bundle.verify()

    def test_failed_binding_gate_is_rejected_even_when_rehashed(self):
        table = json.loads((self.bundle.results / "confirmation_gates.json").read_text())
        table["gates"]["7_dtype_device"]["passed"] = False
        save(self.bundle.results / "confirmation_gates.json", table)
        summary = copy.deepcopy(self.bundle.summary)
        summary["confirmation_gates_sha256"] = sha256(
            self.bundle.results / "confirmation_gates.json")
        self.bundle.save_summary(summary)
        with self.assertRaisesRegex(checker.VerificationError, "7_dtype_device"):
            self.bundle.verify()

    def test_gate7_raw_reference_is_replayed(self):
        rows = [json.loads(line) for line in
                (self.bundle.results / "gate7_cpu_reference.jsonl").read_text().splitlines()]
        rows[0]["conflict"]["answer_logits"] = [20.0, -20.0, -20.0, -20.0,
                                                   -20.0, -20.0, -20.0]
        write_lines(self.bundle.results / "gate7_cpu_reference.jsonl", rows)
        self.bundle.rehash()
        with self.assertRaisesRegex(checker.VerificationError, "gate-7 .*replay|gate-7 table"):
            self.bundle.verify()

    def test_contradictory_raw_technical_evidence_is_rejected(self):
        technical = self.bundle.records[0]["technical"]
        technical["checks"]["conflict_hook"] = False
        technical["failures"] = ["conflict_hook: fabricated failure"]
        technical["passed"] = True
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError,
                                    "technical passed/check/failure evidence"):
            self.bundle.verify()

    def test_contradictory_gate_detail_is_rejected(self):
        path = self.bundle.results / "confirmation_gates.json"
        table = json.loads(path.read_text())
        table["gates"]["4_hooks"]["failed"] = [["confirmation-0000", "conflict_hook"]]
        save(path, table)
        summary = copy.deepcopy(self.bundle.summary)
        summary["confirmation_gates_sha256"] = sha256(path)
        self.bundle.save_summary(summary)
        with self.assertRaisesRegex(checker.VerificationError, "gate-4 evidence contradicts"):
            self.bundle.verify()

    def test_arbitrary_full_logit_audit_is_rejected_even_when_rehashed(self):
        (self.bundle.results / "audit_full_logits.npz").write_bytes(b"not an npz archive")
        self.bundle.rehash()
        with self.assertRaisesRegex(checker.VerificationError, "full-logit audit NPZ"):
            self.bundle.verify()

    def test_altered_full_logit_payload_is_recomputed(self):
        arrays = copy.deepcopy(self.bundle.audit_arrays)
        arrays[0][1][-1] += 10.0
        write_audit_npz(self.bundle.results / "audit_full_logits.npz", arrays)
        self.bundle.rehash()
        with self.assertRaisesRegex(checker.VerificationError, "independent full-logit audit"):
            self.bundle.verify()

    def test_checker_report_is_optional_then_verified_as_a_fixed_point(self):
        report = self.bundle.verify()
        save(self.bundle.results / "checker_report.json", report)
        self.bundle.rehash()
        self.assertEqual(self.bundle.verify(), report)
        fabricated = {**report, "verified": False, "level": "FABRICATED"}
        save(self.bundle.results / "checker_report.json", fabricated)
        self.bundle.rehash()
        with self.assertRaisesRegex(checker.VerificationError, "checker_report.json differs"):
            self.bundle.verify()

    def test_raw_identity_failure_is_rejected_even_when_rehashed(self):
        self.bundle.records[40]["identity"]["same_generation"] = False
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError, "raw identity"):
            self.bundle.verify()

    def test_changed_frozen_code_is_rejected(self):
        path = self.repo / "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py"
        path.write_text(path.read_text() + "\n# changed")
        with self.assertRaisesRegex(checker.VerificationError, "hash mismatch"):
            self.bundle.verify()

    def test_technical_failure_must_be_reported_as_invalid(self):
        record = [r for r in self.bundle.records if r["qualifies"]][40]
        record["technical"] = {"passed": False}
        self.bundle.write_all()
        with self.assertRaisesRegex(checker.VerificationError, "raw technical"):
            self.bundle.verify()
        claimed_valid = copy.deepcopy(WORLD["2_heterogeneous_concentrated"]["summary"])
        claimed_valid.update(manifest_sha256=self.bundle.summary["manifest_sha256"],
                             records_sha256=self.bundle.summary["records_sha256"],
                             confirmation_gates_sha256=self.bundle.summary[
                                 "confirmation_gates_sha256"],
                             timings_sha256=self.bundle.summary["timings_sha256"])
        self.bundle.save_summary(claimed_valid)
        with self.assertRaisesRegex(checker.VerificationError, "raw technical|run status"):
            self.bundle.verify()

    def test_command_line_runs_with_the_standard_library_only(self):
        with tempfile.TemporaryDirectory() as other:
            bundle = Bundle(Path(other), "2_heterogeneous_concentrated", real_code=True)
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(CHECKER), "--results", str(bundle.results),
                 "--repository-root", str(bundle.repo)],
                # The executable checker is the real one; its frozen files live in this
                # deliberately isolated fixture repository.
                capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
