"""Round 1 runner tests (protocol v2) on a tiny, randomly initialised Gemma 2 built locally.

No Gemma file is read and nothing is downloaded. The tests check the patch mechanics
(hidden_states[l] is the input of block l; the hook fires once per forward and writes
only the last prompt position; the identity self-patch reproduces the logits and the
greedy generation), the primary answer-form readout and the descriptive paper readout,
the protocol-v2 pool rule, the design and alignment checks inside a family, the snapshot
hashing, the pilot summary and the artifact index, and that runner records validate
against the records-only checker. They need torch and transformers
(requirements-model.txt); without them they are skipped and say why.
"""
import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
for path in (APP / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import tokenizers  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_TORCH = False

REASON = "runner tests need torch, transformers and tokenizers (requirements-model.txt)"
CELL = {"i_P": 3, "i_L": 1, "i_R": 5, "i_N": 0}
N_GROUPS = 7
LAYER, DIAGNOSTIC_LAYER = 2, 3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeTokenizer:
    """encode() from a fixed table, to exercise every branch of the pool rule."""

    TABLE = {" a": [1], "A": [11], " b": [2], "B": [12, 13], " c": [3], "C": [14], " d": [4], "D": [14],
             " e": [5, 6], "E": [15], " f": [7], "F": [16], " g": [7], "G": [17],
             " x": [30], "X": [40, 41], " y": [31], " z": [32]}

    def encode(self, text, add_special_tokens=False):
        return self.TABLE[text]


@unittest.skipUnless(HAVE_TORCH, REASON)
class PoolRuleTests(unittest.TestCase):
    def test_protocol_v2_pool_rule(self):
        import mixing_runner as mr
        spec = {"categories": ["Key", "Target", "Other"],
                "items": {"Key": ["x", "y"], "Target": ["a", "b", "c", "d", "e", "f", "g"], "Other": ["z"]}}
        pools, dropped, context_ids, answer_ids = mr.round1_pools(FakeTokenizer(), spec, target=1)
        self.assertEqual(pools["Target"], ["a"])
        reasons = {d["entity"]: d["reason"] for d in dropped["Target"]}
        self.assertEqual(reasons, {"b": "answer form not one token", "e": "context form not one token",
                                   "c": "token id not unique", "d": "token id not unique",
                                   "f": "token id not unique", "g": "token id not unique"})
        # The answer form applies to the answered category only.
        self.assertEqual(pools["Key"], ["x", "y"])
        self.assertEqual(answer_ids, {"a": 11})
        record = mr.pools_record(pools, dropped, context_ids, answer_ids, spec, target=1)
        self.assertEqual(record["answer_form_ids"], {"a": 11})
        self.assertEqual(record["context_form_ids"]["Key"], {"x": 30, "y": 31})

    def test_the_locked_pools_follow_the_rule(self):
        lock = json.loads((APP / "SOURCE_LOCK.json").read_text())["round1_entity_pools"]
        self.assertEqual(lock["answered_category"], "Genre")
        self.assertNotIn("trance", lock["pools"]["Genre"])
        self.assertEqual(len(lock["pools"]["Genre"]), 23)
        self.assertEqual(sorted(lock["answer_form_ids"]), sorted(lock["pools"]["Genre"]))
        for form in (lock["answer_form_ids"], lock["context_form_ids"]["Genre"]):
            self.assertEqual(len(set(form.values())), len(form))


@unittest.skipUnless(HAVE_TORCH, REASON)
class TinyGemmaRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import mixing_runner as mr
        from mixing_tiny_model import TINY_SPEC, tiny_model, tiny_tokenizer
        cls.torch, cls.mr = torch, mr
        cls.spec = TINY_SPEC
        cls.tokenizer = tiny_tokenizer()
        cls.model = tiny_model(len(cls.tokenizer), seed=1)
        pools, dropped, context_ids, answer_ids = mr.round1_pools(cls.tokenizer, cls.spec)
        cls.pools, cls.context_ids, cls.answer_ids = pools, context_ids, answer_ids
        cls.runner = mr.Runner(cls.model, cls.tokenizer, cls.spec, pools, context_ids, answer_ids, LAYER,
                               diagnostic_layer=DIAGNOSTIC_LAYER)
        from mixing_round1_design import random_matrix
        cls.G = random_matrix(N_GROUPS, [pools[c] for c in cls.spec["categories"]], random.Random(1000000))

    def prompt_ids(self, query_group=0):
        return self.runner.prompt(self.G, query_group)["input_ids"]

    def test_tiny_pools_keep_every_entity(self):
        self.assertEqual(self.pools, self.spec["items"])
        self.assertEqual(len(set(self.answer_ids.values())), len(self.answer_ids))

    def test_hidden_states_index_is_the_input_of_that_block(self):
        torch = self.torch
        captured = {}

        def grab(module, args, kwargs):
            captured["input"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

        ids = self.prompt_ids()
        handle = self.model.model.layers[LAYER].register_forward_pre_hook(grab, with_kwargs=True)
        try:
            with torch.no_grad():
                out = self.model(ids, output_hidden_states=True, use_cache=False)
        finally:
            handle.remove()
        self.assertEqual(len(out.hidden_states), self.model.config.num_hidden_layers + 1)
        self.assertTrue(torch.equal(captured["input"], out.hidden_states[LAYER]))

    def test_identity_self_patch_reproduces_logits_and_generation(self):
        ids = self.prompt_ids()
        base, _ = self.runner.forward(ids)
        vector = base.hidden_states[LAYER][0, -1]
        patched, report = self.runner.forward(ids, patch=(LAYER, vector))
        self.assertLessEqual(float((patched.logits - base.logits).abs().max()), 1e-6)
        native, _ = self.runner.generate(ids)
        identity, gen_report = self.runner.generate(ids, patch=(LAYER, vector))
        self.assertEqual(native["token_ids"], identity["token_ids"])
        self.assertEqual(report["calls"], 1)
        self.assertEqual(gen_report["writes"], 1)
        self.assertEqual(gen_report["calls"], self.spec["max_new_tokens"])

    def test_hook_fires_once_per_forward_and_writes_only_the_last_position(self):
        torch = self.torch
        ids = self.prompt_ids(query_group=0)
        donor, _ = self.runner.forward(self.prompt_ids(query_group=4))
        base, _ = self.runner.forward(ids)
        vector = donor.hidden_states[LAYER][0, -1]
        patched, report = self.runner.forward(ids, patch=(LAYER, vector))
        T, D = ids.shape[1], self.model.config.hidden_size
        self.assertEqual((report["calls"], report["writes"]), (1, 1))
        self.assertEqual(report["positions"], [T - 1])
        self.assertEqual(report["shapes"], [[1, T, D]])
        self.assertEqual(report["vector_shape"], [D])
        self.assertTrue(self.mr.Runner.check_hook(report, T, D))
        self.assertTrue(torch.equal(patched.logits[0, :-1], base.logits[0, :-1]))
        self.assertGreater(float((patched.logits[0, -1] - base.logits[0, -1]).abs().max()), 1e-4)
        # hidden_states[LAYER] is recorded before the hook, so it stays the recipient's own.
        self.assertTrue(torch.equal(patched.hidden_states[LAYER], base.hidden_states[LAYER]))
        captured = {}

        def grab(module, args, kwargs):
            captured["x"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

        block = self.model.model.layers[LAYER]
        with self.mr.LastTokenPatch(block, vector, T):
            handle = block.register_forward_pre_hook(grab, with_kwargs=True)
            try:
                with torch.no_grad():
                    self.model(ids, use_cache=False)
            finally:
                handle.remove()
        self.assertTrue(torch.equal(captured["x"][0, -1], vector))
        self.assertTrue(torch.equal(captured["x"][0, :-1], base.hidden_states[LAYER][0, :-1]))

    def test_no_hook_remains_after_a_patched_run(self):
        ids = self.prompt_ids()
        base, _ = self.runner.forward(ids)
        self.runner.forward(ids, patch=(LAYER, base.hidden_states[LAYER][0, -1] * 3))
        self.assertEqual(len(self.model.model.layers[LAYER]._forward_pre_hooks), 0)

    def test_primary_and_paper_readouts(self):
        torch = self.torch
        ids = self.prompt_ids()
        context, positions = self.runner.aligned_entity_ids(self.G, ids)
        out, _ = self.runner.forward(ids)
        entities = [g[1] for g in self.G]
        readout = self.runner.readout(out, entities, full=True)
        logits = readout.pop("_full_logits")
        probabilities = torch.softmax(logits, 0)
        answer = [self.answer_ids[e] for e in entities]
        self.assertAlmostEqual(readout["answer_mass_full_vocab"], float(probabilities[answer].sum()), places=5)
        self.assertEqual(readout["answer_logits"], [float(logits[i]) for i in answer])
        paper = readout["descriptive"]["paper_readout"]
        self.assertEqual(paper["entity_logits"], [float(logits[i]) for i in context])
        self.assertAlmostEqual(paper["entity_mass_full_vocab"], float(probabilities[context].sum()), places=5)
        self.assertAlmostEqual(math.log(math.exp(readout["logsumexp_answer_tokens"])
                                        + math.exp(readout["logsumexp_answer_complement"])),
                               readout["logsumexp_full"], places=4)
        self.assertNotEqual(answer, context)
        tokens = ids[0].tolist()
        self.assertEqual([tokens[i] for i in positions], context)

    def test_alignment_mismatch_is_a_technical_failure(self):
        ids = self.prompt_ids()
        swapped = [list(g) for g in self.G]
        swapped[0][1], swapped[1][1] = swapped[1][1], swapped[0][1]
        with self.assertRaises(self.mr.TechnicalError):
            self.runner.aligned_entity_ids([tuple(g) for g in swapped], ids)

    def run_families(self, count, cells=None, prefix="tiny"):
        records = []
        for i in range(count):
            key, cell = cells[i % len(cells)] if cells else ("c1", CELL)
            record, _ = self.runner.family(case_id=f"{prefix}-{i:04d}", draw_index=i, seed=1000000 + i,
                                           cell_key=key, cell=cell, rng=random.Random(1000000 + i),
                                           n=N_GROUPS, audit=(i == 0))
            records.append(record)
        return records

    def test_family_runs_design_checks_hooks_and_controls(self):
        records = self.run_families(3)
        for record in records:
            self.assertTrue(record["technical"]["passed"], record["technical"]["failures"])
            self.assertEqual(record["design_indices"], CELL)
            self.assertEqual(len(record["answer_logits"]), N_GROUPS)
            self.assertNotIn("entity_logits", record)
            self.assertEqual(len(record["descriptive"]["paper_readout"]["entity_logits"]), N_GROUPS)
            self.assertEqual([a["j"] for a in record["agreement"]], [3, 1, 5])
            for a in record["agreement"]:
                i = a["design_indices"]
                self.assertEqual((i["i_P"], i["i_L"], i["i_R"], i["i_N"]), (a["j"],) * 3 + (0,))
                self.assertEqual((a["hook"]["calls"], a["hook"]["writes"]), (1, 1))
                self.assertIn("answer_mass_full_vocab", a)
            self.assertEqual(record["donor_answer"]["recipient_index"], CELL["i_R"])
            self.assertEqual(record["identity"]["max_abs_logit_difference"], 0.0)
            self.assertTrue(record["identity"]["same_generation"])
            self.assertEqual(record["diagnostic"]["layer"], DIAGNOSTIC_LAYER)
            recipient = record["native"]["recipient"]
            self.assertEqual(recipient["correct"],
                             recipient["first_word"] == recipient["answer"] and recipient["first_token_is_answer_form"])
            json.dumps(record, allow_nan=False)
        again = self.run_families(1)[0]
        self.assertEqual(again["matrix"], records[0]["matrix"])
        self.assertEqual(again["answer_logits"], records[0]["answer_logits"])

    def test_records_validate_against_the_records_only_checker(self):
        checker = load("check_mixing_round1_records", APP / "scripts/check_mixing_round1_records.py")
        import mixing_round1_analysis as ra
        records = self.run_families(4)
        for record in records:
            self.assertIsNone(checker.technical_failure(record, CELL, N_GROUPS))
        # Schema check only: a random tiny model is not natively correct, so copies of the
        # four records are marked qualifying and repeated to the rule's N of 200.
        N = 200
        qualifying = []
        for i in range(N):
            r = copy.deepcopy(records[i % 4])
            r.update(case_id=f"confirmation-{i:04d}", draw_index=i,
                     seed=4_000_000 + i, cell_key="c4", qualifies=True)
            qualifying.append(r)
        sizing = ra.n_rule(0.0)
        manifest = {
            "schema_version": 2, "application": "gur-arieh-2510.06182", "round": 1,
            "stage": "schema test", "n_groups": N_GROUPS, "cell_key": "c4",
            "cell": dict(CELL), "N": N,
            "confirmation": {"seed_base": 4_000_000, "case_id_prefix": "confirmation", "cap": 2 * N},
            "N_rule": {"N": sizing["N"], "adequacy_powered": sizing["adequacy_powered"]},
            "rule": {**ra.CONTRACT, "s_min": 0.05, "d_min": 0.20, "agreement_transfer_floor": 0.9},
            "anchors": {"T_W": 0.5, "T_A": 0.95, "d": 0.45, "q_bar_B": [0.5, 0.3, 0.2], "m_B": 100},
            "mean_gate": {"delta": 1.0}, "development_gates": {
                "resolution_rate_B": 1.0, "agreement_resolution_rate_B": 0.95, "agreement_transfer_rate_B": 0.95},
            "code_files_sha256": {name: sha(ROOT / name) for name in checker.REQUIRED_CODE},
        }
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (results / "RUN_STARTED.json").write_text(json.dumps({"manifest_sha256": sha(results / "manifest.json")}))
            (results / "records.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in qualifying))
            summary = ra.analyze_confirmation(manifest, qualifying)
            summary.update(manifest_sha256=sha(results / "manifest.json"),
                           records_sha256=sha(results / "records.jsonl"))
            (results / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            (results / "artifact_hashes.json").write_text(json.dumps({p.name: sha(p) for p in results.iterdir()}))
            report = checker.verify(results, repository_root=ROOT)
        self.assertTrue(report["verified"])
        self.assertEqual(report["N"], N)

    def test_pilot_summary_and_index_run_on_runner_records(self):
        """Smoke test of the pilot-2 gate table and the artifact index on tiny-model
        records. Records are marked qualifying so every branch runs; the numbers mean
        nothing."""
        import mixing_pilot_summary as summary_module
        script = load("run_mixing_pilot", APP / "scripts/run_mixing_pilot.py")
        pilot = copy.deepcopy(script.PILOT)
        pilot["gate7_reference"]["families"] = [0, 1, 2, 3]
        records = [dict(r, qualifies=True) for r in self.run_families(8, pilot["cells"], prefix="pilot2")]
        reference = [self.runner.conflict_only(r) for r in records[:4]]
        pools = self.mr.pools_record(self.pools, {}, self.context_ids, self.answer_ids, self.spec)
        manifest = {"model": {"hashes": {"passed": True, "problems": [], "files": {}}},
                    "gate3_tokens": {"passed": True, "pools_equal_the_lock": True, "pools_kept": {},
                                     "pools_dropped": {}, "dropped_prefix": "<bos>"},
                    "entity_pools": json.loads(json.dumps(pools)), "environment": {"device": "cpu"}}
        full = self.run_families(1, pilot["cells"], prefix="pilot2")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "manifest.json").write_text(json.dumps(manifest))
            (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            (out / "gate7_cpu_reference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reference))
            _, logits = self.runner.family(case_id="pilot2-0000", draw_index=0, seed=1000000, cell_key="c1",
                                           cell=pilot["cells"][0][1], rng=random.Random(1000000), n=N_GROUPS,
                                           audit=True)
            import numpy as np
            np.savez_compressed(out / "audit_full_logits.npz", **{"pilot2-0000": logits.numpy()})
            result = summary_module.summarize(out, pilot)
            index = script.write_index(out)
            self.assertEqual(set(index["data"]), {p.name for p in out.iterdir()} - {"artifact_hashes.json"})
            self.assertEqual(set(index["producers"]), set(script.PRODUCERS))
        self.assertEqual(full[0]["matrix"], records[0]["matrix"])
        self.assertEqual(result["families"], 8)
        for gate in ("1_model_hashes", "2_native_and_yield", "3_tokens", "4_hooks", "5_identity",
                     "6_agreement", "7_dtype_device", "8_support", "9_separation_descriptive"):
            self.assertIn(gate, result["gates"])
        self.assertTrue(result["gates"]["4_hooks"]["passed"])
        self.assertTrue(result["gates"]["5_identity"]["passed"])
        self.assertEqual(result["gates"]["7_dtype_device"]["max_abs_T_difference"], 0.0)
        self.assertEqual(result["gates"]["6_agreement"]["runs"], 24)
        self.assertTrue(result["audit_full_logits"]["passed"])
        self.assertIsNotNone(result["planning_N"])
        self.assertEqual(set(result["answer_mass_by_run_type"]), set(summary_module.RUN_TYPES))
        json.dumps(result, allow_nan=False)


    def test_split_decisions_run_on_runner_records(self):
        """Smoke test of the split-A selection and the split-B values on tiny-model
        records marked qualifying; the numbers mean nothing, the structure and the rules
        are checked."""
        import mixing_round1_analysis as ra
        import mixing_splits
        script = load("run_mixing_split", APP / "scripts/run_mixing_split.py")
        pools = json.loads(json.dumps(self.mr.pools_record(self.pools, {}, self.context_ids, self.answer_ids, self.spec)))
        manifest = {"model": {"hashes": {"passed": True, "problems": [], "files": {}}},
                    "gate3_tokens": {"passed": True, "pools_equal_the_lock": True, "pools_kept": {},
                                     "pools_dropped": {}, "dropped_prefix": "<bos>"},
                    "entity_pools": pools, "environment": {"device": "cpu"}}

        def write(directory, records):
            out = Path(directory)
            (out / "manifest.json").write_text(json.dumps(manifest))
            (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            reference = [self.runner.conflict_only(r) for r in records if r["draw_index"] < 4]
            (out / "gate7_cpu_reference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reference))
            return out

        spec_a = json.loads(json.dumps(script.SPLIT_A))
        spec_a.update(qualifying_per_cell=2, cap_per_cell=4)
        spec_a["gate7_reference"]["families"] = [0, 1, 2, 3]
        records = []
        mixing_splits.generate_families(
            lambda **kw: (dict(self.runner.family(rng=random.Random(kw["seed"]), n=N_GROUPS, **kw)[0], qualifies=True), None),
            script.CELLS, seed_base=1900000, quota=2, cap=4, per_cell=True, prefix="smoke",
            on_record=lambda r, _: records.append(r))
        with tempfile.TemporaryDirectory() as directory:
            decision = mixing_splits.split_a_decision(write(directory, records), spec_a)
        self.assertEqual(decision["quota"]["met"], True)
        self.assertEqual(list(decision["d_by_cell"]), ["c1", "c2", "c3", "c4"])
        expected = ra.select_cell(list(decision["d_by_cell"].items()), spec_a["d_min"])
        self.assertEqual(decision["selection"]["selected"], expected["selected"])
        if decision["status"] == "PROCEED":
            self.assertEqual(decision["selected"], expected["selected"])
        else:
            self.assertIsNone(decision["selected"])
            self.assertTrue(decision["stops"])
        json.dumps(decision, allow_nan=False)

        spec_b = json.loads(json.dumps(script.SPLIT_B))
        spec_b.update(cells=[script.CELLS[0]], qualifying=3, cap=6)
        spec_b["gate7_reference"]["families"] = [0, 1, 2, 3]
        records = []
        mixing_splits.generate_families(
            lambda **kw: (dict(self.runner.family(rng=random.Random(kw["seed"]), n=N_GROUPS, **kw)[0], qualifies=True), None),
            [script.CELLS[0]], seed_base=1900100, quota=3, cap=6, per_cell=False, prefix="smoke",
            on_record=lambda r, _: records.append(r))
        with tempfile.TemporaryDirectory() as directory:
            decision = mixing_splits.split_b_decision(write(directory, records), spec_b)
        self.assertEqual(decision["cell_key"], "c1")
        self.assertIn("s_min_B", decision["split_B"])
        self.assertEqual(decision["status"] == "PROCEED", not decision["stops"])
        if decision["status"] == "PROCEED":
            freeze = decision["split_B"]["freeze"]
            self.assertEqual(freeze["N"], ra.n_rule(decision["split_B"]["unresolved_rate_B"])["N"])
        json.dumps(decision, allow_nan=False)


@unittest.skipUnless(HAVE_TORCH, REASON)
class SnapshotHashTests(unittest.TestCase):
    def make_snapshot(self, root, corrupt=None, drop=None):
        import mixing_runner as mr
        revision = "0" * 40
        base = root / "models--org--tiny"
        (base / "blobs").mkdir(parents=True)
        snapshot = base / "snapshots" / revision
        snapshot.mkdir(parents=True)
        locked = []
        for name in mr.REQUIRED_SNAPSHOT_FILES:
            if name == drop:
                continue
            content = f"content of {name}".encode()
            lfs = name.endswith((".safetensors", ".model")) or name == "tokenizer.json"
            if lfs:
                blob = hashlib.sha256(content).hexdigest()
            else:
                blob = hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()
            stored = content + b"x" if name == corrupt else content
            (base / "blobs" / blob).write_bytes(stored)
            (snapshot / name).symlink_to(Path("../../blobs") / blob)
            entry = {"path": name, "size": len(content), "git_oid": blob if not lfs else "?"}
            if lfs:
                entry["lfs"] = True
            locked.append(entry)
        return revision, locked

    def test_hashes_match_content_addresses_and_detect_changes(self):
        import mixing_runner as mr
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision, locked = self.make_snapshot(root)
            report = mr.snapshot_hashes(mr.snapshot_path("org/tiny", revision, cache=root), locked)
            self.assertTrue(report["passed"], report["problems"])
            self.assertEqual(len(report["files"]), len(mr.REQUIRED_SNAPSHOT_FILES))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision, locked = self.make_snapshot(root, corrupt="model-00001-of-00002.safetensors")
            report = mr.snapshot_hashes(mr.snapshot_path("org/tiny", revision, cache=root), locked)
            self.assertFalse(report["passed"])

    def test_incomplete_snapshot_stops(self):
        import mixing_runner as mr
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision, _ = self.make_snapshot(root, drop="tokenizer.json")
            with self.assertRaises(mr.SnapshotIncomplete):
                mr.snapshot_path("org/tiny", revision, cache=root)


if __name__ == "__main__":
    unittest.main()
