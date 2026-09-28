"""Round 1 runner tests on a tiny, randomly initialised Gemma 2 built locally.

No Gemma file is read and nothing is downloaded. The tests check the patch mechanics
(hidden_states[l] is the input of block l; the hook fires once per forward and writes
only the last prompt position; the identity self-patch reproduces the logits and the
greedy generation), the readout arithmetic, the design and alignment checks inside a
family, the snapshot hashing, and that runner records validate against the records-only
checker's schema. They need torch and transformers (requirements-model.txt); without
them they are skipped and say why.
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


def load_checker():
    spec = importlib.util.spec_from_file_location("check_mixing_round1_records",
                                                  APP / "scripts/check_mixing_round1_records.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        pools, dropped, ids = mr.single_token_pools(cls.tokenizer, cls.spec)
        cls.pools, cls.ids = pools, ids
        cls.runner = mr.Runner(cls.model, cls.tokenizer, cls.spec, pools, ids, LAYER,
                               diagnostic_layer=DIAGNOSTIC_LAYER)
        rng = random.Random(1000000)
        from mixing_round1_design import random_matrix
        cls.G = random_matrix(N_GROUPS, [pools[c] for c in cls.spec["categories"]], rng)

    def prompt_ids(self, query_group=0):
        return self.runner.prompt(self.G, query_group)["input_ids"]

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
        torch = self.torch
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
        donor_ids = self.prompt_ids(query_group=4)
        donor, _ = self.runner.forward(donor_ids)
        base, _ = self.runner.forward(ids)
        vector = donor.hidden_states[LAYER][0, -1]
        patched, report = self.runner.forward(ids, patch=(LAYER, vector))
        T, D = ids.shape[1], self.model.config.hidden_size
        self.assertEqual(report["calls"], 1)
        self.assertEqual(report["writes"], 1)
        self.assertEqual(report["positions"], [T - 1])
        self.assertEqual(report["shapes"], [[1, T, D]])
        self.assertEqual(report["vector_shape"], [D])
        self.assertTrue(self.mr.Runner.check_hook(report, T, D))
        self.assertTrue(torch.equal(patched.logits[0, :-1], base.logits[0, :-1]))
        self.assertGreater(float((patched.logits[0, -1] - base.logits[0, -1]).abs().max()), 1e-4)
        self.assertTrue(torch.equal(patched.hidden_states[LAYER][0, :-1], base.hidden_states[LAYER][0, :-1]))
        # hidden_states[LAYER] is recorded before the hook, so it stays the recipient's own.
        self.assertTrue(torch.equal(patched.hidden_states[LAYER], base.hidden_states[LAYER]))
        # The block input at the last position is the donor vector after the write.
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

    def test_readout_arithmetic(self):
        torch = self.torch
        ids = self.prompt_ids()
        entity_ids, positions = self.runner.aligned_entity_ids(self.G, ids)
        out, _ = self.runner.forward(ids)
        readout = self.runner.readout(out, entity_ids, full=True)
        logits = readout.pop("_full_logits")
        probabilities = torch.softmax(logits, 0)
        self.assertAlmostEqual(readout["entity_mass_full_vocab"],
                               float(probabilities[entity_ids].sum()), places=5)
        self.assertAlmostEqual(math.log(math.exp(readout["logsumexp_entities"]) + math.exp(readout["logsumexp_complement"])),
                               readout["logsumexp_full"], places=4)
        self.assertEqual(len(readout["entity_logits"]), N_GROUPS)
        for a, b in zip(readout["entity_logits_fp32_unembed"], readout["entity_logits"]):
            self.assertAlmostEqual(a, b, places=4)
        tokens = ids[0].tolist()
        self.assertEqual([tokens[i] for i in positions], entity_ids)

    def test_alignment_mismatch_is_a_technical_failure(self):
        ids = self.prompt_ids()
        swapped = [list(g) for g in self.G]
        swapped[0][1], swapped[1][1] = swapped[1][1], swapped[0][1]
        with self.assertRaises(self.mr.TechnicalError):
            self.runner.aligned_entity_ids([tuple(g) for g in swapped], ids)

    def test_single_token_pools_drop_multi_token_entities(self):
        from mixing_tiny_model import MULTI_TOKEN_GENRE
        spec = copy.deepcopy(self.spec)
        spec["items"]["Genre"].append(MULTI_TOKEN_GENRE)
        pools, dropped, _ = self.mr.single_token_pools(self.tokenizer, spec)
        self.assertEqual(dropped, {"Genre": [MULTI_TOKEN_GENRE]})
        self.assertEqual(pools["Genre"], self.spec["items"]["Genre"])

    def run_families(self, count):
        records = []
        for i in range(count):
            record, _ = self.runner.family(case_id=f"tiny-{i:04d}", draw_index=i, seed=1000000 + i,
                                           cell_key="c1", cell=CELL, rng=random.Random(1000000 + i),
                                           n=N_GROUPS, audit=(i == 0))
            records.append(record)
        return records

    def test_family_runs_design_checks_hooks_and_controls(self):
        records = self.run_families(3)
        for record in records:
            self.assertTrue(record["technical"]["passed"], record["technical"]["failures"])
            self.assertEqual(record["design_indices"], CELL)
            self.assertEqual(len(record["entity_logits"]), N_GROUPS)
            self.assertEqual([a["j"] for a in record["agreement"]], [3, 1, 5])
            for a in record["agreement"]:
                i = a["design_indices"]
                self.assertEqual((i["i_P"], i["i_L"], i["i_R"], i["i_N"]), (a["j"],) * 3 + (0,))
                self.assertEqual((a["hook"]["calls"], a["hook"]["writes"]), (1, 1))
            self.assertTrue(record["donor_answer"]["in_recipient"])
            self.assertEqual(record["donor_answer"]["recipient_index"], CELL["i_R"])
            self.assertEqual(record["identity"]["max_abs_logit_difference"], 0.0)
            self.assertTrue(record["identity"]["same_generation"])
            self.assertEqual(record["diagnostic"]["layer"], DIAGNOSTIC_LAYER)
            self.assertIsInstance(record["qualifies"], bool)
            json.dumps(record, allow_nan=False)
        # The same seed gives the same family.
        again = self.run_families(1)[0]
        self.assertEqual(again["matrix"], records[0]["matrix"])
        self.assertEqual(again["entity_logits"], records[0]["entity_logits"])

    def test_records_validate_against_the_records_only_checker(self):
        checker = load_checker()
        import mixing_round1_analysis as ra
        records = self.run_families(4)
        for record in records:
            self.assertIsNone(checker.technical_failure(record, CELL, N_GROUPS))
        # Schema check only: a random tiny model is not natively correct, so the copies
        # are marked qualifying to exercise the checker's full path.
        qualifying = [dict(copy.deepcopy(r), qualifies=True) for r in records]
        manifest = {
            "schema_version": 1, "application": "gur-arieh-2510.06182", "round": 1,
            "stage": "schema test", "n_groups": N_GROUPS, "cell": dict(CELL), "N": len(qualifying),
            "rule": {**ra.CONTRACT, "s_min": 0.05, "d_min": 0.20, "agreement_transfer_floor": 0.9},
            "anchors": {"T_W": 0.5, "T_A": 0.95, "d": 0.45, "q_bar_B": [0.5, 0.3, 0.2], "m_B": 100},
            "mean_gate": {"delta": 1.0}, "development_gates": {
                "resolution_rate_B": 0.95, "agreement_resolution_rate_B": 0.95,
                "agreement_transfer_rate_B": 0.95},
        }

        def sha(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        manifest["code_files_sha256"] = {name: sha(ROOT / name) for name in checker.REQUIRED_CODE}
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (results / "RUN_STARTED.json").write_text(json.dumps({"manifest_sha256": sha(results / "manifest.json")}))
            (results / "records.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in qualifying))
            summary = ra.analyze_confirmation(manifest, qualifying)
            summary.update(manifest_sha256=sha(results / "manifest.json"),
                           records_sha256=sha(results / "records.jsonl"))
            (results / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            (results / "artifact_hashes.json").write_text(json.dumps(
                {p.name: sha(p) for p in results.iterdir()}))
            report = checker.verify(results, repository_root=ROOT)
        self.assertTrue(report["verified"])
        self.assertEqual(report["N"], len(qualifying))


    def test_pilot_summary_runs_on_runner_records(self):
        """Smoke test of the pilot gate table on tiny-model records. Records are marked
        qualifying so every branch runs; the numbers mean nothing."""
        import mixing_pilot_summary as summary_module
        spec = importlib.util.spec_from_file_location("run_mixing_pilot", APP / "scripts/run_mixing_pilot.py")
        pilot_script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot_script)
        pilot = copy.deepcopy(pilot_script.PILOT)
        records = []
        for i in range(8):
            key, cell = pilot["cells"][i % 4]
            record, _ = self.runner.family(case_id=f"pilot-{i:04d}", draw_index=i, seed=1000000 + i,
                                           cell_key=key, cell=cell, rng=random.Random(1000000 + i),
                                           n=N_GROUPS)
            records.append(dict(record, qualifies=True))
        fp32 = [self.runner.conflict_only(r, N_GROUPS) for r in records[:4]]
        manifest = {"model": {"hashes": {"passed": True, "problems": [], "files": {}}},
                    "gate3_tokens": {"passed": True, "pools_kept": {}, "pools_dropped": {},
                                     "dropped_prefix": "<bos>"},
                    "environment": {"device": "cpu"}}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "manifest.json").write_text(json.dumps(manifest))
            (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            (out / "fp32_reference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fp32))
            result = summary_module.summarize(out, pilot)
        self.assertEqual(result["families"], 8)
        for gate in ("1_model_hashes", "2_native_and_yield", "3_tokens", "4_hooks", "5_identity",
                     "6_agreement", "7_dtype", "8_support", "9_separation_descriptive"):
            self.assertIn(gate, result["gates"])
        self.assertTrue(result["gates"]["4_hooks"]["passed"])
        self.assertTrue(result["gates"]["5_identity"]["passed"])
        self.assertEqual(result["gates"]["7_dtype"]["primary_bf16_logits"]["max_abs_T_difference"], 0.0)
        self.assertEqual(result["gates"]["6_agreement"]["runs"], 24)
        self.assertIsNotNone(result["n_rule"])
        self.assertEqual(set(result["readout_validity_diagnostic"]),
                         {"conflict_layer18", "conflict_layer19", "agreement_layer18", "native_recipient"})
        json.dumps(result, allow_nan=False)


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
            path = mr.snapshot_path("org/tiny", revision, cache=root)
            report = mr.snapshot_hashes(path, locked)
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
