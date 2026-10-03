"""Independent preparation audit; synthetic/input-only, never loads a model."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("transfer_preparation_under_test", HERE/"prepare_transfer.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def state(text):
    depth, minimum = 0, 0
    for token in text:
        if token not in "()":
            raise AssertionError("Unexpected token")
        depth += 1 if token == "(" else -1
        minimum = min(minimum, depth)
    return depth, minimum


def phase(text):
    return int(sha_text("value-prefix-v1:"+text), 16) % 5 == 0


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE/"plan.json").read_text())
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)/"prepared"
        cls.metadata = p.prepare(cls.plan, cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def changed_copy(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name)/"prepared"
        shutil.copytree(self.directory, destination)
        return destination

    def test_frozen_counts_seeds_and_all_input_constraints(self):
        recipe = self.metadata["recipe"]
        self.assertEqual((recipe["pool_size"], recipe["per_stratum"], len(recipe["heads"])), (1024, 64, 6))
        self.assertEqual([(r["recipient_seed"], r["donor_seed"]) for r in recipe["heads"]],
                         [(26100301+2*i, 26100302+2*i) for i in range(6)])
        exclusions = json.loads((self.directory/"exclusions.json").read_text())
        banned = set(exclusions["recipient_strings"])
        ids = set()
        for task in self.plan["cohort"]:
            directory = self.directory/p.head_key(task)
            candidates = p.read_rows(directory/"candidates.jsonl")
            donors = p.read_rows(directory/"donor_families.jsonl")
            self.assertEqual((len(candidates), len(donors)), (1024, 128))
            for i, row in enumerate(candidates):
                text = row["recipient"]
                self.assertEqual(row["candidate_index"], i)
                self.assertEqual(row["candidate_id"], sha_text(f"screen-transfer-003:{task['recipient_seed']}:{i}")[:20])
                self.assertNotIn(row["candidate_id"], ids)
                ids.add(row["candidate_id"])
                self.assertEqual((len(text), text.count("("), text.count(")")), (32, 16, 16))
                self.assertLess(state(text)[1], 0)
                self.assertTrue(phase(text))
                self.assertNotIn(text, banned)
            for i, family in enumerate(donors):
                self.assertEqual(family["family_id"], sha_text(f"development:{task['donor_seed']}:{i}")[:20])
                self.assertEqual(family["phase"], "development")
                self.assertEqual(len(family["donors"]), 8)
                cells, prefixes = set(), {}
                for donor in family["donors"]:
                    text, pos, depth, replica = (donor[k] for k in ("string", "position", "balance", "replica"))
                    self.assertEqual((len(text), text.count("("), text.count(")")), (32, 16, 16))
                    self.assertIn(pos, (20, 28))
                    self.assertIn(depth, (-2, 2))
                    self.assertIn(replica, (0, 1))
                    prefix = text[:pos]
                    self.assertEqual(state(prefix), (depth, -4))
                    self.assertEqual(text[pos-1], ")")
                    self.assertTrue(phase(prefix))
                    self.assertEqual(donor["prefix_sha256"], sha_text(prefix))
                    self.assertNotIn(text, banned)
                    self.assertNotIn(prefix, exclusions["prefixes"][str(pos)])
                    self.assertNotIn(prefix, prefixes.setdefault((depth,pos), set()))
                    prefixes[depth,pos].add(prefix)
                    cells.add(donor["cell"])
                self.assertEqual(cells, {f"{s}_{pos}_{r}" for s in ("neg", "pos") for pos in (20, 28) for r in (0, 1)})

    def test_historical_union_includes_all_roles_and_both_positions(self):
        base = json.loads((p.VALUE/"screen_002/inputs/exclusions.json").read_text())
        strings = set(base["recipient_strings"])
        for relative in ("inputs/development_001/cases.jsonl", "screen_002/inputs/candidates.jsonl",
                         "screen_002/inputs/donor_families.jsonl"):
            for row in p.read_rows(p.VALUE/relative):
                strings.add(row["recipient"])
                strings.update(d["string"] for d in row.get("donors", []))
        actual = json.loads((self.directory/"exclusions.json").read_text())
        self.assertEqual(set(actual["recipient_strings"]), strings)
        self.assertEqual(len(strings), 5424)
        for pos, count in ((20, 5054), (28, 3671)):
            expected = {s[:pos] for s in strings if len(s) >= pos}
            self.assertEqual(set(actual["prefixes"][str(pos)]), expected)
            self.assertEqual(len(expected), count)

    def test_cache_free_historical_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = root/"applications/li-saphra-2507.06445/value_followup"
            native = value.parent/"native_followup"
            for relative in ("inputs/development_001/cases.jsonl", "inputs/development_001/exclusions.json",
                             "screen_002/inputs/candidates.jsonl", "screen_002/inputs/donor_families.jsonl",
                             "screen_002/inputs/exclusions.json", "screen_002/inputs/preparation.json"):
                target = value/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p.VALUE/relative, target)
            with patch.multiple(p, ROOT=root, VALUE=value, NATIVE=native):
                result = p.historical_exclusions()
            self.assertEqual(result, json.loads((self.directory/"exclusions.json").read_text()))
            self.assertFalse((native/"cache").exists())

    def test_exact_deterministic_regeneration(self):
        metadata, inputs = p.validate_inputs(self.plan, self.directory, regenerate=True)
        self.assertEqual(metadata, self.metadata)
        self.assertEqual(len(inputs), 6)

    def test_no_overwrite(self):
        before = (self.directory/"preparation.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            p.prepare(self.plan, self.directory)
        self.assertEqual((self.directory/"preparation.json").read_bytes(), before)

    def test_candidate_hash_tampering_is_detected(self):
        directory = self.changed_copy()
        file = directory/p.head_key(self.plan["cohort"][0])/"candidates.jsonl"
        with file.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            p.validate_inputs(self.plan, directory)

    def test_regeneration_catches_data_changed_with_updated_hash(self):
        directory = self.changed_copy()
        relative = p.head_key(self.plan["cohort"][0])+"/candidates.jsonl"
        rows = p.read_rows(directory/relative)
        rows[0]["recipient"], rows[1]["recipient"] = rows[1]["recipient"], rows[0]["recipient"]
        (directory/relative).write_text(p.jsonl(rows))
        metadata = json.loads((directory/"preparation.json").read_text())
        metadata["files"][relative] = p.sha(directory/relative)
        (directory/"preparation.json").write_text(p.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "regeneration"):
            p.validate_inputs(self.plan, directory)

    def test_recipe_seed_and_quota_changes_are_rejected(self):
        for mode in ("seed", "quota", "pool"):
            plan = copy.deepcopy(self.plan)
            if mode == "seed":
                plan["cohort"][0]["recipient_seed"] += 1
            elif mode == "quota":
                plan["screen"]["families_per_stratum"] = 32
            else:
                plan["screen"]["native_candidates_per_head"] = 512
            with self.assertRaisesRegex(ValueError, "recipe"):
                p.validate_inputs(plan, self.directory)

    def test_missing_file_binding_is_detected(self):
        directory = self.changed_copy()
        metadata = json.loads((directory/"preparation.json").read_text())
        del metadata["files"][p.head_key(self.plan["cohort"][0])+"/candidates.jsonl"]
        (directory/"preparation.json").write_text(p.dumps(metadata))
        with self.assertRaises(ValueError):
            p.validate_inputs(self.plan, directory)

    def test_recorded_head_counts_and_seeds_checked(self):
        directory = self.changed_copy()
        metadata = json.loads((directory/"preparation.json").read_text())
        metadata["heads"][p.head_key(self.plan["cohort"][0])]["donor_templates"] = 64
        (directory/"preparation.json").write_text(p.dumps(metadata))
        with self.assertRaises(ValueError):
            p.validate_inputs(self.plan, directory)

    def test_release_bookkeeping_does_not_change_recipe(self):
        plan = copy.deepcopy(self.plan)
        plan["status"] = "reviewed"
        plan["execution_authorized"] = True
        self.assertEqual(p.input_recipe(plan), self.metadata["recipe"])


if __name__ == "__main__":
    unittest.main()
