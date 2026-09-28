"""Source lock of the Gur-Arieh et al. application: structure always, clone when available.

The upstream clone lives outside the repository. Tests that read it run only when
MIXING_MECHS_UPSTREAM points to it; otherwise they are skipped and say why.
"""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
SPEC = importlib.util.spec_from_file_location("mixing_lock_sources", APP / "scripts/lock_sources.py")
lock_sources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lock_sources)
LOCK = json.loads((APP / "SOURCE_LOCK.json").read_text())
UPSTREAM = os.environ.get("MIXING_MECHS_UPSTREAM")


def is_hex(value, length):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


class SourceLockStructureTests(unittest.TestCase):
    def test_pins_paper_code_licence_and_model_revision(self):
        self.assertEqual(LOCK["paper"]["version"], "v2")
        self.assertEqual(LOCK["paper"]["authors"], ["Yoav Gur-Arieh", "Mor Geva", "Atticus Geiger"])
        upstream = LOCK["upstream"]
        self.assertEqual(upstream["commit"], "c53372c606e7cadf2494d2ac7b08e466042052df")
        self.assertTrue(upstream["commit_verified"])
        self.assertEqual(upstream["license"]["spdx"], "MIT")
        self.assertTrue(upstream["license"]["verified"])
        self.assertTrue(all(is_hex(h, 64) for h in upstream["files_sha256"].values()))
        model = LOCK["model"]
        self.assertTrue(is_hex(model["revision"], 40))
        self.assertEqual(model["label"], "not declared upstream")

    def test_model_content_hashes_come_from_gate_one_of_the_pilot(self):
        model = LOCK["model"]
        self.assertTrue(all("sha256" not in entry for entry in model["files_at_revision"]))
        self.assertIn("gate 1", model["content_hashes"])
        manifest = json.loads((APP / "results/pilot/manifest.json").read_text())
        recorded = manifest["model"]["hashes"]
        self.assertTrue(recorded["passed"])
        self.assertEqual(model["content_hashes_gate1"],
                         {k: {"sha256": v["sha256"], "size": v["size"]} for k, v in recorded["files"].items()})
        locked_sizes = {e["path"]: e["size"] for e in model["files_at_revision"]}
        for name, entry in model["content_hashes_gate1"].items():
            self.assertTrue(is_hex(entry["sha256"], 64))
            self.assertEqual(entry["size"], locked_sizes[name])
        self.assertTrue(LOCK["chat_template"]["rendered_template_verified"])
        self.assertTrue(LOCK["chat_template"]["rendered_for_X"].startswith("<bos>"))

    def test_every_brief_fact_has_a_status_and_cited_lines(self):
        ids = {fact["id"] for fact in LOCK["upstream_facts"]}
        self.assertLessEqual({"patch_positions", "readout", "classification", "index_collisions",
                              "case_filtering", "agreement_conditions"}, ids)
        for fact in LOCK["upstream_facts"]:
            self.assertTrue(fact["status"])
            for evidence in fact["code_evidence"]:
                self.assertIsInstance(evidence["line"], int)
                self.assertTrue(evidence["file"] in LOCK["upstream"]["files_sha256"])

    def test_registry_reports_listed_but_undefined_schemas(self):
        registry = LOCK["task_registry"]
        self.assertEqual(len(registry["tasks"]), 10)
        self.assertEqual(len(registry["listed_but_not_defined"]), 4)
        music = next(t for t in registry["tasks"] if t["name"] == "music_performance")
        self.assertEqual(music["categories"], ["Musician", "Genre", "Instrument"])

    def test_openness_check_is_recorded_with_where_it_searched(self):
        check = LOCK["openness_check"]
        self.assertTrue(check["searched"])
        self.assertIn("No STOP", check["result"])


@unittest.skipUnless(UPSTREAM and Path(UPSTREAM, ".git").is_dir(),
                     "set MIXING_MECHS_UPSTREAM to a clone of yoavgur/mixing-mechs")
class SourceLockCloneTests(unittest.TestCase):
    def test_clone_matches_lock(self):
        report = lock_sources.check(UPSTREAM, LOCK)
        self.assertEqual(report["status"], "clone matches lock")
        self.assertEqual(report["tasks"], 10)

    def test_changed_cited_line_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "clone"
            shutil.copytree(UPSTREAM, copy)
            path = copy / "tasks/dist.py"
            lines = path.read_text().splitlines(keepends=True)
            lines[311] = "    token_positions = [-1, -4]\n"
            path.write_text("".join(lines))
            with self.assertRaises(lock_sources.LockError):
                lock_sources.check(copy, LOCK)


if __name__ == "__main__":
    unittest.main()
