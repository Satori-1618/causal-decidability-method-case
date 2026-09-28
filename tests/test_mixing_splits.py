"""Round 1 splits A and B (protocol v2): generation order, quotas, gate-7 cases and the
declared specs. No model; the family runner is a stand-in."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
if str(APP / "src") not in sys.path:
    sys.path.insert(0, str(APP / "src"))
import mixing_splits  # noqa: E402

spec = importlib.util.spec_from_file_location("run_mixing_split", APP / "scripts/run_mixing_split.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)
PROPOSED = {p["id"]: p for p in json.loads((APP / "PROPOSED_VALUES.json").read_text())["proposals"]}


class StandIn:
    """Qualifies unless (cell, j) is listed as failing."""

    def __init__(self, failing=()):
        self.failing, self.calls = set(failing), []

    def __call__(self, case_id, draw_index, seed, cell_key, cell):
        self.calls.append((draw_index, seed, cell_key))
        return {"case_id": case_id, "draw_index": draw_index, "seed": seed, "cell_key": cell_key,
                "qualifies": (cell_key, draw_index) not in self.failing}, None


def generate(runner, **kwargs):
    stored = []
    counts, met = mixing_splits.generate_families(runner, script.CELLS, on_record=lambda r, _: stored.append(r),
                                                  prefix="t", **kwargs)
    return counts, met, stored


class GenerationTests(unittest.TestCase):
    def test_split_A_interleaves_cells_and_stops_each_at_its_quota(self):
        runner = StandIn(failing={("c2", 1), ("c2", 5)})
        counts, met, stored = generate(runner, seed_base=2000000, quota=3, cap=6, per_cell=True)
        self.assertTrue(met)
        self.assertEqual(counts, {"c1": {"generated": 3, "qualifying": 3}, "c2": {"generated": 5, "qualifying": 3},
                                  "c3": {"generated": 3, "qualifying": 3}, "c4": {"generated": 3, "qualifying": 3}})
        self.assertEqual([r["draw_index"] for r in stored][:12], list(range(12)))
        self.assertEqual([(r["draw_index"], r["cell_key"]) for r in stored[12:]], [(13, "c2"), (17, "c2")])
        self.assertTrue(all(r["seed"] == 2000000 + r["draw_index"] for r in stored))
        self.assertTrue(all(script.CELLS[r["draw_index"] % 4][0] == r["cell_key"] for r in stored))

    def test_gate7_cases_of_split_A_are_the_first_eight_families_of_each_cell(self):
        runner = StandIn(failing={("c3", 2)})
        _, _, stored = generate(runner, seed_base=2000000, quota=50, cap=100, per_cell=True)
        declared = set(script.SPLIT_A["gate7_reference"]["families"])
        chosen = [r for r in stored if r["draw_index"] in declared]
        for key, _ in script.CELLS:
            first = [r["draw_index"] for r in stored if r["cell_key"] == key][:8]
            self.assertEqual(sorted(r["draw_index"] for r in chosen if r["cell_key"] == key), first)

    def test_a_quota_not_met_within_the_cap_is_reported(self):
        runner = StandIn(failing={("c4", 3), ("c4", 7), ("c4", 11)})
        counts, met, _ = generate(runner, seed_base=2000000, quota=3, cap=5, per_cell=True)
        self.assertFalse(met)
        self.assertEqual(counts["c4"], {"generated": 5, "qualifying": 2})

    def test_split_B_runs_one_cell_until_its_quota(self):
        stored = []
        runner = StandIn(failing={("c2", 0), ("c2", 3)})
        counts, met = mixing_splits.generate_families(runner, [script.CELLS[1]], seed_base=3000000, quota=5,
                                                      cap=10, per_cell=False, prefix="b",
                                                      on_record=lambda r, _: stored.append(r))
        self.assertTrue(met)
        self.assertEqual([r["draw_index"] for r in stored], list(range(7)))
        self.assertEqual(counts["c2"], {"generated": 7, "qualifying": 5})
        with self.assertRaises(ValueError):
            mixing_splits.generate_families(runner, script.CELLS, seed_base=3000000, quota=5, cap=10,
                                            per_cell=False, prefix="b", on_record=lambda r, _: None)


class DeclaredSpecTests(unittest.TestCase):
    def test_seed_blocks_are_disjoint_and_the_confirmation_block_is_untouched(self):
        blocks = {"pilot1": 1000000, "pilot2": 1100000, "smoke": script.COMMON["smoke_seed_base"],
                  "A": script.SPLIT_A["seed_base"], "B": script.SPLIT_B["seed_base"],
                  "confirmation": script.COMMON["confirmation_seed_base_untouched"]}
        self.assertEqual((blocks["A"], blocks["B"], blocks["confirmation"]), (2000000, 3000000, 4000000))
        spans = sorted((base, base + 1000) for base in blocks.values())
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])))
        self.assertLessEqual(script.SPLIT_A["cap_per_cell"] * 4, 1000)
        self.assertLessEqual(script.SPLIT_B["cap"], 1000)

    def test_specs_match_the_declared_values(self):
        sizes = PROPOSED["split_sizes"]["value"]
        self.assertEqual(script.SPLIT_A["qualifying_per_cell"], sizes["A_per_candidate_cell"])
        self.assertEqual(script.SPLIT_A["cap_per_cell"], sizes["A_cap_per_cell"])
        self.assertEqual(script.SPLIT_B["qualifying"], sizes["B_selected_cell"])
        self.assertEqual(script.SPLIT_B["cap"], sizes["B_cap"])
        self.assertEqual(script.SPLIT_A["gate7_reference"]["families"], list(range(32)))
        self.assertEqual(script.SPLIT_B["gate7_reference"]["families"], list(range(32)))
        cells = [c["cell"] for c in PROPOSED["candidate_cells"]["value"]]
        self.assertEqual([cell for _, cell in script.CELLS], cells)
        delta = PROPOSED["delta_procedure"]["value"]
        self.assertEqual(script.COMMON["delta"], {"false_invalid_rate": delta["false_invalid_rate"],
                                                  "resamples": delta["resamples"], "seed": delta["seed"]})
        self.assertTrue((APP / "SPLIT_A_B_PROTOCOL.md").is_file())

    def test_producers_include_the_split_code(self):
        self.assertIn("applications/gur-arieh-2510.06182/src/mixing_splits.py", script.PRODUCERS)
        self.assertIn("applications/gur-arieh-2510.06182/scripts/run_mixing_split.py", script.PRODUCERS)


if __name__ == "__main__":
    unittest.main()
