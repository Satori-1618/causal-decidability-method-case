"""Round 1 design checks and the proposed-values file (Gur-Arieh et al.). No model."""
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
if str(APP / "src") not in sys.path:
    sys.path.insert(0, str(APP / "src"))
import mixing_round1_analysis as ra  # noqa: E402
import mixing_round1_design as design  # noqa: E402

PROPOSED = json.loads((APP / "PROPOSED_VALUES.json").read_text())
LOCK = json.loads((APP / "SOURCE_LOCK.json").read_text())
POOLS = [[f"person{i}" for i in range(23)], [f"genre{i}" for i in range(24)],
         [f"instrument{i}" for i in range(23)]]


def proposal(identifier):
    return next(p for p in PROPOSED["proposals"] if p["id"] == identifier)


class DesignIndexCheckerTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(20260928)
        self.cells = [c["cell"] for c in proposal("candidate_cells")["value"]]

    def test_conflict_case_realizes_exactly_the_frozen_cell(self):
        for cell in self.cells:
            for _ in range(50):
                G = design.random_matrix(7, POOLS, self.rng)
                case = design.target_rebind(G, cell, 7)
                with self.subTest(cell=cell):
                    self.assertEqual(design.design_indices(**case), cell)

    def test_agreement_control_points_P_L_and_R_to_the_same_group(self):
        for cell in self.cells:
            for j in (cell["i_P"], cell["i_L"], cell["i_R"]):
                for _ in range(30):
                    G = design.random_matrix(7, POOLS, self.rng)
                    case = design.agreement_control(G, j, cell["i_N"], self.rng)
                    indices = design.design_indices(**case)
                    self.assertEqual((indices["i_P"], indices["i_L"], indices["i_R"]), (j, j, j))
                    self.assertEqual(indices["i_N"], cell["i_N"])
                    moved = [i for i in range(7) if i != j]
                    self.assertTrue(all(case["donor"][i] != G[i] for i in moved))
                    self.assertEqual(sorted(case["donor"]), sorted(G))

    def test_absent_answer_gives_no_reflexive_index(self):
        G = design.random_matrix(7, POOLS, self.rng)
        case = design.target_rebind(G, self.cells[0], 7)
        donor = [list(g) for g in case["donor"]]
        donor[self.cells[0]["i_P"]][design.TARGET] = "absent genre"
        case["donor"] = [tuple(g) for g in donor]
        self.assertIsNone(design.design_indices(**case)["i_R"])

    def test_collisions_and_window_violations_are_refused(self):
        G = design.random_matrix(7, POOLS, self.rng)
        with self.assertRaises(ValueError):
            design.target_rebind(G, {"i_P": 3, "i_L": 1, "i_R": 1, "i_N": 0}, 7)
        with self.assertRaises(ValueError):
            design.target_rebind(G, {"i_P": 3, "i_L": 2, "i_R": 5, "i_N": 0}, 7)

    def test_upstream_sampler_shares_at_n_7(self):
        shares = design.upstream_draw_shares(7)
        self.assertAlmostEqual(shares["collision"], 1 / 5)
        self.assertGreater(shares["inadmissible"], 0.5)


class ProposedValuesTests(unittest.TestCase):
    def test_everything_is_a_proposal_awaiting_approval(self):
        self.assertIn("awaiting", PROPOSED["status"])
        self.assertFalse(PROPOSED["approved"])
        self.assertTrue(all(p["status"] in ("proposed", "fixed by the brief") for p in PROPOSED["proposals"]))

    def test_values_fixed_by_the_brief_match_the_analyzer_contract(self):
        self.assertEqual(proposal("w")["value"], ra.CONTRACT["w"])
        self.assertEqual(proposal("per_label_tail")["value"], ra.CONTRACT["label_tail"])
        self.assertEqual(ra.CONTRACT["alpha"] / (2 * ra.CONTRACT["family_size"]), ra.CONTRACT["label_tail"])

    def test_candidate_cells_are_admissible_middle_cells_and_at_most_five(self):
        cells = proposal("candidate_cells")["value"]
        n = proposal("n_groups")["value"]
        self.assertLessEqual(len(cells), 5)
        self.assertEqual(len({json.dumps(c["cell"], sort_keys=True) for c in cells}), len(cells))
        for entry in cells:
            ra.check_cell(entry["cell"], n, ra.CONTRACT["w"])
            self.assertEqual(entry["cell"]["i_P"], (n - 1) // 2)

    def test_proposed_task_exists_upstream_with_the_declared_query(self):
        task = proposal("task")["value"]
        registered = next(t for t in LOCK["task_registry"]["tasks"] if t["name"] == task["schema"])
        self.assertEqual(registered["categories"][task["t_entity"] - 1], registered["queries"][task["query"]])

    def test_power_orientation_of_the_brief_reproduces(self):
        self.assertAlmostEqual(design.power(150, 0.90)["adequate"], 0.831, places=3)
        self.assertAlmostEqual(design.power(200, 0.70)["excluded"], 0.842, places=3)
        n_power = proposal("N_and_power")["value"]
        self.assertGreaterEqual(design.power(n_power["N"], 0.70)["excluded"], n_power["declared_power"])

    def test_gates(self):
        self.assertTrue(design.yield_gate(200, 380, 0.5)["passed"])
        self.assertFalse(design.yield_gate(200, 420, 0.5)["passed"])
        pairs = [((0.61, True, ["lexical"]), (0.615, True, ["lexical"]))]
        self.assertTrue(design.dtype_gate(pairs, 0.01)["passed"])
        pairs.append(((0.40, True, ["lexical"]), (0.40, True, ["positional"])))
        self.assertFalse(design.dtype_gate(pairs, 0.01)["passed"])


if __name__ == "__main__":
    unittest.main()
