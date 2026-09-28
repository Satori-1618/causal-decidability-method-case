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
    def test_approved_for_the_pilot_only_and_final_approval_pending(self):
        self.assertIn("PILOT ONLY", PROPOSED["status"])
        self.assertFalse(PROPOSED["approved"])
        self.assertTrue(PROPOSED["approved_for_pilot"])
        self.assertEqual(PROPOSED["approved_for_pilot_on"], "2026-09-28")
        self.assertIn("pending", PROPOSED["final_approval"])
        allowed = ("fixed by the brief", "fixed by the user (2026-09-28)",
                   "approved for the pilot only (2026-09-28); final approval pending",
                   "approved for the pilot on 2026-09-28")
        self.assertTrue(all(p["status"].startswith(allowed) for p in PROPOSED["proposals"]))

    def test_layer_is_fixed_by_declaration_and_19_is_only_a_diagnostic(self):
        layer = proposal("layer")
        self.assertIn("fixed by declaration", layer["status"])
        self.assertEqual(layer["value"]["layer"], 18)
        self.assertIn("never a STOP", layer["value"]["diagnostic"])
        self.assertNotIn("check", layer["value"])

    def test_s_min_wording_states_the_order_statistic_property(self):
        text = proposal("s_min_rule")["justification"]
        self.assertIn("1/(m + 1)", text)
        self.assertNotIn("at most 1% of unpatched runs", text)
        ra_values = [i / 1000 for i in range(50)]
        self.assertEqual(ra.upper_order_statistic(ra_values, 0.99), max(ra_values))

    def test_agreement_anchor_and_its_gate_match_the_analyzer(self):
        entry = proposal("agreement_anchor")
        self.assertEqual(entry["value"]["agreement_resolution_floor"], ra.CONTRACT["agreement_resolution_floor"])
        self.assertIn("mean over j in {P, L, R}", entry["value"]["T_A"])

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
        self.assertGreaterEqual(design.power(200, 0.70)["excluded"], design.N_RULE["declared_power"])

    def test_n_rule_branches(self):
        self.assertIn("at most 2%", proposal("N_rule")["value"]["rule"])
        low = design.n_rule(0.01)
        self.assertEqual((low["N"], low["status"], low["flag"]), (200, "PROCEED", None))
        edge = design.n_rule(0.02)
        self.assertEqual(edge["N"], 200)
        self.assertLess(edge["exclusion_power"], 0.80)
        self.assertIsNotNone(edge["flag"])
        self.assertEqual(design.n_rule(0.025)["N"], 300)
        self.assertEqual(design.n_rule(0.04)["N"], 400)
        five = design.n_rule(0.05)
        self.assertEqual((five["N"], five["adequacy_powered"]), (500, True))
        high = design.n_rule(0.08)
        self.assertEqual((high["N"], high["adequacy_powered"], high["status"]), (500, False, "PROCEED"))
        self.assertGreaterEqual(high["exclusion_power"], 0.80)
        worst = design.n_rule(0.20)
        self.assertEqual((worst["N"], worst["adequacy_powered"], worst["status"]), (500, False, "STOP"))

    def test_gates(self):
        self.assertTrue(design.yield_gate(200, 380, 0.5)["passed"])
        self.assertFalse(design.yield_gate(200, 420, 0.5)["passed"])
        pairs = [((0.61, True, ["lexical"]), (0.615, True, ["lexical"]))]
        self.assertTrue(design.dtype_gate(pairs, 0.01)["passed"])
        pairs.append(((0.40, True, ["lexical"]), (0.40, True, ["positional"])))
        self.assertFalse(design.dtype_gate(pairs, 0.01)["passed"])


if __name__ == "__main__":
    unittest.main()
