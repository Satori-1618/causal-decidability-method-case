"""Round 0 (RETROSPECTIVE) structural tests for the Gur-Arieh et al. application. No model."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
for path in (APP / "src", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import mixing_round0 as r0  # noqa: E402


class RoundZeroTests(unittest.TestCase):
    def setUp(self):
        self.result = r0.reconstruct()
        self.conditions = self.result["conditions"]

    def test_target_rebind_separates_P_L_N_and_groups_R_with_A(self):
        c = self.conditions["sec3.2_layer_l_point_P_figure1"]
        self.assertEqual(c["equivalence_groups"], [["A", "R"], ["L"], ["N"], ["P"]])
        self.assertEqual(c["not_separated_pairs"], [["A", "R"]])

    def test_collision_of_lexical_and_reflexive_index_merges_L_into_the_group(self):
        c = self.conditions["sec3.2_layer_l_collision_iL_eq_iR"]
        self.assertIn(["A", "L", "R"], c["equivalence_groups"])

    def test_window_needs_admissible_design_indices(self):
        figure1 = self.conditions["sec3.2_layer_l_window_w1_figure1"]
        self.assertIn(["L", "P"], figure1["not_separated_pairs"])
        self.assertIn(["P", "R"], figure1["not_separated_pairs"])
        admissible = self.conditions["sec3.2_layer_l_window_w1_admissible_n7"]
        self.assertEqual(admissible["not_separated_pairs"], [["A", "R"]])

    def test_absent_answer_makes_R_undefined_and_A_a_distinct_point(self):
        c = self.conditions["sec3.4_layer_l_readout_sees_absent"]
        self.assertEqual(c["undefined_not_evaluable"], ["R"])
        self.assertNotIn("R", c["evaluable"])
        self.assertIn(["A"], c["equivalence_groups"])
        self.assertEqual(c["not_separated_pairs"], [])

    def test_in_context_readout_cannot_test_the_answer_copy(self):
        c = self.conditions["sec3.4_layer_l_in_context_readout"]
        self.assertEqual(c["outside_scope"], ["A"])
        self.assertEqual(c["undefined_not_evaluable"], ["R"])

    def test_next_layer_is_a_different_intervention(self):
        c = self.conditions["sec3.4_layer_l_plus_1"]
        self.assertEqual(c["evaluable"], [])
        self.assertEqual(sorted(c["outside_scope"]), sorted(r0.CANDIDATES))

    def test_undefined_is_never_a_number_and_gives_no_separation(self):
        predictions = r0.target_rebind(2, 1, 3, 4, 4)
        predictions["R"] = r0.undefined("test")
        analysed = r0.analyse(predictions)
        self.assertNotIn("R", {name for pair in analysed["separated_pairs"] for name in pair})
        self.assertNotIn("R", {name for group in analysed["equivalence_groups"] for name in group})
        self.assertNotIn("entities", predictions["R"])

    def test_incomplete_or_unknown_predictions_are_refused(self):
        predictions = r0.target_rebind(2, 1, 3, 4, 4)
        del predictions["N"]
        with self.assertRaises(ValueError):
            r0.analyse(predictions)
        predictions = r0.target_rebind(2, 1, 3, 4, 4)
        predictions["N"] = {"type": "number", "entities": [0.5]}
        with self.assertRaises(ValueError):
            r0.analyse(predictions)

    def test_labels_and_handover(self):
        self.assertEqual(self.result["label"], "RETROSPECTIVE")
        self.assertEqual(self.result["compatible_after_round0"]["compatible_untested"], ["R"])
        self.assertIn("single cases", self.result["still_open_handed_to_round1"])

    def test_stored_result_matches_reconstruction(self):
        stored = json.loads((APP / "results/round0_retrospective/round0.json").read_text())
        self.assertEqual(stored, json.loads(json.dumps(self.result)))


if __name__ == "__main__":
    unittest.main()
