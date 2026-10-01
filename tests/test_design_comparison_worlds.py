"""Development-family checks only; reserved structural families stay unrun."""
import importlib.util
import json
import unittest
from unittest import mock
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

PATH = Path(__file__).resolve().parents[1] / "applications/design-comparison/worlds.py"
if torch is not None:
    spec = importlib.util.spec_from_file_location("design_comparison_worlds", PATH)
    worlds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worlds)


@unittest.skipIf(torch is None, "PyTorch is an optional application dependency")
class CircuitWorldTests(unittest.TestCase):
    def test_executed_graphs_match_independent_predictions_and_controls(self):
        for family in worlds.DEVELOPMENT_FAMILIES:
            for amplitude in (0.01, 0.3, 1.0, 2.0):
                case = worlds.public_case(17, family, amplitude)
                self.assertTrue(case["controls"]["all_passed"])
                self.assertEqual(json.loads(json.dumps(case)), case)
                for candidate, expected in case["predictions"].items():
                    observed = worlds.observe_truth(case, candidate)
                    self.assertLessEqual(max(abs(a-b) for a, b in zip(expected, observed)),
                                         case["numerical_bound"])

    def test_full_patch_cannot_distinguish_but_channel_context_cross_can(self):
        case = worlds.public_case(20, "linear", 1.0)
        signatures = {tuple(values[:worlds.MANDATORY_COUNT]) for values in case["predictions"].values()}
        self.assertEqual(len(signatures), 1)
        signatures = {tuple(values) for values in case["predictions"].values()}
        self.assertEqual(len(signatures), len(worlds.CANDIDATES) - 1)
        self.assertEqual(case["predictions"]["channel_a"], case["predictions"]["channel_a_alias"])

    def test_actual_hidden_intervention_has_expected_selectivity(self):
        case = worlds.public_case(1, "linear", 1.0)
        a = worlds.observe_truth(case, "channel_a")
        b = worlds.observe_truth(case, "channel_b")
        ia, ib = worlds.MENU.index("read_a:c0:d1"), worlds.MENU.index("read_b:c0:d1")
        self.assertGreater(a[ia], 0)
        self.assertEqual(a[ib], 0)
        self.assertEqual(b[ia], 0)
        self.assertGreater(b[ib], 0)
        self.assertTrue(all(value == 0 for index, value in enumerate(worlds.observe_truth(case, "joint")) if index >= worlds.MANDATORY_COUNT))

    def test_outside_candidate_is_not_silently_in_declared_space(self):
        case = worlds.public_case(1, "relu_offset", 1.0)
        outside = worlds.observe_truth(case, worlds.OUTSIDE_CANDIDATE)
        self.assertNotIn(worlds.OUTSIDE_CANDIDATE, case["predictions"])
        self.assertTrue(all(abs(outside[1] - pred[1]) > case["numerical_bound"]
                            for pred in case["predictions"].values()))

    def test_public_case_has_no_truth_label_and_is_repeatable(self):
        first = worlds.public_case(71, "linear", 0.4)
        self.assertEqual(first, worlds.public_case(71, "linear", 0.4))
        self.assertNotIn("truth", first)
        self.assertNotIn("truth_name", first)
        self.assertNotEqual(first["case_id"], worlds.public_case(72, "linear", 0.4)["case_id"])

    def test_invalid_inputs_fail(self):
        for amplitude in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                worlds.public_case(0, "linear", amplitude)
        with self.assertRaises(ValueError):
            worlds.public_case(0, "unknown", 1.0)
        with self.assertRaises(ValueError):
            worlds.observe_truth(worlds.public_case(0, "linear", 1.0), "unknown")

    def test_reserved_structures_are_blocked_before_execution(self):
        for family in worlds.HELDOUT_FAMILIES:
            with self.assertRaisesRegex(ValueError, "Reserved structural family"):
                worlds.public_case(0, family, 1.0)

    def test_qualification_costs_match_executed_calls(self):
        forwards = 0
        original_forward = worlds.NeuralCircuit.forward

        def counted_forward(circuit, *args, **kwargs):
            nonlocal forwards
            forwards += 1
            return original_forward(circuit, *args, **kwargs)

        with mock.patch.object(worlds.NeuralCircuit, "forward", counted_forward), \
                mock.patch.object(worlds, "_analytic_cell", wraps=worlds._analytic_cell) as formulas:
            case = worlds.public_case(11, "linear", 0.3)
        controls = case["controls"]
        graphs = len(worlds.CANDIDATES) + 1
        self.assertEqual(forwards, graphs * (2 * len(worlds.MENU) + 2 * 3))
        self.assertEqual(forwards, 304)
        self.assertEqual(controls["qualification_graph_forward_count"], forwards)
        self.assertEqual(formulas.call_count, (len(worlds.CANDIDATES) + graphs) * len(worlds.MENU))
        self.assertEqual(formulas.call_count, 240)
        self.assertEqual(controls["analytic_formula_evaluation_count"], formulas.call_count)


if __name__ == "__main__":
    unittest.main()
