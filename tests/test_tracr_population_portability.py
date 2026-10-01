"""The release adapter must not turn numerical portability into a relaxed decision."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portable", ROOT / "scripts/check_tracr_records_portable.py")
portable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable)


class PopulationPortabilityTests(unittest.TestCase):
    def setUp(self):
        self.stored = json.loads((ROOT / "applications/tracr/results/confirmation_001/summary.json").read_text())["population_decision"]

    def test_exact_rational_boundary_and_last_digit_tail_drift(self):
        actual = copy.deepcopy(self.stored)
        actual["null_upper_tail"] += 1e-12
        report = portable.check_population(actual, self.stored)
        self.assertEqual(report["integer_boundary_and_decision"], "exactly_verified")

    def test_changed_decision_bounds_or_material_tail_are_rejected(self):
        for field, value in (("adequate", False), ("successes", 127),
                             ("null_upper_tail", .05), ("null_upper_tail", float("nan"))):
            actual = copy.deepcopy(self.stored)
            actual[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                portable.check_population(actual, self.stored)
        tampered = copy.deepcopy(self.stored)
        tampered["bounds"]["lower_successes"] += 1
        tampered["bounds"]["lower_rate"] = tampered["bounds"]["lower_successes"] / tampered["population_size"]
        with self.assertRaises(ValueError):
            portable.check_population(tampered, tampered)
