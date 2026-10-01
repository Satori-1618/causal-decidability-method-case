"""The six synthetic worlds of the Round 1 brief (Sec. 7): analyzer verification only."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
if str(APP / "src") not in sys.path:
    sys.path.insert(0, str(APP / "src"))
import mixing_synthetic_worlds as worlds  # noqa: E402
import query_route_analysis  # noqa: E402


class SyntheticWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {name: worlds.run_world(name) for name in worlds.WORLDS}
        cls.outcomes = {name: worlds.outcome(r) for name, r in cls.results.items()}

    def test_1_w_band_inside_is_adequate_and_outside_is_excluded(self):
        inside, outside = self.outcomes["1a_w_band_inside"], self.outcomes["1b_w_band_outside"]
        self.assertEqual(inside["statuses"]["W_T"], "adequate")
        self.assertEqual(inside["level"], "S3 (A_T excluded)")
        self.assertEqual(outside["statuses"]["W_T"], "excluded")
        self.assertEqual(outside["level"], "S3 (both excluded)")
        self.assertTrue(inside["mean_gate_passed"] and outside["mean_gate_passed"])

    def test_2_heterogeneous_concentrated_cases_earn_the_between_case_sentence(self):
        o = self.outcomes["2_heterogeneous_concentrated"]
        self.assertEqual(o["statuses"], {"W_T": "excluded", "A_T": "adequate"})
        self.assertEqual(o["level"], "S3 (W_T excluded only)")
        self.assertTrue(o["between_case_earned"])
        self.assertLess(o["max_U"], 0.8)

    def test_3_half_and_half_excludes_both(self):
        o = self.outcomes["3_half_and_half"]
        self.assertEqual(o["statuses"], {"W_T": "excluded", "A_T": "excluded"})
        self.assertFalse(o["between_case_earned"])

    def test_4_below_resolution_stops_at_S1(self):
        o = self.outcomes["4_below_resolution"]
        self.assertEqual((o["development"], o["level"]), ("STOP", "S1"))
        self.assertIn("resolution", o["reason"])

    def test_5_stale_anchor_is_invalid_and_blocks_the_sentence(self):
        o = self.outcomes["5_stale_anchor"]
        self.assertEqual(o["run_status"], "INVALID")
        self.assertIn("stale anchor", o["invalid_reason"])
        self.assertEqual(o["level"], "S1")
        self.assertFalse(o["between_case_earned"])
        # Without the gate A_T would look adequate: the gate, not the profile, blocks it.
        self.assertEqual(o["descriptive_statuses_ignoring_gate"]["A_T"], "adequate")

    def test_6_unresolved_heavy_never_excludes_through_unresolved_cases(self):
        o = self.outcomes["6_unresolved_heavy"]
        self.assertEqual(o["u"], 70)
        self.assertEqual(o["statuses"]["W_T"], "undecided")
        old_rule_upper = query_route_analysis.clopper_pearson(o["k_W"], o["N"], 0.05, 2)[1]
        self.assertLess(old_rule_upper, 0.8)

    def test_stored_outcomes_match(self):
        stored = json.loads((APP / "results/synthetic_worlds/worlds.json").read_text())
        self.assertEqual(stored["label"], "SYNTHETIC: analyzer verification, not evidence")
        self.assertEqual(stored["worlds"], json.loads(json.dumps(self.outcomes)))


if __name__ == "__main__":
    unittest.main()
