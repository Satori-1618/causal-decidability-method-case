import importlib.util
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import analyze
import countermodels as cm


class AccuracyParsingTests(unittest.TestCase):
    def test_count_reconstruction_from_float_serialization(self):
        self.assertEqual(analyze.count("0.8230000000000001"),823)

    def test_noncount_and_invalid_range_rejected(self):
        for value in ("0.8232","1.001","-0.001"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                analyze.count(value)

    def test_reporting_band_is_strict_and_not_zero_equivalence(self):
        self.assertEqual(analyze.state(10),"within_band")
        self.assertEqual(analyze.state(11),"improved")
        self.assertEqual(analyze.state(-11),"damaged")
        self.assertEqual(analyze.state(1,band=0),"improved")

    def test_both_is_not_silently_sign_matching(self):
        self.assertEqual(analyze.kind(True,True),"both")


class AttentionWitnessTests(unittest.TestCase):
    def test_mean_operator_really_is_mean_native_attention(self):
        self.assertEqual(cm.MEAN,tuple((a+b)/2 for a,b in zip(cm.NATIVE_A,cm.NATIVE_B)))
        self.assertNotEqual(cm.MEAN,cm.UNIFORM)
        self.assertEqual(sum(cm.MEAN),1)

    def test_identical_existing_outputs_opposite_removal_effects(self):
        a,b=cm.witnesses([779,823,827])
        for op in ("native","uniform","mean"):
            self.assertEqual(a["predictions"][op],b["predictions"][op])
        self.assertGreater(a["counts"]["zero"],779)
        self.assertLess(b["counts"]["zero"],779)

    def test_witnesses_work_for_more_than_one_selected_example(self):
        for targets in ([500,600,650],[800,850,830],[400,300,350],[300,300,300]):
            with self.subTest(targets=targets):
                a,b=cm.witnesses(targets,step=50)
                self.assertEqual(a["counts"]["native"],targets[0])
                self.assertEqual(a["counts"]["uniform"],targets[1])
                self.assertEqual(b["counts"]["mean"],targets[2])

    def test_simplex_invariance_and_zero_break(self):
        a,b=cm.witnesses([779,823,827])
        for weights in [(F(1),F(0),F(0)),(F(1,7),F(2,7),F(4,7)),cm.UNIFORM]:
            ya=a["rest"]+sum(x*y for x,y in zip(weights,a["values"][0]))
            yb=b["rest"]+sum(x*y for x,y in zip(weights,b["values"][0]))
            self.assertEqual(ya,yb)
        self.assertNotEqual(a["rest"],b["rest"])

    def test_countermodel_reference_must_be_feasible(self):
        for targets in ([0,1,2],[1000,900,950],[-1,2,3]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                cm.witnesses(targets)

    def test_actual_preflight_preserves_tie_then_separates(self):
        report=cm.report([779,823,827])
        self.assertEqual(len(report["existing_design"]["identical_mean_groups"]),1)
        self.assertEqual(len(report["design_with_zero_reference"]["identical_mean_groups"]),2)
        self.assertIsNone(report["design_with_zero_reference"]["all_pairs_clear_planning_screen"])


class SourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records,cls.all_heads=analyze.load_records()

    def test_complete_cohort_and_no_head_selection(self):
        self.assertEqual(len(self.records),1620)
        self.assertEqual(len([r for r in self.records if r["n_layer"]>=2]),1350)
        self.assertEqual(len(self.all_heads),270)

    def test_counterexample_does_not_turn_into_measured_zero(self):
        self.assertTrue(all(r["zero_reference_measured"] is False for r in self.records))

    def test_reporting_uses_model_aggregation(self):
        rows=[{"model_id":"a","uniform_change_count":100,"mean_change_count":100},
              {"model_id":"a","uniform_change_count":100,"mean_change_count":100},
              {"model_id":"b","uniform_change_count":0,"mean_change_count":0}]
        out=analyze.group_summary(rows)
        self.assertEqual(out["uniform"]["model_weighted_mean_pp"],5)
        self.assertNotEqual(out["uniform"]["head_weighted_mean_pp"],5)


if __name__ == "__main__":
    unittest.main()
