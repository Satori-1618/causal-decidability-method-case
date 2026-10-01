"""Post hoc descriptive companion: reproduces frozen counts; standard library, stored records."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'applications/makelov-2311.17030/scripts/describe_post_hoc.py'
spec = importlib.util.spec_from_file_location('describe_post_hoc', SCRIPT)
describe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(describe)


class PostHocDescription(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = describe.describe()

    def test_frozen_tolerances_reproduce_published_counts(self):
        r2 = self.report['round2_tolerance_sensitivity']['successes_by_kappa']
        r3 = self.report['round3a_tolerance_sensitivity']['successes_by_epsilon_nat']
        self.assertEqual(r2['0.25'], {'transfer': 0, 'joint_dependence': 0, 'preservation': 8})
        self.assertEqual(r3['0.25'], {'position_only': 301, 'identity_only': 0})

    def test_counts_never_fall_as_the_tolerance_widens(self):
        for rows in (self.report['round2_tolerance_sensitivity']['successes_by_kappa'],
                     self.report['round3a_tolerance_sensitivity']['successes_by_epsilon_nat']):
            ordered = [rows[k] for k in sorted(rows, key=float)]
            for profile in ordered[0]:
                counts = [row[profile] for row in ordered]
                self.assertEqual(counts, sorted(counts), profile)

    def test_round2_refuses_tolerances_that_merge_endpoint_profiles(self):
        with self.assertRaisesRegex(ValueError, 'two profiles'):
            describe.round2_profile_counts([], .5)

    def test_q1_description_covers_every_directed_case(self):
        q1 = self.report['read_source_q1']
        self.assertEqual(q1['n_directed_cases_not_independent'], 128)
        self.assertEqual(q1['directed_cases_with_both_fractions_in_0_1'], 128)
        self.assertEqual(q1['directed_cases_where_coefficient_magnitude_order_matches_B_versus_A'],
                         128)
        self.assertEqual(
            q1['directed_cases_where_larger_coefficient_magnitude_gives_larger_absolute_effect'], 128)

    def test_scalar_name_contrast_is_shared_by_both_recipient_orders(self):
        diagnostics = self.report['round3a_mean_contrast_diagnostics']
        self.assertLess(diagnostics['maximum_scalar_level_name_contrast_gap_between_recipient_orders'],
                        1e-12)
        self.assertEqual(diagnostics['families_with_opposite_sign_I_across_recipient_orders'], 491)
        self.assertEqual(diagnostics['families_with_same_sign_J_across_recipient_orders'], 498)

    def test_mismatch_with_stored_counts_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / 'round2'
            shutil.copytree(describe.ROUND2, copy, ignore=shutil.ignore_patterns('directions.npz'))
            summary = json.loads((copy / 'summary.json').read_text())
            summary['profiles']['preservation']['successes'] = 9
            (copy / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                describe.describe(round2=copy)


if __name__ == '__main__':
    unittest.main()
