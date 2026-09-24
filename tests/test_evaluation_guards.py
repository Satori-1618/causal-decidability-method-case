"""Fail-closed scientific contracts and invariance to equivalent candidate aliases."""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from causal_decidability.evaluate import compare, evaluate, load_rows  # noqa: E402


class EvaluationGuards(unittest.TestCase):
    def data(self, values):
        return load_rows((str(i), 'r', condition, value)
                         for i, cells in enumerate(values)
                         for condition, value in cells.items())

    def test_tested_self_anchor_is_rejected_by_both_entry_points(self):
        data = self.data([{'probe': float(i)} for i in range(12)])
        candidates = {'tautology': {'probe': 'same_as:probe'}, 'zero': {'probe': 0}}
        with self.assertRaisesRegex(ValueError, 'self-anchor'):
            evaluate(data, candidates, ['probe'], 'absolute', tolerance=0)
        with self.assertRaisesRegex(ValueError, 'self-anchor'):
            compare(data, candidates, ['probe'])

    def test_an_untested_self_anchor_is_not_an_evaluation_condition(self):
        # Full is an anchor, not a tested prediction: the Makelov use remains valid.
        data = self.data([{'full': i + 1.0, 'probe': i + 1.0} for i in range(12)])
        candidate = {'anchored': {'probe': 'same_as:full', 'full': 'same_as:full'}}
        out = evaluate(data, candidate, ['probe'], 'absolute', tolerance=.1,
                       scope='pooled', resamples=200)
        self.assertEqual(out['status'], {'anchored': 'adequate'})

    def test_identical_numeric_aliases_do_not_change_adequacy_or_intervals(self):
        # Previously adding aliases widened the interval and changed adequate to undecided.
        data = self.data([{'probe': 1.0 if i < 2 else 0.0} for i in range(12)])
        original = evaluate(data, {'zero': {'probe': 0}}, ['probe'], 'signed',
                            tolerance=.45, scope='pooled', resamples=10_000)
        aliases = {'zero': {'probe': 0},
                   **{f'alias{i}': {'probe': 0.0} for i in range(5)}}
        expanded = evaluate(data, aliases, ['probe'], 'signed', tolerance=.45,
                            scope='pooled', resamples=10_000,
                            equivalence_groups=[list(aliases)])
        self.assertEqual(original['status']['zero'], 'adequate')
        self.assertEqual(set(expanded['status'].values()), {'adequate'})
        self.assertEqual(expanded['contract']['interval_family_size'], 1)
        for row in expanded['rows']:
            self.assertEqual(row['interval'], original['rows'][0]['interval'])
        self.assertEqual(expanded['retained'], sorted(aliases))

    def test_anchor_aliases_share_one_decision_and_one_comparison(self):
        data = self.data([{'baseline': float(i), 'full': i + 2.0,
                           'probe': i + .1 * (i % 4)} for i in range(18)])
        original_candidates = {'B': {'probe': 'same_as:baseline'},
                               'C': {'probe': 'same_as:full'}}
        original = evaluate(data, original_candidates, ['probe'], 'absolute',
                            tolerance=.5, scope='pooled', resamples=300, permutations=300)
        aliased = {**original_candidates, 'A_alias': {'probe': 'same_as:baseline'}}
        expanded = evaluate(data, aliased, ['probe'], 'absolute', tolerance=.5,
                            scope='pooled', resamples=300, permutations=300,
                            equivalence_groups=[('B', 'A_alias')])
        self.assertEqual(expanded['contract']['interval_family_size'], 2)
        self.assertEqual(expanded['status']['A_alias'], original['status']['B'])
        self.assertEqual(expanded['status']['B'], original['status']['B'])
        self.assertEqual(expanded['status']['C'], original['status']['C'])
        self.assertEqual(len(expanded['comparisons']), 1)
        for key in ('interval', 'sign_test_p', 'sign_flip_p', 'interval_family_size'):
            self.assertEqual(expanded['comparisons'][0][key], original['comparisons'][0][key])
        self.assertEqual(expanded['comparisons'][0]['pair_members']['A_alias'],
                         ['A_alias', 'B'])
        self.assertEqual(set(expanded['retained']),
                         {name for name, status in expanded['status'].items()
                          if status != 'excluded'})

    def test_contradictory_equivalence_declaration_is_rejected(self):
        data = self.data([{'probe': 0.0} for _ in range(12)])
        candidates = {'a': {'probe': 0}, 'b': {'probe': 10}}
        with self.assertRaisesRegex(ValueError, 'incompatible tested prediction rules'):
            evaluate(data, candidates, ['probe'], 'absolute', tolerance=.1,
                     equivalence_groups=[('a', 'b')])
        with self.assertRaisesRegex(ValueError, 'incompatible tested prediction rules'):
            compare(data, candidates, ['probe'], equivalence_groups=[('a', 'b')])

    def test_matching_observations_do_not_make_different_anchor_rules_equivalent(self):
        data = self.data([{'baseline': 0.0, 'full': 0.0, 'probe': 0.0}
                          for _ in range(12)])
        candidates = {'a': {'probe': 'same_as:baseline'},
                      'b': {'probe': 'same_as:full'}}
        with self.assertRaisesRegex(ValueError, 'incompatible tested prediction rules'):
            compare(data, candidates, ['probe'], equivalence_groups=[('a', 'b')])

    def test_untested_metadata_does_not_split_equivalent_rules(self):
        data = self.data([{'probe': 0.0} for _ in range(12)])
        candidates = {'a': {'probe': 0, 'unused': 'same_as:missing'},
                      'b': {'probe': 0.0}}
        out = evaluate(data, candidates, ['probe'], 'absolute', tolerance=.1,
                       resamples=100, equivalence_groups=[('a', 'b')])
        self.assertEqual(out['status'], {'a': 'adequate', 'b': 'adequate'})
        self.assertEqual(out['comparisons'], [])
        self.assertEqual(compare(data, candidates, ['probe'],
                                 equivalence_groups=[('a', 'b')]), [])

    def test_all_ties_report_no_untied_evidence_not_nan_or_equivalence(self):
        data = self.data([{'probe': 0.0} for _ in range(12)])
        out = compare(data, {'a': {'probe': -1}, 'b': {'probe': 1}}, ['probe'],
                      resamples=100, permutations=100)[0]
        self.assertEqual(out['ties'], 12)
        self.assertEqual(out['sign_test_p'], 1.0)
        self.assertEqual(out['sign_test_status'], 'no_untied_units')
        self.assertEqual(out['sign_flip_p'], 1.0)
        self.assertNotIn('adequate', out)

    def test_nominal_intervals_and_unadjusted_p_values_are_explicit(self):
        data = self.data([{'probe': float(i % 3)} for i in range(12)])
        candidates = {str(i): {'probe': i} for i in range(3)}
        out = compare(data, candidates, ['probe'], resamples=100, permutations=100)
        self.assertEqual(len(out), 3)
        for row in out:
            self.assertEqual(row['interval_family_size'], 3)
            self.assertEqual(row['p_value_family_size_per_test'], 3)
            self.assertEqual(row['p_value_adjustment'], 'none')
            self.assertIn('coverage not guaranteed', row['interval_calibration'])

    def test_invalid_measurements_cannot_become_an_all_ties_success(self):
        candidates = {'a': {'probe': 0}, 'b': {'probe': 1}}
        for bad in (math.nan, math.inf):
            data = {str(i): {'r': {'probe': bad}} for i in range(12)}
            with self.assertRaisesRegex(ValueError, 'non-finite measurement'):
                compare(data, candidates, ['probe'])
            with self.assertRaisesRegex(ValueError, 'non-finite measurement'):
                evaluate(data, candidates, ['probe'], 'absolute', tolerance=.1)
        with self.assertRaisesRegex(ValueError, 'no data'):
            compare({}, candidates, ['probe'])
        with self.assertRaisesRegex(ValueError, 'unique'):
            compare(self.data([{'probe': 0.0} for _ in range(12)]), candidates,
                    ['probe', 'probe'])

    def test_finite_measurements_with_overflowing_errors_fail_closed(self):
        data = self.data([{'probe': 1e308} for _ in range(12)])
        candidates = {'a': {'probe': -1e308}, 'b': {'probe': -9e307}}
        with self.assertRaisesRegex(ValueError, 'non-finite derived statistic'):
            compare(data, candidates, ['probe'], resamples=20, permutations=20)
        with self.assertRaisesRegex(ValueError, 'non-finite derived statistic'):
            evaluate(data, candidates, ['probe'], 'absolute', tolerance=.1,
                     resamples=20, permutations=20)

    def test_finite_loss_with_overflowing_tolerance_contrast_fails_closed(self):
        data = self.data([{'probe': 1e307} for _ in range(12)])
        with self.assertRaisesRegex(ValueError, 'non-finite derived statistic'):
            evaluate(data, {'a': {'probe': 0}}, ['probe'], 'signed',
                     tolerance=1.75e308, resamples=20)

    def test_overflow_in_relative_tolerance_is_not_false_adequacy(self):
        data = self.data([{'probe': 1.0, 'low': -1e308, 'high': 1e308}
                          for _ in range(12)])
        with self.assertRaisesRegex(ValueError, 'non-finite derived statistic'):
            evaluate(data, {'a': {'probe': 0}}, ['probe'], 'absolute',
                     tolerance_fraction=.1, scale=('low', 'high'), resamples=20)


if __name__ == '__main__':
    unittest.main()
