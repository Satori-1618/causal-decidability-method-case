"""Planning changes must clear both modeled floors, not just the larger one."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from causal_decidability import calculator as calc  # noqa: E402


class PlanningGuidance(unittest.TestCase):
    def plan(self, **changes):
        args = dict(predictions={'a': 0.0, 'b': 1.0}, n=10, sigma=10.0,
                    dtype='bfloat16', readout_scale=100.0)
        args.update(changes)
        return args, calc.decidability(**args)

    def check_recommendation(self, args, result):
        """Execute the suggested declarations through the public calculator again."""
        lever = result['lever']
        updated = dict(args)
        if lever['replicates_needed'] is not None:
            updated['n'] = lever['replicates_needed']
        if lever['recommended_dtype'] is not None:
            updated['dtype'] = lever['recommended_dtype']
        again = calc.decidability(**updated)
        self.assertEqual(again['decidable'], lever['clears_both_floors'])
        self.assertAlmostEqual(again['statistical_floor'], lever['projected_statistical_floor'])
        self.assertAlmostEqual(again['numerical_floor'], lever['projected_numerical_floor'])
        self.assertAlmostEqual(again['ratio'], lever['projected_ratio'])
        return again

    def test_audited_both_floor_regression(self):
        args, result = self.plan()
        lever = result['lever']
        self.assertEqual(result['binding_floor'], 'statistical')
        self.assertGreater(result['numerical_floor'], result['separation'])
        self.assertEqual(lever['kind'], 'units_and_precision')
        self.assertEqual(lever['blocking_floors'], ['statistical', 'numerical'])
        self.assertFalse(lever['units_alone_sufficient'])
        self.assertFalse(lever['precision_alone_sufficient'])
        self.assertFalse(calc.decidability(**dict(args, n=lever['replicates_needed']))['decidable'])
        self.assertFalse(calc.decidability(**dict(args, dtype=lever['recommended_dtype']))['decidable'])
        self.assertTrue(self.check_recommendation(args, result)['decidable'])
        self.assertEqual(lever['remaining_blockers'], [])

    def test_both_floors_with_numerical_one_binding(self):
        args, result = self.plan(sigma=0.8)
        self.assertEqual(result['binding_floor'], 'numerical')
        self.assertGreater(result['statistical_floor'], 1.0)
        self.assertEqual(result['lever']['kind'], 'units_and_precision')
        self.assertTrue(self.check_recommendation(args, result)['decidable'])

    def test_single_statistical_blocker(self):
        args, result = self.plan(dtype='float32', readout_scale=1.0)
        lever = result['lever']
        self.assertEqual(lever['kind'], 'units_or_noise')
        self.assertTrue(lever['units_alone_sufficient'])
        self.assertIsNone(lever['extra_mantissa_bits'])
        self.assertIsNone(lever['recommended_dtype'])
        self.assertTrue(self.check_recommendation(args, result)['decidable'])

    def test_single_numerical_blocker_and_torch_dtype_alias(self):
        for dtype in ('bfloat16', 'torch.bfloat16'):
            with self.subTest(dtype=dtype):
                args, result = self.plan(sigma=0.0, dtype=dtype)
                lever = result['lever']
                self.assertEqual(lever['kind'], 'precision')
                self.assertTrue(lever['precision_alone_sufficient'])
                self.assertIsNone(lever['replicates_needed'])
                self.assertEqual(lever['recommended_dtype'], 'float16')
                self.assertTrue(self.check_recommendation(args, result)['decidable'])

    def test_statistical_boundary_requires_an_actual_increase(self):
        args, result = self.plan(n=1, sigma=1.0, noise_factor=1.0, z=1.0,
                                 readout_scale=0.0)
        self.assertEqual(result['ratio'], 1.0)
        self.assertFalse(result['decidable'])
        self.assertEqual(result['lever']['replicates_needed'], 2)
        self.assertTrue(self.check_recommendation(args, result)['decidable'])

    def test_numerical_boundary_requires_extra_bits(self):
        gap = calc.numerical_floor('bfloat16', 100.0)
        args, result = self.plan(predictions={'a': 0.0, 'b': gap}, sigma=0.0)
        self.assertEqual(result['ratio'], 1.0)
        self.assertFalse(result['decidable'])
        self.assertEqual(result['lever']['extra_mantissa_bits'], 1)
        self.assertTrue(self.check_recommendation(args, result)['decidable'])

    def test_both_at_exact_boundary_require_both_changes(self):
        gap = calc.numerical_floor('bfloat16', 100.0)
        args, result = self.plan(predictions={'a': 0.0, 'b': gap}, n=1,
                                 sigma=gap, noise_factor=1.0, z=1.0)
        self.assertEqual(result['ratio'], 1.0)
        self.assertEqual(result['lever']['kind'], 'units_and_precision')
        self.assertTrue(self.check_recommendation(args, result)['decidable'])

    def test_zero_floors_and_zero_separation_are_distinct(self):
        _, result = self.plan(sigma=0, readout_scale=0)
        self.assertEqual(result['ratio'], math.inf)
        self.assertEqual(result['lever']['kind'], 'none')
        self.assertTrue(result['lever']['clears_both_floors'])
        _, no_separation = self.plan(predictions={'a': 0, 'b': 0}, sigma=0,
                                     readout_scale=0)
        self.assertEqual(no_separation['ratio'], 0)
        self.assertEqual(no_separation['lever']['remaining_blockers'], ['structural'])
        self.assertFalse(no_separation['lever']['clears_both_floors'])

    def test_no_comparison_is_not_a_sample_size_problem(self):
        _, result = self.plan(equivalence_groups=[('a', 'b')])
        self.assertIsNone(result['closest_rivals'])
        self.assertIn('no distinct candidate groups', result['lever']['detail'])
        self.assertIsNone(result['lever']['replicates_needed'])

    def test_unsupported_precision_is_not_reported_as_solved(self):
        args, result = self.plan(dtype='float64', readout_scale=1e20, sigma=0)
        lever = result['lever']
        self.assertIsNone(lever['recommended_dtype'])
        self.assertGreater(lever['extra_mantissa_bits'], 0)
        self.assertEqual(lever['remaining_blockers'], ['numerical'])
        self.assertIn('no supported dtype', lever['detail'])
        self.assertFalse(self.check_recommendation(args, result)['decidable'])

    def test_large_gap_needs_no_change(self):
        args, result = self.plan(predictions={'a': 0, 'b': 1000})
        self.assertEqual(result['lever']['kind'], 'none')
        self.assertTrue(self.check_recommendation(args, result)['decidable'])

    def test_derived_overflow_is_an_invalid_plan(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.plan(sigma=1e308)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.plan(predictions={'a': -1e308, 'b': 1e308})


class PlanningCLI(unittest.TestCase):
    def run_cli(self, *extra):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        return subprocess.run([sys.executable, '-m', 'causal_decidability',
                               '--prediction', 'a=0', '--prediction', 'b=1',
                               '--n', '10', '--sigma', '10', '--dtype', 'bfloat16',
                               '--readout-scale', '100', *extra],
                              cwd=ROOT, env=env, capture_output=True, text=True)

    def test_cli_warns_and_reports_both_changes(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('planning heuristic', result.stdout)
        self.assertIn('both changes are needed', result.stdout)
        self.assertIn('not a power calculation', result.stdout)
        self.assertIn('identification guarantee', result.stdout)

    def test_json_keeps_machine_readable_limitations_and_recommendations(self):
        result = self.run_cli('--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload['decidable'])
        self.assertIn('Planning heuristic', payload['interpretation'])
        self.assertEqual(payload['lever']['kind'], 'units_and_precision')
        self.assertGreater(payload['lever']['projected_ratio'], 1.0)
        self.assertTrue(payload['lever']['clears_both_floors'])


if __name__ == '__main__':
    unittest.main()
