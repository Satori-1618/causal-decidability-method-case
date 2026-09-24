"""A declared-bounds teaching example, never a substitute for measured LLM evidence."""
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))

import iterative_path_test as demo  # noqa: E402


class IterativePathTest(unittest.TestCase):
    def setUp(self):
        self.cells = {'y00': 1, 'y10': 3, 'y01': 1, 'y11': 1}

    def test_four_worlds_have_distinct_and_limited_outcomes(self):
        worlds = {w['name']: w for w in demo.demonstration()['worlds']}
        self.assertEqual(len(worlds), 4)
        self.assertEqual(worlds['route']['outcome'], 'selected_queries_endpoint_adequate')
        self.assertEqual(worlds['bypass']['outcome'], 'bypass_endpoint_adequate')
        self.assertEqual(worlds['intermediate']['outcome'], 'no_candidate_fits')
        self.assertEqual(worlds['intermediate']['retained_candidates'], [])
        self.assertEqual(worlds['insufficient_resolution']['outcome'], 'insufficient_resolution')
        for world in worlds.values():
            self.assertIn('NOT LLM RESULTS', world['source'])
            self.assertEqual(set(world['cells']), set(demo.CELLS))
            self.assertEqual(set(world['contrasts']), {'T', 'R', 'K'})
            self.assertEqual(world['cells']['y01'], world['cells']['y00'])

    def test_existing_design_functions_show_what_the_extra_arm_buys(self):
        design = demo.analyze_world(self.cells)['design']
        self.assertEqual(design['before_groups'], [['bypass', 'selected_queries']])
        self.assertEqual(design['after_groups'], [['bypass'], ['selected_queries']])
        self.assertEqual(design['newly_separated_pairs'], [('bypass', 'selected_queries')])

    def test_additive_global_and_clamp_offsets_cancel(self):
        # Pure algebra: cancellation does not establish a valid query self-clamp.
        # The added clamp offset breaks its off-patch identity, so do not authorize
        # a mechanistic verdict merely because the differences are unchanged.
        original = demo.analyze_world(self.cells)
        shifted = {name: value + 7 + (11 if name in ('y01', 'y11') else 0)
                   for name, value in self.cells.items()}
        changed = demo.analyze_world(shifted, controls_passed=False)
        for contrast in ('T', 'R', 'K'):
            self.assertAlmostEqual(original['contrasts'][contrast], changed['contrasts'][contrast])
        self.assertEqual(changed['outcome'], 'invalid_control')

    def test_patch_by_clamp_interaction_does_not_cancel(self):
        original = demo.analyze_world(self.cells)
        changed = demo.analyze_world(dict(self.cells, y11=self.cells['y11'] + .7))
        self.assertAlmostEqual(changed['contrasts']['T'], original['contrasts']['T'])
        self.assertAlmostEqual(changed['contrasts']['R'] - original['contrasts']['R'], .7)
        self.assertAlmostEqual(changed['contrasts']['K'] - original['contrasts']['K'], -.7)
        self.assertEqual(changed['outcome'], 'no_candidate_fits')

    def test_worst_case_error_bounds_hold_and_are_attainable(self):
        error = .125  # exactly representable; all 16 extreme error assignments
        base = {'y00': 1, 'y10': 3, 'y01': 1.5, 'y11': 2.5}
        result = demo.analyze_world(base, per_cell_error=error)
        maxima = {name: 0 for name in ('T', 'R', 'K')}
        for signs in itertools.product((-1, 1), repeat=4):
            perturbed = {name: base[name] + sign * error
                         for name, sign in zip(demo.CELLS, signs)}
            actual = demo.analyze_world(perturbed, per_cell_error=0)['contrasts']
            for name in maxima:
                delta = abs(actual[name] - result['contrasts'][name])
                maxima[name] = max(maxima[name], delta)
                self.assertLessEqual(delta, result['deterministic_radii'][name])
        self.assertEqual(maxima, {'T': 2 * error, 'R': 2 * error, 'K': 4 * error})

    def test_separation_gate_is_strict_at_its_boundary(self):
        # 2*.25 + 6*.125 = 1.25; equality must not count as resolved.
        cells = {'y00': 0, 'y10': 1.25, 'y01': 0, 'y11': 0}
        result = demo.analyze_world(cells, tolerance=.25, per_cell_error=.125)
        self.assertEqual(result['resolution_gate']['threshold'], 1.25)
        self.assertFalse(result['resolution_gate']['passed'])
        self.assertEqual(result['outcome'], 'insufficient_resolution')
        changed = demo.analyze_world(dict(cells, y10=1.25 + 1e-9),
                                    tolerance=.25, per_cell_error=.125)
        self.assertTrue(changed['resolution_gate']['passed'])
        self.assertEqual(changed['outcome'], 'selected_queries_endpoint_adequate')

    def test_a_lone_retained_candidate_is_not_automatically_a_winner(self):
        result = demo.analyze_world(dict(self.cells, y11=1.19))
        self.assertTrue(result['resolution_gate']['passed'])
        self.assertEqual(result['candidate_status']['selected_queries'], 'undecided')
        self.assertEqual(result['candidate_status']['bypass'], 'excluded')
        self.assertEqual(result['retained_candidates'], ['selected_queries'])
        self.assertEqual(result['outcome'], 'insufficient_resolution')

    def test_invalid_controls_block_a_scientific_verdict(self):
        result = demo.analyze_world(self.cells, controls_passed=False)
        self.assertEqual(result['outcome'], 'invalid_control')
        self.assertEqual(set(result['candidate_status'].values()), {'not_assessed'})
        self.assertEqual(result['retained_candidates'], list(demo.CANDIDATES))

    def test_zero_total_effect_does_not_identify_either_endpoint(self):
        result = demo.analyze_world(dict(self.cells, y10=self.cells['y00']), per_cell_error=0)
        self.assertEqual(result['contrasts']['T'], 0)
        self.assertEqual(result['design']['newly_separated_pairs'], [])
        self.assertEqual(result['outcome'], 'insufficient_resolution')
        self.assertEqual(set(result['candidate_status'].values()), {'not_assessed'})

    def test_negative_total_effect_is_also_testable(self):
        result = demo.analyze_world({name: -value for name, value in self.cells.items()})
        self.assertEqual(result['outcome'], 'selected_queries_endpoint_adequate')

    def test_invalid_and_overflowing_inputs_are_refused(self):
        for value in (math.nan, math.inf, -math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                demo.analyze_world(dict(self.cells, y11=value))
        for kwargs in ({'per_cell_error': -.1}, {'tolerance': -.1},
                       {'per_cell_error': math.inf}, {'controls_passed': 'yes'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                demo.analyze_world(self.cells, **kwargs)
        with self.assertRaises(ValueError):
            demo.analyze_world(dict(self.cells, y00=-1e308, y10=1e308))
        with self.assertRaises(ValueError):
            demo.analyze_world(self.cells, per_cell_error=1e308)
        with self.assertRaises(ValueError):
            demo.analyze_world({'y00': 0})

    def test_standalone_isolated_json_needs_no_site_packages(self):
        result = subprocess.run([sys.executable, '-I', '-S',
                                 str(ROOT / 'examples' / 'iterative_path_test.py'), '--json'],
                                cwd='/', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn('NOT LLM RESULTS', payload['notice'])
        self.assertEqual(len(payload['worlds']), 4)
        self.assertEqual(len(payload['guard_examples']), 2)
        self.assertIn('no empirical coverage claim', payload['bounds'])


if __name__ == '__main__':
    unittest.main()
