"""Check the planning example's decisions and its fail-closed input semantics."""
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('preflight_example', ROOT / 'examples/causal_preflight.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestCausalPreflight(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(MODULE.EXAMPLE.read_text())

    def result(self, **kwargs):
        return MODULE.check(self.config, **kwargs)

    def test_example_distinguishes_design_and_measurement_changes(self):
        small = self.result(selected=['full_patch'], n=64)
        large = self.result(selected=['full_patch'], n=1000000)
        self.assertEqual(small['identical_mean_groups'], [['A', 'B']])
        self.assertEqual(large['pairs'][0]['status'], 'identical_declared_means')
        self.assertEqual(small['pairs_separated_by_all_optional_cells'], [('A', 'B')])
        self.assertEqual(self.result(n=64)['pairs'][0]['status'], 'planning_overlap')
        self.assertEqual(self.result(n=256)['pairs'][0]['status'], 'planning_separated')

    def test_more_units_reduce_only_statistical_radius(self):
        a = self.result(n=64)['pairs'][0]['cells'][1]
        b = self.result(n=256)['pairs'][0]['cells'][1]
        self.assertAlmostEqual(a['statistical_radius'], 2 * b['statistical_radius'])
        self.assertEqual(a['numerical_allowance'], b['numerical_allowance'])

    def test_one_separating_pair_does_not_certify_all_rivals(self):
        self.config['predictions']['C'] = [2.0, 0.3]
        result = self.result(n=256)
        self.assertFalse(result['all_pairs_clear_planning_screen'])
        self.assertEqual([p['status'] for p in result['pairs']],
                         ['planning_separated', 'planning_separated', 'identical_declared_means'])

    def test_strict_boundary_and_numeric_floor(self):
        # Binary-exact values ensure this tests the strict rule, not decimal rounding.
        self.config['predictions'] = {'A': [0, 0], 'B': [0, 0.25]}
        for cell in self.config['cells']:
            cell['sd_per_unit'] = 0
            cell['numerical_allowance'] = 0.125
        self.assertEqual(self.result(n=64)['pairs'][0]['status'],
                         'numerical_allowance_blocks_this_screen')
        self.assertEqual(self.result(n=1000000)['pairs'][0]['status'],
                         'numerical_allowance_blocks_this_screen')
        self.config['predictions']['B'][1] = math.nextafter(0.25, math.inf)
        self.assertEqual(self.result()['pairs'][0]['status'], 'planning_separated')

    def test_missing_calibration_is_not_a_pass(self):
        for key in ('sd_per_unit', 'numerical_allowance'):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config['cells'][1][key] = None
                result = MODULE.check(config, n=256)
                self.assertEqual(result['pairs'][0]['status'], 'resolution_unknown')
                self.assertEqual(result['identical_mean_groups'], [['A'], ['B']])
        self.config['n'] = None
        self.assertEqual(self.result()['pairs'][0]['status'], 'resolution_unknown')

    def test_optional_menu_preserves_multiplicity(self):
        result = self.result(selected=['path_a_only'], n=256)
        self.assertEqual(result['simultaneous_cell_count'], 2)
        self.assertAlmostEqual(result['pairs'][0]['cells'][0]['statistical_radius'], 0.0420263, places=6)

    def test_pilot_status_reaches_human_output(self):
        self.config['cells'][1]['sd_kind'] = 'pilot_estimate'
        result = self.result()
        self.assertIn('not covered', result['calibration'])
        output = io.StringIO()
        with redirect_stdout(output):
            MODULE.display(result)
        self.assertIn('pilot SD: provisional', output.getvalue())

    def test_invalid_structural_tables_rejected(self):
        for table in ({'A': [1, 2]}, {'A': [], 'B': []},
                      {'A': [1, 2], 'B': [1]},
                      {'A': [1, math.nan], 'B': [1, 2]},
                      {'A': [1, math.inf], 'B': [1, 2]},
                      {'A': [1, True], 'B': [1, 2]}):
            with self.subTest(table=table):
                self.config['predictions'] = table
                with self.assertRaises(ValueError):
                    self.result()

    def test_invalid_units_cells_and_alpha_rejected(self):
        for n in (0, -1, True, 3.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                self.result(n=n)
        for cells in ([], ['full_patch', 'full_patch'], ['absent'], [True]):
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                self.result(selected=cells)
        for alpha in (0, 1, -1, math.nan, 1e-300):
            config = copy.deepcopy(self.config)
            config['alpha'] = alpha
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                MODULE.check(config)
        self.config['independent_unit'] = ' '
        with self.assertRaises(ValueError):
            self.result()

    def test_invalid_calibration_and_overflow_rejected(self):
        for key in ('sd_per_unit', 'numerical_allowance'):
            for value in (-0.1, math.nan, math.inf, True):
                config = copy.deepcopy(self.config)
                config['cells'][1][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    MODULE.check(config)
        self.config['predictions'] = {'A': [1e308, 0], 'B': [-1e308, 0]}
        with self.assertRaises(ValueError):
            self.result()


if __name__ == '__main__':
    unittest.main()
