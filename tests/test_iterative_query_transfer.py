"""The fifth arm separates responses without manufacturing graph identification."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))
import iterative_query_transfer as demo


class IterativeQueryTransfer(unittest.TestCase):
    def test_four_arms_are_identical_but_reverse_transplant_separates(self):
        result = demo.demonstration()
        worlds = {w['name']: w for w in result['worlds']}
        a, b = worlds['transfer']['cells'], worlds['joint_dependence']['cells']
        self.assertEqual([a[k] for k in 'ABCD'], [b[k] for k in 'ABCD'])
        self.assertEqual((a['E'], b['E']), (1, 0))
        self.assertIn(('joint_dependence', 'transfer'), result['newly_separated_pairs'])
        self.assertIn(['cancelled_extra_paths', 'joint_dependence', 'transfer'],
                      result['before_A_to_D'])

    def test_delivered_analyzer_resolves_all_three_profiles_per_constructed_pair(self):
        worlds = {w['name']: w for w in demo.demonstration()['worlds']}
        for name in ('transfer', 'joint_dependence', 'preservation'):
            with self.subTest(name=name):
                fits = worlds[name]['profile_fits']
                self.assertEqual([p for p, fit in fits.items() if fit], [name])
                # Do not fake a large independent sample by replicating the table.
                self.assertEqual(worlds[name]['population_status'], 'unresolved')

    def test_cancelled_extra_paths_remain_unidentified_by_all_seven_arms(self):
        result = demo.demonstration()
        worlds = {w['name']: w for w in result['worlds']}
        self.assertEqual(worlds['transfer']['cells'], worlds['cancelled_extra_paths']['cells'])
        self.assertIn(['cancelled_extra_paths', 'transfer'], result['after_A_to_E'])
        self.assertEqual(demo.cancellation(1, 1), 1)
        self.assertEqual(demo.cancellation(1, 1, u_override=0), 0)
        self.assertEqual(result['outside_menu_counterexample']['transfer_output'], 1)

    def test_partial_response_can_fit_none_instead_of_forcing_a_winner(self):
        world = demo.analyze_world('partial', lambda x, m: .5 * x + .5 * m, 'Y=(X+M)/2')
        self.assertFalse(any(world['profile_fits'].values()))
        self.assertEqual(world['contrasts']['R'], .5)
        self.assertEqual(world['contrasts']['S'], .5)

    def test_self_clamp_controls_still_fail_closed_in_the_shipped_analyzer(self):
        # A and C have the same intervention inputs; a stateful counterfeit fails.
        count = 0
        def counterfeit(x, m):
            nonlocal count
            count += 1
            return m + int(count == 3)
        with self.assertRaisesRegex(ValueError, 'identity exceeds tolerance'):
            demo.analyze_world('bad', counterfeit, 'not a stable structural function')

    def test_isolated_cli_runs_outside_repository_without_site_packages(self):
        for flag in ([], ['--json']):
            result = subprocess.run([sys.executable, '-I', '-S',
                                     str(ROOT / 'examples/iterative_query_transfer.py'), *flag],
                                    cwd='/', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            if flag:
                value = json.loads(result.stdout)
                self.assertIn('NOT LLM RESULTS', value['notice'])
                self.assertIn('not empirically verified', value['scope'])
            else:
                self.assertIn('WITHOUT the original patch', result.stdout)
                self.assertIn('not all possible causal graphs', result.stdout)


if __name__ == '__main__':
    unittest.main()
