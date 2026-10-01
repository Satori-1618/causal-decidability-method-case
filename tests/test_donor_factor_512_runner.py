"""A larger cap must not authorize a changed contract, pilot or failed planning."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import pytest

pytest.importorskip('torch', reason='frozen runner imports require .[test-hooks]')

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'applications/makelov-2311.17030/scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('factor_runner_512', SCRIPTS/'run_donor_factor_512.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AmendmentAuthorizationTests(unittest.TestCase):
    def decision(self):
        original = json.loads((runner.ROOT/'results/donor_factor_planning/planning.json').read_text())
        decision = copy.deepcopy(original)
        decision.update({'schema_version': 'donor-factor-planning-512-v1',
                         'status': 'SELECTED', 'selected_n': 512,
                         'amendment': runner.amendment_binding(),
                         'planner_code_sha256': runner.sha256(runner.ROOT/'src/donor_factor_planning_512.py')})
        decision['settings']['candidate_sizes'].append(512)
        decision['candidates'].append({'n': 512, 'passes': True, 'coverage': {'passes': True},
                                      'normal_reference': {'passes': True},
                                      'bootstrap_checks': {d: {'passes': True}
                                                           for d in ('gaussian', 'empirical')}})
        return decision

    def test_only_grid_changes_and_all_original_dependencies_remain(self):
        contract = copy.deepcopy(runner.CONTRACT)
        self.assertEqual(contract['planning']['n_grid'].pop(), 512)
        self.assertEqual(contract, runner.ORIGINAL_CONTRACT)
        self.assertEqual(runner.ORIGINAL_CONTRACT['planning']['n_grid'], [128, 192, 256, 384])
        self.assertEqual(len(runner.code_files()), 13)
        self.assertTrue(set(runner.original_code_files()) <= set(runner.code_files()))

    def test_must_bind_original_pilot_stop_and_all_gates(self):
        cases = runner.ROOT/'results/donor_factor_development/cases.json'
        good = self.decision()
        self.assertEqual(runner.validate_planning(good, cases), 512)
        for key, value in [('schema_version', 'donor-factor-planning-v1'),
                           ('status', 'STOP'), ('status', 'TEST_ONLY'), ('selected_n', 384),
                           ('protocol_matches_development_size', False),
                           ('protocol_matches_frozen_simulation_counts', False),
                           ('amendment', {}), ('planner_code_sha256', '0'*64)]:
            bad = copy.deepcopy(good)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                runner.validate_planning(bad, cases)
        for distribution in ('gaussian', 'empirical'):
            bad = copy.deepcopy(good)
            bad['candidates'][-1]['bootstrap_checks'][distribution]['passes'] = False
            with self.subTest(distribution=distribution), self.assertRaises(RuntimeError):
                runner.validate_planning(bad, cases)
        bad = copy.deepcopy(good)
        bad['development_binding']['records.jsonl'] = '0'*64
        with self.assertRaises(RuntimeError):
            runner.validate_planning(bad, cases)
        bad = copy.deepcopy(good)
        bad['candidates'][0]['passes'] = True
        with self.assertRaises(RuntimeError):
            runner.validate_planning(bad, cases)


if __name__ == '__main__':
    unittest.main()
