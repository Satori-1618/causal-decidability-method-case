"""The confirmation preparer must reject test-only or unbound planning files."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'applications/makelov-2311.17030/scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('factor_runner', SCRIPTS/'run_donor_factor.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PlanningAuthorizationTests(unittest.TestCase):
    def test_status_counts_bound_files_and_actual_candidate_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('cases.json', 'records.jsonl', 'summary.json', 'manifest.json'):
                (root/name).write_text('{}\n')
            decision = {'status': 'SELECTED', 'selected_n': 192,
                        'protocol_matches_frozen_simulation_counts': True,
                        'protocol_matches_development_size': True, 'n_development_units': 32,
                        'candidates': [{'n': 192, 'passes': True, 'coverage': {'passes': True},
                                        'normal_reference': {'passes': True},
                                        'bootstrap_checks': {d: {'passes': True}
                                                            for d in ('gaussian', 'empirical')}}],
                        'development_binding': {p.name: runner.sha256(p) for p in root.iterdir()},
                        'planner_code_sha256': runner.sha256(runner.ROOT/'src/donor_factor_planning.py')}
            self.assertEqual(runner.validate_planning(decision, root/'cases.json'), 192)
            for key, value in (('status', 'TEST_ONLY'), ('status', 'STOP'),
                               ('protocol_matches_frozen_simulation_counts', False),
                               ('protocol_matches_development_size', False),
                               ('n_development_units', 31)):
                bad = copy.deepcopy(decision)
                bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    runner.validate_planning(bad, root/'cases.json')
            bad = copy.deepcopy(decision)
            bad['candidates'][0]['bootstrap_checks']['empirical']['passes'] = False
            with self.assertRaises(RuntimeError):
                runner.validate_planning(bad, root/'cases.json')
            (root/'records.jsonl').write_text('changed\n')
            with self.assertRaises(RuntimeError):
                runner.validate_planning(decision, root/'cases.json')


if __name__ == '__main__':
    unittest.main()
