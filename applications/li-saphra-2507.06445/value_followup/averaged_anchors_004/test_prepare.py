"""Sampling and freeze tests only: no neural runtime, checkpoint, or outcomes."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('averaged_prepare_test', HERE / 'prepare.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class InputPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((HERE / 'inputs/preparation.json').read_text())
        cls.exclusions = json.loads((HERE / 'inputs/exclusions.json').read_text())
        cls.candidates = p.read_rows(HERE / 'inputs/candidates.jsonl')
        cls.families = p.read_rows(HERE / 'inputs/donor_families.jsonl')
        cls.plan = json.loads((HERE / 'plan.json').read_text())

    def test_standard_library_only_import(self):
        script = "import importlib.util,sys;s=importlib.util.spec_from_file_location('p',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable, '-B', '-S', '-c', script, str(HERE / 'prepare.py')], check=True)

    def test_full_frozen_manifest_and_plan(self):
        manifest, candidates, families = p.validate_inputs(self.plan, HERE / 'inputs', regenerate=False)
        self.assertEqual((len(candidates), len(families)), (2048, 256))
        self.assertEqual(manifest['statistics']['donor_draws'], 4096)
        self.assertEqual(manifest['statistics']['global_cross_role_prefix_overlap'], {'20': 0, '28': 0})

    def test_deterministic_clean_regeneration(self):
        self.assertEqual(p.validate(HERE / 'inputs'), self.manifest)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                p.prepare(directory)

    def test_fixed_historical_support_and_role_pool(self):
        self.assertEqual(self.exclusions, p.historical_exclusions())
        s = self.manifest['statistics']
        self.assertEqual((s['historical_full_strings'], s['historical_prefixes']['20'], s['historical_prefixes']['28']),
                         (18475, 16753, 18447))
        self.assertEqual(s['exact_fresh_p20_support_before_suffix_check']['2']['confirmation_calibration'], 1045)
        self.assertEqual(s['exact_fresh_p20_support_before_suffix_check']['2']['confirmation_target'], 1099)

    def test_both_prefix_bans_apply_even_when_p28_is_not_read(self):
        exclusions = copy.deepcopy(self.exclusions)
        donor = next(d for d in self.families[0]['donors'] if d['position'] == 20)
        exclusions['prefixes']['28'].append(donor['string'][:28])
        with self.assertRaisesRegex(ValueError, 'prefix leakage'):
            p.validate_records(self.candidates, self.families, exclusions)

    def test_role_assignment_cannot_leak_across_read_positions(self):
        donor = self.families[0]['donors'][0]
        self.assertEqual(p.role_of(donor['string'][:20]), p.role_of(donor['string'][:28]))
        self.assertEqual(p.role_of(donor['string'][:20]), p.role_of(donor['string'][:20] + '))))(((('))
        families = copy.deepcopy(self.families)
        d = families[0]['donors'][0]
        d['role'] = 'target' if d['role'] == 'calibration' else 'calibration'
        d['cell'] = d['role'] + '_' + '_'.join(d['cell'].split('_')[1:])
        with self.assertRaises(ValueError):
            p.validate_records(self.candidates, families, self.exclusions)

    def test_four_distinct_measured_prefixes_per_family_cell(self):
        for family in self.families:
            for balance in (-2, 2):
                for position in (20, 28):
                    donors = [d for d in family['donors'] if (d['balance'], d['position']) == (balance, position)]
                    self.assertEqual(len({d['prefix_sha256'] for d in donors}), 4)
        families = copy.deepcopy(self.families)
        first, second = families[0]['donors'][:2]
        second['string'], second['prefix_sha256'] = first['string'], first['prefix_sha256']
        with self.assertRaisesRegex(ValueError, 'Repeated measured prefix'):
            p.validate_records(self.candidates, families, self.exclusions)

    def test_recipient_observations_are_not_accepted_as_inputs(self):
        candidates = copy.deepcopy(self.candidates)
        candidates[0]['margin'] = 0.0
        with self.assertRaisesRegex(ValueError, 'Invalid recipient'):
            p.validate_records(candidates, self.families, self.exclusions)

    def test_sampling_plan_change_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan['sampling']['recipient_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'Plan sampling differs'):
            p.validate_inputs(plan, HERE / 'inputs', regenerate=False)

    def test_bounded_failure_not_silent_resampling(self):
        recipe = dict(p.RECIPE, attempt_limit_per_draw=0)
        with self.assertRaisesRegex(RuntimeError, 'Recipient sampling exhausted'):
            p.draw_candidates(self.exclusions, recipe)
        with self.assertRaisesRegex(RuntimeError, 'Donor sampling exhausted'):
            p.draw_families(self.exclusions, recipe)

    def test_file_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'inputs'
            shutil.copytree(HERE / 'inputs', target)
            with (target / 'candidates.jsonl').open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'Prepared file changed'):
                p.validate(target, regenerate=False)

    def test_unexpected_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'inputs'
            shutil.copytree(HERE / 'inputs', target)
            (target / 'scores.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'Unexpected input inventory'):
                p.validate(target, regenerate=False)

    def test_reuse_and_overlap_reported_without_posthoc_deduplication(self):
        s = p.statistics(self.candidates, self.families, self.exclusions)
        self.assertEqual(s['maximum_measured_prefix_reuse'], 4)
        self.assertEqual(s['unique_donor_strings'], 4094)
        self.assertEqual(s['new_recipient_donor_full_string_overlap'], 0)
        self.assertEqual(s['new_recipient_donor_prefix_overlap'], {'20': 11, '28': 0})


if __name__ == '__main__':
    unittest.main()
