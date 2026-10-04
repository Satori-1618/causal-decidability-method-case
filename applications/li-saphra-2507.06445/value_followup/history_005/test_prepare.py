"""Input construction, finite support and freshness tests; never load a model."""
import copy
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('history_prepare_tests', HERE / 'prepare.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def plan_fixture():
    return {'round': 'history_005', 'model': {'model': 'a9g0io1r', 'layer_one_based': 2, 'head_one_based': 1},
            'sampling': {'phase': 'confirmation', 'recipient_seed': 26100451, 'donor_seed': 26100452,
                         'native_candidates': 2048, 'families': 256, 'length': 32, 'opens': 16, 'closes': 16,
                         'recipient_requires_negative_prefix': True, 'role_rule': p.ROLE_RULE,
                         'role_assignment_uses_measurements': False, 'no_global_deduplication': True,
                         'historical_full_strings': 24617, 'historical_prefix20': 22548, 'historical_prefix28': 24587}}


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = p.read(HERE / 'inputs/preparation.json')
        cls.exclusions = p.read(HERE / 'inputs/exclusions.json')
        cls.support = p.read(HERE / 'inputs/eligible_stems.json')
        cls.candidates = p.read_rows(HERE / 'inputs/candidates.jsonl')
        cls.families = p.read_rows(HERE / 'inputs/donor_families.jsonl')

    def validate(self, families=None, exclusions=None, candidates=None):
        p.validate_records(self.candidates if candidates is None else candidates,
                           self.families if families is None else families,
                           self.exclusions if exclusions is None else exclusions, self.support)

    def test_import_does_not_load_neural_packages(self):
        code = 'import importlib.util,sys;s=importlib.util.spec_from_file_location("p",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert "torch" not in sys.modules;assert "numpy" not in sys.modules'
        subprocess.run([sys.executable, '-B', '-S', '-c', code, str(HERE / 'prepare.py')], check=True)

    def test_full_input_validation_and_plan_interface(self):
        manifest, candidates, families = p.validate_inputs(plan_fixture(), HERE / 'inputs', regenerate=False)
        self.assertEqual((len(candidates), len(families), manifest['statistics']['donor_draws']), (2048, 256, 4096))

    def test_deterministic_byte_regeneration(self):
        self.assertEqual(p.validate(HERE / 'inputs', regenerate=True), self.manifest)

    def test_support_matches_independent_combinatorial_enumeration(self):
        banned20 = set(self.exclusions['prefixes']['20']); banned28 = set(self.exclusions['prefixes']['28'])
        expected = {'calibration': [], 'target': []}; raw = 0
        # Ending at -4 after18 steps requires exactly seven opening brackets.
        for opening_indices in combinations(range(18), 7):
            opens = set(opening_indices)
            stem = ''.join('(' if i in opens else ')' for i in range(18))
            balance = 0; low = 0
            for symbol in stem:
                balance += 1 if symbol == '(' else -1; low = min(low, balance)
            if low < -4:
                continue
            self.assertEqual(balance, -4); raw += 1
            if stem + '((' in banned20:
                continue
            prefixes = [stem + core + tail for core in ('(())((', '(()()(') for tail in ('()()', '(())')]
            if any(text in banned28 or int(hashlib.sha256(('value-prefix-v1:' + text).encode()).hexdigest(), 16) % 5 == 0 for text in prefixes):
                continue
            parity = int(hashlib.sha256(('matched-history-005-role:' + stem + '((').encode()).hexdigest(), 16) % 2
            expected[('calibration', 'target')[parity]].append(stem)
        self.assertEqual(raw, 13260)
        self.assertEqual({role: sorted(values) for role, values in expected.items()}, self.support['pools'])
        self.assertEqual(self.support['role_pool_counts'], {'calibration': 2538, 'target': 2620})

    def test_all_factorial_edits_are_the_two_declared_swaps(self):
        for family in self.families:
            for role in ('calibration', 'target'):
                for replica in (0, 1):
                    quartet = {(d['recency'], d['ending']): d['string'] for d in family['donors'] if (d['role'], d['replica']) == (role, replica)}
                    self.assertEqual(len({s[:21] for s in quartet.values()}), 1)
                    self.assertEqual(len({s[28:] for s in quartet.values()}), 1)
                    for ending in ('alt', 'close'):
                        self.assertEqual([i + 1 for i, (a, b) in enumerate(zip(quartet[6, ending], quartet[10, ending])) if a != b], [22, 23])
                    for recency in (6, 10):
                        self.assertEqual([i + 1 for i, (a, b) in enumerate(zip(quartet[recency, 'alt'], quartet[recency, 'close'])) if a != b], [26, 27])
        self.validate()

    def test_no_calibration_target_prefix_overlap(self):
        donors = [d for f in self.families for d in f['donors']]
        for length in (20, 28):
            left = {d['string'][:length] for d in donors if d['role'] == 'calibration'}
            right = {d['string'][:length] for d in donors if d['role'] == 'target'}
            self.assertFalse(left & right)

    def test_wrong_history_factor_label_rejected(self):
        families = copy.deepcopy(self.families)
        families[0]['donors'][0]['last_minimum_position'] = 18
        with self.assertRaisesRegex(ValueError, 'factor labels'):
            self.validate(families=families)

    def test_same_bracket_counts_but_unmatched_future_suffix_rejected(self):
        families = copy.deepcopy(self.families)
        donor = families[0]['donors'][0]
        other_suffix = next(suffix for suffix in p.SUFFIXES if suffix != donor['string'][28:])
        donor['string'] = donor['string'][:28] + other_suffix
        with self.assertRaisesRegex(ValueError, 'Quartet stem/future matching'):
            self.validate(families=families)

    def test_repeated_stem_within_family_rejected(self):
        families = copy.deepcopy(self.families)
        quartet = [d for d in families[0]['donors'] if d['role'] == 'calibration' and d['replica'] == 0]
        for source in quartet:
            target = next(d for d in families[0]['donors'] if (d['role'], d['replica'], d['recency'], d['ending']) == ('calibration', 1, source['recency'], source['ending']))
            for key in ('string', 'prefix_sha256', 'stem_sha256'):
                target[key] = source[key]
        with self.assertRaisesRegex(ValueError, 'four distinct matched stems'):
            self.validate(families=families)

    def test_missing_factorial_cell_rejected(self):
        families = copy.deepcopy(self.families); families[0]['donors'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete factorial'):
            self.validate(families=families)

    def test_historical_shorter_prefix_ban_enforced(self):
        exclusions = copy.deepcopy(self.exclusions)
        exclusions['prefixes']['20'].append(self.families[0]['donors'][0]['string'][:20])
        with self.assertRaisesRegex(ValueError, 'freshness'):
            self.validate(exclusions=exclusions)

    def test_candidate_measurement_field_rejected(self):
        candidates = copy.deepcopy(self.candidates); candidates[0]['margin'] = 0.
        with self.assertRaisesRegex(ValueError, 'Invalid candidate'):
            self.validate(candidates=candidates)

    def test_plan_seed_change_rejected(self):
        plan = plan_fixture(); plan['sampling']['donor_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'Plan sampling differs'):
            p.validate_inputs(plan, HERE / 'inputs', regenerate=False)

    def test_exhaustion_and_overwrite_are_fail_closed(self):
        recipe = dict(p.RECIPE, candidate_attempt_limit_per_draw=0)
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            p.draw_candidates(self.exclusions, recipe)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                p.prepare(directory)

    def test_tampered_prepared_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'inputs'; shutil.copytree(HERE / 'inputs', target)
            with (target / 'donor_families.jsonl').open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'Prepared data changed'):
                p.validate(target, regenerate=False)

    def test_reuse_and_one_new_prefix_overlap_are_reported(self):
        s = self.manifest['statistics']
        self.assertEqual(s['maximum_measured_prefix_reuse'], 3)
        self.assertEqual(s['new_recipient_donor_full_string_overlap'], 0)
        self.assertEqual(s['new_recipient_donor_prefix_overlap'], {'20': 1, '28': 0})
        self.assertEqual(set(s['factorial_cell_draws'].values()), {256})


if __name__ == '__main__':
    unittest.main()
