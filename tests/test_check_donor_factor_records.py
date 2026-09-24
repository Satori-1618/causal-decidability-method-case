"""Synthetic bundles exercise record verification; they are not model evidence."""
import copy
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'applications/makelov-2311.17030/scripts/check_donor_factor_records.py'
spec = importlib.util.spec_from_file_location('check_donor_factor_records', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
from donor_factor_analysis import analyze_records
from donor_factor_sampling import historical_exclusions, sample_iid_families


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_lines(path, values):
    path.write_text(''.join(json.dumps(v, allow_nan=False)+'\n' for v in values))


class ToyTokenizer:
    bos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [{'Ada': 10, 'Bea': 11}.get(word, 100+sum(map(ord, word)))
                for word in re.findall(r'\w+|[^\w\s]', text)]


def measurement(precision, position):
    baseline = [[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]]
    effects = {'00': 0., '01': .5, '10': 0., '11': .5, 'zero': 0.}
    outputs = {c: [[1.+d, 0.], [1.+d, 0.], [-1., 0.], [-1., 0.]] for c, d in effects.items()}
    tolerance = 64 * 2.**(-23 if precision == 'float32' else -52) * 1.5
    fidelity = {c: {'calls': 1, 'passed': True, 'other_positions_unchanged': True,
                    'insertion_error_max_per_item': [0.]*4,
                    'rounding_budget_per_item': [tolerance]*4,
                    'actual_delta_l2_per_item': [abs(d), abs(d), 0., 0.]}
                for c, d in effects.items()}
    panels = [{'recipient_index': r, 'baseline_margin': 1.,
               'donor_indices': copy.deepcopy(checker.DONOR_INDICES[r]),
               'patched_margins': {c: outputs[c][r][0] for c in checker.CELLS},
               'alphas': {c: effects[c] for c in checker.CELLS},
               'delta_l2': {c: abs(effects[c]) for c in checker.CELLS},
               'controls': {'passed': True, 'identity_tolerance': 2*tolerance}}
              for r in (0, 1)]
    sources = {c: [checker.DONOR_INDICES[r][c] for r in (0, 1)] for c in checker.CELLS}
    sources['zero'] = [0, 1]
    return {'baseline_margins': [1., 1., -1., -1.], 'panels': panels,
            'baseline_answer_logits': baseline, 'patched_answer_logits': outputs,
            'source_indices': sources, 'source_activations_sha256': 'a'*64,
            'source_activation_sha256_per_prompt': ['b'*64]*4,
            'controls': {'passed': True, 'identity_tolerance': 2*tolerance,
                         'answer_logit_identity_tolerance': tolerance,
                         'identity_errors': {'self_vs_baseline': 0., 'zero_vs_baseline': 0.,
                                             'untouched_batch_rows': 0.},
                         'identity_exact': True, 'fidelity': fidelity,
                         'site': checker.SITE, 'absolute_position': position, 'zero_alphas': [0., 0.]},
            'model_forward_calls': 6, 'model_prompt_evaluations': 24}


class DonorFactorRecordChecker(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        self.app = self.repo/'applications/makelov-2311.17030'
        self.results = self.app/'results/constructed'
        self.results.mkdir(parents=True)
        source = self.app/'artifacts/makelov_source'
        data = {'names': ['old0', 'old1', 'Ada', 'Bea'], 'objects': ['old', 'book'],
                'places': ['old', 'room'], 'prefixes': ['', '', 'Then, '],
                'templates': ['old0', 'old1', '{name_A} and {name_B} met. {name_C} gives a {object} in {place}']}
        for key, values in data.items():
            save(source/f'data/{key}.json', values)
        for index, name in enumerate(checker.HISTORICAL_CASE_FILES):
            save(self.app/name, [{'prompts': [f'history {index} A', f'history {index} B']}])
        excluded, provenance = historical_exclusions(self.app)
        self.cases, sampling = sample_iid_families(
            source, ToyTokenizer(), n=4, seed=24092431, stage='development',
            excluded_prompts=excluded, exclusion_provenance=provenance, tokenizer_fingerprint='synthetic')
        self.records = [{'case_id': c['case_id'], 'content_family_id': c['content_family_id'],
                         'precisions': {p: measurement(p, c['position']) for p in checker.PRECISIONS}}
                        for c in self.cases]
        code = {}
        for name in checker.CODE_PATHS:
            path = self.app/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# Synthetic code hash fixture; not experiment evidence.\n'+name)
            code[str(path.relative_to(self.repo))] = checker.sha256(path)
        self.contract = {'profile_residual_tolerance': .25, 'profile_coverage': .8,
                         'profile_family_alpha': .025, 'profiles': list(checker.PROFILES),
                         'mean_practical_boundary': .10, 'mean_family_alpha': .025,
                         'mean_contrasts': list(checker.CONTRASTS), 'mean_bootstrap_draws': 20000,
                         'mean_bootstrap_seed': 320260924, 'secondary_bootstrap_seed': 320260925,
                         'bootstrap_endpoint_stability_budget': .01, 'numerical_discrepancy_budget': .01}
        save(self.results/'cases.json', self.cases)
        self.manifest = {'schema_version': 1, 'experiment': 'donor_factor_3a', 'stage': 'development',
                         'n_families': 4, 'seed': 24092431, 'sampling': sampling,
                         'source': {'files': sampling['sampling_definition']['source_files']},
                         'code_files_sha256': code, 'cases_sha256': checker.sha256(self.results/'cases.json'),
                         'directions_sha256': 'd'*64, 'qualification_sha256': 'e'*64,
                         'planning_input': None, 'model_snapshot_revision': 'synthetic',
                         'model_files_sha256': {'missing_model.bin': 'f'*64},
                         'site': checker.SITE, 'precisions': list(checker.PRECISIONS), 'contract': self.contract}
        save(self.results/'manifest.json', self.manifest)
        frozen = checker.sha256(self.results/'manifest.json')
        save(self.results/'RUN_STARTED.json', {'manifest_sha256': frozen})
        self.write_records()
        self.summary = analyze_records(self.records)
        self.summary.update(stage='development', confirmation=False, manifest_sha256=frozen,
                            records_sha256=checker.sha256(self.results/'records.jsonl'),
                            model_forward_calls=48, model_prompt_evaluations=192)
        self.save_summary()

    def tearDown(self):
        self.temporary.cleanup()

    def write_records(self):
        write_lines(self.results/'records.jsonl', self.records)
        for precision in checker.PRECISIONS:
            write_lines(self.results/f'measurements_{precision}.jsonl',
                        [{'case_id': record['case_id'], **record['precisions'][precision]}
                         for record in self.records])

    def rehash(self):
        save(self.results/'artifact_hashes.json', {p.name: checker.sha256(p) for p in self.results.iterdir()
                                                  if p.name != 'artifact_hashes.json'})

    def save_summary(self):
        self.summary['records_sha256'] = checker.sha256(self.results/'records.jsonl')
        save(self.results/'summary.json', self.summary)
        self.rehash()

    def verify(self, **kwargs):
        return checker.verify(self.results, repository_root=self.repo,
                              model_snapshot=self.repo/'absent_model', **kwargs)

    def test_complete_bundle_and_full_bootstrap_pass_without_loading_originals(self):
        report = self.verify(full_bootstrap=True)
        self.assertTrue(report['verified'])
        self.assertEqual(report['independent_profile_counts'], {'position_only': 4, 'identity_only': 0})
        self.assertEqual(report['mean_statuses'], {'I': 'equivalent', 'P': 'positive_relevant', 'J': 'equivalent'})
        self.assertTrue(report['full_seeded_bootstrap_reproduced'])
        self.assertIn('model:missing_model.bin', report['optional_original_inputs_unavailable'])
        self.assertIn('directions.npz', report['optional_original_inputs_unavailable'])
        with self.assertRaisesRegex(checker.VerificationError, 'original inputs unavailable'):
            self.verify(require_original_inputs=True)

    def test_rehashed_margin_corruption_is_detected_from_raw_logits(self):
        self.records[0]['precisions']['float32']['panels'][0]['patched_margins']['10'] += .1
        self.write_records()
        self.save_summary()
        with self.assertRaisesRegex(checker.VerificationError, 'margin disagrees with logits'):
            self.verify()

    def test_rehashed_profile_count_and_cp_corruption_are_independently_detected(self):
        original = copy.deepcopy(self.summary)
        for field in ('successes', 'interval', 'status'):
            self.summary = copy.deepcopy(original)
            profile = self.summary['profiles']['position_only']
            profile[field] = {'successes': 3, 'interval': [.2, .9], 'status': 'excluded'}[field]
            self.save_summary()
            with self.subTest(field=field), self.assertRaisesRegex(checker.VerificationError, 'independent'):
                self.verify()

    def test_rehashed_mean_corruption_is_detected_without_bootstrap_dependency(self):
        self.summary['mean_effects']['P']['precisions']['float64']['mean'] = .6
        self.save_summary()
        with self.assertRaisesRegex(checker.VerificationError, 'independent mean'):
            self.verify()

    def test_forged_bootstrap_endpoints_need_full_reproduction(self):
        # These endpoints retain the same decision/stability and cannot be rejected
        # by the documented standard-library mean/decision checks alone.
        for precision in checker.PRECISIONS:
            entry = self.summary['mean_effects']['P']['precisions'][precision]
            entry['interval'] = entry['secondary_interval'] = [.4, .6]
        self.save_summary()
        self.assertFalse(self.verify()['full_seeded_bootstrap_reproduced'])
        with self.assertRaisesRegex(checker.VerificationError, 'analyzer reproduction'):
            self.verify(full_bootstrap=True)

    def test_wrong_donor_map_and_changed_nonrecipient_are_rejected_despite_passed_flags(self):
        original = copy.deepcopy(self.records)
        for kind in ('map', 'unchanged_row', 'audit'):
            self.records = copy.deepcopy(original)
            raw = self.records[0]['precisions']['float64']
            if kind == 'map':
                raw['source_indices']['10'] = [2, 2]
            elif kind == 'unchanged_row':
                raw['patched_answer_logits']['10'][2][0] += .1
            else:
                raw['controls']['fidelity']['10']['actual_delta_l2_per_item'][2] = .1
            self.write_records()
            self.save_summary()
            with self.subTest(kind=kind), self.assertRaises(checker.VerificationError):
                self.verify()

    def test_missing_independent_unit_cannot_be_removed_silently(self):
        self.records.pop()
        self.write_records()
        self.save_summary()
        with self.assertRaisesRegex(checker.VerificationError, 'omit/reorder/duplicate'):
            self.verify()

    def test_original_source_and_code_corruption_are_detected(self):
        source_path = self.app/'artifacts/makelov_source/data/names.json'
        source_path.write_text(source_path.read_text()+'\n')
        with self.assertRaisesRegex(checker.VerificationError, 'hash mismatch: source:'):
            self.verify()
        code_path = self.app/'src/donor_factor.py'
        code_path.write_text(code_path.read_text()+'\n')
        with self.assertRaisesRegex(checker.VerificationError, 'hash mismatch: code:'):
            self.verify()

    def test_unrehashed_mutation_and_missing_chain_file_are_detected(self):
        with (self.results/'records.jsonl').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(checker.VerificationError, 'hash mismatch'):
            self.verify()
        (self.results/'measurements_float64.jsonl').unlink()
        self.rehash()
        with self.assertRaisesRegex(checker.VerificationError, 'omits required'):
            self.verify()

    def test_direct_binomial_cp_matches_closed_form_extremes(self):
        for n in (4, 32, 128, 192, 256, 384):
            low, high = checker.independent_cp(n, n)
            self.assertAlmostEqual(low, (.025/4)**(1/n), places=13)
            self.assertEqual(high, 1.)
            low, high = checker.independent_cp(0, n)
            self.assertEqual(low, 0.)
            self.assertAlmostEqual(high, 1-(.025/4)**(1/n), places=13)


if __name__ == '__main__':
    unittest.main()
