"""512 amendment binding tests; constructed logits are never model evidence."""
import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPLICATION = ROOT/'applications/makelov-2311.17030'
SCRIPT = APPLICATION/'scripts/check_donor_factor_512_records.py'
spec = importlib.util.spec_from_file_location('check_donor_factor_512_records', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class DonorFactor512RecordChecker(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        prepared = APPLICATION/'results/donor_factor_confirmation_512'
        self.manifest = checker.load(prepared/'manifest.json')
        self.plan_path = self.repo/self.manifest['planning_input']['path']
        self.plan = checker.load(ROOT/self.manifest['planning_input']['path'])
        paths = [checker.STOP_PATH, checker.PROTOCOL_PATH, self.manifest['planning_input']['path']]
        development = Path(self.manifest['development_input']['path']).parent
        paths.extend(str(development/name) for name in self.plan['development_binding'])
        for name in paths:
            target = self.repo/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)

    def tearDown(self):
        self.temporary.cleanup()

    def rebind_plan(self):
        save(self.plan_path, self.plan)
        self.manifest['planning_input']['sha256'] = checker.sha256(self.plan_path)

    def audit(self):
        return checker.audit_amendment(self.manifest, self.repo)

    def test_real_frozen_bindings_and_saved_planning_gates_pass(self):
        report = self.audit()
        self.assertEqual(report['selected_n'], 512)
        self.assertEqual(report['coverage_required_successes'], 433)
        self.assertTrue(report['original_four_candidates_unchanged'])
        self.assertTrue(report['development_binding_unchanged'])
        for result in report['bootstrap_checks'].values():
            self.assertTrue(result['passes'])
            self.assertGreater(result['smallest_mc_lower_bound'], .8)

    def test_old_stop_binding_and_amendment_identity_cannot_change(self):
        original = copy.deepcopy(self.manifest)
        for key in ('id', 'stop'):
            self.manifest = copy.deepcopy(original)
            if key == 'id':
                self.manifest['amendment']['id'] = 'unapproved-new-amendment'
            else:
                self.manifest['amendment']['original_stop']['sha256'] = '0'*64
            with self.subTest(key=key), self.assertRaises(checker.VerificationError):
                self.audit()

    def test_original_four_candidates_must_remain_exactly_unchanged(self):
        self.plan['candidates'][0]['normal_reference']['directions']['positive']['P']['detections'] += 1
        self.rebind_plan()
        with self.assertRaisesRegex(checker.VerificationError, 'original four'):
            self.audit()

    def test_rehashed_mc_interval_corruption_is_detected_independently(self):
        report = self.plan['candidates'][-1]['bootstrap_checks']['empirical']
        report['directions']['positive']['P']['monte_carlo_95_percent_interval'][0] -= .01
        self.rebind_plan()
        with self.assertRaisesRegex(checker.VerificationError, 'MC CP interval differs'):
            self.audit()

    def test_rehashed_pass_flags_cannot_override_a_failed_mc_gate(self):
        entry = self.plan['candidates'][-1]['bootstrap_checks']['gaussian']['directions']['negative']['P']
        entry.update(detections=750, probability=.75,
                     monte_carlo_95_percent_interval=checker.mc_interval(750, 1000), passes=True)
        self.rebind_plan()
        with self.assertRaisesRegex(checker.VerificationError, 'MC gate differs'):
            self.audit()

    def test_changed_development_rules_and_frozen_code_are_rejected(self):
        original = copy.deepcopy(self.manifest)
        for kind in ('contract', 'code'):
            self.manifest = copy.deepcopy(original)
            if kind == 'contract':
                self.manifest['contract']['mean_practical_boundary'] = .2
            else:
                path = checker.APPLICATION_PREFIX+'src/donor_factor.py'
                self.manifest['code_files_sha256'][path] = '0'*64
            with self.subTest(kind=kind), self.assertRaises(checker.VerificationError):
                self.audit()

    def test_planning_artifact_is_required_even_when_original_inputs_are_optional(self):
        self.plan_path.unlink()
        with self.assertRaisesRegex(checker.VerificationError, 'missing required file: 512 planning'):
            self.audit()

    def test_512_raw_record_integration_and_full_bootstrap(self):
        # Reuse the earlier test's explicitly fabricated measurements, attached to
        # the actual frozen 512 case IDs. No cached experiment outcomes are used.
        fixture_spec = importlib.util.spec_from_file_location(
            'donor_factor_checker_test_fixture', ROOT/'tests/test_check_donor_factor_records.py')
        fixture = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture)
        from donor_factor_analysis import analyze_records
        cases_path = APPLICATION/'results/donor_factor_confirmation_512/cases.json'
        cases = checker.load(cases_path)
        results = self.repo/'synthetic_512_records'
        results.mkdir()
        shutil.copyfile(cases_path, results/'cases.json')
        # Preserve the original manifest's exact bytes and original hash bindings.
        original_manifest = APPLICATION/'results/donor_factor_confirmation_512/manifest.json'
        shutil.copyfile(original_manifest, results/'manifest.json')
        records = [{'case_id': case['case_id'], 'content_family_id': case['content_family_id'],
                    'precisions': {p: fixture.measurement(p, case['position']) for p in checker.PRECISIONS}}
                   for case in cases]
        fixture.write_lines(results/'records.jsonl', records)
        for precision in checker.PRECISIONS:
            fixture.write_lines(results/f'measurements_{precision}.jsonl',
                                [{'case_id': record['case_id'], **record['precisions'][precision]}
                                 for record in records])
        frozen = checker.sha256(results/'manifest.json')
        save(results/'RUN_STARTED.json', {'manifest_sha256': frozen})
        summary = analyze_records(records)
        summary.update(stage='confirmation', confirmation=True, manifest_sha256=frozen,
                       records_sha256=checker.sha256(results/'records.jsonl'),
                       model_forward_calls=12*512, model_prompt_evaluations=48*512)
        save(results/'summary.json', summary)
        save(results/'artifact_hashes.json', {path.name: checker.sha256(path) for path in results.iterdir()})
        report = checker.verify(results, repository_root=self.repo,
                                model_snapshot=self.repo/'unavailable_model', full_bootstrap=True)
        self.assertEqual(report['n_families'], 512)
        self.assertEqual(report['independent_profile_counts'], {'position_only': 512, 'identity_only': 0})
        self.assertTrue(report['full_seeded_bootstrap_reproduced'])
        self.assertEqual(report['amendment_audit']['selected_n'], 512)
        # Rehashed summary corruption still fails scientific record checks.
        summary['profiles']['position_only']['successes'] -= 1
        save(results/'summary.json', summary)
        artifacts = checker.load(results/'artifact_hashes.json')
        artifacts['summary.json'] = checker.sha256(results/'summary.json')
        save(results/'artifact_hashes.json', artifacts)
        with self.assertRaisesRegex(checker.VerificationError, 'independent count'):
            checker.verify(results, repository_root=self.repo, model_snapshot=self.repo/'unavailable_model')


if __name__ == '__main__':
    unittest.main()
