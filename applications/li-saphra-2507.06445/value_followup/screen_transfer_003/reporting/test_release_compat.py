"""Schema-only regression tests; no model imports or inference."""
import copy
import difflib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('release_compat_under_test', HERE/'analyze_release_compat.py')
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class ReleaseCompatibilityTests(unittest.TestCase):
    def valid_records(self):
        plan = json.loads((HERE.parent/'plan.json').read_text())
        plan['status'] = 'released_for_execution'
        plan['execution_authorized'] = True
        plan['provenance']['new_input_pools_prepared'] = True
        release = {
            'status': 'approved', 'execution_authorized': True,
            'final_review': {'status': 'PASS', 'reviewer': 'Synthetic test reviewer',
                             'reviewed_commit': 'a'*40},
            'user_execution_authorization': 'Synthetic test authorization',
        }
        return plan, release

    def test_full_valid_release_schema_accepted(self):
        compat.release_validator()(*self.valid_records())

    def test_wrong_nested_label_is_still_rejected(self):
        plan, release = self.valid_records()
        release['final_review']['status'] = 'approved'
        with self.assertRaises(ValueError):
            compat.release_validator()(plan, release)

    def test_missing_authorization_and_pending_review_rejected(self):
        plan, release = self.valid_records()
        variants = []
        wrong = copy.deepcopy(release); wrong['user_execution_authorization'] = ''; variants.append(wrong)
        wrong = copy.deepcopy(release); wrong['final_review']['status'] = 'pending'; variants.append(wrong)
        wrong = copy.deepcopy(release); wrong['execution_authorized'] = False; variants.append(wrong)
        wrong = copy.deepcopy(release); wrong['final_review']['reviewed_commit'] = 'unknown'; variants.append(wrong)
        wrong = copy.deepcopy(release); wrong['final_review']['reviewer'] = ''; variants.append(wrong)
        for wrong in variants:
            with self.subTest(release=wrong), self.assertRaises(ValueError):
                compat.release_validator()(plan, wrong)

    def test_transformation_changes_exactly_one_literal(self):
        source = (HERE.parent/'analyze_transfer.py').read_bytes()
        executed = compat.transformed_source(source)
        self.assertEqual(executed, source.replace(compat.OLD.encode(), compat.NEW.encode(), 1))
        changed = [line for line in difflib.ndiff(source.decode().splitlines(), executed.decode().splitlines())
                   if line.startswith(('- ', '+ '))]
        self.assertEqual(len(changed), 2)
        self.assertEqual((HERE.parent/'analyze_transfer.py').read_bytes(), source)

    def test_other_analyzer_source_rejected(self):
        source = (HERE.parent/'analyze_transfer.py').read_bytes()
        with self.assertRaises(ValueError):
            compat.transformed_source(source+b'\n')

    def test_missing_or_multiple_replacement_sites_rejected(self):
        for source in ('no nested review check', compat.OLD+'\n'+compat.OLD):
            with self.subTest(source=source), self.assertRaises(ValueError):
                compat.replace_review_label(source)


if __name__ == '__main__':
    unittest.main()
