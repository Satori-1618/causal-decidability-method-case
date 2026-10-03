"""Checks for a review-only freeze: integrity, drift and unauthorized release."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from verify_plan import HERE, verify_contract, verify_hashes


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((HERE/'plan.json').read_text())

    def test_current_contract(self):
        verify_contract(self.plan)

    def test_no_implicit_execution_release(self):
        self.plan['execution_authorized'] = True
        with self.assertRaisesRegex(ValueError, 'review-only'):
            verify_contract(self.plan)

    def test_budget_cannot_drift(self):
        self.plan['screen']['native_candidates_per_head'] += 1
        with self.assertRaisesRegex(ValueError, 'disagree'):
            verify_contract(self.plan)

    def test_measurement_scope_cannot_expand(self):
        self.plan['intervention']['secondary_target_transfers'] = True
        with self.assertRaisesRegex(ValueError, 'secondary'):
            verify_contract(self.plan)

    def test_reference_cannot_be_a_transfer_head(self):
        self.plan['cohort'][0]['model'] = self.plan['reference']['model']
        with self.assertRaisesRegex(ValueError, 'Reference'):
            verify_contract(self.plan)

    def test_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'protocol.txt'
            path.write_text('frozen')
            files = {'protocol.txt': hashlib.sha256(path.read_bytes()).hexdigest()}
            verify_hashes(directory, files)
            path.write_text('modified')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                verify_hashes(directory, files)

    def test_source_cannot_escape_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'leaves repository'):
                verify_hashes(directory, {'../outside': 'not-a-hash'})


if __name__ == '__main__':
    unittest.main()
