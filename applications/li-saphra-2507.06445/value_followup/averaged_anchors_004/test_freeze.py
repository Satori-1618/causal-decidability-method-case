"""Release-schema tests use explicit fixtures, never the live approval state."""
import copy
import json
from pathlib import Path
import unittest

import analysis
import verify_freeze as freeze


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((freeze.HERE/'plan.json').read_text())
        self.plan.update(status='awaiting_final_pre_run_review', execution_authorized=False)
        self.pending = {'status':'pending', 'execution_authorized':False,
                        'final_review':{'status':'pending','reviewer':None,'reviewed_commit':None},
                        'user_execution_authorization':None}

    def released(self):
        plan = copy.deepcopy(self.plan)
        plan.update(status='released_for_execution', execution_authorized=True)
        release = {'status':'approved','execution_authorized':True,
                   'final_review':{'status':'PASS','reviewer':'fixture reviewer','reviewed_commit':'a'*40},
                   'user_execution_authorization':'fixture authorization'}
        return plan, release

    def test_current_scientific_contract(self):
        freeze.verify_contract(self.plan)
        self.assertEqual(self.plan['sampling']['families'], analysis.N)
        self.assertEqual(self.plan['target_start']['minimum_separating_families'], analysis.MIN_SEPARATING)
        self.assertEqual(self.plan['inference']['candidate_tail_alpha'], analysis.ALPHA)

    def test_pending_fixture_blocks(self):
        with self.assertRaisesRegex(ValueError, 'Execution blocked'):
            freeze.require_release(self.plan, self.pending)

    def test_plan_flip_does_not_authorize(self):
        plan,_ = self.released()
        with self.assertRaisesRegex(ValueError, 'Execution blocked'):
            freeze.require_release(plan, self.pending)

    def test_exact_pass_schema_is_accepted(self):
        freeze.require_release(*self.released())

    def test_wrong_nested_approved_label_is_rejected(self):
        plan,release = self.released()
        release['final_review']['status']='approved'
        with self.assertRaisesRegex(ValueError, 'Execution blocked'):
            freeze.require_release(plan, release)

    def test_missing_user_authorization_is_rejected(self):
        plan,release = self.released()
        release['user_execution_authorization']=''
        with self.assertRaisesRegex(ValueError, 'Execution blocked'):
            freeze.require_release(plan, release)

    def test_wrong_head_partition_rejected(self):
        self.plan['model'].update(model_heads=2,head_width=32)
        with self.assertRaisesRegex(ValueError, 'architecture'):
            freeze.verify_contract(self.plan)

    def test_budget_drift_rejected(self):
        self.plan['sampling']['native_candidates']+=1
        with self.assertRaisesRegex(ValueError, 'sampling'):
            freeze.verify_contract(self.plan)

    def test_start_drift_rejected(self):
        self.plan['target_start']['minimum_separating_families']=127
        with self.assertRaisesRegex(ValueError, 'start rule'):
            freeze.verify_contract(self.plan)

    def test_boundary_drift_rejected(self):
        self.plan['intervention']['scientific_tolerance_nat']=.2
        with self.assertRaisesRegex(ValueError, 'boundary'):
            freeze.verify_contract(self.plan)


if __name__=='__main__':
    unittest.main()
