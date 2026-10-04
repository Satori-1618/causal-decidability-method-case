"""A past release or a flipped plan flag must not authorize this new round."""
import copy
import json
import unittest

import verify_freeze as freeze


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((freeze.HERE / 'plan.json').read_text())
        self.plan.update(status='awaiting_final_pre_run_review', execution_authorized=False)
        self.pending = {'round': 'history_005', 'status': 'pending', 'execution_authorized': False,
                        'final_review': {'status': 'pending', 'reviewer': None, 'reviewed_commit': None},
                        'user_execution_authorization': None}

    def released_fixture(self):
        plan = copy.deepcopy(self.plan)
        plan.update(status='released_for_execution', execution_authorized=True)
        return plan, {'round': 'history_005', 'status': 'approved', 'execution_authorized': True,
                      'final_review': {'status': 'PASS', 'reviewer': 'test fixture', 'reviewed_commit': 'a' * 40},
                      'user_execution_authorization': 'Test fixture, never a live release'}

    def test_pending_and_plan_flag_alone_block(self):
        with self.assertRaisesRegex(ValueError, 'Execution blocked'):
            freeze.require_release(self.plan, self.pending)
        plan, _ = self.released_fixture()
        with self.assertRaisesRegex(ValueError, 'Execution blocked'):
            freeze.require_release(plan, self.pending)

    def test_previous_round_release_cannot_be_reused(self):
        plan, release = self.released_fixture()
        release['round'] = 'averaged_anchors_004'
        with self.assertRaisesRegex(ValueError, 'Execution blocked'):
            freeze.require_release(plan, release)

    def test_complete_review_and_release_fixture(self):
        freeze.require_release(*self.released_fixture())

    def test_shipped_analysis_and_plan_have_the_same_decision_constants(self):
        import analysis
        self.assertEqual(self.plan['sampling']['families'], analysis.N)
        self.assertEqual(self.plan['target_start']['minimum_separating_families'], analysis.MIN_SEPARATING)
        self.assertEqual(self.plan['intervention']['numerical_allowance_nat'], analysis.PRECISION)
        self.assertEqual(self.plan['intervention']['meaningful_error_advantage_nat'], analysis.MEANINGFUL_ADVANTAGE)
        self.assertEqual(self.plan['intervention']['comparison_numerical_guard_nat'], analysis.COMPARISON_GUARD)
        self.assertEqual(self.plan['intervention']['robust_error_advantage_nat'], analysis.ROBUST_ADVANTAGE)
        self.assertEqual(self.plan['inference']['pairs'], [list(pair) for pair in analysis.PAIRS])
        self.assertEqual(self.plan['inference']['pair_alpha'], analysis.PAIR_ALPHA)

    def test_missing_user_or_review_blocks(self):
        for field in ('authorization', 'reviewer', 'commit', 'verdict'):
            plan, release = self.released_fixture()
            if field == 'authorization':
                release['user_execution_authorization'] = ''
            elif field == 'reviewer':
                release['final_review']['reviewer'] = None
            elif field == 'commit':
                release['final_review']['reviewed_commit'] = 'short'
            else:
                release['final_review']['status'] = 'pending'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Execution blocked'):
                freeze.require_release(plan, release)

    def test_scientific_boundaries_are_not_silently_repaired(self):
        for field, value in [('positions', [20]), ('meaningful_error_advantage_nat', .01),
                             ('numerical_allowance_nat', .01), ('dtypes', ['float32'])]:
            plan = copy.deepcopy(self.plan)
            plan['intervention'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'boundary'):
                freeze.verify_contract(plan)

    def test_adequacy_or_equivalence_cannot_silently_be_claimed(self):
        for field in ('absolute_adequacy_test', 'equivalence_test'):
            plan = copy.deepcopy(self.plan)
            plan['inference'][field] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'error family'):
                freeze.verify_contract(plan)

    def test_sampling_start_architecture_and_budget(self):
        for section, field, value in [('sampling', 'families', 255),
                                      ('target_start', 'minimum_separating_families', 63),
                                      ('model', 'model_heads', 2),
                                      ('cost', 'neural_retries_authorized', True)]:
            plan = copy.deepcopy(self.plan)
            plan[section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                freeze.verify_contract(plan)


if __name__ == '__main__':
    unittest.main()
