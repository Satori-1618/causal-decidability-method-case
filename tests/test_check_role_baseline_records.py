"""Synthetic records only: the independent checker never imports a model."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1]/'applications/makelov-2311.17030/scripts/check_role_baseline_records.py'
SPEC = importlib.util.spec_from_file_location('independent_stage_a', PATH)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def replay(raw, precision):
    logits = raw['rows'][0]['name_logits'][:]
    raw['controls'] = {'passed': True, 'token_replay_passed': True, 'no_padding': True,
        'replay_row_id': raw['rows'][0]['row_id'], 'replay_name_logits': logits,
        'batch_vs_single_max_name_logit_error': 0.,
        'batch_vs_single_tolerance': 64*2.**(-23 if precision == 'float32' else -52)*max(1., *map(abs, logits))}


def world():
    cases, records = [], []
    for i in range(32):
        frozen, rows = [], []
        for w in checker.WORDINGS:
            for c, (binding, form) in checker.CONTEXTS.items():
                for q in checker.ROLES:
                    correct = binding[checker.ROLES.index(q)]
                    row = {'row_id': f'{w}/{c}/{q}', 'wording': w, 'context_id': c,
                           'form': form, 'query': q, 'correct_name_index': correct,
                           'correct_answer_token_id': 10+correct, 'position': 4,
                           'token_ids': [0]+[10+n for n in checker.mention_order(c)]+[99],
                           'name_token_positions': [1, 2, 3]}
                    frozen.append(row)
                    rows.append({**row, 'name_logits': [4. if j == correct else 0. for j in range(3)],
                                 'name_probability_mass': .8, 'full_vocab_argmax_id': 10+correct})
        cases.append({'case_id': f'family-{i}', 'names': ['Ada', 'Bea', 'Cal'],
                      'answer_token_ids': [10, 11, 12], 'bos_token_id': 0, 'rows': frozen})
        precisions = {p: {'rows': copy.deepcopy(rows), 'prompt_evaluations': 49,
                         'model_forward_calls': 4} for p in checker.PRECISIONS}
        for p, raw in precisions.items():
            replay(raw, p)
        records.append({'case_id': f'family-{i}', 'precisions': precisions})
    return cases, records


class IndependentStageATests(unittest.TestCase):
    def test_known_triangle_geometry_and_all_groups(self):
        result = checker.recompute(*world())
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(result['competence_groups']), 18)
        self.assertEqual(result['potential_geometry']['successes'], 32)
        geometry = result['families']['family-0']['precisions']['float32']['potential_geometry']['fit']
        # The name target equals the native answer, so its line is {0}.
        self.assertAlmostEqual(geometry['diagnostics']['name'][0]['gap'], math.sqrt(32))
        self.assertAlmostEqual(geometry['diagnostics']['position'][0]['gap'], math.sqrt(24))
        self.assertAlmostEqual(geometry['diagnostics']['switch'][0]['gap'], math.sqrt(8))
        self.assertAlmostEqual(geometry['diagnostics']['no_op'][0]['gap'], math.sqrt(32))

    def test_exact_29_family_rule_preserves_weak_draws(self):
        for count in (3, 4):
            cases, records = world()
            for record in records[:count]:
                for p, raw in record['precisions'].items():
                    for row in raw['rows']:
                        row['name_logits'] = [.1 if j == row['correct_name_index'] else 0. for j in range(3)]
                    replay(raw, p)
            result = checker.recompute(cases, records)
            self.assertEqual(result['potential_geometry']['successes'], 32-count)
            self.assertEqual(result['status'], 'PASS' if count == 3 else 'STOP_GEOMETRY')

    def test_tied_correct_max_fails_and_family_first_mean_is_exact(self):
        cases, records = world()
        for record in records[:4]:
            for raw in record['precisions'].values():
                target = next(r for r in raw['rows'] if r['row_id'] == 'fit/d1/receiver')
                target['name_logits'] = [4., 4., 0.]
        result = checker.recompute(cases, records)
        self.assertEqual(result['failed_competence_groups'], ['receiver|fit|received'])
        group = result['competence_groups']['receiver|fit|received']['precisions']['float32']
        self.assertEqual(group['mean_family_accuracy'], 28/32)
        self.assertEqual(group['family_scores'], [0.]*4+[1.]*28)

    def test_derived_precision_failure_despite_small_native_discrepancies(self):
        cases, records = world()
        rows = {r['row_id']: r for r in records[0]['precisions']['float32']['rows']}
        rows['fit/b1_gave/observer']['name_logits'][0] += .006
        rows['fit/b1_gave/receiver']['name_logits'][2] += .006
        result = checker.recompute(cases, records)
        self.assertEqual(result['status'], 'INVALIDNUMERICS')
        gaps = result['families']['family-0']['numerical_resolution']['discrepancies_nat']
        self.assertLess(max(v for k, v in gaps.items() if k.startswith('row.')), .01)
        self.assertGreater(max(gaps.values()), .01)

    def test_replay_scalar_forgery_and_incomplete_grid_fail(self):
        for mutation in ('error', 'tolerance', 'grid', 'label'):
            cases, records = world()
            raw = records[0]['precisions']['float64']
            if mutation == 'error':
                raw['controls']['replay_name_logits'][0] += .1
            elif mutation == 'tolerance':
                raw['controls']['batch_vs_single_tolerance'] = 1.
            elif mutation == 'grid':
                raw['rows'].pop()
            else:
                raw['rows'][0]['correct_name_index'] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                checker.recompute(cases, records)

    def test_zero_direction_is_not_an_arbitrary_line(self):
        self.assertEqual(checker.projection([1., 2., 3.], [0., 0., 0.]), [0., 0., 0.])
        self.assertAlmostEqual(checker.distance([1., 2., 3.], [0., 0., 0.]), math.sqrt(14/3))

    def test_artifact_tampering_is_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ('manifest.json', 'cases.json', 'records.jsonl', 'summary.json', 'RUN_STARTED.json',
                     'measurements_float32.jsonl', 'measurements_float64.jsonl')
            for name in names:
                (root/name).write_text('{}')
            (root/'artifact_hashes.json').write_text(json.dumps({n: checker.digest(root/n) for n in names}))
            (root/'records.jsonl').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                checker.verify_directory(root)


if __name__ == '__main__':
    unittest.main()
