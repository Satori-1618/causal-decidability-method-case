import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import makelov_table1 as m  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = _load_script('analyze_makelov_table1')
SOURCE = ROOT / 'artifacts' / 'makelov_source'


class SamplingPort(unittest.TestCase):
    """The dataset must be the authors' own, not merely one drawn the same way."""

    def setUp(self):
        if not (SOURCE / 'data' / 'names.json').exists():
            self.skipTest('pinned upstream inputs not present')
        self.dist = m.load_distribution(SOURCE)

    def test_split_is_the_held_out_half_and_third_template(self):
        self.assertEqual(len(self.dist['names']), 108)
        self.assertEqual(len(self.dist['templates']), 1)
        self.assertEqual(self.dist['prefixes'][self.dist['prefix_len']], 'Then, ')

    def test_resampling_abb_to_bab_swaps_the_first_two_names_only(self):
        prompt = {'names': ('Zoe', 'Evan', 'Evan'), 'template': '{name_A}{name_B}{name_C}',
                  'obj': 'o', 'place': 'p', 'prefix': ''}
        out = m.resample_pattern(prompt, 'ABB', 'BAB')
        self.assertEqual(out['names'], ('Evan', 'Zoe', 'Evan'))

    def test_resampling_consumes_no_randomness(self):
        prompt = {'names': ('Zoe', 'Evan', 'Evan'), 'template': '{name_A}',
                  'obj': 'o', 'place': 'p', 'prefix': ''}
        rng = random.Random(0)
        before = rng.getstate()
        m.resample_pattern(prompt, 'ABB', 'BAB')
        self.assertEqual(before, rng.getstate())

    def test_patched_answer_is_the_base_subject_when_the_subject_moves(self):
        rng = random.Random(42)
        records = m.sample_das(rng, self.dist, ['ABB'], ['BAB'], 5, ['A', 'B'])
        for record in records:
            self.assertNotEqual(record['base_s1_pos'], record['source_s1_pos'])
            self.assertEqual(record['patched_answer_names'],
                             [record['base_s_name'], record['base_io_name']])

    def test_discarding_the_first_draw_is_what_selects_the_patching_set(self):
        """The notebook seeds once and builds TEST_DATASET before PATCHING_DATASET."""
        dataset = m.build_patching_dataset(self.dist, ['A', 'B'], samples_per_combination=4)
        rng = random.Random(42)
        without_skip = m.sample_das(rng, self.dist, ['ABB'], ['BAB'], 4, ['A', 'B'])
        self.assertNotEqual(dataset[0]['base_sentence'], without_skip[0]['base_sentence'])

    def test_patching_dataset_is_deterministic_and_two_halves(self):
        a = m.build_patching_dataset(self.dist, ['A', 'B'], samples_per_combination=3)
        b = m.build_patching_dataset(self.dist, ['A', 'B'], samples_per_combination=3)
        self.assertEqual([r['base_sentence'] for r in a],
                         [r['base_sentence'] for r in b])
        self.assertEqual([r['half'] for r in a],
                         ['ABB->BAB'] * 3 + ['BAB->ABB'] * 3)

    def test_symbol_order_changes_which_name_plays_a(self):
        a = m.build_patching_dataset(self.dist, ['A', 'B'], samples_per_combination=2)
        b = m.build_patching_dataset(self.dist, ['B', 'A'], samples_per_combination=2)
        self.assertNotEqual(a[0]['base_sentence'], b[0]['base_sentence'])
        self.assertEqual(sorted(set(a[0]['base_names'])), sorted(set(b[0]['base_names'])))


class DirectionPatch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.base = torch.randn(4, 6)
        self.source = torch.randn(4, 6)
        self.v = torch.randn(6)

    def test_patch_sets_the_component_to_the_source_and_leaves_the_rest(self):
        out = m.direction_patch(self.base, self.source, self.v)
        unit = self.v / self.v.norm()
        torch.testing.assert_close(out @ unit, self.source @ unit)
        # everything the patch moved lies along v, so the change is rank one
        change = out - self.base
        along = (change @ unit).unsqueeze(-1) * unit
        torch.testing.assert_close(change, along, atol=1e-5, rtol=1e-5)

    def test_patch_is_invariant_to_rescaling_the_direction(self):
        """Renormalising a sub-direction before patching cannot change the result."""
        plain = m.direction_patch(self.base, self.source, self.v)
        scaled = m.direction_patch(self.base, self.source, self.v * 7.3)
        unit = m.direction_patch(self.base, self.source, self.v / self.v.norm())
        torch.testing.assert_close(plain, scaled, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(plain, unit, atol=1e-5, rtol=1e-5)

    def test_shape_mismatches_are_rejected(self):
        with self.assertRaises(ValueError):
            m.direction_patch(self.base, self.source[:2], self.v)
        with self.assertRaises(ValueError):
            m.direction_patch(self.base, self.source, self.v[:3])


class Decomposition(unittest.TestCase):
    def setUp(self):
        # a (4, 2) map of rank 1: one output direction is dead
        self.w = torch.tensor([[1., 0.], [0., 0.], [0., 0.], [0., 0.]])
        self.v = torch.tensor([1., 2., 0., 3.])

    def test_svd_basis_drops_the_dead_direction(self):
        parts, audit = m.decompose(self.v, self.w, basis=m.ROW_BASIS_SVD)
        self.assertEqual(audit['rank'], 1)
        torch.testing.assert_close(parts['full'], parts['rowspace'] + parts['nullspace'])
        torch.testing.assert_close(parts['nullspace'] @ self.w,
                                   torch.zeros(2), atol=1e-6, rtol=0)

    def test_qr_basis_keeps_every_column_as_upstream_does(self):
        parts, audit = m.decompose(self.v, self.w, basis=m.ROW_BASIS_QR)
        self.assertEqual(audit['rank'], self.w.shape[1])
        torch.testing.assert_close(parts['full'], parts['rowspace'] + parts['nullspace'])
        torch.testing.assert_close(parts['nullspace'] @ self.w,
                                   torch.zeros(2), atol=1e-6, rtol=0)

    def test_bases_disagree_exactly_on_the_dead_direction(self):
        qr, _ = m.decompose(self.v, self.w, basis=m.ROW_BASIS_QR)
        svd, _ = m.decompose(self.v, self.w, basis=m.ROW_BASIS_SVD)
        self.assertGreater(float((qr['rowspace'] - svd['rowspace']).norm()), 1e-3)

    def test_unknown_basis_is_rejected(self):
        with self.assertRaises(ValueError):
            m.decompose(self.v, self.w, basis='pca')


class Aggregation(unittest.TestCase):
    def test_columns_follow_the_upstream_metric_definitions(self):
        results = {c: {'io': torch.tensor([3., 1.]), 's': torch.tensor([1., 2.]),
                       'interchange': torch.tensor([1, 0])} for c in m.CONDITIONS}
        rows = m.aggregate(results)
        self.assertEqual([r['condition'] for r in rows], list(m.CONDITIONS))
        self.assertAlmostEqual(rows[0]['logit_diff'], 0.5)
        self.assertAlmostEqual(rows[0]['accuracy'], 0.5)

    def test_clean_accuracy_is_flagged_as_an_upstream_placeholder(self):
        self.assertTrue(m.PUBLISHED_ROWS['clean']['accuracy_is_placeholder'])
        self.assertNotIn('accuracy_is_placeholder', m.PUBLISHED_ROWS['rowspace'])

    def test_published_nullspace_logit_diff_equals_published_clean(self):
        self.assertEqual(m.PUBLISHED_ROWS['nullspace']['logit_diff'],
                         m.PUBLISHED_ROWS['clean']['logit_diff'])


class PairedResolution(unittest.TestCase):
    def test_unpaired_reference_recovers_the_published_reading(self):
        ref = analyze.unpaired_reference()
        self.assertAlmostEqual(ref['ratio'], 1.610, places=3)

    def test_mcnemar_uses_only_the_discordant_pairs(self):
        a = np.array([1.] * 13 + [0.] * 1987)
        b = np.array([1.] * 6 + [0.] * 1994)
        out = analyze.accuracy_contrast(a, b)
        self.assertEqual(out['table'], {'both_correct': 6.0, 'a_only': 7.0,
                                        'b_only': 0.0, 'neither': 1987.0})
        self.assertAlmostEqual(out['separation'], 0.0035)
        self.assertAlmostEqual(out['paired']['exact_binomial_p'], 2 / 2 ** 7)
        self.assertLess(out['paired']['resolution'], out['unpaired']['resolution'])

    def test_exact_binomial_p_is_two_sided(self):
        self.assertAlmostEqual(analyze.exact_mcnemar_p(5, 5), 1.0)
        self.assertAlmostEqual(analyze.exact_mcnemar_p(3, 0), 0.25)

    def test_perfectly_correlated_arms_have_zero_paired_resolution(self):
        a = np.arange(50, dtype=float)
        out = analyze.logit_contrast(a + 2.0, a)
        self.assertAlmostEqual(out['separation'], 2.0)
        self.assertAlmostEqual(out['paired']['resolution'], 0.0)
        self.assertGreater(out['unpaired']['resolution'], 0.0)

    def test_bootstrap_tracks_the_closed_form_paired_resolution(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=400)
        b = a + rng.normal(scale=0.2, size=400)
        out = analyze.logit_contrast(a, b)
        self.assertAlmostEqual(out['bootstrap_paired']['resolution'],
                               out['paired']['resolution'], places=3)


class ExportShape(unittest.TestCase):
    """The reported run must carry every cell the paired analysis needs."""

    DIRECTORY = ROOT / 'results' / 'makelov_replication_001'

    def setUp(self):
        if not (self.DIRECTORY / 'per_example.jsonl').exists():
            self.skipTest('replication export not present')

    def test_every_pair_appears_once_per_condition(self):
        rows = [json.loads(line) for line
                in (self.DIRECTORY / 'per_example.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 4 * 2000)
        for condition in m.CONDITIONS:
            indices = sorted(r['index'] for r in rows if r['condition'] == condition)
            self.assertEqual(indices, list(range(2000)))

    def test_logit_difference_columns_are_consistent(self):
        for line in (self.DIRECTORY / 'per_example.jsonl').read_text().splitlines()[:200]:
            row = json.loads(line)
            self.assertAlmostEqual(row['logit_diff_base_ordering'],
                                   row['logit_base_io'] - row['logit_base_s'], places=4)
            self.assertAlmostEqual(row['logit_diff_patched_ordering'],
                                   -row['logit_diff_base_ordering'], places=4)

    def test_gate_passed_on_the_reported_run(self):
        aggregate = json.loads((self.DIRECTORY / 'aggregate.json').read_text())
        self.assertTrue(aggregate['gate']['passed'])
        self.assertEqual(aggregate['config']['symbol_order'], 'AB')
        self.assertEqual(aggregate['config']['row_basis'], m.ROW_BASIS_QR)


if __name__ == '__main__':
    unittest.main()
