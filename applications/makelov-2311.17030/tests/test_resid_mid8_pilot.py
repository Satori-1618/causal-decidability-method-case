"""The resid_mid.8 pilot must fix Freeze B's inputs without revealing E(row) or E(null).

None of these tests loads GPT-2 or runs a forward pass at the preregistered site.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import makelov_table1 as M  # noqa: E402
import resid_mid8_pilot as P  # noqa: E402

SOURCE = ROOT / 'artifacts' / 'makelov_source'
RUNNER = ROOT / 'scripts' / 'run_resid_mid8_pilot.py'


def _cells(n, ld, hit):
    return {'ld': torch.tensor(ld[:n], dtype=torch.float32),
            'interchange': torch.tensor(hit[:n], dtype=torch.int64)}


def _effect(full_l, row_l, null_l, full_a, row_a, null_a):
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    return {'L': {'full': t(full_l), 'row': t(row_l), 'null': t(null_l)},
            'A': {'full': t(full_a), 'row': t(row_a), 'null': t(null_a)}}


def _numbers(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _numbers(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _numbers(value)
    elif isinstance(obj, float):
        yield obj


class Direction(unittest.TestCase):

    def test_pinned_payload_is_a_unit_768_vector(self):
        v = P.load_direction(SOURCE / 'das_resid_mid.joblib')
        self.assertEqual(v.shape, (768,))
        self.assertAlmostEqual(float(v.norm()), 1.0, places=4)

    def test_header_is_the_validated_mlp8_header_with_only_the_shape_changed(self):
        mlp8 = (SOURCE / 'das_mlp8.joblib').read_bytes()[:len(P._DIRECTION_HEADER)]
        self.assertEqual(mlp8.replace(b'M\x00\x0c\x85', b'M\x00\x03\x85'), P._DIRECTION_HEADER)

    def test_any_other_file_is_rejected_not_unpickled(self):
        with self.assertRaises(ValueError):
            P.load_direction(SOURCE / 'das_mlp8.joblib')


class Decomposition(unittest.TestCase):

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.w_qs = [torch.randn(768, 64, generator=g) for _ in P.NAME_MOVERS]
        self.v = torch.randn(768, generator=g)

    def test_split_is_orthogonal_and_null_is_invisible_to_the_queries(self):
        parts, audit = P.decompose_namemovers(self.v, self.w_qs)
        self.assertEqual(audit['rank'], 192)
        self.assertLess(abs(audit['row_dot_null']), 1e-3)
        self.assertLess(audit['null_through_queries'] / audit['vector_norm'], 1e-4)
        self.assertLess(audit['split_error'], 1e-6)
        self.assertTrue(torch.equal(parts['full'], self.v))

    def test_matches_upstream_cell_8_line_for_line(self):
        q, _ = torch.linalg.qr(torch.cat(self.w_qs, dim=1))
        parts, _ = P.decompose_namemovers(self.v, self.w_qs)
        self.assertTrue(torch.equal(parts['row'], self.v @ q @ q.T))


class Effects(unittest.TestCase):

    def test_a_patch_that_changes_nothing_has_effect_exactly_zero_on_both_readouts(self):
        clean = _cells(4, [3.0, 2.5, -0.5, 1.0], [0, 1, 0, 0])
        cells = {c: dict(clean) for c in P.CONDITIONS}
        effect = P.effects(cells)
        for readout in P.READOUTS:
            for condition in ('full', 'row', 'null'):
                self.assertTrue(torch.equal(effect[readout][condition],
                                            torch.zeros(4, dtype=torch.float64)))

    def test_signs_follow_the_prereg(self):
        cells = {'clean': _cells(2, [3.0, 3.0], [0, 0]),
                 'full': _cells(2, [1.0, 2.0], [1, 0]),
                 'row': _cells(2, [3.0, 3.0], [0, 0]),
                 'null': _cells(2, [3.0, 3.0], [0, 0])}
        effect = P.effects(cells)
        self.assertEqual(effect['L']['full'].tolist(), [2.0, 1.0])
        self.assertEqual(effect['A']['full'].tolist(), [1.0, 0.0])


class Blinding(unittest.TestCase):
    """§7: the output is the same whatever E(row) and E(null) are, SDs held fixed."""

    def setUp(self):
        g = torch.Generator().manual_seed(1)
        self.full = (torch.randn(200, generator=g, dtype=torch.float64) + 1.5).tolist()
        self.row = (torch.randn(200, generator=g, dtype=torch.float64) * 0.4).tolist()
        self.null = (torch.randn(200, generator=g, dtype=torch.float64) * 0.3).tolist()
        self.hits = [float(i % 5 == 0) for i in range(200)]

    def test_shifting_a_component_mean_leaves_the_record_unchanged(self):
        a = P.blinded_summary(_effect(self.full, self.row, self.null,
                                      self.hits, self.hits, self.hits))
        shift = [x + 0.8125 for x in self.row]
        b = P.blinded_summary(_effect(self.full, shift, self.null,
                                      self.hits, self.hits, self.hits))
        self.assertEqual(a['mean_E_full'], b['mean_E_full'])
        for x, y in zip(_numbers(a['sd']), _numbers(b['sd'])):
            self.assertAlmostEqual(x, y, places=12)

    def test_no_component_or_contrast_mean_appears_anywhere(self):
        row = [x + 0.123456 for x in self.row]
        row_hits = [float(i % 7 == 0) for i in range(200)]
        null_hits = [float(i % 3 == 0) for i in range(200)]
        effect = _effect(self.full, row, self.null, self.hits, row_hits, null_hits)
        record = P.blinded_summary(effect)
        forbidden = []
        for readout in P.READOUTS:
            for component in P.COMPONENTS:
                e = effect[readout][component]
                forbidden += [float(e.mean()), float((effect[readout]['full'] - e).mean())]
        for value in _numbers(record):
            for secret in forbidden:
                self.assertNotAlmostEqual(value, secret, places=9)

    def test_readout_a_is_blinded_only_up_to_p_versus_one_minus_p(self):
        """Disclosed in the prereg clarification: a {-1,0,1} contrast is not
        location-invariant. With clean interchange at zero and E(full) = 1 for every
        pair, rates p and 1 - p give the same record, and nothing else does."""
        ones = [1.0] * 200
        low = [float(i < 40) for i in range(200)]
        high = [float(i < 160) for i in range(200)]
        other = [float(i < 60) for i in range(200)]
        base = dict(full_l=self.full, row_l=self.row, null_l=self.null,
                    full_a=ones, null_a=ones)
        rec = {k: P.blinded_summary(_effect(row_a=v, **base))
               for k, v in (('low', low), ('high', high), ('other', other))}
        for x, y in zip(_numbers(rec['low']), _numbers(rec['high'])):
            self.assertAlmostEqual(x, y, places=12)
        self.assertNotEqual(rec['low']['sd']['A']['row'], rec['other']['sd']['A']['row'])

    def test_only_allowlisted_keys_survive_validation(self):
        summary = P.blinded_summary(_effect(self.full, self.row, self.null,
                                            self.hits, self.hits, self.hits))
        audit = dict.fromkeys(P.DECOMPOSITION_KEYS, 0.0)
        P.validate_record({'summary': summary, 'decomposition': audit, 'provenance': {}})
        leaky = dict(summary, mean_E_row={'L': 0.1, 'A': 0.0})
        with self.assertRaises(ValueError):
            P.validate_record({'summary': leaky, 'decomposition': audit, 'provenance': {}})
        with self.assertRaises(ValueError):
            P.validate_record({'summary': summary, 'decomposition': audit,
                               'provenance': {}, 'per_pair': []})
        with self.assertRaises(ValueError):
            P.validate_record({'summary': summary, 'decomposition': audit,
                               'provenance': {'effects': list(range(200))}})


class Data(unittest.TestCase):

    def setUp(self):
        if not (SOURCE / 'data' / 'names.json').exists():
            self.skipTest('pinned upstream inputs not present')
        self.dist = M.load_distribution(SOURCE)

    def test_pilot_is_200_pairs_split_evenly_and_deterministic(self):
        a = P.build_pilot_dataset(self.dist)
        b = P.build_pilot_dataset(self.dist)
        self.assertEqual([r['pair_id'] for r in a], [r['pair_id'] for r in b])
        self.assertEqual(sum(r['half'] == 'ABB->BAB' for r in a), 100)
        self.assertEqual(sum(r['half'] == 'BAB->ABB' for r in a), 100)

    def test_symbol_order_changes_the_pairs(self):
        pinned = M.build_patching_dataset(self.dist, ['B', 'A'], seed=P.PILOT_SEED,
                                          samples_per_combination=P.PILOT_PER_COMBINATION)
        other = M.build_patching_dataset(self.dist, ['A', 'B'], seed=P.PILOT_SEED,
                                         samples_per_combination=P.PILOT_PER_COMBINATION)
        self.assertEqual([r['pair_id'] for r in P.build_pilot_dataset(self.dist)],
                         [r['pair_id'] for r in pinned])
        self.assertNotEqual([r['pair_id'] for r in pinned], [r['pair_id'] for r in other])

    def test_pinned_order_is_what_upstream_gets_under_hashseed_zero(self):
        code = ('import sys; sys.path.insert(0, "src"); import resid_mid8_pilot as P; '
                'print("".join(P.upstream_symbol_order()))')
        env = dict(os.environ, PYTHONHASHSEED='0')
        out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), ''.join(P.SYMBOL_ORDER))


class Gates(unittest.TestCase):

    def test_site_name_is_transformer_lens_resid_mid_8(self):
        from transformer_lens import utils
        self.assertEqual(utils.get_act_name('resid_mid', layer=8), P.SITE_NAME)

    def test_runner_refuses_without_the_hash_seed(self):
        env = {k: v for k, v in os.environ.items() if k != 'PYTHONHASHSEED'}
        out = subprocess.run([sys.executable, str(RUNNER)], cwd=ROOT, env=env,
                             capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn('PYTHONHASHSEED=0', out.stderr)

    def test_pinned_prereg_and_calculator_hashes_match_the_tree(self):
        self.assertEqual(M.sha256(ROOT / 'src' / 'decidability.py'), P.CALCULATOR_SHA)
        self.assertEqual(M.sha256(ROOT / 'PREREG_RESID_MID8.md'), P.PREREG_SHA)


if __name__ == '__main__':
    unittest.main()
