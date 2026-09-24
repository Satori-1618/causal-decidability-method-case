"""Freeze B must be the §8 call on the pilot record, and scoring must be §9–§10 exactly."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import resid_mid8_freeze as F  # noqa: E402

PILOT = ROOT / 'results' / 'resid_mid8_pilot' / 'pilot.json'
PREDICTIONS = ROOT / 'results' / 'resid_mid8_freeze_b' / 'predictions.json'


class Prediction(unittest.TestCase):

    def test_simultaneous_quantile_is_the_pinned_one(self):
        self.assertAlmostEqual(F.Z, 2.8070, places=4)

    def test_freeze_b_is_reproduced_from_the_pilot_record(self):
        if not (PILOT.exists() and PREDICTIONS.exists()):
            self.skipTest('pilot or Freeze B not present')
        frozen = json.loads(PREDICTIONS.read_text())
        again = F.predict(json.loads(PILOT.read_text())['summary'])
        for key in ('z', 'primary', 'crossover', 'secondary'):
            self.assertEqual(json.loads(json.dumps(again[key])), frozen[key])

    def test_ratio_is_separation_over_the_statistical_floor_in_float32(self):
        summary = {'mean_E_full': {'L': 4.0, 'A': 0.5},
                   'sd': {r: {x: {'d_inert': 1.0, 'd_all': 2.0} for x in F.COMPONENTS}
                          for r in F.READOUTS}}
        out = F.predict(summary)
        cell = next(c for c in out['primary'] if c['readout'] == 'L' and c['n'] == 100)
        self.assertEqual(cell['sigma'], 2.0)
        self.assertEqual(cell['binding_floor'], 'statistical')
        self.assertAlmostEqual(cell['ratio'], 4.0 / (F.Z * 2.0 / 10.0), places=9)
        self.assertEqual(len(out['primary']), 28)


class Realized(unittest.TestCase):

    def test_both_endpoints_compatible_is_undecided(self):
        got = F.realized([0.0, 2.0, -1.0, 1.0], [1.0, -1.0, 2.0, 0.0])
        self.assertTrue(got['inert_compatible'] and got['all_compatible'])
        self.assertFalse(got['decided'])

    def test_excluding_one_endpoint_decides(self):
        got = F.realized([0.1, -0.1, 0.0, 0.05], [3.0, 3.1, 2.9, 3.0])
        self.assertTrue(got['inert_compatible'])
        self.assertFalse(got['all_compatible'])
        self.assertTrue(got['decided'])

    def test_excluding_both_endpoints_also_decides(self):
        got = F.realized([1.0, 1.1, 0.9, 1.0], [1.0, 0.9, 1.1, 1.0])
        self.assertFalse(got['inert_compatible'] or got['all_compatible'])
        self.assertTrue(got['decided'])

    def test_zero_spread_at_zero_is_compatible(self):
        self.assertTrue(F.realized([0.0] * 5, [1.0] * 5)['inert_compatible'])


class Scoring(unittest.TestCase):

    def _primary(self, ratios):
        return [{'readout': 'L', 'component': 'row', 'n': i, 'ratio': r,
                 'decidable': r > 1, 'outside_band': not (0.5 <= r <= 2.0)}
                for i, r in enumerate(ratios)]

    def _realized(self, decided):
        return {('L', 'row', i): {'decided': d} for i, d in enumerate(decided)}

    def test_supported_needs_eighty_percent_and_no_outside_miss(self):
        primary = self._primary([3.0] * 8 + [1.5, 1.5])
        out = F.score(primary, self._realized([True] * 8 + [False, False]))
        self.assertEqual(out['verdict'], 'supported')

    def test_any_outside_band_miss_contradicts(self):
        primary = self._primary([3.0] * 10)
        out = F.score(primary, self._realized([True] * 9 + [False]))
        self.assertEqual(out['verdict'], 'contradicted')

    def test_inside_band_misses_below_eighty_percent_are_inconclusive(self):
        primary = self._primary([3.0] * 7 + [1.5] * 3)
        out = F.score(primary, self._realized([True] * 7 + [False] * 3))
        self.assertEqual(out['verdict'], 'inconclusive')


class Order(unittest.TestCase):

    def test_interleaving_balances_every_even_prefix(self):
        data = ([{'half': 'ABB->BAB', 'i': i} for i in range(5)]
                + [{'half': 'BAB->ABB', 'i': i} for i in range(5)])
        out = F.interleave(data)
        for n in (2, 4, 6, 8, 10):
            self.assertEqual(sum(r['half'] == 'ABB->BAB' for r in out[:n]), n // 2)
        self.assertEqual([r['i'] for r in out[::2]], list(range(5)))

    def test_blocks_are_disjoint_and_consecutive(self):
        inert = [0.0, 0.0, 5.0, 5.1, 0.0, 0.0, 9.0]
        all_ = [1.0, 1.0, 0.0, 0.1, 1.0, 1.0, 9.0]
        out = F.block_rate(inert, all_, 2)
        self.assertEqual(out['blocks'], 3)


class Gates(unittest.TestCase):

    def test_confirmation_refuses_without_the_hash_seed(self):
        env = {k: v for k, v in os.environ.items() if k != 'PYTHONHASHSEED'}
        out = subprocess.run(
            [sys.executable, str(ROOT / 'scripts' / 'run_resid_mid8_confirmation.py')],
            cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn('PYTHONHASHSEED=0', out.stderr)


if __name__ == '__main__':
    unittest.main()
