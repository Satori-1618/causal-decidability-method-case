"""The calculator this application ran with, pinned here byte for byte, must give the
same numbers as the core package on main. Its docstring differs; its arithmetic may not."""
import itertools
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
CORE = APP.parents[1] / 'src'
sys.path.insert(0, str(APP / 'src'))
import decidability as pinned  # noqa: E402

PINNED_SHA = 'ab0fd838a9a6b354d72d83c8b9730b095c0a443a4eeae756f2d1491eb17365f8'


class Equivalence(unittest.TestCase):

    def test_pinned_copy_is_the_one_that_ran(self):
        self.assertEqual(pinned.source_sha256(), PINNED_SHA)

    def test_core_calculator_agrees_on_a_grid(self):
        if not (CORE / 'causal_decidability' / 'calculator.py').exists():
            self.skipTest('core package not present')
        sys.path.insert(0, str(CORE))
        from causal_decidability import calculator as core
        grid = itertools.product((0.0, 0.01, 0.73, 4.8), (1, 3, 20, 2000), (0.0, 0.5, 1.5),
                                 ('bfloat16', 'float16', 'float32', 'float64'),
                                 (1.0, 20.0), (1, 768), (1.0, 2 ** 0.5), (2, 3))
        for gap, n, sigma, dtype, scale, depth, noise, sig in grid:
            args = dict(predictions={'a': 0.0, 'b': gap, 'c': gap / 3}, n=n, sigma=sigma,
                        dtype=dtype, readout_scale=scale, depth=depth,
                        equivalence_groups=[('b', 'c')], noise_factor=noise,
                        signatures=sig)
            self.assertEqual(pinned.decidability(**args), core.decidability(**args))


if __name__ == '__main__':
    unittest.main()
