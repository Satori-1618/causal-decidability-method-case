"""The calculator's precision table must agree with the dtype metadata, for every dtype."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import decidability   # noqa: E402

DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
          'float32': torch.float32, 'float64': torch.float64}


class HalfUlpMatchesTheDtype(unittest.TestCase):
    def test_every_entry_equals_half_of_torch_eps(self):
        for name, dtype in DTYPES.items():
            for key in (name, f'torch.{name}'):
                self.assertEqual(decidability.half_ulp(key), torch.finfo(dtype).eps / 2,
                                 f'{key}: table and dtype metadata disagree')

    def test_bfloat16_is_the_entry_that_was_wrong(self):
        """It listed 8 bits and so halved the bfloat16 numerical floor."""
        self.assertEqual(decidability.MANTISSA_BITS['bfloat16'], 7)
        self.assertEqual(decidability.half_ulp('bfloat16'), 2.0 ** -8)

    def test_it_bounds_the_measured_worst_case_rounding_of_a_cast(self):
        x = torch.linspace(1.0, 2.0, 100001, dtype=torch.float64)
        for name, dtype in DTYPES.items():
            if dtype is torch.float64:
                continue
            worst = ((x.to(dtype).to(torch.float64) - x).abs() / x).max().item()
            self.assertLessEqual(worst, decidability.half_ulp(name) * (1 + 1e-9), name)
            self.assertGreater(worst, decidability.half_ulp(name) / 2, name)


if __name__ == '__main__':
    unittest.main()
