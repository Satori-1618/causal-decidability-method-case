"""Check actual source indexing, fixed read/write and identity with a small fake model."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'applications/makelov-2311.17030/src'))
from donor_factor import MAPS, SITE, run_family


class FakeModel:
    cfg = SimpleNamespace(n_layers=12, d_model=768, device='cpu')

    def __init__(self, double_call=False):
        self.double_call = double_call

    def activation(self, tokens):
        # Distinct donor states; token positions besides P-1 must stay untouched.
        return torch.stack((tokens.double(), tokens.double()**2), dim=-1)

    def output(self, h):
        out = torch.zeros((*h.shape[:2], 6), dtype=h.dtype)
        out[..., 1] = h[..., 0] + 2*h[..., 1]
        out[..., 2] = -.5*h[..., 0]
        return out

    def run_with_cache(self, tokens, names_filter):
        assert names_filter == [SITE]
        h = self.activation(tokens)
        return self.output(h), {SITE: h}

    def run_with_hooks(self, tokens, fwd_hooks):
        h = self.activation(tokens)
        for site, hook in fwd_hooks:
            assert site == SITE
            h = hook(h)
            if self.double_call:
                h = hook(h)
        return self.output(h)


def inputs():
    case = {'token_ids': [[9, 1], [9, 2], [9, 3], [9, 4]], 'position': 1,
            'answer_token_ids': [1, 2], 'recipient_indices': [0, 1],
            'donor_indices': list(MAPS)}
    directions = {'full': np.array([.6, .8]), 'read_null': np.array([.1, .2]),
                  'read_row': np.array([.5, .6])}
    return case, directions


class DonorFactorHookTests(unittest.TestCase):
    def test_all_eight_source_assignments_and_output_orientation(self):
        case, directions = inputs()
        result = run_family(FakeModel(), case, directions)
        h = np.array([[t[-1], t[-1]**2] for t in case['token_ids']], dtype=float)
        for r, panel in enumerate(result['panels']):
            for cell, donor in MAPS[r].items():
                alpha = float((h[donor]-h[r]) @ directions['read_null'])
                expected = panel['baseline_margin'] + alpha * (1.5*.6+2*.8)
                self.assertAlmostEqual(panel['alphas'][cell], alpha)
                self.assertAlmostEqual(panel['patched_margins'][cell], expected)
        self.assertEqual(result['model_forward_calls'], 6)
        self.assertTrue(result['controls']['identity_exact'])
        for panel in result['panels']:
            self.assertEqual(panel['alphas']['00'], 0)
            self.assertEqual(panel['delta_l2']['00'], 0)

    def test_mapping_position_and_hook_count_fail_closed(self):
        case, directions = inputs()
        bad = copy.deepcopy(case)
        bad['donor_indices'][1] = dict(MAPS[0])
        with self.assertRaises(ValueError):
            run_family(FakeModel(), bad, directions)
        bad = copy.deepcopy(case)
        bad['position'] = 0
        with self.assertRaises(ValueError):
            run_family(FakeModel(), bad, directions)
        with self.assertRaises(RuntimeError):
            run_family(FakeModel(double_call=True), case, directions)


if __name__ == '__main__':
    unittest.main()
