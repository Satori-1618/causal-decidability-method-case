import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import makelov_read_source as m


class ReadSourceIntervention(unittest.TestCase):
    def setUp(self):
        self.w = np.array([[1., 0.], [0., 0.], [0., 0.]])
        self.d, self.audit = m.decompose_direction(np.array([1., 2., 0.]), self.w)

    def test_rank_deficient_output_does_not_create_extra_visible_direction(self):
        self.assertEqual(self.audit['rank'], 1)
        np.testing.assert_allclose(self.d['read_null'] @ self.w, 0, atol=1e-14)
        np.testing.assert_allclose(self.d['read_row']+self.d['read_null'], self.d['full'])

    def test_null_read_can_actuate_visible_output_with_common_write(self):
        base = torch.zeros(1, 3, dtype=torch.float64)
        donor = torch.tensor([[0., 5., 0.]], dtype=torch.float64)
        delta = m.patch_deltas(base, donor, self.d)
        self.assertGreater(abs(float((delta['read_null'] @ torch.tensor(self.w))[0, 0])), 0.1)
        torch.testing.assert_close(delta['read_row'], torch.zeros_like(base))
        torch.testing.assert_close(delta['full'], delta['read_null'])

    def test_nonrenormalized_read_components_sum_to_full_patch(self):
        torch.manual_seed(4)
        b, d = torch.randn(7, 3), torch.randn(7, 3)
        changes = m.patch_deltas(b, d, self.d)
        torch.testing.assert_close(changes['full'], changes['read_row']+changes['read_null'])
        self.assertLess(np.linalg.norm(self.d['read_row']), 1)
        self.assertLess(np.linalg.norm(self.d['read_null']), 1)

    def test_absolute_position_survives_appended_tokens(self):
        x = torch.randn(2, 9, 3)
        d = torch.ones(2, 3)
        audit = {}
        patched = m.position_hook(4, d, audit)(x)
        torch.testing.assert_close(patched[:, 4], x[:, 4]+d)
        self.assertTrue(torch.equal(patched[:, 5:], x[:, 5:]))
        self.assertTrue(torch.equal(patched[:, :4], x[:, :4]))
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['calls'], 1)

    def test_invalid_position_rejected(self):
        for pos in (-1, 5):
            with self.assertRaises(ValueError):
                m.position_hook(pos, torch.zeros(1, 3), {})(torch.zeros(1, 5, 3))

    def test_same_donor_is_exact_identity(self):
        h = torch.randn(5, 3)
        for delta in m.patch_deltas(h, h, self.d).values():
            self.assertTrue(torch.equal(delta, torch.zeros_like(h)))

    def test_pinned_vector_loaded_without_pickle(self):
        root = Path(__file__).resolve().parents[1]
        v = m.load_published_vector(root/'artifacts/makelov_source/das_mlp8.joblib')
        self.assertEqual(v.shape, (3072,))
        # Published float32 artifact is only approximately unit norm. The common
        # write vector is normalized ONCE by decompose_direction, not by the loader.
        self.assertLess(abs(np.linalg.norm(v)-1.), 1e-3)


if __name__ == '__main__':
    unittest.main()
