import itertools
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import design


class DesignTests(unittest.TestCase):
    def test_path_counts_against_exhaustive_small_paths(self):
        for length in (6, 8, 10):
            for target in (-2, 0, 2):
                count = sum(design.prefix_state(''.join(p)) == (target, -4)
                            for p in itertools.product('()', repeat=length))
                self.assertEqual(design.completion_count(length, 0, False, target), count)

    def test_prefix_sampling_constraints(self):
        rng = random.Random(24)
        for p, d in itertools.product(design.POSITIONS, design.BALANCES):
            for _ in range(20):
                prefix = design.draw_prefix(rng,p,d)
                self.assertEqual(design.prefix_state(prefix),(d,-4))
                self.assertEqual((len(prefix),prefix[-1]),(p,')'))
                self.assertLessEqual(prefix.count('('),16)
                self.assertLessEqual(prefix.count(')'),16)

    def test_phase_is_about_prefix_not_suffix(self):
        prefix = design.draw_prefix(random.Random(8),20,2)
        self.assertEqual(design.phase_of((prefix+'()()')[:20]),design.phase_of((prefix+'))((')[:20]))

    def test_two_phases_disjoint_and_family_controls(self):
        banned={'recipient_strings': [], 'prefixes': {'20': [], '28': []}}
        a=design.generate(42,8,'development',banned)
        b=design.generate(43,8,'confirmation',banned)
        prefixes=lambda rows:{d['prefix_sha256'] for f in rows for d in f['donors']}
        self.assertFalse(prefixes(a)&prefixes(b))
        for family in a+b:design.validate_family(family)
        bad=dict(a[0]);bad['donors']=a[0]['donors'][:-1]
        with self.assertRaises(ValueError):design.validate_family(bad)

    def test_forecasts_share_anchors_and_test_replicas(self):
        p=design.predictions(dict(zip(design.ANCHORS,(1.,3.))))
        self.assertEqual(p['H_state']['neg_28_0'],1.)
        self.assertEqual(p['H_position']['neg_28_0'],3.)
        self.assertEqual(p['H_state']['pos_20_0'],3.)
        self.assertEqual(p['H_position']['pos_20_0'],1.)
        self.assertEqual(p['H_state']['neg_20_1'],1.)
        self.assertEqual(p['H_position']['pos_28_1'],3.)
        self.assertEqual(len(p['H_state']),6)


if __name__=='__main__':unittest.main()
