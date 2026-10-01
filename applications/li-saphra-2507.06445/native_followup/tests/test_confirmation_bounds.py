import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_confirmation import kl_interval
from prepare_confirmation import rotations,valid,tasks


class BoundTests(unittest.TestCase):
    def test_boundary_solutions(self):
        a=.05/288
        lo,hi=kl_interval(0,128,a)
        self.assertEqual(lo,0)
        self.assertAlmostEqual(hi,1-a**(1/128))
        lo,hi=kl_interval(128,128,a)
        self.assertAlmostEqual(lo,a**(1/128))
        self.assertEqual(hi,1)

    def test_declared_hit_counts(self):
        a=.05/288
        for n,want in [(128,120),(256,230),(512,446)]:
            got=next(k for k in range(n+1) if kl_interval(k,n,a)[0]>=.8)
            self.assertEqual(got,want)

    def test_small_finite_population_coverage(self):
        # Exhaust the hypergeometric support; no iid substitution in this test.
        N,n,a=24,10,.05
        for successes in range(N+1):
            p=successes/N
            misslo=misshi=0.
            for k in range(max(0,n-(N-successes)),min(n,successes)+1):
                prob=math.comb(successes,k)*math.comb(N-successes,n-k)/math.comb(N,n)
                lo,hi=kl_interval(k,n,a)
                misslo+=prob*(lo>p)
                misshi+=prob*(hi<p)
            self.assertLessEqual(misslo,a+1e-12)
            self.assertLessEqual(misshi,a+1e-12)

    def test_orbit_equivalence_and_scope(self):
        self.assertEqual(rotations('())('),rotations(')(()'))
        self.assertTrue(any(valid(x) for x in rotations('())(')))
        cohort=tasks()
        self.assertEqual(len(cohort),36)
        self.assertTrue(all(t['role']!='transfer_cohort' or (t['init_seed'],t['shuffle_seed'])!=(365,220) for t in cohort))


if __name__=='__main__':
    unittest.main()
