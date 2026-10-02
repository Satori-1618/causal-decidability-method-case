from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze import analyze_family


def case(kind='state',gap=1.):
    cells={}
    for s in ('neg','pos'):
        for p in (20,28):
            for r in (0,1):
                v=float((s=='pos') if kind=='state' else (p==28))*gap
                snap={'patched_margin':v,'margin_change':v,'a_r':.1}
                cells[f'{s}_{p}_{r}']={'float32':dict(snap),'float64':dict(snap)}
    return {'family_id':'synthetic','cells':cells}


class AnalysisTests(unittest.TestCase):
    def test_opposite_known_profiles(self):
        for kind,winner in [('state','H_state'),('position','H_position')]:
            r=analyze_family(case(kind))
            self.assertTrue(r['eligible'])
            self.assertEqual(r['max_prediction_error'][winner],0.)
            self.assertEqual(max(r['max_prediction_error'].values()),1.)

    def test_same_label_replica_can_defeat_both(self):
        row=case()
        for dtype in ('float32','float64'):row['cells']['neg_20_1'][dtype]['patched_margin']=3.
        r=analyze_family(row)
        self.assertEqual(r['max_same_label_prefix_difference'],3.)
        self.assertTrue(all(e>=3. for e in r['max_prediction_error'].values()))

    def test_eligibility_strict_and_small_gap_not_equivalence(self):
        self.assertFalse(analyze_family(case(gap=.202))['eligible'])
        self.assertTrue(analyze_family(case(gap=.2020001))['eligible'])

    def test_precision_on_difference(self):
        row=case()
        row['cells']['neg_28_0']['float32']['patched_margin']+=.0007
        row['cells']['neg_20_0']['float32']['patched_margin']-=.0007
        with self.assertRaisesRegex(ValueError,'precision'):analyze_family(row)


if __name__=='__main__':unittest.main()
