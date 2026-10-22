import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_screen import exact_bounds, primary_counts, summarize, TAIL_ALPHA


def synthetic(accepted_hits=32, rejected_hits=0, state_misses=0):
    rows=[]
    for stratum, hits in [('accepted', accepted_hits), ('rejected', rejected_hits)]:
        for i in range(32):
            baseline = 1. if stratum == 'accepted' else 9.
            gap = 1. if i < hits else 0.01
            cells={}
            for sign in ['neg', 'pos']:
                for position in [20, 28]:
                    for replica in [0, 1]:
                        name=f'{sign}_{position}_{replica}'
                        if stratum == 'rejected' and name not in ['neg_20_0', 'pos_28_0']:
                            continue
                        value = baseline + (gap if sign == 'pos' else 0.)
                        if stratum == 'accepted' and i < state_misses and name == 'neg_20_1':
                            value += .2
                        snapshot={'native_margin': baseline, 'patched_margin': value,
                                  'margin_change': value-baseline, 'a_r': .03}
                        cells[name]={'float32': dict(snapshot), 'float64': dict(snapshot)}
            rows.append({'family_id':f'{stratum}_{i}', 'stratum':stratum, 'cells':cells})
    return rows


class ScreenAnalysisTests(unittest.TestCase):
    def test_exact_extremes_are_not_degenerate(self):
        self.assertAlmostEqual(exact_bounds(0, 32)[1], 1-TAIL_ALPHA**(1/32))
        self.assertAlmostEqual(exact_bounds(32, 32)[0], TAIL_ALPHA**(1/32))
        self.assertEqual(exact_bounds(0,32)[0], 0.)
        self.assertEqual(exact_bounds(32,32)[1], 1.)

    def test_symmetric_exact_intervals(self):
        for k in range(33):
            a, b=exact_bounds(k,32), exact_bounds(32-k,32)
            self.assertAlmostEqual(a[0], 1-b[1])
            self.assertAlmostEqual(a[1], 1-b[0])

    def test_countermodels_strong_equal_reverse(self):
        strong=primary_counts(32,0)
        self.assertEqual(strong['decision'], 'supports_at_least_25pp_enrichment')
        self.assertEqual(primary_counts(16,16)['directional_status'], 'unresolved')
        reverse=primary_counts(0,32)
        self.assertEqual(reverse['directional_status'], 'negative_enrichment')
        self.assertEqual(reverse['decision'], 'excludes_25pp_enrichment')

    def test_weaker_claim_is_not_practical_success(self):
        result=primary_counts(16,0)
        self.assertEqual(result['directional_status'], 'positive_enrichment')
        self.assertEqual(result['decision'], 'unresolved_at_25pp')

    def test_primary_can_pass_while_secondary_fails(self):
        result=summarize(synthetic(state_misses=8))
        self.assertEqual(result['primary']['decision'], 'supports_at_least_25pp_enrichment')
        self.assertFalse(result['secondary']['confirmation_start_gate'])
        self.assertEqual(result['secondary']['candidates']['H_state']['definite_hits'],24)

    def test_secondary_threshold_not_relaxed(self):
        self.assertTrue(summarize(synthetic(state_misses=3))['secondary']['confirmation_start_gate'])
        self.assertFalse(summarize(synthetic(state_misses=4))['secondary']['confirmation_start_gate'])

    def test_missing_case_and_wrong_stratum_fail_closed(self):
        rows=synthetic()
        with self.assertRaises(ValueError): summarize(rows[:-1])
        rows[0]['stratum']='rejected'
        with self.assertRaises(ValueError): summarize(rows)

    def test_cross_dtype_separation_disagreement_blocks(self):
        rows=synthetic()
        rows[0]['cells']['pos_28_0']['float64']['patched_margin']=1.2020001
        rows[0]['cells']['pos_28_0']['float32']['patched_margin']=1.2019999
        with self.assertRaisesRegex(ValueError, 'Numerically'): summarize(rows)

    def test_no_silent_screen_reassignment(self):
        rows=synthetic()
        rows[0]['cells']['neg_20_0']['float64']['native_margin']=9.
        with self.assertRaisesRegex(ValueError, 'stratum'): summarize(rows)


if __name__ == '__main__':
    unittest.main()
