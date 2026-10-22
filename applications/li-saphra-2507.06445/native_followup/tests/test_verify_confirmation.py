"""Boundary and tamper tests for the independent confirmation verifier."""
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / 'verify_confirmation.py'
SPEC = importlib.util.spec_from_file_location('independent_confirmation_verifier', PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def family(gap=.6, residual=0.):
    result = []
    for condition in ('valid', 'invalid_open', 'invalid_close'):
        row = {'family_id': 'synthetic', 'condition': condition}
        for dtype in ('float32', 'float64'):
            row[dtype + '_native'] = 0.
            row[dtype + '_routing_only'] = gap
            row[dtype + '_within_token'] = gap
            row[dtype + '_token_mass'] = residual if condition == 'valid' else 0.
        result.append(row)
    return result


FREEZE = {'gap_threshold': .502, 'prediction_tolerance': .25,
          'numerical_tolerance': .001, 'alpha': .05,
          'one_sided_bound_count': 288, 'minimum_eligible_families': 128,
          'adequacy': .8}


class IndependentVerifierTests(unittest.TestCase):
    def test_endpoint_bounds_have_closed_forms(self):
        alpha, n = .05 / 288, 512
        self.assertEqual(VERIFY.likelihood_interval(0, n, alpha)[0], 0.)
        self.assertAlmostEqual(VERIFY.likelihood_interval(0, n, alpha)[1], 1 - alpha**(1/n), places=14)
        self.assertAlmostEqual(VERIFY.likelihood_interval(n, n, alpha)[0], alpha**(1/n), places=14)
        self.assertEqual(VERIFY.likelihood_interval(n, n, alpha)[1], 1.)

    def test_frozen_precision_counts(self):
        for n, required in ((128, 120), (256, 230), (512, 446)):
            self.assertGreaterEqual(VERIFY.likelihood_interval(required, n, .05 / 288)[0], .8)
            self.assertLess(VERIFY.likelihood_interval(required - 1, n, .05 / 288)[0], .8)

    def test_family_max_does_not_hide_valid_member_failure(self):
        good = VERIFY.recompute_stage(family(), 'stage2', FREEZE)
        bad = VERIFY.recompute_stage(family(residual=.4), 'stage2', FREEZE)
        self.assertEqual(good['candidates']['H_W']['definite_hits'], 1)
        self.assertEqual(bad['candidates']['H_W']['possible_hits'], 0)
        self.assertEqual(bad['candidates']['H_W']['status'], 'insufficient_eligible_families')

    def test_gap_is_strict(self):
        self.assertEqual(VERIFY.recompute_stage(family(gap=.502), 'stage2', FREEZE)['eligible_families'], 0)
        self.assertEqual(VERIFY.recompute_stage(family(gap=math.nextafter(.502, math.inf)), 'stage2', FREEZE)['eligible_families'], 1)

    def test_numerical_guard_band(self):
        for residual, definite, possible in ((.249, 1, 1), (.25, 0, 1), (.252, 0, 0)):
            result = VERIFY.recompute_stage(family(residual=residual), 'stage2', FREEZE)['candidates']['H_W']
            self.assertEqual((result['definite_hits'], result['possible_hits']), (definite, possible))

    def test_forecast_mutation_rejected(self):
        measured = family()
        predicted = [{'family_id': r['family_id'], 'condition': r['condition'],
                      'predictions': {'within_token': {'within_token': .6, 'token_mass': 0.},
                                      'token_mass': {'within_token': 0., 'token_mass': .6}}} for r in measured]
        VERIFY.verify_prediction_rows(predicted, measured, 'stage2')
        predicted[1]['predictions']['within_token']['token_mass'] = .01
        with self.assertRaisesRegex(ValueError, 'forecast differs'):
            VERIFY.verify_prediction_rows(predicted, measured, 'stage2')

    def test_orbit_population_not_just_row_count(self):
        canonical = '(' * 16 + ')' * 16
        identifier = hashlib.sha256(canonical.encode()).hexdigest()[:20]
        rows = [dict(family_id=identifier, condition='valid', string=canonical, valid=True),
                dict(family_id=identifier, condition='invalid_open', string=canonical[1:] + canonical[:1], valid=False),
                dict(family_id=identifier, condition='invalid_close', string=canonical[16:] + canonical[:16], valid=False)]
        with tempfile.TemporaryDirectory() as temporary, patch.object(VERIFY, 'HERE', Path(temporary)):
            self.assertEqual(VERIFY.verify_case_population(rows, {'generator': {'families': 1}}), 0)
            corrupted = copy.deepcopy(rows)
            corrupted[1]['valid'] = True
            with self.assertRaisesRegex(ValueError, 'validity label'):
                VERIFY.verify_case_population(corrupted, {'generator': {'families': 1}})


if __name__ == '__main__':
    unittest.main()
