"""Geometry counterexamples and development-fit invariants, without model evidence."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

MODULE = (Path(__file__).resolve().parents[1]
          /'applications/makelov-2311.17030/src/role_geometry.py')
spec = importlib.util.spec_from_file_location('role_geometry', MODULE)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def scale(vector, factor):
    return tuple(factor*x for x in vector)


class RoleGeometryTests(unittest.TestCase):
    def test_pairwise_order_common_offset_invariance_and_scaling(self):
        a, b = (1.5, -.5, 2.), (-2., 1., .5)
        margins_a, margins_b = geometry.pairwise_margins(a), geometry.pairwise_margins(b)
        self.assertEqual(margins_a, (2., -.5, -2.5))
        self.assertEqual(margins_a[0]-margins_a[1]+margins_a[2], 0.)
        distance = geometry.rms_distance(margins_a, margins_b)
        for offset_a, offset_b in ((16., -32.), (-8., 64.)):
            shifted_a = geometry.pairwise_margins(tuple(x+offset_a for x in a))
            shifted_b = geometry.pairwise_margins(tuple(x+offset_b for x in b))
            self.assertEqual(shifted_a, margins_a)
            self.assertEqual(shifted_b, margins_b)
            self.assertEqual(geometry.rms_distance(shifted_a, shifted_b), distance)
        for multiplier in (-3., 2.):
            self.assertAlmostEqual(geometry.rms_distance(scale(margins_a, multiplier),
                                                        scale(margins_b, multiplier)),
                                   abs(multiplier)*distance)

    def test_rms_uses_all_three_unweighted_pairwise_coordinates(self):
        self.assertAlmostEqual(geometry.rms_distance((1., 2., 1.), (0., 0., 0.)), math.sqrt(2.))
        self.assertEqual(geometry.rms_distance((1., 2., 1.), (1., 2., 1.)), 0.)
        self.assertAlmostEqual(geometry.rms_distance((0., 0., 0.), (1., 2., 1.)), math.sqrt(2.))

    def test_oracle_recovers_arbitrary_amplitude_including_negative_gain(self):
        direction = geometry.pairwise_margins((3., 1., 0.))
        for gain in (-7.25, -1., 0., .001, .75, 3., 1000.):
            with self.subTest(gain=gain):
                observed = scale(direction, gain)
                fit = geometry.closest_line_fit(observed, direction)
                self.assertAlmostEqual(fit['gain'], gain)
                self.assertAlmostEqual(fit['distance'], 0., places=10)
                self.assertFalse(fit['zero_direction'])
                for actual, expected in zip(fit['projection'], observed):
                    self.assertAlmostEqual(actual, expected, places=10)

    def test_analytic_projection_is_orthogonal_and_minimizes_distance(self):
        observed, direction = (0., 1., 1.), (1., 1., 0.)
        fit = geometry.closest_line_fit(observed, direction)
        self.assertEqual(fit['gain'], .5)
        self.assertEqual(fit['projection'], (.5, .5, 0.))
        self.assertAlmostEqual(fit['distance'], math.sqrt(.5))
        residual = tuple(a-b for a, b in zip(observed, fit['projection']))
        self.assertAlmostEqual(sum(a*b for a, b in zip(residual, direction)), 0.)
        for offset in (-1., -.1, .1, 1.):
            competitor = geometry.rms_distance(observed, scale(direction, fit['gain']+offset))
            self.assertGreater(competitor, fit['distance'])

    def test_two_name_collinearity_for_three_candidates_and_three_name_separation(self):
        # Duplicating the second name as a third coordinate adds no information:
        # every nonzero two-name scalar prediction spans the same oracle line.
        observed = geometry.pairwise_margins((3., 0., 0.))
        two_name_candidates = [geometry.pairwise_margins((x, 0., 0.)) for x in (1., -2., .5)]
        for direction in two_name_candidates:
            self.assertAlmostEqual(geometry.closest_line_fit(observed, direction)['distance'], 0.)
        # With three genuinely distinct candidate names, the three one-name score
        # directions are different lines in the two-dimensional margin plane.
        three_name_candidates = [geometry.pairwise_margins(logits) for logits in
                                 ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))]
        distances = [geometry.closest_line_fit(observed, d)['distance'] for d in three_name_candidates]
        self.assertAlmostEqual(distances[0], 0.)
        self.assertGreater(distances[1], 0.)
        self.assertGreater(distances[2], 0.)

    def test_zero_direction_retains_error_and_has_no_identifiable_gain(self):
        for observed in ((3., 2., -1.), (0., 0., 0.)):
            fit = geometry.closest_line_fit(observed, (0., 0., 0.))
            self.assertIsNone(fit['gain'])
            self.assertTrue(fit['zero_direction'])
            self.assertEqual(fit['projection'], (0., 0., 0.))
            self.assertEqual(fit['distance'], geometry.rms_distance(observed, (0., 0., 0.)))
        with self.assertRaisesRegex(ValueError, 'unidentifiable'):
            geometry.fit_bounded_gain([[((1., 1., 0.), (0., 0., 0.))]])

    def test_bounded_gain_reports_raw_optimum_saturation_and_boundary_cases(self):
        target = (1., 1., 0.)
        for raw, gain, saturation in ((-.5, 0., 'lower'), (0., 0., None),
                                      (.4, .4, None), (1., 1., None), (2., 1., 'upper')):
            with self.subTest(raw=raw):
                fit = geometry.fit_bounded_gain([[(scale(target, raw), target)]])
                self.assertAlmostEqual(fit['raw_gain'], raw)
                self.assertAlmostEqual(fit['gain'], gain)
                self.assertEqual(fit['saturation'], saturation)
                self.assertIs(fit['saturated'], saturation is not None)
                self.assertAlmostEqual(fit['rms_error'],
                                       geometry.rms_distance(scale(target, raw), scale(target, gain)))

    def test_analytic_gain_uses_response_target_cross_product(self):
        target = (1., 1., 0.)
        families = [[(scale(target, .2), target)], [(scale(target, .8), scale(target, 2.))]]
        fit = geometry.fit_bounded_gain(families)
        self.assertAlmostEqual(fit['raw_gain'], (.2+2*.8)/(1+4))
        for gain in (0., .25, .5, 1.):
            loss = math.sqrt(sum(geometry.rms_distance(row[0], scale(row[1], gain))**2
                                 for family in families for row in family)/2)
            self.assertGreater(loss, fit['rms_error'])

    def test_replicating_rows_within_a_family_does_not_increase_family_weight(self):
        target = (1., 1., 0.)
        one = [((0., 0., 0.), target)]
        two = [(target, target)]
        original = geometry.fit_bounded_gain([one, two])
        repeated = geometry.fit_bounded_gain([one, two*100])
        self.assertEqual(original['gain'], .5)
        self.assertEqual(repeated['gain'], .5)
        self.assertAlmostEqual(original['rms_error'], repeated['rms_error'])
        self.assertEqual(repeated['n_families'], 2)
        self.assertEqual(repeated['n_rows'], 101)
        # Also duplicate an entire heterogeneous within-family row collection.
        mixed = [((0., 0., 0.), target), (scale(target, 2.), scale(target, 2.))]
        first = geometry.fit_bounded_gain([mixed, two])
        second = geometry.fit_bounded_gain([mixed*17, two])
        self.assertAlmostEqual(first['gain'], second['gain'])
        self.assertAlmostEqual(first['rms_error'], second['rms_error'])

    def test_individual_zero_targets_are_retained_in_the_loss(self):
        target = (1., 1., 0.)
        fit = geometry.fit_bounded_gain([[(scale(target, .5), target), (target, (0., 0., 0.))]])
        self.assertEqual(fit['gain'], .5)
        self.assertEqual(fit['n_rows'], 2)
        self.assertAlmostEqual(fit['rms_error'], math.sqrt(1/3))

    def test_inputs_are_not_mutated_and_large_or_tiny_scales_do_not_break_fit(self):
        families = [[([.5, .5, 0.], [1., 1., 0.])]]
        before = copy.deepcopy(families)
        geometry.fit_bounded_gain(families)
        self.assertEqual(families, before)
        for magnitude in (1e-200, 1e200):
            target = (magnitude, magnitude, 0.)
            observed = scale(target, .5)
            fit = geometry.fit_bounded_gain([[(observed, target)]])
            self.assertAlmostEqual(fit['gain'], .5)
            self.assertEqual(fit['rms_error'], 0.)
            self.assertAlmostEqual(geometry.closest_line_fit(observed, target)['gain'], .5)
            self.assertAlmostEqual(geometry.rms_distance(target, (0., 0., 0.))/magnitude,
                                   math.sqrt(2/3))

    def test_nonfinite_wrong_shape_and_empty_input_fail_closed(self):
        invalid_vectors = [None, [], [1., 2.], [1., 2., 3., 4.], '123', {'A': 1, 'B': 2, 'C': 3},
                           [True, 1., 2.], [complex(1), 1., 2.], [math.nan, 1., 2.],
                           [1., math.inf, 2.], [1., 2., -math.inf], [10**1000, 1, 2]]
        valid = (1., 1., 0.)
        for invalid in invalid_vectors:
            functions = [lambda: geometry.validate_logits(invalid),
                         lambda: geometry.pairwise_margins(invalid),
                         lambda: geometry.rms_distance(valid, invalid),
                         lambda: geometry.closest_line_fit(invalid, valid),
                         lambda: geometry.closest_line_fit(valid, invalid),
                         lambda: geometry.fit_bounded_gain([[(valid, invalid)]])]
            for index, function in enumerate(functions):
                with self.subTest(invalid=repr(invalid), function=index), self.assertRaises(ValueError):
                    function()
        for invalid in ([], [[]], None, {}, 'families', [[(valid,)]], [[(valid, valid, valid)]]):
            with self.subTest(families=invalid), self.assertRaises(ValueError):
                geometry.fit_bounded_gain(invalid)
        with self.assertRaises(ValueError):
            geometry.pairwise_margins((1e308, -1e308, 0.))


if __name__ == '__main__':
    unittest.main()
