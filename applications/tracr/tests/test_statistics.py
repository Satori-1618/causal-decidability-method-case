"""Exact finite-population inversion checked against combinatorial enumeration."""
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "src"))
sys.path.insert(0, str(APP.parents[1] / "src"))

from tracr_demo.design import FAMILY_POPULATION_SIZE  # noqa: E402
from tracr_demo.statistics import (evaluate_confirmation, finite_population_bounds,
                                   plan_confirmation)  # noqa: E402


def exact_tail(observed, n, population, successful, upper):
    low = max(0, n - population + successful)
    high = min(n, successful)
    return sum((Fraction(math.comb(successful, k) * math.comb(population - successful, n - k),
                         math.comb(population, n))
                for k in range(low, high + 1) if (k >= observed if upper else k <= observed)),
               Fraction(0))


class FinitePopulationStatistics(unittest.TestCase):
    def test_small_population_bounds_equal_exhaustive_integer_inversion(self):
        for population in (5, 10, 20):
            for n in (1, population // 2, population):
                for observed in range(n + 1):
                    with self.subTest(population=population, n=n, observed=observed):
                        alpha = Fraction(1, 20)
                        lower = min(k for k in range(population + 1)
                                    if exact_tail(observed, n, population, k, True) > alpha)
                        upper = max(k for k in range(population + 1)
                                    if exact_tail(observed, n, population, k, False) > alpha)
                        result = finite_population_bounds(observed, n, population, alpha=.05)
                        self.assertEqual(result["lower_successes"], lower)
                        self.assertEqual(result["upper_successes"], upper)

    def test_bound_coverage_for_every_small_population_success_count(self):
        population, n = 12, 6
        alpha = Fraction(1, 20)
        for successful in range(population + 1):
            lower_misses, upper_misses = Fraction(0), Fraction(0)
            for observed in range(max(0, n - population + successful), min(n, successful) + 1):
                probability = Fraction(math.comb(successful, observed)
                                       * math.comb(population - successful, n - observed),
                                       math.comb(population, n))
                bounds = finite_population_bounds(observed, n, population)
                if bounds["lower_successes"] > successful:
                    lower_misses += probability
                if bounds["upper_successes"] < successful:
                    upper_misses += probability
            self.assertLessEqual(lower_misses, alpha)
            self.assertLessEqual(upper_misses, alpha)

    def test_planned_count_and_power_match_small_population_enumeration(self):
        population, n = 20, 10
        plan = plan_confirmation(population, n, minimum_success_rate=.5, planning_success_rate=.9)
        required = min(s for s in range(n + 1)
                       if exact_tail(s, n, population, 10, True) <= Fraction(1, 20))
        self.assertEqual(plan["required_successes"], required)
        self.assertAlmostEqual(plan["null_rejection_probability"],
                               float(exact_tail(required, n, population, 10, True)))
        self.assertAlmostEqual(plan["power"], float(exact_tail(required, n, population, 18, True)))

    def test_main_design_size_power_and_only_fixed_sample_evaluation(self):
        self.assertEqual(FAMILY_POPULATION_SIZE, 119750400)
        population = FAMILY_POPULATION_SIZE - 8
        plan = plan_confirmation(population)
        self.assertEqual(plan["required_successes"], 126)
        self.assertEqual(plan["maximum_failures"], 2)
        self.assertLessEqual(plan["null_rejection_probability"], .05)
        self.assertGreater(plan["power"], .97)
        self.assertEqual(plan["statistical_looks"], 1)
        for successes, adequate in ((125, False), (126, True), (128, True)):
            result = evaluate_confirmation(successes, 128, population_size=population)
            self.assertEqual(result["adequate"], adequate)
        for incomplete in (64, 127, 129):
            with self.assertRaises(ValueError):
                evaluate_confirmation(incomplete, incomplete, population_size=population)

    def test_no_success_all_success_and_census_boundaries(self):
        self.assertEqual(finite_population_bounds(0, 5, 20)["lower_successes"], 0)
        self.assertEqual(finite_population_bounds(5, 5, 20)["upper_successes"], 20)
        for successes in range(11):
            bounds = finite_population_bounds(successes, 10, 10)
            self.assertEqual(bounds["lower_successes"], successes)
            self.assertEqual(bounds["upper_successes"], successes)
        plan = plan_confirmation(100, 1)
        self.assertIsNone(plan["required_successes"])
        self.assertEqual(plan["power"], 0)

    def test_invalid_counts_and_rates_fail_closed(self):
        for args in ((True, 5, 10), (-1, 5, 10), (6, 5, 10), (1, 0, 10),
                     (1, 11, 10), (1, 5, 0), (1, 5.0, 10)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                finite_population_bounds(*args)
        for alpha in (0, 1, -.1, math.nan, math.inf, True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                finite_population_bounds(1, 5, 10, alpha=alpha)
        for kwargs in (dict(minimum_success_rate=1), dict(planning_success_rate=.9),
                       dict(planning_success_rate=math.nan)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_confirmation(200, **kwargs)


if __name__ == "__main__":
    unittest.main()
