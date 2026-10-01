"""One fixed-look, finite-population adequacy test for whole reversal families.

The frozen generator samples uniformly without replacement after development
signatures are excluded. One success is the compound result of every required
family condition and gate; coordinates, stages and dtypes are not replicates.
Positive-control qualification does not create a second population claim.

Inversion is over the integer number of successful families in the remaining
population. SciPy's hypergeom tail routines evaluate the exact sampling law (no
normal or iid-binomial approximation). Near a decision boundary, rational
arithmetic resolves floating-point ambiguity. No outcomes are used by planning.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
"""
from fractions import Fraction
import math

from scipy.stats import hypergeom


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _rate(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value < 1):
        raise ValueError(f"{name} must lie strictly between zero and one")


def _design(population_size, sample_size, alpha):
    _integer(population_size, "population_size")
    _integer(sample_size, "sample_size")
    if not 1 <= sample_size <= population_size:
        raise ValueError("sample size must lie between one and population size")
    _rate(alpha, "alpha")


def _tail(successes, sample_size, population_size, population_successes, upper):
    if upper:
        return float(hypergeom.sf(successes - 1, population_size, population_successes, sample_size))
    return float(hypergeom.cdf(successes, population_size, population_successes, sample_size))


def _reject(successes, sample_size, population_size, population_successes, alpha, upper):
    tail = _tail(successes, sample_size, population_size, population_successes, upper)
    if not math.isfinite(tail):
        raise ValueError("non-finite hypergeometric probability")
    if math.isclose(tail, alpha, rel_tol=1e-12, abs_tol=0):
        low = max(0, sample_size - population_size + population_successes)
        high = min(sample_size, population_successes)
        if upper:
            low = max(low, successes)
        else:
            high = min(high, successes)
        numerator = sum(math.comb(population_successes, k)
                        * math.comb(population_size - population_successes, sample_size - k)
                        for k in range(low, high + 1))
        return Fraction(numerator, math.comb(population_size, sample_size)) <= Fraction(str(alpha))
    return tail <= alpha


def finite_population_bounds(successes, sample_size, population_size, *, alpha=.05):
    """Separate exact one-sided 1-alpha lower and upper success-rate bounds.

    The pair is not a joint 1-alpha confidence interval. Each boundary is the
    first/last integer population count not rejected by its corresponding tail.
    The primary adequacy claim uses only the lower bound.
    """
    _design(population_size, sample_size, alpha)
    _integer(successes, "successes")
    if not 0 <= successes <= sample_size:
        raise ValueError("success count must lie between zero and sample size")
    low, high = 0, population_size
    while low < high:
        middle = (low + high) // 2
        if _reject(successes, sample_size, population_size, middle, alpha, True):
            low = middle + 1
        else:
            high = middle
    lower = low
    low, high = 0, population_size
    while low < high:
        middle = (low + high + 1) // 2
        if _reject(successes, sample_size, population_size, middle, alpha, False):
            high = middle - 1
        else:
            low = middle
    upper = low
    return dict(lower_successes=lower, upper_successes=upper,
                lower_rate=lower / population_size, upper_rate=upper / population_size,
                alpha=alpha, coverage="each bound separately is one-sided 1-alpha")


def _population_count(rate, population_size, *, round_up=False):
    fraction = Fraction(str(rate)) * population_size
    return math.ceil(fraction) if round_up else math.floor(fraction)


def plan_confirmation(population_size, sample_size=128, *, alpha=.05,
                      minimum_success_rate=.95, planning_success_rate=.995):
    """Determine the smallest passing count and power before a single fixed look.

    H0 is finite-population success fraction <= minimum_success_rate. The null
    count is floored; the planning alternative is rounded up to the first
    attainable count at least planning_success_rate, and its actual rate is
    reported. population_size must exclude all distinct development signatures.
    """
    _design(population_size, sample_size, alpha)
    _rate(minimum_success_rate, "minimum_success_rate")
    _rate(planning_success_rate, "planning_success_rate")
    if planning_success_rate <= minimum_success_rate:
        raise ValueError("planning success rate must exceed the adequacy threshold")
    null_count = _population_count(minimum_success_rate, population_size)
    planned_count = _population_count(planning_success_rate, population_size, round_up=True)
    low, high = 0, sample_size + 1
    while low < high:
        middle = (low + high) // 2
        if _reject(middle, sample_size, population_size, null_count, alpha, True):
            high = middle
        else:
            low = middle + 1
    required = low if low <= sample_size else None
    size = _tail(required, sample_size, population_size, null_count, True) if required is not None else 0.0
    power = _tail(required, sample_size, population_size, planned_count, True) if required is not None else 0.0
    return dict(population_size=population_size, sample_size=sample_size,
                alpha=alpha, minimum_success_rate=minimum_success_rate,
                null_population_successes=null_count,
                required_successes=required,
                maximum_failures=sample_size - required if required is not None else None,
                null_rejection_probability=size,
                planning_success_rate=planning_success_rate,
                planning_population_successes=planned_count,
                attainable_planning_success_rate=planned_count / population_size,
                power=power, statistical_looks=1,
                unit="whole sequence family; compound success across required conditions and gates")


def evaluate_confirmation(successes, sample_size, *, population_size,
                          planned_sample_size=128, alpha=.05, minimum_success_rate=.95):
    """Evaluate only the complete frozen sample; no interim adequacy decisions."""
    _integer(planned_sample_size, "planned_sample_size")
    if sample_size != planned_sample_size:
        raise ValueError("adequacy requires exactly the frozen number of families, without replacement")
    _rate(minimum_success_rate, "minimum_success_rate")
    bounds = finite_population_bounds(successes, sample_size, population_size, alpha=alpha)
    null_count = _population_count(minimum_success_rate, population_size)
    passed = bounds["lower_successes"] > null_count
    return dict(successes=successes, sample_size=sample_size, population_size=population_size,
                minimum_success_rate=minimum_success_rate, bounds=bounds,
                null_upper_tail=_tail(successes, sample_size, population_size, null_count, True),
                adequate=passed, status="adequate" if passed else "not_demonstrated",
                statistical_looks=1)
