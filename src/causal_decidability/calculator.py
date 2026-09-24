"""The pre-experiment decidability calculator. Zero forwards, zero observations.

WHAT IT ANSWERS. Not "is my effect large enough to see" (power) but "do my rivals predict
things far enough apart that the answer can say which one is right". The two are
independent: an intervention can have perfect power and zero separation.

  separation   the smallest distance between the predictions of two rivals that are not
               declared always-together, UNDER THE OPERATOR ACTUALLY EXECUTED. An operator
               that severs the channel makes every rival predict the same thing and the
               separation is zero whatever the effect size.

  floors       two of them, reported separately and never summed, because they have
               different levers:
                 statistical  z * sigma * sqrt(noise_factor^2 / n)
                 numerical    combination * u(dtype) * readout_scale * sqrt(1 + depth/3)
               The resolution is the larger.

  ratio        separation / resolution. A design diagnostic under the declared model;
               it is neither a guarantee above 1 nor an impossibility result below 1.

THE NUMERICAL FLOOR, derived rather than fitted. One readout value of magnitude P is
accumulated in `depth` chunks. The elementwise cast contributes one relative rounding
u = 2^-(mantissa+1); each chunk-add rounds the running total, and `depth` independent
roundings uniform on +-uP have variance depth * (uP)^2 / 3. Their quadrature is
u * P * sqrt(1 + depth/3). The estimand is a fixed combination of `combination^2` such
readouts, so the floor is combination * u * P * sqrt(1 + depth/3).

This is an ESTIMATE of the floor, not a bound. It has no fitted constant: `combination`
and `depth` are counted off the estimand and the implementation, P is declared. In the
synthetic benchmark (branch validation/synthetic-benchmark) it over-predicted the measured
bfloat16 error by a median factor of about 27 in the worlds whose verdict it decided; keep
a measured precision check on the final estimand.

WHAT IT IS NOT. Not a causality test, not an identification guarantee, not a bound on the
error rate of anything. It is a statement about the design, computed before the design is
executed, and it is falsifiable: docs/validation.md reports how often it predicts what the
run then does.
"""
import hashlib
import math
from statistics import NormalDist

#: STORED (fraction) mantissa bits, excluding the implicit leading bit. The relative
#: half-ULP is then 2**-(bits + 1), which equals torch.finfo(dtype).eps / 2.
#: bfloat16 has 7. An earlier version listed 8 -- the precision including the implicit bit
#: -- while every other entry used the stored count, so the bfloat16 numerical floor came
#: out at half its value. tests/test_decidability_dtype.py now checks every entry against
#: the dtype metadata so the convention cannot drift again.
MANTISSA_BITS = {'bfloat16': 7, 'float16': 10, 'float32': 23, 'float64': 52,
                 'torch.bfloat16': 7, 'torch.float16': 10, 'torch.float32': 23,
                 'torch.float64': 52}
DEFAULT_ALPHA = 0.01
#: eight readout values enter a difference of two second differences; their coefficients
#: are +-1/2, so sqrt(8) is a factor-2 conservative reading of the quadrature.
DEFAULT_COMBINATION = math.sqrt(8.0)
#: a paired contrast of two independently measured panels
DEFAULT_NOISE_FACTOR = math.sqrt(2.0)


def _finite(name, value, minimum=None):
    """Refuse NaN and infinities: a non-finite declaration is an error, never a verdict."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be at least {minimum}, got {value!r}')
    return value


def half_ulp(dtype):
    """Relative half-ULP of this dtype. The only thing the numerical floor needs from it."""
    if dtype not in MANTISSA_BITS:
        raise ValueError(f'unknown dtype {dtype!r}')
    return 2.0 ** -(MANTISSA_BITS[dtype] + 1)


def bonferroni_z(alpha=DEFAULT_ALPHA, signatures=3):
    """Two-sided simultaneous quantile over `signatures` distinct predictions."""
    if not 0.0 < alpha < 1.0 or signatures < 1:
        raise ValueError('invalid interval declaration')
    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * signatures))


def separation(predictions, equivalence_groups=()):
    """Smallest distance between two rivals that are not declared always-together.

    Zero means the experiment cannot distinguish them however it comes out -- the
    grand-mean case. Returned with the pair that produces it, because that pair is what
    the analyst has to change.
    """
    for name, value in predictions.items():
        _finite(f'prediction {name!r}', value)
    names = sorted(predictions)
    groups = [set(g) for g in equivalence_groups]

    def together(a, b):
        return any({a, b} <= g for g in groups)

    pairs = [(abs(predictions[a] - predictions[b]), a, b)
             for i, a in enumerate(names) for b in names[i + 1:] if not together(a, b)]
    if not pairs:
        return 0.0, None, []
    pairs.sort()
    return pairs[0][0], (pairs[0][1], pairs[0][2]), [
        {'candidates': [a, b], 'distance': d} for d, a, b in pairs]


def statistical_floor(n, sigma, noise_factor=DEFAULT_NOISE_FACTOR, z=None,
                      alpha=DEFAULT_ALPHA, signatures=3):
    """From the declared sampling model only. No data."""
    _finite('n', n, 1)
    _finite('sigma', sigma, 0)
    _finite('noise_factor', noise_factor, 0)
    z = bonferroni_z(alpha, signatures) if z is None else _finite('z', z, 0)
    return z * sigma * noise_factor / math.sqrt(n)


def numerical_floor(dtype, readout_scale, depth=1, combination=DEFAULT_COMBINATION):
    """From the dtype only. No data. See the module docstring for the derivation."""
    _finite('readout_scale', readout_scale, 0)
    _finite('depth', depth, 1)
    _finite('combination', combination, 0)
    return combination * half_ulp(dtype) * readout_scale * math.sqrt(1.0 + depth / 3.0)


def decidability(predictions, n, sigma, dtype, readout_scale, depth=1,
                 equivalence_groups=(), noise_factor=DEFAULT_NOISE_FACTOR,
                 combination=DEFAULT_COMBINATION, alpha=DEFAULT_ALPHA, signatures=3,
                 z=None):
    """The whole calculation. Inputs are declarations; nothing here may be an observation.

    Returns separation, both floors separately, the ratio and planning guidance. A
    recommendation must clear both floors: changing only the currently larger one may
    leave the other above the separation. The legacy ``decidable`` field means only
    that this heuristic's ratio is strictly greater than one, not empirical power.
    """
    gap, closest, pairs = separation(predictions, equivalence_groups)
    stat = statistical_floor(n, sigma, noise_factor, z, alpha, signatures)
    num = numerical_floor(dtype, readout_scale, depth, combination)
    for name, value in [('separation', gap), ('statistical floor', stat),
                        ('numerical floor', num)]:
        _finite(name, value, 0)
    resolution = max(stat, num)
    binding = 'statistical' if stat >= num else 'numerical'
    ratio = (0.0 if gap == 0 else gap / resolution if resolution > 0 else math.inf)
    decidable = gap > 0 and ratio > 1.0
    out = {'separation': gap, 'closest_rivals': list(closest) if closest else None,
           'pairwise': pairs, 'statistical_floor': stat, 'numerical_floor': num,
           'resolution': resolution, 'binding_floor': binding, 'ratio': ratio,
           'decidable': decidable, 'n': n, 'sigma': sigma, 'dtype': dtype,
           'readout_scale': readout_scale, 'depth': depth,
           'interpretation': 'Planning heuristic, not a power calculation, error bound, '
                             'or identification guarantee. Check measured precision '
                             'and use a declared inference rule after the run.'}
    out['lever'] = _lever(
        gap, ratio, n, decidable, stat, num, dtype, readout_scale, depth, combination,
        lambda count: statistical_floor(count, sigma, noise_factor, z, alpha, signatures),
        has_pair=closest is not None)
    return out


def _lever(gap, ratio, n, decidable, stat, num, dtype, readout_scale, depth,
           combination, stat_at, has_pair=True):
    """Target strict ratio > 1 under the same floor model, without retuning it.

    ``replicates_needed`` is the required *total* number of independent units.
    ``extra_mantissa_bits`` describes a modeled precision requirement; an actually
    supported dtype is recommended separately. If no supported dtype suffices, the
    projected design keeps that blocker and never claims the change solves it.
    """
    common = {'replicates_needed': None, 'extra_mantissa_bits': None,
              'recommended_dtype': None, 'target': 'ratio > 1 (strict)',
              'blocking_floors': [], 'remaining_blockers': [],
              'units_alone_sufficient': False, 'precision_alone_sufficient': False}
    if gap == 0.0:
        detail = ('two rivals predict the same value under this operator; no '
                  'number of units and no dtype changes that' if has_pair else
                  'no distinct candidate groups are left to compare; declare a '
                  'comparison before planning its resolution')
        return dict(common, kind='intervention_or_candidates', detail=detail,
                    remaining_blockers=['structural'], projected_ratio=0.0,
                    clears_both_floors=False)
    if decidable:
        return dict(common, kind='none',
                    detail=f'separation exceeds the modeled resolution by {ratio:.2f}x; '
                           'no planning change is suggested, but inference is still required',
                    projected_statistical_floor=stat, projected_numerical_floor=num,
                    projected_ratio=ratio, clears_both_floors=True)

    def clears(floor):
        # Use the actual ratio criterion, including equality and floating-point edges.
        return floor == 0 or (floor < gap and gap / floor > 1.0)

    stat_blocked, num_blocked = not clears(stat), not clears(num)
    blockers = [name for name, blocked in [('statistical', stat_blocked),
                                          ('numerical', num_blocked)] if blocked]
    needed, bits, recommended_dtype = None, None, None
    projected_stat, projected_num = stat, num
    details = []
    if stat_blocked:
        # Equality is not enough. ceil(threshold) would return the unchanged n at
        # ratio == 1; floor(threshold) + 1 targets a strict crossing instead.
        try:
            threshold = n * (stat / gap) ** 2
        except OverflowError as error:
            raise ValueError('required sample size exceeds the finite planning range') from error
        _finite('required sample-size threshold', threshold, 0)
        needed = max(math.floor(threshold) + 1, math.floor(n) + 1)
        projected_stat = stat_at(needed)
        while not clears(projected_stat):
            # The mathematical minimum can round back onto the boundary; jump a
            # small conservative amount rather than incrementing an enormous integer
            # whose float representation would remain unchanged.
            needed = max(needed + 1, math.ceil(needed * (1.0 + 1e-12)))
            projected_stat = stat_at(needed)
        details.append(f'increase total independent units from {n} to {needed} '
                       'to clear the statistical floor (or reduce the declared contrast SD)')
    if num_blocked:
        bits = max(1, math.floor(math.log2(num) - math.log2(gap)) + 1)
        while not clears(math.ldexp(num, -bits)):
            bits += 1
        for candidate in ('bfloat16', 'float16', 'float32', 'float64'):
            if MANTISSA_BITS[candidate] <= MANTISSA_BITS[dtype]:
                continue
            candidate_floor = numerical_floor(candidate, readout_scale, depth, combination)
            if clears(candidate_floor):
                recommended_dtype, projected_num = candidate, candidate_floor
                break
        if recommended_dtype:
            details.append(f'use {recommended_dtype} to clear the numerical floor '
                           f'(modeled minimum: {bits} extra mantissa bits)')
        else:
            details.append(f'no supported dtype clears the numerical floor '
                           f'(modeled minimum: {bits} extra mantissa bits); '
                           'redesign the measurement/intervention and recompute both floors')
    if stat_blocked and num_blocked:
        kind = 'units_and_precision'
        details.append('both changes are needed; more units alone and higher precision '
                       'alone each leave the other floor unresolved')
    else:
        kind = 'units_or_noise' if stat_blocked else 'precision'
    remaining = [name for name, value in [('statistical', projected_stat),
                                         ('numerical', projected_num)] if not clears(value)]
    projected_resolution = max(projected_stat, projected_num)
    return dict(common, kind=kind, detail='; '.join(details),
                replicates_needed=needed, extra_mantissa_bits=bits,
                recommended_dtype=recommended_dtype, blocking_floors=blockers,
                remaining_blockers=remaining,
                units_alone_sufficient=stat_blocked and not num_blocked,
                precision_alone_sufficient=(num_blocked and not stat_blocked
                                            and recommended_dtype is not None),
                projected_statistical_floor=projected_stat,
                projected_numerical_floor=projected_num,
                projected_ratio=(gap / projected_resolution
                                 if projected_resolution else math.inf),
                clears_both_floors=not remaining)


def source_sha256():
    with open(__file__, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()
