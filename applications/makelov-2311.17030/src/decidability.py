"""A1 -- the pre-experiment decidability calculator. Zero forwards, zero observations.

Written AFTER src/ablation2_worlds.py was frozen at
5fd26d856c7bb91b4655ac340ebe2b4acd9c05a5f0d4aff972ef174d02e633c3.

WHAT IT ANSWERS. Not "is my effect large enough to see" (power) but "do my rivals predict
things far enough apart that the answer can say which one is right". The two are
independent: the grand-mean case of this project had perfect power and zero separation.

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
and `depth` are counted off the estimand and the implementation, P is declared. Whether it
is any good is measured -- scripts/run_ablation2.py records the share of worlds whose
observed working-vs-reference gap stays under it, on development AND on held-out seeds.

WHAT IT IS NOT. Not a causality test, not an identification guarantee, not a bound on the
error rate of anything. It is a statement about the design, computed before the design is
executed, and it is falsifiable: D1 in AGENT_BRIEF_LEAN_METHOD.md section 6 asks whether
it predicts what the run then does.
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
    if n < 1 or sigma < 0 or noise_factor < 0:
        raise ValueError('invalid declared sampling model')
    z = bonferroni_z(alpha, signatures) if z is None else z
    return z * sigma * noise_factor / math.sqrt(n)


def numerical_floor(dtype, readout_scale, depth=1, combination=DEFAULT_COMBINATION):
    """From the dtype only. No data. See the module docstring for the derivation."""
    if readout_scale < 0 or depth < 1 or combination < 0:
        raise ValueError('invalid declared readout')
    return combination * half_ulp(dtype) * readout_scale * math.sqrt(1.0 + depth / 3.0)


def decidability(predictions, n, sigma, dtype, readout_scale, depth=1,
                 equivalence_groups=(), noise_factor=DEFAULT_NOISE_FACTOR,
                 combination=DEFAULT_COMBINATION, alpha=DEFAULT_ALPHA, signatures=3,
                 z=None):
    """The whole calculation. Inputs are declarations; nothing here may be an observation.

    Returns separation, both floors separately, the ratio, the verdict and the lever that
    moves it.
    """
    gap, closest, pairs = separation(predictions, equivalence_groups)
    stat = statistical_floor(n, sigma, noise_factor, z, alpha, signatures)
    num = numerical_floor(dtype, readout_scale, depth, combination)
    resolution = max(stat, num)
    binding = 'statistical' if stat >= num else 'numerical'
    ratio = (0.0 if gap == 0 else gap / resolution if resolution > 0 else math.inf)
    decidable = gap > 0 and ratio > 1.0
    out = {'separation': gap, 'closest_rivals': list(closest) if closest else None,
           'pairwise': pairs, 'statistical_floor': stat, 'numerical_floor': num,
           'resolution': resolution, 'binding_floor': binding, 'ratio': ratio,
           'decidable': decidable, 'n': n, 'sigma': sigma, 'dtype': dtype,
           'readout_scale': readout_scale, 'depth': depth}
    out['lever'] = _lever(gap, ratio, binding, n, decidable)
    return out


def _lever(gap, ratio, binding, n, decidable):
    """What to change, and by how much. The point of the number is that it names a move."""
    if gap == 0.0:
        return {'kind': 'intervention_or_candidates',
                'detail': 'two rivals predict the same value under this operator; no '
                          'number of units and no dtype changes that',
                'replicates_needed': None, 'extra_mantissa_bits': None}
    if decidable:
        return {'kind': 'none', 'detail': f'separation exceeds the resolution by {ratio:.2f}x',
                'replicates_needed': None, 'extra_mantissa_bits': None}
    if binding == 'statistical':
        needed = math.ceil(n / ratio ** 2)
        return {'kind': 'units_or_noise',
                'detail': f'the statistical floor binds; n must rise from {n} to about '
                          f'{needed}, or sigma fall by {1 / ratio:.2f}x',
                'replicates_needed': needed, 'extra_mantissa_bits': None}
    bits = math.ceil(math.log2(1.0 / ratio))
    return {'kind': 'precision',
            'detail': f'the numerical floor binds; the readout needs about {bits} more '
                      f'mantissa bits, or a shallower accumulation',
            'replicates_needed': None, 'extra_mantissa_bits': bits}


# --- the benchmark's own declared readout scale -----------------------------------
# Specific to the generator in src/ablation_worlds.py, and derived from its DECLARED
# constants, never from an observation or from the true node share.

def benchmark_readout_scale(gain, interaction_magnitude, dim=2560, node_norm=24.8):
    """Magnitude of one panel readout of that generator.

    Linear path: gain * <unit readout direction, node vector> ~ gain * ||a|| / sqrt(dim).
    Quadratic path: its contribution to a panel value is at most 2 * |downstream|, and the
    largest downstream share the CANDIDATE SPACE allows is the whole interaction, so
    2 * |I|. The true share is not used: it is what the run is trying to find out.
    """
    return gain * node_norm / math.sqrt(dim) + 2.0 * abs(interaction_magnitude)


def source_sha256():
    with open(__file__, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()
