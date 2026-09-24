"""Step 3, measured: the resolution your own per-unit values support for a contrast.

When both conditions are measured on the same units (the same prompts, the same
base/source pairs), the relevant resolution is the paired one. It is usually much finer
than what published means suggest, because the two conditions move together. Aggregated
means cannot give it; that is the reason to export per-unit values.

Two readout types:

    binary       an indicator per unit (did the patch flip the prediction). With discordant
                 counts b and c, the difference is (b - c) / n with standard error
                 sqrt(b + c - (b - c)^2 / n) / n, and the exact McNemar test is reported
                 beside it, because normal intervals are unreliable on few discordant units.
    continuous   a real value per unit (a logit difference, a margin): the standard error
                 of the per-unit difference.

The ratios returned here are **separation over one standard error**. The calculator's
resolution is a quantile times that error (z * SE), so its ratio is smaller by z.
"""
import math
from statistics import fmean, stdev


def _pair(a, b):
    a, b = [float(x) for x in a], [float(x) for x in b]
    if len(a) != len(b):
        raise ValueError('a and b must cover the same units, in the same order')
    if len(a) < 2:
        raise ValueError('need at least two units')
    if not all(math.isfinite(x) for x in a + b):
        raise ValueError('non-finite per-unit value: a failed measurement, not data')
    return a, b


def exact_mcnemar_p(only_a, only_b):
    """Two-sided exact binomial test on the discordant units; nan if there are none.

    Computed in exact integer arithmetic and divided once, so it stays finite for any
    number of discordant units (a float 2 ** n overflows from n = 1024).
    """
    if min(only_a, only_b) < 0 or int(only_a) != only_a or int(only_b) != only_b:
        raise ValueError('discordant counts must be non-negative integers')
    total = int(only_a + only_b)
    if total == 0:
        return float('nan')
    tail = sum(math.comb(total, i) for i in range(int(min(only_a, only_b)) + 1))
    return min(1.0, 2 * tail / (1 << total))


def _ratio(separation, se):
    if se > 0:
        return abs(separation) / se
    return 0.0 if separation == 0 else math.inf


def paired_binary(a, b):
    """Paired resolution of a contrast between two 0/1 arrays over the same units."""
    a, b = _pair(a, b)
    if not all(x in (0.0, 1.0) for x in a + b):
        raise ValueError('binary readouts must be 0 or 1')
    n = len(a)
    only_a = sum(1 for x, y in zip(a, b) if x == 1.0 and y == 0.0)
    only_b = sum(1 for x, y in zip(a, b) if x == 0.0 and y == 1.0)
    separation = (only_a - only_b) / n
    paired_se = math.sqrt(max(only_a + only_b - (only_a - only_b) ** 2 / n, 0.0)) / n
    p_a, p_b = fmean(a), fmean(b)
    unpaired_se = math.sqrt(p_a * (1 - p_a) / n + p_b * (1 - p_b) / n)
    return {'readout': 'binary', 'n': n, 'separation': separation,
            'paired_se': paired_se, 'unpaired_se': unpaired_se,
            'paired_ratio': _ratio(separation, paired_se),
            'unpaired_ratio': _ratio(separation, unpaired_se),
            'discordant': {'a_only': only_a, 'b_only': only_b},
            'exact_mcnemar_p': exact_mcnemar_p(only_a, only_b)}


def paired_continuous(a, b):
    """Paired resolution of a contrast between two real-valued arrays over the same units."""
    a, b = _pair(a, b)
    n = len(a)
    diff = [x - y for x, y in zip(a, b)]
    separation = fmean(diff)
    paired_se = stdev(diff) / math.sqrt(n)
    unpaired_se = math.sqrt(stdev(a) ** 2 / n + stdev(b) ** 2 / n)
    return {'readout': 'continuous', 'n': n, 'separation': separation,
            'paired_se': paired_se, 'unpaired_se': unpaired_se,
            'paired_ratio': _ratio(separation, paired_se),
            'unpaired_ratio': _ratio(separation, unpaired_se)}


def paired(a, b):
    """0/1 arrays go to the binary form, anything else to the continuous one."""
    a, b = _pair(a, b)
    if all(x in (0.0, 1.0) for x in a + b):
        return paired_binary(a, b)
    return paired_continuous(a, b)
