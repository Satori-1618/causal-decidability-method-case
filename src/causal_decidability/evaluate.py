"""Steps 3 and 4 on measured data, under an explicit confirmation contract.

Three declarations carry the scientific weight and have to be fixed together, before the
data are seen. The function refuses to guess any of them:

**Loss: what a prediction error is.**

- ``'absolute'``: the mean absolute error per unit, formed per repeat and condition
  *before* anything is averaged. It asks how well a candidate predicts the individual
  cases. Errors of +2 and -2 count as 2 each; they do not cancel.
- ``'signed'``: the mean signed error per unit. It asks whether a candidate is right on
  average, a question about systematic deviation. Opposite errors cancel here by design,
  so a candidate can pass with large case-by-case errors. The signed mean is compared
  with ``[-tolerance, +tolerance]`` through the two contrasts ``mean - tolerance`` and
  ``mean + tolerance``, both resampled.

**Scope: where the loss is taken.** Per tested condition, or ``'pooled'`` over all tested
conditions. The Makelov read-source pilot used the absolute loss, pooled.

**Tolerance: when a loss is small enough.** Either an absolute number in readout units,
fixed from an independent justification, or a fraction of a reference gap between two
conditions, such as the full effect. A relative tolerance is estimated from the same
units as the loss, so it is resampled *together* with the loss: the bootstrap statistic
is ``loss - tolerance`` on each resample.

**Uncertainty rule.** A percentile bootstrap over units: complete units are resampled,
with all their repeats and conditions. The intervals are made simultaneous by Bonferroni
over all declared prediction-class-by-scope tests. They are **nominal**: a percentile
bootstrap has no guaranteed coverage, least of all for small or discrete samples, and Bonferroni does not
repair that. Fewer than ``min_units`` units (default 10) are refused.

**Status per candidate and scope.** With ``T = loss - tolerance`` and its interval
``[lo, hi]``:

- ``excluded``: ``lo > 0``, so the loss exceeds the tolerance with confidence;
- ``adequate``: ``hi <= 0``, so the loss is within the tolerance with confidence;
- ``undecided``: otherwise.

A candidate is excluded if it is excluded in any scope, and adequate only if it is
adequate in every scope. The **compatible set** is every candidate not excluded; declared
equivalence groups with identical operative predictions are decided once. A tested
condition cannot also supply its own predicted value. Being the only candidate left is
not the same as being shown adequate, and the output keeps the two apart.

**Comparisons.** For the absolute loss, each pair of candidates is also compared directly
on the per-unit loss difference, by ``compare``, which needs no tolerance and can be
called on its own. It reports an exact sign test (how often each candidate wins a case;
assumes independent units and, under the null, a win probability of one half on average
over the untied units), a sign-flip permutation test of the mean difference
(exact only under symmetry of the differences), and a nominal bootstrap interval.

The rule assumes the units are independent draws. It says nothing about units the design
did not sample, or about candidates nobody declared.
"""
import itertools
import math
import random
from statistics import fmean as _fmean

from .compatible_set import classify, groups_of
from .paired import exact_mcnemar_p

ANCHOR = 'same_as:'
LOSSES = ('absolute', 'signed')
STATUSES = ('excluded', 'adequate', 'undecided')


def _finite(value):
    if not math.isfinite(value):
        raise ValueError('non-finite derived statistic: rescale the data before inference')
    return value


def fmean(values):
    """Fail closed if arithmetic overflows despite finite input measurements."""
    try:
        return _finite(_fmean(values))
    except OverflowError as error:
        raise ValueError('non-finite derived statistic: rescale the data before inference') from error


def load_rows(rows):
    """``(unit, repeat, condition, value)`` rows -> ``{unit: {repeat: {condition: value}}}``.

    A repeat is a measurement of the whole unit that belongs together, such as one swap
    direction of a base pair; use a constant when there is only one. A duplicated
    ``(unit, repeat, condition)`` and a non-finite value are refused.
    """
    data = {}
    for unit, repeat, condition, value in rows:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'non-finite value for unit {unit!r}, repeat {repeat!r}, '
                             f'condition {condition!r}: a failed measurement, not data')
        cell = data.setdefault(unit, {}).setdefault(repeat, {})
        if condition in cell:
            raise ValueError(f'duplicate measurement for unit {unit!r}, repeat {repeat!r}, '
                             f'condition {condition!r}')
        cell[condition] = value
    if not data:
        raise ValueError('no data')
    return data


def _canonical_spec(spec, name, condition):
    """A prediction rule, not its realised value, determines declared equivalence."""
    if isinstance(spec, str):
        if not spec.startswith(ANCHOR):
            raise ValueError(f'{name!r}/{condition!r}: use a number or "{ANCHOR}<condition>"')
        anchor = spec[len(ANCHOR):]
        if not anchor:
            raise ValueError(f'{name!r}/{condition!r}: an anchor needs a condition name')
        if anchor == condition:
            raise ValueError(f'{name!r}/{condition!r}: a tested condition cannot anchor '
                             'its own prediction; that would be a vacuous self-anchor')
        return 'anchor', anchor
    value = float(spec)
    if not math.isfinite(value):
        raise ValueError(f'{name!r}/{condition!r}: non-finite prediction')
    return 'number', value


def _prediction(spec, values, name, condition):
    kind, value = _canonical_spec(spec, name, condition)
    return values[value] if kind == 'anchor' else value


def _check(data, candidates, tested, scale):
    if not candidates or not tested:
        raise ValueError('declare at least one candidate and one tested condition')
    if not data:
        raise ValueError('no data')
    if len(set(tested)) != len(tested):
        raise ValueError('tested conditions must be unique')
    needed = set(tested) | set(scale or ())
    signatures = {}
    for name, specs in candidates.items():
        missing = set(tested) - set(specs)
        if missing:
            raise ValueError(f'candidate {name!r} makes no prediction for {sorted(missing)}')
        operative = tuple((c, _canonical_spec(specs[c], name, c)) for c in tested)
        signatures[name] = operative
        needed |= {value for _, (kind, value) in operative if kind == 'anchor'}
    for unit, repeats in data.items():
        if not repeats:
            raise ValueError(f'unit {unit!r} has no measurements')
        for repeat, values in repeats.items():
            if not needed <= set(values):
                raise ValueError(f'unit {unit!r}, repeat {repeat!r} lacks '
                                 f'{sorted(needed - set(values))}')
            if not all(math.isfinite(values[c]) for c in needed):
                raise ValueError(f'unit {unit!r}, repeat {repeat!r}: '
                                 'non-finite measurement, not evidence for a candidate')
    return signatures


def _prediction_groups(candidates, signatures, equivalence_groups):
    """Validate classes from declared tested rules, never from coincident observations."""
    groups = groups_of(candidates, equivalence_groups)
    for group in groups:
        if len(set(group)) != len(group):
            raise ValueError('a candidate may appear only once in an equivalence group')
        first = signatures[group[0]]
        if any(signatures[name] != first for name in group[1:]):
            raise ValueError(f'declared equivalence group {group!r} has incompatible '
                             'tested prediction rules; identical observed values do '
                             'not establish equivalence')
    return groups


def _check_uncertainty(alpha, resamples, permutations, min_units):
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0, 1)')
    for name, value, minimum in (('resamples', resamples, 1),
                                 ('permutations', permutations, 0),
                                 ('min_units', min_units, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer at least {minimum}')


def unit_losses(data, candidates, tested, loss, scopes):
    """``{candidate: {scope: [loss per unit]}}``, units in sorted order."""
    out = {}
    for name, specs in candidates.items():
        out[name] = {}
        for scope in scopes:
            conditions = tested if scope == 'pooled' else [scope]
            per_unit = []
            for unit in sorted(data):
                errors = [values[c] - _prediction(specs[c], values, name, c)
                          for values in data[unit].values() for c in conditions]
                per_unit.append(fmean(abs(e) for e in errors) if loss == 'absolute'
                                else fmean(errors))
            out[name][scope] = per_unit
    return out


def _quantiles(draws, level):
    draws = sorted(draws)
    k = len(draws)
    lo = draws[max(0, math.floor((1 - level) / 2 * k))]
    hi = draws[min(k - 1, math.ceil((1 + level) / 2 * k) - 1)]
    return lo, hi


def _sign_flip_p(diffs, permutations, rng):
    observed = abs(fmean(diffs))
    n = len(diffs)
    if n <= 16:
        signs = itertools.product((1, -1), repeat=n)
        hits = total = 0
        for s in signs:
            total += 1
            hits += abs(fmean(d * x for d, x in zip(diffs, s))) >= observed - 1e-12
        return hits / total
    hits = sum(abs(fmean(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12
               for _ in range(permutations))
    return (hits + 1) / (permutations + 1)


def evaluate(data, candidates, tested, loss, tolerance=None, tolerance_fraction=None,
             scale=None, scope='per_condition', alpha=0.05, resamples=10_000,
             permutations=20_000, seed=0, min_units=10, equivalence_groups=()):
    """Evaluate declared candidates on per-unit data under an explicit contract.

    ``data`` comes from ``load_rows``. ``candidates`` maps each name to
    ``{tested condition: number or "same_as:<condition>"}``. Give exactly one of
    ``tolerance`` (absolute, readout units) or ``tolerance_fraction`` together with
    ``scale = (condition_a, condition_b)``. ``scope`` is ``'per_condition'`` or
    ``'pooled'``.
    """
    if loss not in LOSSES:
        raise ValueError(f'loss must be one of {LOSSES}; there is no default on purpose')
    if scope not in ('per_condition', 'pooled'):
        raise ValueError("scope must be 'per_condition' or 'pooled'")
    if (tolerance is None) == (tolerance_fraction is None):
        raise ValueError('declare exactly one of tolerance and tolerance_fraction')
    if tolerance_fraction is not None and (scale is None or len(scale) != 2):
        raise ValueError('a relative tolerance needs scale = (condition_a, condition_b)')
    declared = tolerance if tolerance is not None else tolerance_fraction
    if not (math.isfinite(declared) and declared >= 0):
        raise ValueError('the tolerance must be finite and non-negative')
    _check_uncertainty(alpha, resamples, permutations, min_units)
    signatures = _check(data, candidates, tested, scale)
    groups = _prediction_groups(candidates, signatures, equivalence_groups)
    representatives = {g[0]: candidates[g[0]] for g in groups}
    n = len(data)
    if n < min_units:
        raise ValueError(f'{n} units: the bootstrap is unreliable below {min_units}; '
                         f'collect more units or lower min_units knowingly')

    scopes = ['pooled'] if scope == 'pooled' else list(tested)
    losses = unit_losses(data, representatives, tested, loss, scopes)
    units = sorted(data)
    gaps = ([fmean(abs(v[scale[0]] - v[scale[1]]) for v in data[u].values()) for u in units]
            if tolerance_fraction is not None else None)

    def statistic(index, name, s):
        # (loss, tolerance, loss - tolerance, loss + tolerance) on one resample. For the
        # signed loss the mean keeps its sign: it is compared with [-tolerance, +tolerance]
        # through both contrasts, never through its absolute value, whose resampled
        # distribution lies above zero even for an unbiased candidate.
        mean_loss = fmean(losses[name][s][i] for i in index)
        tol = (tolerance if gaps is None
               else tolerance_fraction * fmean(gaps[i] for i in index))
        return tuple(_finite(v) for v in
                     (mean_loss, tol, mean_loss - tol, mean_loss + tol))

    tests = [(name, s) for name in representatives for s in scopes]
    level = 1 - alpha / len(tests)
    rng = random.Random(seed)
    upper = {t: [] for t in tests}
    lower = {t: [] for t in tests}
    everyone = list(range(n))
    for _ in range(resamples):
        index = rng.choices(everyone, k=n)
        for name, s in tests:
            _, _, above, below = statistic(index, name, s)
            upper[(name, s)].append(above)
            lower[(name, s)].append(below)

    representative_rows = {}
    for name, s in tests:
        mean_loss, tol, above, below = statistic(everyone, name, s)
        lo, hi = _quantiles(upper[(name, s)], level)
        if loss == 'absolute':
            verdict = 'excluded' if lo > 0 else 'adequate' if hi <= 0 else 'undecided'
            row = {'margin': above, 'interval': (lo, hi)}
        else:
            lo_b, hi_b = _quantiles(lower[(name, s)], level)
            verdict = ('excluded' if lo > 0 or hi_b < 0 else
                       'adequate' if hi <= 0 and lo_b >= 0 else 'undecided')
            row = {'margin': abs(mean_loss) - tol, 'interval': (lo, hi),
                   'interval_of_loss_plus_tolerance': (lo_b, hi_b)}
        representative_rows[(name, s)] = {'scope': s, 'loss': mean_loss, 'tolerance': tol,
                                          **row, 'status': verdict}
    representative_of = {name: g[0] for g in groups for name in g}
    rows = [{'candidate': name, **representative_rows[(representative_of[name], s)]}
            for name in candidates for s in scopes]
    status = {}
    for name in candidates:
        verdicts = [r['status'] for r in rows if r['candidate'] == name]
        status[name] = ('excluded' if 'excluded' in verdicts else
                        'adequate' if all(v == 'adequate' for v in verdicts) else 'undecided')

    kept = [g for g in groups if status[g[0]] != 'excluded']
    comparisons = (compare(data, candidates, tested, scope=scope, alpha=alpha,
                           resamples=resamples, permutations=permutations, seed=seed,
                           min_units=min_units, equivalence_groups=groups)
                   if loss == 'absolute' and len(groups) > 1 else [])

    contract = {'loss': loss, 'scope': scope,
                'tolerance': ({'absolute': tolerance} if tolerance is not None else
                              {'fraction': tolerance_fraction, 'of_gap_between': list(scale)}),
                'alpha': alpha, 'intervals': 'percentile bootstrap over units, nominal, '
                f'Bonferroni over {len(tests)} prediction-class-by-scope tests',
                'interval_calibration': 'nominal bootstrap; coverage not guaranteed',
                'interval_family_size': len(tests), 'prediction_classes': groups,
                'comparison_p_value_adjustment': 'none; declare the comparison family',
                'resamples': resamples,
                'permutations': permutations, 'seed': seed, 'units': n}
    return {'contract': contract, 'rows': rows, 'status': status,
            'retained': sorted(m for g in kept for m in g),
            'retained_groups': kept,
            'outcome': classify(groups, kept),
            'excluded_by': {name: [r['scope'] for r in rows
                                   if r['candidate'] == name and r['status'] == 'excluded']
                            for name in candidates if status[name] == 'excluded'},
            'comparisons': comparisons}


def compare(data, candidates, tested, scope='pooled', alpha=0.05, resamples=10_000,
            permutations=20_000, seed=0, min_units=10, equivalence_groups=()):
    """Which candidate predicts better? Pairwise, on the per-unit absolute loss; no tolerance.

    For each pair and scope, on the per-unit loss differences ``d = loss_a - loss_b``:

    - ``sign_test_p``: exact two-sided binomial test on the number of units each candidate
      wins (ties dropped). Its question is how often a candidate wins a case. It assumes
      independent units and, under the null, a win probability of one half on average
      over the untied units. If that probability varies between units, the test is
      conservative (Hoeffding 1956, Theorem 5).
    - ``sign_flip_p``: sign-flip permutation test of the mean difference. It is exact only
      if the differences are symmetric about zero under the null; equal expected losses do
      not imply that, so declare and justify the assumption before relying on it.
    - ``interval``: nominal percentile-bootstrap interval of the mean difference,
      Bonferroni over all pairs of declared prediction classes and scopes.

    The p-values are not adjusted across pairs; declare the family or hierarchy in advance.
    Declared equivalents must have identical tested prediction rules. Each class has one
    representative in the comparisons; ``pair_members`` preserves its aliases. With one
    class there is no comparison to make and the result is empty. All ties yield p = 1
    and ``sign_test_status = 'no_untied_units'``, not evidence of equivalence.
    """
    if scope not in ('per_condition', 'pooled'):
        raise ValueError("scope must be 'per_condition' or 'pooled'")
    if len(candidates) < 2:
        raise ValueError('a comparison needs at least two candidates')
    _check_uncertainty(alpha, resamples, permutations, min_units)
    signatures = _check(data, candidates, tested, None)
    groups = _prediction_groups(candidates, signatures, equivalence_groups)
    representatives = {g[0]: candidates[g[0]] for g in groups}
    members = {g[0]: g for g in groups}
    n = len(data)
    if n < min_units:
        raise ValueError(f'{n} units: below {min_units}')
    if len(groups) == 1:
        return []
    scopes = ['pooled'] if scope == 'pooled' else list(tested)
    losses = unit_losses(data, representatives, tested, 'absolute', scopes)
    everyone = list(range(n))
    pairs = list(itertools.combinations(sorted(representatives), 2))
    level = 1 - alpha / (len(pairs) * len(scopes))
    out = []
    for a, b in pairs:
        for s in scopes:
            # Common draws keep a comparison invariant to alias names and the ordering
            # of other declared pairs. Bonferroni does not require independent draws.
            rng = random.Random(seed)
            diffs = [x - y for x, y in zip(losses[a][s], losses[b][s])]
            wins_a, wins_b = sum(d < 0 for d in diffs), sum(d > 0 for d in diffs)
            boot = [fmean(diffs[i] for i in rng.choices(everyone, k=n))
                    for _ in range(resamples)]
            out.append({
                'pair': (a, b), 'pair_members': {a: members[a], b: members[b]},
                'scope': s, 'units': n,
                'mean_difference': fmean(diffs), 'interval': _quantiles(boot, level),
                'interval_calibration': 'nominal bootstrap; coverage not guaranteed',
                'interval_family_size': len(pairs) * len(scopes),
                'p_value_adjustment': 'none',
                'p_value_family_size_per_test': len(pairs) * len(scopes),
                'units_favouring': {a: wins_a, b: wins_b}, 'ties': n - wins_a - wins_b,
                'sign_test_p': (exact_mcnemar_p(wins_a, wins_b)
                                if wins_a + wins_b else 1.0),
                'sign_test_status': 'tested' if wins_a + wins_b else 'no_untied_units',
                'sign_flip_p': _sign_flip_p(diffs, permutations, rng),
                'sign_flip_method': ('exact enumeration' if n <= 16 else
                                     f'Monte Carlo, {permutations} flips; '
                                     f'smallest reportable p = 1/{permutations + 1}'),
                'assumptions': {'sign_test_p': 'independent units; under the null an untied '
                                               'unit wins with probability 1/2 on average '
                                               '(conservative if it varies)',
                                'sign_flip_p': 'independent units; per-unit differences '
                                               'symmetric about zero under the null'}})
    return out
