"""Three-name response geometry and a development-only scalar gain fit.

The coordinate order is always (A-B, A-C, B-C). These are three redundant
coordinates for two independent logit contrasts. Common logit offsets disappear.
Distances are RMS over these three coordinates, in logit units.

An unrestricted line fit is an optimistic per-cell oracle: its gain is chosen
after seeing that cell's response, including negative gains. It is not a held-out
predictor. A bounded gain may instead be fitted once on development families and
frozen by the caller. These mathematical comparisons do not identify a unique
mechanism or representation. No data splitting or model inference happens here.
"""

import math
from collections.abc import Mapping
from numbers import Real


def _items(value, label):
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f'{label} must be an iterable of numeric vectors, not text or a mapping')
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(f'{label} must be iterable') from exc


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label} must be a finite real number')
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{label} exceeds finite floating-point range') from exc
    if not math.isfinite(number):
        raise ValueError(f'{label} must be finite')
    return number


def _vector(value, label):
    values = _items(value, label)
    if len(values) != 3:
        raise ValueError(f'{label} must contain exactly three coordinates')
    return tuple(_finite(x, f'{label}[{i}]') for i, x in enumerate(values))


def validate_logits(logits):
    """Return exactly three finite real logits as a tuple, rejecting bools."""
    return _vector(logits, 'logits')


def pairwise_margins(logits):
    """Return (A-B, A-C, B-C); adding one constant to all logits has no effect."""
    a, b, c = validate_logits(logits)
    return _vector((a-b, a-c, b-c), 'computed pairwise margins')


def rms_distance(left_margins, right_margins):
    """RMS Euclidean distance between finite three-coordinate margin vectors.

    Callers provide margins, not raw logits. Coordinates are not independently
    normalized or reweighted. Dividing before hypot avoids squaring large values.
    """
    left, right = _vector(left_margins, 'left margins'), _vector(right_margins, 'right margins')
    root_three = math.sqrt(3.)
    differences = _vector((a/root_three-b/root_three for a, b in zip(left, right)),
                          'scaled margin differences')
    return _finite(math.hypot(*differences), 'RMS distance')


def _rescale_ratio(coefficient, numerator_scale, denominator_scale):
    """Compute coefficient * scale ratio without avoidable intermediate overflow."""
    if coefficient == 0 or numerator_scale == 0:
        return 0.
    a, ae = math.frexp(numerator_scale)
    b, be = math.frexp(denominator_scale)
    try:
        result = math.ldexp(coefficient*a/b, ae-be)
    except OverflowError as exc:
        raise ValueError('fitted gain exceeds finite floating-point range') from exc
    if result == 0:
        raise ValueError('nonzero fitted gain is below representable floating-point range')
    return _finite(result, 'fitted gain')


def closest_line_fit(observed_margins, direction_margins):
    """Closest point on {g * direction: g in R}, allowing any signed gain.

    The analytic gain is dot(observed, direction)/dot(direction, direction).
    RMS scaling does not change that minimizer. A zero direction defines the
    singleton {0}; its gain is unidentified (None), and its residual is retained.
    The per-cell oracle is deliberately more flexible than a frozen gain model.
    """
    observed = _vector(observed_margins, 'observed margins')
    direction = _vector(direction_margins, 'direction margins')
    direction_scale, observed_scale = max(map(abs, direction)), max(map(abs, observed))
    if direction_scale == 0:
        projection = (0., 0., 0.)
        return {'gain': None, 'projection': projection,
                'distance': rms_distance(observed, projection), 'zero_direction': True}
    if observed_scale == 0:
        return {'gain': 0., 'projection': (0., 0., 0.), 'distance': 0., 'zero_direction': False}
    unit_direction = tuple(x/direction_scale for x in direction)
    unit_observed = tuple(y/observed_scale for y in observed)
    coefficient = (math.fsum(x*y for x, y in zip(unit_direction, unit_observed))
                   / math.fsum(x*x for x in unit_direction))
    gain = _rescale_ratio(coefficient, observed_scale, direction_scale)
    projection = _vector((observed_scale*(coefficient*x) for x in unit_direction),
                         'line projection')
    return {'gain': gain, 'projection': projection,
            'distance': rms_distance(observed, projection), 'zero_direction': False}


def fit_bounded_gain(families):
    """Fit one development gain in [0,1], giving each whole family equal weight.

    ``families`` is a nonempty iterable of nonempty families. Each family contains
    pairs ``(observed_response_margins, target_direction_margins)``. The loss is
    mean_families(mean_rows(RMS(observed - gain * target)**2)). Replicating every
    row within a family therefore does not increase that family's weight.

    Return both the unconstrained raw_gain and constrained gain, with diagnostics
    when the raw optimum lies outside the bounds. Exactly 0 or 1 is a boundary
    optimum, not saturation. All-zero target directions make gain unidentifiable
    and raise ValueError; individual zero directions remain in the loss. The caller
    owns grouping, train/test separation and freezing any development estimate.
    """
    parsed = []
    for fi, family in enumerate(_items(families, 'families')):
        rows = []
        for ri, pair in enumerate(_items(family, f'family {fi}')):
            values = _items(pair, f'family {fi} row {ri}')
            if len(values) != 2:
                raise ValueError(f'family {fi} row {ri} must contain an observed/target pair')
            rows.append((_vector(values[0], f'family {fi} row {ri} observed'),
                         _vector(values[1], f'family {fi} row {ri} target')))
        if not rows:
            raise ValueError(f'family {fi} must contain at least one observed/target pair')
        parsed.append(rows)
    if not parsed:
        raise ValueError('at least one development family is required')
    target_scale = max(abs(x) for family in parsed for _, target in family for x in target)
    if target_scale == 0:
        raise ValueError('gain is unidentifiable: every target direction is zero')
    observed_scale = max(abs(y) for family in parsed for observed, _ in family for y in observed)
    denominator = math.fsum(math.fsum(
        math.fsum((x/target_scale)**2 for x in target)/3 for _, target in family)/len(family)
        for family in parsed)/len(parsed)
    if observed_scale == 0:
        raw_gain = 0.
    else:
        numerator = math.fsum(math.fsum(
            math.fsum((x/target_scale)*(y/observed_scale) for y, x in zip(observed, target))/3
            for observed, target in family)/len(family) for family in parsed)/len(parsed)
        raw_gain = _rescale_ratio(numerator/denominator, observed_scale, target_scale)
    gain = min(1., max(0., raw_gain))
    saturation = 'lower' if raw_gain < 0 else 'upper' if raw_gain > 1 else None
    errors = [[rms_distance(observed, tuple(gain*x for x in target))
               for observed, target in family] for family in parsed]
    error_scale = max(error for family in errors for error in family)
    rms_error = (error_scale*math.sqrt(math.fsum(
        math.fsum((error/error_scale)**2 for error in family)/len(family)
        for family in errors)/len(errors)) if error_scale else 0.)
    return {'gain': gain, 'raw_gain': raw_gain, 'saturated': saturation is not None,
            'saturation': saturation, 'n_families': len(parsed),
            'n_rows': sum(map(len, parsed)), 'rms_error': _finite(rms_error, 'fit RMS error')}
