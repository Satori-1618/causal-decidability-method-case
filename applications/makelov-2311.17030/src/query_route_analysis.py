"""Round-2 query-route response profiles; no model inference or graph identification.

One record is one independently sampled base-pair draw. Its two reciprocal
directions and its two precision evaluations are components of a single success,
never additional statistical units. Repeated *content* is allowed under iid
sampling with replacement; draw IDs must be unique.

The Clopper--Pearson construction uses binomial tails, evaluated with floating
point arithmetic and bisection. It is an exact-binomial interval construction,
not a guarantee about numerical error, sampling validity, or causal semantics.
"""

import math
from collections import Counter
from collections.abc import Mapping


PRECISIONS = ("float32", "float64")
CELLS = tuple("ABCDEFG")
PROFILES = ("transfer", "joint_dependence", "preservation")


class InvalidControlsError(ValueError):
    """A technical failure blocks analysis of the run, rather than losing a case."""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def binomial_tail(k, n, p, side):
    """Inclusive P(X <= k) or P(X >= k), without subtracting a CDF from one."""
    n = _integer(n, "n")
    k = _integer(k, "k")
    p = _finite(p, "p")
    if k > n or not 0 <= p <= 1 or side not in ("lower", "upper"):
        raise ValueError("require 0 <= k <= n, 0 <= p <= 1, and lower/upper side")
    if p == 0:
        return float(side == "lower" or k == 0)
    if p == 1:
        return float(side == "upper" or k == n)
    indices = range(k + 1) if side == "lower" else range(k, n + 1)
    log_p, log_q = math.log(p), math.log1p(-p)
    log_n_factorial = math.lgamma(n + 1)
    terms = [math.exp(log_n_factorial - math.lgamma(j + 1)
                      - math.lgamma(n - j + 1) + j * log_p
                      + (n - j) * log_q) for j in indices]
    return min(1.0, max(0.0, math.fsum(terms)))


def clopper_pearson(successes, n, alpha=.05, family_size=3):
    """Two-sided CP with alpha/(2*family_size) in each tail.

    Bonferroni controls the family error without independence between profiles;
    independent, identically sampled base-pair successes are still required.
    """
    n = _integer(n, "n", 1)
    successes = _integer(successes, "successes")
    family_size = _integer(family_size, "family_size", 1)
    alpha = _finite(alpha, "alpha")
    if successes > n or not 0 < alpha < 1:
        raise ValueError("require successes <= n and 0 < alpha < 1")
    tail = alpha / (2 * family_size)
    if tail == 0:
        raise ValueError("the requested tail probability is below float resolution")

    if successes == 0:
        lower = 0.0
    elif successes == n:
        lower = math.exp(math.log(tail) / n)
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if binomial_tail(successes, n, mid, "upper") > tail:
                hi = mid
            else:
                lo = mid
        lower = (lo + hi) / 2

    if successes == n:
        upper = 1.0
    elif successes == 0:
        upper = -math.expm1(math.log(tail) / n)
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if binomial_tail(successes, n, mid, "lower") > tail:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return lower, upper


def _validated_precision(raw, label):
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping")
    cells = raw.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != set(CELLS):
        raise ValueError(f"{label}.cells must contain exactly A through G")
    parsed = {}
    for name in CELLS:
        values = cells[name]
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"{label}.{name} must contain both directional margins")
        parsed[name] = [_finite(v, f"{label}.{name}[{i}]")
                        for i, v in enumerate(values)]
    controls = raw.get("controls")
    if not isinstance(controls, Mapping) or controls.get("passed") is not True:
        raise InvalidControlsError(f"{label}: controls.passed must be true")
    tolerance = _finite(controls.get("identity_tolerance"),
                        f"{label}.identity_tolerance")
    if tolerance < 0:
        raise ValueError(f"{label}.identity_tolerance must be nonnegative")
    errors = {}
    for check, left, right in (("off_own_q0", "C", "A"),
                               ("on_own_q1", "F", "B"),
                               ("zero_mlp", "G", "A")):
        errors[check] = [_finite(abs(a - b), f"{label}.{check} error")
                         for a, b in zip(parsed[left], parsed[right])]
        if any(e > tolerance for e in errors[check]):
            raise InvalidControlsError(f"{label}: {check} identity exceeds tolerance")
    contrasts = []
    for i in range(2):
        t = _finite(parsed["B"][i] - parsed["A"][i], f"{label}.T")
        r = _finite(parsed["D"][i] - parsed["C"][i], f"{label}.R")
        s = _finite(parsed["E"][i] - parsed["C"][i], f"{label}.S")
        interaction = _finite(t - r - s, f"{label}.T-R-S")
        contrasts.append({"T": t, "R": r, "S": s,
                          "interaction_T_minus_R_minus_S": interaction})
    return {"cells": parsed, "contrasts": contrasts,
            "identity_tolerance": tolerance, "identity_errors": errors}


def _profile_fits(contrasts, kappa):
    t, r, s = (contrasts[name] for name in ("T", "R", "S"))
    tolerance = _finite(kappa * abs(t), "profile tolerance")
    errors = {
        "transfer": (abs(r), abs(_finite(s - t, "S-T"))),
        "joint_dependence": (abs(r), abs(s)),
        "preservation": (abs(_finite(r - t, "R-T")), abs(s)),
    }
    return {profile: {"errors_R_S": list(values), "tolerance": tolerance,
                      "fits": all(e <= tolerance for e in values)}
            for profile, values in errors.items()}


def analyze_records(records, kappa=.25, numerical_fraction=.025,
                    coverage=.8, alpha=.05):
    """Validate a complete run and classify three operational response profiles.

    Required schema per draw::

        {'pair_id': 'unique-draw-id', 'precisions': {
          'float32': {'cells': {'A': [swap0, swap1], ..., 'G': [...]},
                      'controls': {'passed': True, 'identity_tolerance': ...}},
          'float64': { ... }}}

    A/B are patch off/on without query insertion; C/D use off-cache queries;
    E is patch off with on-cache queries; F is patch on with its own on-cache
    queries; G is the zero-delta MLP identity. Extra record/control metadata is
    allowed. Technical failure raises before any result is returned. Numerically
    unstable pairs stay in n with zero successes for all profiles.
    """
    kappa = _finite(kappa, "kappa")
    numerical_fraction = _finite(numerical_fraction, "numerical_fraction")
    coverage = _finite(coverage, "coverage")
    alpha = _finite(alpha, "alpha")
    if not 0 <= kappa < .5 or numerical_fraction < 0:
        raise ValueError("require 0 <= kappa < .5 and numerical_fraction >= 0")
    if not 0 < coverage < 1 or not 0 < alpha < 1:
        raise ValueError("coverage and alpha must lie strictly between zero and one")
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be an iterable of base-pair mappings")
    try:
        records = list(records)
    except TypeError as exc:
        raise ValueError("records must be iterable") from exc
    if not records:
        raise ValueError("at least one base-pair record is required")

    seen = set()
    output = []
    failures = Counter()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("each record must be a mapping")
        pair_id = raw.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise ValueError("pair_id must be a unique nonempty draw ID")
        seen.add(pair_id)
        precisions = raw.get("precisions")
        if not isinstance(precisions, Mapping) or set(precisions) != set(PRECISIONS):
            raise ValueError(f"{pair_id}: require float32 and float64 evaluations")
        evaluated = {p: _validated_precision(precisions[p], f"{pair_id}.{p}")
                     for p in PRECISIONS}
        directions = []
        for i in range(2):
            c32, c64 = (evaluated[p]["contrasts"][i] for p in PRECISIONS)
            t = min(abs(c32["T"]), abs(c64["T"]))
            same_sign = ((c32["T"] > 0 and c64["T"] > 0)
                         or (c32["T"] < 0 and c64["T"] < 0))
            differences = {c: _finite(abs(c32[c] - c64[c]), f"{pair_id}.delta_{c}")
                           for c in ("T", "R", "S")}
            budget = _finite(numerical_fraction * t, "numerical budget")
            reason = ("zero_anchor" if t == 0 else
                      "anchor_sign_disagreement" if not same_sign else
                      "cross_precision_disagreement" if max(differences.values()) > budget
                      else None)
            if reason:
                failures[reason] += 1
            directions.append({
                "direction_index": i,
                "resolution": {"passed": reason is None, "failure_reason": reason,
                               "min_abs_T": t, "discrepancies": differences,
                               "maximum_allowed_discrepancy": budget},
                "profiles_by_precision": {
                    p: _profile_fits(evaluated[p]["contrasts"][i], kappa)
                    for p in PRECISIONS},
            })
        resolved = all(d["resolution"]["passed"] for d in directions)
        successes = {
            profile: resolved and all(d["profiles_by_precision"][p][profile]["fits"]
                                      for d in directions for p in PRECISIONS)
            for profile in PROFILES}
        if sum(successes.values()) > 1:
            raise ValueError("non-exclusive profile result; numerical resolution insufficient")
        output.append({"pair_id": pair_id, "precisions": evaluated,
                       "directions": directions, "resolved": resolved,
                       "profile_successes": successes})

    n = len(output)
    results = {}
    for profile in PROFILES:
        count = sum(item["profile_successes"][profile] for item in output)
        lo, hi = clopper_pearson(count, n, alpha, len(PROFILES))
        status = ("adequate" if lo > coverage else
                  "excluded" if hi < coverage else "unresolved")
        results[profile] = {"successes": count, "n": n, "fraction": count / n,
                            "interval": [lo, hi], "status": status}
    adequate = [p for p in PROFILES if results[p]["status"] == "adequate"]
    outcome = ("profile_adequate" if adequate else
               "none_meets_coverage" if all(v["status"] == "excluded"
                                            for v in results.values()) else "unresolved")
    return {
        "contract": {"kappa": kappa, "numerical_fraction": numerical_fraction,
                     "coverage": coverage, "family_alpha": alpha,
                     "number_of_profiles": len(PROFILES),
                     "per_tail_alpha": alpha / (2 * len(PROFILES))},
        "n_base_pairs": n,
        "n_resolved_pairs": sum(item["resolved"] for item in output),
        "n_unresolved_pairs": sum(not item["resolved"] for item in output),
        "direction_resolution_failure_counts": dict(failures),
        "profiles": results, "adequate_profiles": adequate, "outcome": outcome,
        "assumptions_and_scope": [
            "Binomial coverage requires iid base-pair draws from the frozen generator; "
            "this function does not establish that sampling assumption.",
            "Both swaps and both precision evaluations form one success; "
            "weak or unstable pairs remain in the denominator.",
            "Bonferroni simultaneous Clopper-Pearson intervals cover three profile "
            "probabilities; profile indicators need not be mutually independent.",
            "Cross-precision agreement is an instrument-stability criterion, "
            "not a bound relative to exact arithmetic.",
            "Profiles describe intervention responses, not unique graphs, "
            "mechanism shares, native computation, or semantic content.",
        ],
        "records": output,
    }
