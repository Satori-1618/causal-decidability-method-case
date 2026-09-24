"""Frozen Round-3A factorial readouts and whole-family inference.

The independent unit contains both recipient panels and both precisions. Margins
always favor that panel's recipient-correct name. Percentile bootstrap intervals
are nominal, not finite-sample coverage guarantees. No model inference occurs here.
"""

import copy
import math
from collections.abc import Mapping

import numpy as np

from query_route_analysis import InvalidControlsError, clopper_pearson


PRECISIONS = ("float32", "float64")
CELLS = ("00", "01", "10", "11")
CONTRASTS = ("I", "P", "J")
PROFILES = ("position_only", "identity_only")
MEAN_EPSILON = .10
INVARIANCE_EPSILON = .25
COVERAGE = .80
PROFILE_ALPHA = .025
MEAN_ALPHA = .025
NUMERICAL_BUDGET = .01
MC_ENDPOINT_BUDGET = .01
PRIMARY_SEED = 320260924
SECONDARY_SEED = 320260925


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def interval_decision(interval):
    """Strict mean boundaries; an endpoint exactly at +/-epsilon is unresolved."""
    lo, hi = interval
    if lo > MEAN_EPSILON:
        return "positive_relevant"
    if hi < -MEAN_EPSILON:
        return "negative_relevant"
    if lo > -MEAN_EPSILON and hi < MEAN_EPSILON:
        return "equivalent"
    return "unresolved"


def _controls(raw, label):
    if not isinstance(raw, Mapping) or raw.get("passed") is not True:
        raise InvalidControlsError(f"{label}.passed must be true")


def _validated_precision(raw, label):
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping")
    _controls(raw.get("controls"), f"{label}.controls")
    tolerance = _finite(raw["controls"].get("identity_tolerance"),
                        f"{label}.controls.identity_tolerance")
    if tolerance < 0:
        raise ValueError("identity_tolerance must be nonnegative")
    baselines = raw.get("baseline_margins")
    if not isinstance(baselines, (list, tuple)) or len(baselines) != 4:
        raise ValueError(f"{label}.baseline_margins must have four entries")
    baselines = [_finite(v, f"{label}.baseline_margins[{i}]")
                 for i, v in enumerate(baselines)]
    panels = raw.get("panels")
    if not isinstance(panels, (list, tuple)) or len(panels) != 2:
        raise ValueError(f"{label}.panels must contain two recipient panels")
    parsed = {}
    for panel in panels:
        if not isinstance(panel, Mapping):
            raise ValueError(f"{label}.panel must be a mapping")
        index = _integer(panel.get("recipient_index"), "recipient_index")
        if index not in (0, 1) or index in parsed:
            raise ValueError("require exactly recipient_index 0 and 1")
        prefix = f"{label}.panel[{index}]"
        _controls(panel.get("controls"), f"{prefix}.controls")
        baseline = _finite(panel.get("baseline_margin"), f"{prefix}.baseline")
        if baseline != baselines[index]:
            raise InvalidControlsError(f"{prefix}: panel baseline does not match recorded baseline")
        if "source_indices" in panel and panel["source_indices"] != (
                [0, 1, 2, 3] if index == 0 else [1, 0, 3, 2]):
            raise ValueError(f"{prefix}: source_indices does not match the frozen donor map")
        expected_donors = dict(zip(CELLS, [0, 1, 2, 3] if index == 0 else [1, 0, 3, 2]))
        if "donor_indices" in panel and panel["donor_indices"] != expected_donors:
            raise ValueError(f"{prefix}: donor_indices does not match the frozen donor map")
        mappings = {}
        for field in ("patched_margins", "alphas"):
            values = panel.get(field)
            if not isinstance(values, Mapping) or set(values) != set(CELLS):
                raise ValueError(f"{prefix}.{field} requires exactly {CELLS}")
            mappings[field] = {c: _finite(values[c], f"{prefix}.{field}.{c}")
                               for c in CELLS}
        d = {c: _finite(mappings["patched_margins"][c] - baseline,
                        f"{prefix}.delta.{c}") for c in CELLS}
        identity_errors = {"self_delta": abs(d["00"]),
                           "self_alpha": abs(mappings["alphas"]["00"])}
        if max(identity_errors.values()) > tolerance:
            raise InvalidControlsError(f"{prefix}: donor-self identity exceeds tolerance")
        simple = {"identity_at_p0": d["10"] - d["00"],
                  "identity_at_p1": d["11"] - d["01"],
                  "position_at_i0": d["01"] - d["00"],
                  "position_at_i1": d["11"] - d["10"]}
        simple = {k: _finite(v, f"{prefix}.{k}") for k, v in simple.items()}
        contrasts = {"I": simple["identity_at_p0"] / 2 + simple["identity_at_p1"] / 2,
                     "P": simple["position_at_i0"] / 2 + simple["position_at_i1"] / 2,
                     "J": simple["identity_at_p1"] - simple["identity_at_p0"]}
        contrasts = {k: _finite(v, f"{prefix}.{k}") for k, v in contrasts.items()}
        residuals = {"position_only": [simple["identity_at_p0"], simple["identity_at_p1"]],
                     "identity_only": [simple["position_at_i0"], simple["position_at_i1"]]}
        result = copy.deepcopy(dict(panel))
        result.update({"baseline_margin": baseline, **mappings, "deltas": d,
                       "simple_effects": simple, "contrasts": contrasts,
                       "identity_errors": identity_errors,
                       "profiles": {p: {"residuals": values,
                                          "fits": all(abs(v) <= INVARIANCE_EPSILON
                                                      for v in values)}
                                    for p, values in residuals.items()}})
        parsed[index] = result
    result = copy.deepcopy(dict(raw))
    result.update({"baseline_margins": baselines,
                   "panels": [parsed[i] for i in (0, 1)],
                   "unit_means": {c: _finite(parsed[0]["contrasts"][c] / 2
                                             + parsed[1]["contrasts"][c] / 2,
                                             f"{label}.unit_mean.{c}")
                                  for c in CONTRASTS}})
    return result


def bootstrap_intervals(values, draws, seed):
    """Percentile intervals over rows, sharing draws across all trailing axes.

    Kept bounded in memory; a complete unit is always sampled as a whole.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires finite unit-by-estimand values")
    draws = _integer(draws, "bootstrap_draws", 2)
    _integer(seed, "seed")
    rng = np.random.default_rng(seed)
    samples = np.empty((draws,) + values.shape[1:], dtype=np.float64)
    batch_size = max(1, min(256, 1_000_000 // values.size))
    for start in range(0, draws, batch_size):
        end = min(draws, start + batch_size)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        samples[start:end] = values[indices].mean(axis=1)
    if not np.isfinite(samples).all():
        raise ValueError("bootstrap means must be finite")
    tail = MEAN_ALPHA / (2 * len(CONTRASTS))
    return np.quantile(samples, [tail, 1 - tail], axis=0)


def analyze_records(records, bootstrap_draws=20000, seed=PRIMARY_SEED,
                    secondary_seed=SECONDARY_SEED):
    """Validate, preserve and analyze complete families; never remove failures.

    Required input is the runner's case_id/precisions schema, with four finite
    baseline_margins, two panels, and all four patched_margins and alphas per
    panel. Technical failures raise before any scientific summary is returned.
    Both profile and mean inference use the original number of independent units.
    A mean claim additionally needs >=2 units, both precision decisions, stable
    secondary-seed endpoints/decisions and resolved relevant numeric quantities.
    """
    _integer(bootstrap_draws, "bootstrap_draws", 2)
    _integer(seed, "seed")
    _integer(secondary_seed, "secondary_seed")
    if seed == secondary_seed:
        raise ValueError("secondary_seed must differ from seed")
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be an iterable of case mappings")
    try:
        records = list(records)
    except TypeError as exc:
        raise ValueError("records must be iterable") from exc
    if not records:
        raise ValueError("at least one independent unit is required")
    evaluated, seen = [], set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("each record must be a mapping")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("case_id must be a unique nonempty draw ID")
        seen.add(case_id)
        precisions = raw.get("precisions")
        if not isinstance(precisions, Mapping) or set(precisions) != set(PRECISIONS):
            raise ValueError("require exactly float32 and float64 evaluations")
        parsed = {p: _validated_precision(precisions[p], f"{case_id}.{p}")
                  for p in PRECISIONS}
        differences = {}
        for index in (0, 1):
            for field in ("deltas", "simple_effects", "contrasts"):
                a, b = (parsed[p]["panels"][index][field] for p in PRECISIONS)
                for name in a:
                    differences[f"panel{index}.{field}.{name}"] = _finite(
                        abs(a[name] - b[name]), f"{case_id}.discrepancy.{field}.{name}")
        for c in CONTRASTS:
            differences[f"unit_mean.{c}"] = _finite(
                abs(parsed["float32"]["unit_means"][c]
                    - parsed["float64"]["unit_means"][c]), f"{case_id}.discrepancy.{c}")
        resolved = max(differences.values()) <= NUMERICAL_BUDGET
        result = copy.deepcopy(dict(raw))
        result.update({"precisions": parsed, "resolved": resolved,
                       "resolution": {"passed": resolved, "discrepancies": differences,
                                      "maximum_allowed_discrepancy": NUMERICAL_BUDGET},
                       "profile_successes": {
                           profile: resolved and all(panel["profiles"][profile]["fits"]
                               for p in PRECISIONS for panel in parsed[p]["panels"])
                           for profile in PROFILES}})
        evaluated.append(result)
    n = len(evaluated)
    values = np.array([[[r["precisions"][p]["unit_means"][c] for c in CONTRASTS]
                        for p in PRECISIONS] for r in evaluated], dtype=np.float64)
    primary = bootstrap_intervals(values, bootstrap_draws, seed)
    secondary = bootstrap_intervals(values, bootstrap_draws, secondary_seed)
    means = values.mean(axis=0)
    if not np.isfinite(means).all():
        raise ValueError("population sample means must be finite")
    mean_results = {}
    for ci, c in enumerate(CONTRASTS):
        by_precision = {}
        for pi, p in enumerate(PRECISIONS):
            interval = primary[:, pi, ci].tolist()
            repeat_interval = secondary[:, pi, ci].tolist()
            shift = float(np.max(np.abs(primary[:, pi, ci] - secondary[:, pi, ci])))
            decision, repeated = interval_decision(interval), interval_decision(repeat_interval)
            by_precision[p] = {"mean": float(means[pi, ci]), "interval": interval,
                               "decision": decision, "secondary_interval": repeat_interval,
                               "secondary_decision": repeated, "mc_endpoint_shift": shift,
                               "mc_stable": shift <= MC_ENDPOINT_BUDGET and decision == repeated}
        discrepant = [r["case_id"] for r in evaluated if not r["resolved"]]
        interval_discrepancy = float(max(np.max(np.abs(primary[:, 0, ci] - primary[:, 1, ci])),
                                         np.max(np.abs(secondary[:, 0, ci] - secondary[:, 1, ci]))))
        reasons = []
        if n < 2:
            reasons.append("fewer_than_two_independent_units")
        if discrepant or interval_discrepancy > NUMERICAL_BUDGET:
            reasons.append("cross_precision_disagreement")
        if not all(v["mc_stable"] for v in by_precision.values()):
            reasons.append("bootstrap_monte_carlo_instability")
        decisions = {v["decision"] for v in by_precision.values()}
        if len(decisions) != 1:
            reasons.append("precision_decisions_disagree")
        mean_results[c] = {"status": next(iter(decisions)) if not reasons else "unresolved",
                           "precisions": by_precision, "blocked_reasons": reasons,
                           "discrepant_case_ids": discrepant,
                           "maximum_interval_precision_discrepancy": interval_discrepancy}
    profiles = {}
    for profile in PROFILES:
        k = sum(r["profile_successes"][profile] for r in evaluated)
        lo, hi = clopper_pearson(k, n, PROFILE_ALPHA, len(PROFILES))
        profiles[profile] = {"successes": k, "n": n, "fraction": k / n,
                             "interval": [lo, hi],
                             "status": "adequate" if lo > COVERAGE else
                                       "excluded" if hi < COVERAGE else "unresolved"}
    return {"schema_version": "donor-factor-analysis-v1", "n_units": n,
            "n_resolved_units": sum(r["resolved"] for r in evaluated),
            "n_unresolved_units": sum(not r["resolved"] for r in evaluated),
            "profiles": profiles, "mean_effects": mean_results, "records": evaluated,
            "contract": {"mean_epsilon_nat": MEAN_EPSILON,
                         "invariance_epsilon_nat": INVARIANCE_EPSILON,
                         "coverage": COVERAGE, "profile_family_alpha": PROFILE_ALPHA,
                         "mean_family_alpha": MEAN_ALPHA, "numerical_budget_nat": NUMERICAL_BUDGET,
                         "bootstrap_method": "whole-unit percentile Bonferroni",
                         "bootstrap_draws_per_seed": bootstrap_draws,
                         "primary_seed": seed, "secondary_seed": secondary_seed,
                         "mc_endpoint_budget_nat": MC_ENDPOINT_BUDGET,
                         "mc_requires_identical_decisions": True},
            "assumptions_and_scope": [
                "Independent identically sampled complete families are the statistical units.",
                "Two recipient panels and two precisions are not independent observations.",
                "Profile CP intervals are simultaneous exact-binomial constructions.",
                "Mean percentile bootstrap intervals have nominal, not guaranteed, coverage.",
                "Adequate coarse invariance can coexist with practically relevant mean dependence.",
                "Both invariant profiles may fit when all effects are small.",
                "Signed panel effects can cancel in a unit mean; mean equivalence does not imply negligible effects in every panel.",
                "Controlled donor dependence does not identify a unique semantic mechanism."]}
