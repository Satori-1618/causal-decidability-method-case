"""Round-3A 512-cap amendment; all frozen statistical rules remain unchanged.

This separately versioned copy appends n=512 after the original four candidates.
The original 384-cap planner and its STOP outcome remain part of the record.

No model calls. Development families determine covariance, never the population
or scientific boundaries. The bootstrap check uses a fixed index bank, as does
the frozen seeded analyzer. Simulated datasets are independent conditional on
that bank; Monte Carlo binomial bounds concern this deterministic seeded rule.
"""

from collections.abc import Mapping
from statistics import NormalDist

import numpy as np

from donor_factor_analysis import (CONTRASTS, COVERAGE, MEAN_ALPHA, MEAN_EPSILON,
                                   PRECISIONS, PROFILE_ALPHA, PROFILES,
                                   PRIMARY_SEED, _integer, analyze_records)
from query_route_analysis import binomial_tail, clopper_pearson


CANDIDATE_SIZES = (128, 192, 256, 384, 512)
PLANNING_SEED = 320260926
GAUSSIAN_SIMULATIONS = 10000
BOOTSTRAP_SIMULATIONS = 1000
BOOTSTRAP_DRAWS = 2000
DEVELOPMENT_UNITS = 32
SD_INFLATION = 1.5
ALTERNATIVE_MEAN = .20
TARGET_POWER = .80
PROFILE_ALTERNATIVE = .90


def exact_coverage_plan(n):
    """Smallest success count with CP lower > .8 and its power at p=.9."""
    _integer(n, "n", 1)
    tail = PROFILE_ALPHA / (2 * len(PROFILES))
    lo, hi = 0, n + 1
    while lo < hi:
        mid = (lo + hi) // 2
        sufficient = mid <= n and binomial_tail(mid, n, COVERAGE, "upper") < tail
        if sufficient:
            hi = mid
        else:
            lo = mid + 1
    threshold = lo if lo <= n else None
    power = binomial_tail(threshold, n, PROFILE_ALTERNATIVE, "upper") if threshold else 0.0
    return {"n": n, "required_successes": threshold,
            "power_at_90_percent_success": power, "passes": power >= TARGET_POWER}


def _power_report(counts, simulations, require_mc_bound):
    results = {}
    for direction, key in enumerate(("positive", "negative")):
        results[key] = {}
        for ci, contrast in enumerate(CONTRASTS):
            count = int(counts[direction, ci])
            bounds = clopper_pearson(count, simulations, alpha=.05, family_size=1)
            results[key][contrast] = {
                "detections": count, "simulations": simulations,
                "probability": count / simulations,
                "monte_carlo_95_percent_interval": list(bounds),
                "passes": bounds[0] > TARGET_POWER if require_mc_bound
                          else count / simulations >= TARGET_POWER}
    return {"directions": results,
            "passes": all(v["passes"] for side in results.values() for v in side.values()),
            "criterion": "lower endpoint of two-sided 95% MC CP interval > .80"
                         if require_mc_bound else "estimated detection probability >= .80",
            "precision_requirement": "both precisions must detect each contrast"}


def _counts_from_intervals(intervals):
    """Intervals shape (2, simulations, 2 precisions, 3 contrasts), centered at 0."""
    boundary_gap = ALTERNATIVE_MEAN - MEAN_EPSILON
    positive = np.all(intervals[0] > -boundary_gap, axis=1)
    negative = np.all(intervals[1] < boundary_gap, axis=1)
    return np.stack((positive.sum(axis=0), negative.sum(axis=0)))


def _normal_power(n, covariance_root, simulations, seed):
    rng = np.random.default_rng(seed)
    critical = NormalDist().inv_cdf(1 - MEAN_ALPHA / (2 * len(CONTRASTS)))
    counts = np.zeros((2, len(CONTRASTS)), dtype=np.int64)
    for start in range(0, simulations, 128):
        size = min(128, simulations - start)
        samples = rng.standard_normal((size, n, 6)) @ covariance_root.T
        means = samples.mean(axis=1).reshape(size, 2, 3)
        radius = (critical * samples.std(axis=1, ddof=1) / np.sqrt(n)).reshape(size, 2, 3)
        counts += _counts_from_intervals(np.stack((means - radius, means + radius)))
    return _power_report(counts, simulations, require_mc_bound=False)


def _bootstrap_weights(n, draws, seed=PRIMARY_SEED):
    """Exactly the same row-index schedule as the main bootstrap, compressed."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(draws, n))
    weights = np.zeros((draws, n), dtype=np.float64)
    np.add.at(weights, (np.arange(draws)[:, None], indices), 1 / n)
    return weights


def _bootstrap_power(n, covariance_root, empirical, simulations, draws, seed,
                     distribution):
    rng = np.random.default_rng(seed)
    weights = _bootstrap_weights(n, draws)
    counts = np.zeros((2, len(CONTRASTS)), dtype=np.int64)
    tail = MEAN_ALPHA / (2 * len(CONTRASTS))
    # One fixed bank is the actual deterministic seeded bootstrap rule, not a
    # shared random data-generating component. Independent datasets give iid trials.
    for start in range(0, simulations, 32):
        size = min(32, simulations - start)
        if distribution == "gaussian":
            samples = rng.standard_normal((size, n, 6)) @ covariance_root.T
        elif distribution == "empirical":
            indices = rng.integers(0, len(empirical), size=(size, n))
            samples = empirical[indices]
        else:
            raise ValueError("distribution must be gaussian or empirical")
        flat = samples.transpose(1, 0, 2).reshape(n, size * 6)
        boot_means = (weights @ flat).reshape(draws, size, 2, 3)
        intervals = np.quantile(boot_means, [tail, 1 - tail], axis=0)
        counts += _counts_from_intervals(intervals)
    report = _power_report(counts, simulations, require_mc_bound=True)
    report.update({"distribution": distribution, "bootstrap_draws": draws,
                   "fixed_bootstrap_seed": PRIMARY_SEED, "simulation_seed": seed})
    return report


def plan_confirmation(summary_or_records, gaussian_simulations=GAUSSIAN_SIMULATIONS,
                      bootstrap_simulations=BOOTSTRAP_SIMULATIONS,
                      bootstrap_draws=BOOTSTRAP_DRAWS, seed=PLANNING_SEED):
    """Choose the first eligible n, or STOP; never run/authorize confirmation.

    Reduced simulation counts and development sets other than exactly 32 units
    are software tests only and cannot produce a SELECTED result. Full planning
    still does not replace a public
    freeze or authorize model inference. Both signs and all three contrasts
    must pass each mean gate. Candidate sizes and scientific constants are fixed.
    """
    _integer(gaussian_simulations, "gaussian_simulations", 2)
    _integer(bootstrap_simulations, "bootstrap_simulations", 2)
    _integer(bootstrap_draws, "bootstrap_draws", 2)
    _integer(seed, "seed")
    # Revalidate raw surfaces even when a previously computed summary is supplied.
    records = (summary_or_records.get("records") if isinstance(summary_or_records, Mapping)
               else summary_or_records)
    summary = analyze_records(records)
    protocol_counts = (gaussian_simulations == GAUSSIAN_SIMULATIONS
                       and bootstrap_simulations == BOOTSTRAP_SIMULATIONS
                       and bootstrap_draws == BOOTSTRAP_DRAWS and seed == PLANNING_SEED)
    result = {"schema_version": "donor-factor-planning-512-v1", "status": "STOP",
              "selected_n": None, "n_development_units": summary["n_units"],
              "candidates": [], "protocol_matches_frozen_simulation_counts": protocol_counts,
              "protocol_matches_development_size": summary["n_units"] == DEVELOPMENT_UNITS,
              "confirmation_authorized": False,
              "settings": {"candidate_sizes": list(CANDIDATE_SIZES), "sd_inflation": SD_INFLATION,
                           "alternative_means_nat": [-ALTERNATIVE_MEAN, ALTERNATIVE_MEAN],
                           "practical_boundary_nat": MEAN_EPSILON, "target_power_each_contrast": TARGET_POWER,
                           "profile_alternative_probability": PROFILE_ALTERNATIVE,
                           "gaussian_simulations": gaussian_simulations,
                           "bootstrap_simulations_per_distribution": bootstrap_simulations,
                           "bootstrap_draws": bootstrap_draws, "seed": seed,
                           "mean_family_alpha": MEAN_ALPHA, "profile_family_alpha": PROFILE_ALPHA},
              "assumptions_and_scope": [
                  "Planning alternatives and inflated development covariance are assumptions, not power guarantees.",
                  "Detection target is separate for each I/P/J contrast and each sign, not joint power.",
                  "Both precision decisions must detect; paired covariance is preserved in each simulation.",
                  "The bootstrap check tests nominal percentile intervals with 2000 draws; actual inference uses 20000 and an independent-seed stability gate.",
                  "MC binomial intervals are conditional on the fixed seeded bootstrap index bank.",
                  "No Monte Carlo, numerical, technical, or population-validity gate is guaranteed on new data.",
                  "Sample-size selection does not authorize confirmation before the required public freeze."]}
    if summary["n_units"] < 2 or summary["n_unresolved_units"]:
        result["stop_reason"] = "development requires >=2 complete numerically resolved independent units"
        return result
    values = np.array([[[r["precisions"][p]["unit_means"][c] for c in CONTRASTS]
                        for p in PRECISIONS] for r in summary["records"]]).reshape(-1, 6)
    centered = values - values.mean(axis=0)
    covariance = np.cov(centered, rowvar=False, ddof=1) * SD_INFLATION**2
    if not np.isfinite(covariance).all():
        raise ValueError("inflated development covariance must be finite")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    # Correct empirical n/(n-1) to match the inflated ddof=1 covariance exactly.
    empirical = centered * SD_INFLATION * np.sqrt(len(values) / (len(values) - 1))
    result.update({"covariance_order": [f"{p}.{c}" for p in PRECISIONS for c in CONTRASTS],
                   "inflated_development_covariance": covariance.tolist(),
                   "inflated_sd_by_precision": {
                       p: {c: float(np.sqrt(covariance[pi * 3 + ci, pi * 3 + ci]))
                           for ci, c in enumerate(CONTRASTS)} for pi, p in enumerate(PRECISIONS)}})
    for index, n in enumerate(CANDIDATE_SIZES):
        candidate = {"n": n, "coverage": exact_coverage_plan(n),
                     "normal_reference": _normal_power(n, covariance_root, gaussian_simulations,
                                                        seed + 100 * index),
                     "bootstrap_checks": None, "passes": False}
        result["candidates"].append(candidate)
        if not candidate["coverage"]["passes"] or not candidate["normal_reference"]["passes"]:
            continue
        checks = {distribution: _bootstrap_power(
                      n, covariance_root, empirical, bootstrap_simulations, bootstrap_draws,
                      seed + 100 * index + offset, distribution)
                  for offset, distribution in ((1, "gaussian"), (2, "empirical"))}
        candidate["bootstrap_checks"] = checks
        candidate["passes"] = all(check["passes"] for check in checks.values())
        if candidate["passes"]:
            eligible = protocol_counts and summary["n_units"] == DEVELOPMENT_UNITS
            result.update({"selected_n": n, "status": "SELECTED" if eligible else "TEST_ONLY"})
            return result
    result["stop_reason"] = "no prespecified n <=512 meets coverage and every mean planning gate"
    return result
