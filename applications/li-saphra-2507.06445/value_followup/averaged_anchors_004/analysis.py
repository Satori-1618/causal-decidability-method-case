"""Frozen, model-free arithmetic for averaged-anchor development 004.

This is not an execution runner or a records/provenance verifier. It consumes
calibration and target margins only after the separately specified controls.
No torch, checkpoint, outcome-file or network access occurs on import.
"""
from functools import lru_cache
import math

N = 256
MIN_SEPARATING = 128
ADEQUACY = .90
ALPHA = .01
PRECISION = .001
SEPARATION = .202
DEFINITE = .099
POSSIBLE = .101
CELLS = tuple(f"{s}_{p}_{r}" for s in ("neg", "pos") for p in (20, 28) for r in (0, 1))
CALIBRATION_CELLS = tuple("calibration_" + c for c in CELLS)
TARGET_CELLS = tuple("target_" + c for c in CELLS)
DTYPES = ("float32", "float64")
CANDIDATES = ("B_avg", "P_avg", "B_single4", "B_legacy2")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "Expected a finite number")
    return float(value)


def count(value, n, label):
    require(type(value) is int and 0 <= value <= n, label + " must be a count")
    return value


def grid(values, role):
    expected = CALIBRATION_CELLS if role == "calibration" else TARGET_CELLS
    require(isinstance(values, dict) and set(values) == set(expected), "Expected the complete eight-cell " + role + " grid")
    return {cell: number(values[cell]) for cell in expected}


def forecasts(calibration):
    """Four fixed rules, all predicting the SAME eight held-out target cells.

    B_single4 changes only replication: use calibration_0 at each of four cells
    and the same across-position averaging as B_avg. B_legacy2 changes coverage
    as well and is retained only as a descriptive historical comparator.
    """
    c = {key.removeprefix("calibration_"): value for key, value in grid(calibration, "calibration").items()}
    means = {f"{s}_{p}": (c[f"{s}_{p}_0"] + c[f"{s}_{p}_1"])/2
             for s in ("neg", "pos") for p in (20, 28)}
    predictions = {name: {} for name in CANDIDATES}
    for cell in CELLS:
        s, p, _ = cell.split("_")
        predictions["B_avg"]["target_"+cell] = (means[s+"_20"] + means[s+"_28"])/2
        predictions["P_avg"]["target_"+cell] = (means["neg_"+p] + means["pos_"+p])/2
        predictions["B_single4"]["target_"+cell] = (c[s+"_20_0"] + c[s+"_28_0"])/2
        predictions["B_legacy2"]["target_"+cell] = c["neg_20_0" if s == "neg" else "pos_28_0"]
    return predictions


def calibration_result(calibration):
    require(isinstance(calibration, dict) and set(calibration) == set(DTYPES), "Both calibration dtypes are required")
    pred = {dtype: forecasts(calibration[dtype]) for dtype in DTYPES}
    signed = {dtype: {cell: pred[dtype]["B_avg"][cell] - pred[dtype]["P_avg"][cell]
                       for cell in TARGET_CELLS} for dtype in DTYPES}
    discrepancy = max(abs(signed["float32"][cell] - signed["float64"][cell]) for cell in TARGET_CELLS)
    gaps = {dtype: max(abs(x) for x in signed[dtype].values()) for dtype in DTYPES}
    require(discrepancy <= PRECISION, "Forecast-contrast precision gate failed")
    require((gaps["float32"] > SEPARATION) == (gaps["float64"] > SEPARATION),
            "Forecast-separation dtype classification disagrees")
    return {"forecasts": pred, "forecast_gap": gaps["float64"],
            "forecast_contrast_dtype_error": discrepancy,
            "separating": gaps["float64"] > SEPARATION}


def start_rule(calibration_results):
    require(len(calibration_results) == N, "Exactly 256 complete calibration families are required")
    require(all(type(r.get("separating")) is bool for r in calibration_results), "Invalid calibration result")
    n = sum(r["separating"] for r in calibration_results)
    return {"families": N, "separating": n, "minimum_separating": MIN_SEPARATING,
            "start_targets": n >= MIN_SEPARATING,
            "status": "start_targets" if n >= MIN_SEPARATING else "insufficient_forecast_separation"}


def family_result(calibration, targets):
    result = calibration_result(calibration)
    require(isinstance(targets, dict) and set(targets) == set(DTYPES), "Both target dtypes are required")
    t = {dtype: grid(targets[dtype], "target") for dtype in DTYPES}
    errors, cell_errors, precision = {}, {}, {}
    for candidate in CANDIDATES:
        signed = {dtype: {cell: t[dtype][cell] - result["forecasts"][dtype][candidate][cell]
                          for cell in TARGET_CELLS} for dtype in DTYPES}
        cell_errors[candidate] = {cell: abs(signed["float64"][cell]) for cell in TARGET_CELLS}
        errors[candidate] = max(cell_errors[candidate].values())
        precision[candidate] = max(abs(signed["float32"][cell]-signed["float64"][cell]) for cell in TARGET_CELLS)
    require(max(precision.values()) <= PRECISION, "Prediction-error precision gate failed")
    within = {dtype: {f"{s}_{p}": t[dtype][f"target_{s}_{p}_1"] - t[dtype][f"target_{s}_{p}_0"]
                       for s in ("neg", "pos") for p in (20, 28)} for dtype in DTYPES}
    within_precision = max(abs(within["float32"][cell] - within["float64"][cell])
                           for cell in within["float64"])
    require(within_precision <= PRECISION, "Within-cell contrast precision gate failed")
    witness = {dtype: {cell: abs(delta) > SEPARATION for cell, delta in within[dtype].items()}
               for dtype in DTYPES}
    definite_witness = {cell: witness["float32"][cell] and witness["float64"][cell]
                        for cell in within["float64"]}
    possible_witness = {cell: witness["float32"][cell] or witness["float64"][cell]
                        for cell in within["float64"]}
    ambiguous_witness = [cell for cell in within["float64"]
                         if witness["float32"][cell] != witness["float64"][cell]]
    result.update({"max_error": errors, "cell_errors": cell_errors,
                   "prediction_error_dtype_discrepancy": precision,
                   "definite_hits": {name: errors[name] <= DEFINITE for name in CANDIDATES},
                   "possible_hits": {name: errors[name] <= POSSIBLE for name in CANDIDATES},
                   "within_cell_signed_differences": within["float64"],
                   "within_cell_dtype_discrepancy": within_precision,
                   "within_cell_witnesses": definite_witness,
                   "within_cell_possible_witnesses": possible_witness,
                   "within_cell_numerically_unresolved": ambiguous_witness,
                   "extra_prefix_dependence_witness": any(definite_witness.values()),
                   "extra_prefix_dependence_possible_witness": any(possible_witness.values())})
    return result


def binomial_mass(n, k, p):
    if p == 0:
        return float(k == 0)
    if p == 1:
        return float(k == n)
    return math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                    +k*math.log(p)+(n-k)*math.log1p(-p))


def binomial_tail(n, p, k, upper=True):
    return min(1., math.fsum(binomial_mass(n, j, p) for j in
                            (range(k, n+1) if upper else range(k+1))))


@lru_cache(maxsize=None)
def exact_bounds(k, n, alpha=ALPHA):
    """Each one-sided Clopper-Pearson bound spends alpha, not alpha/2."""
    require(type(n) is int and n > 0, "Positive integer denominator required")
    count(k, n, "successes")
    require(0 < number(alpha) < .5, "Invalid bound alpha")
    lower, upper = 0., 1.
    if k:
        lo, hi = 0., 1.
        for _ in range(64):
            mid = (lo+hi)/2
            if binomial_tail(n, mid, k) < alpha:
                lo = mid
            else:
                hi = mid
        lower = (lo+hi)/2
    if k < n:
        lo, hi = 0., 1.
        for _ in range(64):
            mid = (lo+hi)/2
            if binomial_tail(n, mid, k, upper=False) > alpha:
                lo = mid
            else:
                hi = mid
        upper = (lo+hi)/2
    return lower, upper


def candidate_result(definite, possible, n):
    count(definite, n, "definite hits")
    count(possible, n, "possible hits")
    require(definite <= possible, "Definite hits cannot exceed possible hits")
    lo = exact_bounds(definite, n)[0]
    hi = exact_bounds(possible, n)[1]
    return {"families": n, "definite_hits": definite, "possible_hits": possible,
            "definite_rate": definite/n, "possible_rate": possible/n,
            "adequacy_threshold": ADEQUACY, "interval": [lo, hi],
            "status": "adequate" if lo > ADEQUACY else "excluded" if hi < ADEQUACY else "unresolved"}


def paired_result(gains, losses, n=N):
    count(gains, n, "gains")
    count(losses, n, "losses")
    require(gains+losses <= n, "Discordant counts exceed total families")
    p = binomial_tail(gains+losses, .5, gains) if gains+losses else 1.
    return {"families": n, "robust_gains": gains, "robust_losses": losses,
            "robust_point_difference": (gains-losses)/n,
            "one_sided_exact_mcnemar_p": p, "alpha": ALPHA,
            "improvement_supported": p <= ALPHA,
            "interpretation": "Average definite-hit probability exceeds four-cell single-prefix possible-hit probability.",
            "non_rejection": "Unresolved improvement; not equivalence or proof that averaging is useless."}


def cohort_result(rows):
    require(len(rows) == N, "Exactly 256 target families are required; no post-target exclusion")
    gate = start_rule(rows)
    require(gate["start_targets"], "Target data violate the frozen start rule")
    eligible = [row for row in rows if row["separating"]]
    candidates = {name: candidate_result(sum(r["definite_hits"][name] for r in eligible),
                                        sum(r["possible_hits"][name] for r in eligible), len(eligible))
                  for name in ("B_avg", "P_avg")}
    gains = sum(r["definite_hits"]["B_avg"] and not r["possible_hits"]["B_single4"] for r in rows)
    losses = sum(not r["definite_hits"]["B_avg"] and r["possible_hits"]["B_single4"] for r in rows)
    next_candidates = [name for name, result in candidates.items()
                       if result["definite_rate"] >= ADEQUACY
                       and candidates["P_avg" if name == "B_avg" else "B_avg"]["status"] == "excluded"]
    return {"start": gate, "candidates_on_separating_families": candidates,
            "paired_averaging_on_all_families": paired_result(gains, losses),
            "within_cell_witness_families_descriptive": sum(r["extra_prefix_dependence_witness"] for r in rows),
            "within_cell_possible_witness_families_descriptive": sum(r["extra_prefix_dependence_possible_witness"] for r in rows),
            "within_cell_numerically_unresolved_families_descriptive": sum(bool(r["within_cell_numerically_unresolved"]) for r in rows),
            "future_confirmation_planning_candidates": next_candidates,
            "future_confirmation_permission": False,
            "gate_scope": "A development feasibility condition only. Point accuracy >=90% is not a population adequacy finding.",
            "familywise_error_budget": {"candidate_bounds": 4*ALPHA, "paired_test": ALPHA, "total": 5*ALPHA}}
