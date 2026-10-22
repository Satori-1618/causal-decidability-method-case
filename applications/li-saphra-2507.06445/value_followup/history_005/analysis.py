"""Frozen model-free arithmetic for the matched-history development round.

The estimand is a centered four-cell contrast within each shared-stem quartet.
Comparisons concern frequencies of meaningful predictive wins/losses, not
absolute adequacy, mean-loss superiority or unique causal identification.
"""
import math

N = 256
MIN_SEPARATING = 64
PRECISION = .001
MEANINGFUL_ADVANTAGE = .020
COMPARISON_GUARD = .002
ROBUST_ADVANTAGE = .022
PAIR_ALPHA = .05 / 3
DIRECTIONAL_ALPHA = .05 / 6
DTYPES = ("float32", "float64")
RECENCIES = ("t6", "t10")
ENDINGS = ("alt", "close")
CELLS = tuple(f"{t}_{e}_{r}" for t in RECENCIES for e in ENDINGS for r in (0, 1))
CALIBRATION_CELLS = tuple("calibration_" + c for c in CELLS)
TARGET_CELLS = tuple("target_" + c for c in CELLS)
CANDIDATES = ("H_recency", "H_suffix", "H_constant")
DESCRIPTIVE_CANDIDATE = "H_cell"
PAIRS = (("H_recency", "H_suffix"), ("H_recency", "H_constant"),
         ("H_suffix", "H_constant"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "Expected a finite number")
    return float(value)


def pair_id(first, second):
    return first + "__vs__" + second


def grid(values, role):
    require(role in ("calibration", "target"), "Unknown grid role")
    keys = CALIBRATION_CELLS if role == "calibration" else TARGET_CELLS
    require(isinstance(values, dict) and set(values) == set(keys), "Incomplete " + role + " grid")
    return {key: number(values[key]) for key in keys}


def centered_grid(values, role):
    """Fixed linear transformation, separately within the two matched stems."""
    raw = grid(values, role)
    means = {str(r): math.fsum(raw[f"{role}_{t}_{e}_{r}"] for t in RECENCIES for e in ENDINGS)/4
             for r in (0, 1)}
    centered = {key: value-means[key.rsplit("_", 1)[1]] for key, value in raw.items()}
    require(all(math.isfinite(x) for x in centered.values()), "Non-finite centered grid")
    return {"raw": raw, "stem_means": means, "centered": centered}


def forecasts(calibration):
    """Use calibration alone to predict each of eight centered target cells."""
    centered = centered_grid(calibration, "calibration")["centered"]
    means = {(t, e): (centered[f"calibration_{t}_{e}_0"] + centered[f"calibration_{t}_{e}_1"])/2
             for t in RECENCIES for e in ENDINGS}
    prediction = {name: {} for name in CANDIDATES + (DESCRIPTIVE_CANDIDATE,)}
    for t in RECENCIES:
        for e in ENDINGS:
            for r in (0, 1):
                key = f"target_{t}_{e}_{r}"
                prediction["H_recency"][key] = math.fsum(means[t, ending] for ending in ENDINGS)/2
                prediction["H_suffix"][key] = math.fsum(means[recency, e] for recency in RECENCIES)/2
                prediction["H_constant"][key] = 0.
                prediction["H_cell"][key] = means[t, e]
    return prediction


def calibration_result(calibration):
    require(isinstance(calibration, dict) and set(calibration) == set(DTYPES), "Both calibration dtypes required")
    pred = {dtype: forecasts(calibration[dtype]) for dtype in DTYPES}
    gaps = {dtype: {} for dtype in DTYPES}
    separation, possible, discrepancies = {}, {}, {}
    for first, second in PAIRS:
        key = pair_id(first, second)
        differences = {dtype: {cell: pred[dtype][first][cell]-pred[dtype][second][cell]
                              for cell in TARGET_CELLS} for dtype in DTYPES}
        for dtype in DTYPES:
            gaps[dtype][key] = max(abs(x) for x in differences[dtype].values())
        discrepancies[key] = max(abs(differences["float64"][cell]-differences["float32"][cell]) for cell in TARGET_CELLS)
        separation[key] = all(gaps[dtype][key] > ROBUST_ADVANTAGE for dtype in DTYPES)
        possible[key] = any(gaps[dtype][key] > ROBUST_ADVANTAGE for dtype in DTYPES)
    precision = max(discrepancies.values())
    require(precision <= PRECISION, "Calibration signed-forecast-contrast precision failed")
    ambiguous = [key for key in separation if separation[key] != possible[key]]
    return {"forecasts": pred, "forecast_gap": max(gaps["float64"].values()),
            "forecast_contrast_dtype_error": precision,
            "pair_forecast_gaps_by_dtype": gaps,
            "pair_forecast_contrast_dtype_discrepancies": discrepancies,
            "pair_separation": separation, "pair_possible_separation": possible,
            "pair_numerically_unresolved_separation": ambiguous,
            "separating": any(separation.values())}


def start_rule(results):
    require(isinstance(results, list) and len(results) == N, "Exactly 256 calibration families required")
    require(all(type(row.get("separating")) is bool for row in results), "Invalid calibration flags")
    n = sum(row["separating"] for row in results)
    keys = [pair_id(*pair) for pair in PAIRS]
    return {"families": N, "separating": n, "minimum_separating": MIN_SEPARATING,
            "start_targets": n >= MIN_SEPARATING,
            "status": "start_targets" if n >= MIN_SEPARATING else "insufficient_design_yield",
            "pair_separating_families": {key: sum(row["pair_separation"][key] for row in results) for key in keys},
            "pair_numerically_unresolved_families": {key: sum(key in row["pair_numerically_unresolved_separation"] for row in results) for key in keys},
            "interpretation": "Operational potential-coverage minimum, not accuracy or a power guarantee."}


def factorial(values, role):
    """Per-stem contrasts retain interaction; means across stems are descriptive."""
    raw = grid(values, role)
    result = {}
    for r in (0, 1):
        y = {(t, e): raw[f"{role}_{t}_{e}_{r}"] for t in RECENCIES for e in ENDINGS}
        row = {t: math.fsum(y[t, e] for e in ENDINGS)/2 for t in RECENCIES}
        col = {e: math.fsum(y[t, e] for t in RECENCIES)/2 for e in ENDINGS}
        result[str(r)] = {"recency_delta_t10_minus_t6": row["t10"]-row["t6"],
                          "suffix_delta_close_minus_alt": col["close"]-col["alt"],
                          "interaction": (y["t10", "close"]-y["t10", "alt"])
                                         -(y["t6", "close"]-y["t6", "alt"])}
    return result


def family_result(calibration, targets):
    result = calibration_result(calibration)
    require(isinstance(targets, dict) and set(targets) == set(DTYPES), "Both target dtypes required")
    target_data = {dtype: centered_grid(targets[dtype], "target") for dtype in DTYPES}
    cal_data = {dtype: centered_grid(calibration[dtype], "calibration") for dtype in DTYPES}
    names = CANDIDATES + (DESCRIPTIVE_CANDIDATE,)
    signed = {dtype: {name: {cell: target_data[dtype]["centered"][cell]-result["forecasts"][dtype][name][cell]
                            for cell in TARGET_CELLS} for name in names} for dtype in DTYPES}
    error = {dtype: {name: max(abs(value) for value in signed[dtype][name].values()) for name in names}
             for dtype in DTYPES}
    precision = {name: max(abs(signed["float64"][name][cell]-signed["float32"][name][cell]) for cell in TARGET_CELLS)
                 for name in names}
    require(max(precision[name] for name in CANDIDATES) <= PRECISION,
            "Main centered prediction-error precision failed")
    comparisons = {}
    for first, second in PAIRS:
        key = pair_id(first, second)
        delta = {dtype: error[dtype][second]-error[dtype][first] for dtype in DTYPES}
        reference = delta["float64"]
        outcome = "win" if reference > ROBUST_ADVANTAGE else "loss" if reference < -ROBUST_ADVANTAGE else "neutral"
        # The .002 guard comes from two .001 signed-error allowances. There
        # is no requirement to agree about the arbitrary guarded boundary.
        if outcome == "win":
            require(delta["float32"] > MEANINGFUL_ADVANTAGE, "Guarded win lacks cross-dtype meaningful advantage")
        if outcome == "loss":
            require(delta["float32"] < -MEANINGFUL_ADVANTAGE, "Guarded loss lacks cross-dtype meaningful advantage")
        comparisons[key] = {"first": first, "second": second, "error_advantage_by_dtype": delta,
                            "dtype_discrepancy": abs(delta["float64"]-delta["float32"]),
                            "outcome": outcome}
    # Raw level predictions use the mean CALIBRATION baseline only. Targets
    # supply no intercept, and these errors never enter the centered tests.
    raw_errors = {}
    for name in names:
        baseline = math.fsum(cal_data["float64"]["stem_means"].values())/2
        raw_errors[name] = max(abs(target_data["float64"]["raw"][cell]
                              -(result["forecasts"]["float64"][name][cell]+baseline)) for cell in TARGET_CELLS)
    result.update({"centered_targets": {dtype: target_data[dtype]["centered"] for dtype in DTYPES},
                   "target_stem_means": {dtype: target_data[dtype]["stem_means"] for dtype in DTYPES},
                   "calibration_stem_means": {dtype: cal_data[dtype]["stem_means"] for dtype in DTYPES},
                   "signed_centered_errors": signed, "centered_max_error_by_dtype": error,
                   "max_error": error["float64"], "prediction_error_dtype_discrepancy": precision,
                   "paired_comparisons": comparisons, "raw_max_errors_descriptive": raw_errors,
                   "H_cell_numerical_status": "supported" if precision["H_cell"] <= PRECISION else "unresolved",
                   "factorial_contrasts_descriptive": {
                       role: {dtype: factorial(values[dtype], role) for dtype in DTYPES}
                       for role, values in (("calibration", calibration), ("target", targets))}})
    return result


def binomial_mass(n, k, p):
    if p == 0:
        return float(k == 0)
    if p == 1:
        return float(k == n)
    return math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                    + k*math.log(p)+(n-k)*math.log1p(-p))


def binomial_upper(n, p, k):
    return min(1., math.fsum(binomial_mass(n, j, p) for j in range(k, n+1)))


def pair_result(wins, losses, first, second, n=N):
    require(type(n) is int and n > 0, "Positive family count required")
    require(type(wins) is int and type(losses) is int and min(wins, losses) >= 0
            and wins+losses <= n, "Invalid robust outcome counts")
    trials = wins+losses
    probability = min(1., 2*binomial_upper(trials, .5, max(wins, losses))) if trials else 1.
    direction = (first if wins > losses else second) if probability <= PAIR_ALPHA else None
    return {"first": first, "second": second, "families": n,
            "robust_wins": wins, "robust_losses": losses, "neutral": n-trials,
            "non_neutral_trials": trials, "win_minus_loss_fraction": (wins-losses)/n,
            "two_sided_exact_sign_p": probability, "pair_alpha": PAIR_ALPHA,
            "direction_supported": direction,
            "status": "direction_supported" if direction is not None else "unresolved",
            "interpretation": "More frequent meaningful predictive wins than losses; not mean-loss superiority, absolute adequacy or equivalence."}


def distribution(values):
    v = sorted(number(x) for x in values)
    require(bool(v), "Empty descriptive distribution")
    def q(p):
        index = (len(v)-1)*p
        low = int(index)
        return v[low] + (v[min(low+1, len(v)-1)]-v[low])*(index-low)
    return {"n": len(v), "mean": math.fsum(v)/len(v), "median": q(.5),
            "q1": q(.25), "q3": q(.75), "min": v[0], "max": v[-1]}


def cohort_result(rows):
    gate = start_rule(rows)
    require(gate["start_targets"], "Targets violate the frozen start rule")
    results = {}
    for first, second in PAIRS:
        key = pair_id(first, second)
        outcomes = [row["paired_comparisons"][key]["outcome"] for row in rows]
        require(set(outcomes) <= {"win", "loss", "neutral"}, "Invalid paired outcome")
        results[key] = pair_result(outcomes.count("win"), outcomes.count("loss"), first, second)
    return {"start": gate, "pair_results": results, "familywise_alpha_upper_bound": 3*PAIR_ALPHA,
            "centered_max_error_distributions_descriptive": {
                name: distribution([row["max_error"][name] for row in rows])
                for name in CANDIDATES + (DESCRIPTIVE_CANDIDATE,)},
            "raw_max_error_distributions_descriptive": {
                name: distribution([row["raw_max_errors_descriptive"][name] for row in rows])
                for name in CANDIDATES + (DESCRIPTIVE_CANDIDATE,)},
            "H_cell_numerically_unresolved_families": sum(row["H_cell_numerical_status"] != "supported" for row in rows),
            "absolute_adequacy_claim": False, "unique_mechanism_identification": False,
            "automatic_continuation_authorized": False,
            "scope": "Within-stem tail-bundle comparisons on the fixed selected head. Shared-stem offsets and absolute-margin adequacy are outside the primary estimand."}
