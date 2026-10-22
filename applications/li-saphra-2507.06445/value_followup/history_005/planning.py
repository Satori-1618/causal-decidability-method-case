"""Hypothetical exact planning only; no model or measured outcomes are read."""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

from analysis import (N, MIN_SEPARATING, PAIR_ALPHA, DIRECTIONAL_ALPHA,
                      ROBUST_ADVANTAGE, binomial_mass, binomial_upper, require)

SCREEN_BUDGET = 2048
CALCULATED_FLOAT_FIELDS = frozenset({
    "probability_at_least_64_of_256", "probability_at_least_256_of_2048",
    "assumed_neutral_probability", "assumed_win_minus_loss_probability",
    "favorable_direction_power", "opposite_direction_probability",
    "either_direction_probability"})


def assert_table_matches(actual, expected, path="root"):
    """Permit only tiny arithmetic variation, never changed scientific inputs.

    Known calculated float fields get absolute tolerance1e-11, relative0.
    All declared constants/assumptions, field sets, sequence lengths, integers,
    strings, booleans and unknown fields remain exact, including their types.
    This is an artifact portability check, not a tolerance in a decision rule.
    """
    require(type(actual) is type(expected), "Planning type differs at " + path)
    if isinstance(actual, dict):
        require(set(actual) == set(expected), "Planning fields differ at " + path)
        for key in actual:
            assert_table_matches(actual[key], expected[key], path + "." + key)
    elif isinstance(actual, list):
        require(len(actual) == len(expected), "Planning length differs at " + path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_table_matches(left, right, path + "[" + str(index) + "]")
    elif isinstance(actual, float) and path.rsplit(".", 1)[-1] in CALCULATED_FLOAT_FIELDS:
        require(math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, abs_tol=1e-11, rel_tol=0.),
                "Calculated planning value differs at " + path)
    else:
        require(actual == expected, "Planning value differs at " + path)


@lru_cache(maxsize=None)
def minimum_wins(discordants):
    require(type(discordants) is int and discordants >= 0, "Invalid non-neutral count")
    return next((k for k in range(discordants+1)
                 if binomial_upper(discordants, .5, k) <= DIRECTIONAL_ALPHA), discordants+1)


def paired_power(win, loss, n=N):
    require(type(n) is int and n > 0, "Positive family count required")
    require(0 <= win <= 1 and 0 <= loss <= 1 and win+loss <= 1, "Invalid hypothetical outcome probabilities")
    total = win+loss
    if not total:
        favorable, opposite = 0., 0.
    else:
        favorable = math.fsum(binomial_mass(n, m, total)
                              * binomial_upper(m, win/total, minimum_wins(m)) for m in range(n+1))
        opposite = math.fsum(binomial_mass(n, m, total)
                             * binomial_upper(m, loss/total, minimum_wins(m)) for m in range(n+1))
    return {"families": n, "assumed_robust_win_probability": win,
            "assumed_robust_loss_probability": loss, "assumed_neutral_probability": 1-total,
            "assumed_win_minus_loss_probability": win-loss,
            "favorable_direction_power": min(1., favorable),
            "opposite_direction_probability": min(1., opposite),
            "either_direction_probability": min(1., favorable+opposite),
            "assumptions": "IID complete family outcomes at the guarded .022-nat threshold; no screening/start/technical failure included."}


def planning_report():
    return {
        "status": "hypothetical_planning_only_no_measured_outcomes",
        "constants": {"native_budget": SCREEN_BUDGET, "families": N,
                      "minimum_potentially_separating": MIN_SEPARATING,
                      "robust_advantage_nat": ROBUST_ADVANTAGE,
                      "pair_alpha": PAIR_ALPHA, "directional_alpha": DIRECTIONAL_ALPHA,
                      "pairs": 3, "familywise_alpha_upper_bound": 3*PAIR_ALPHA},
        "new_start_probability": [
            {"assumed_any_pair_potential_coverage": p,
             "probability_at_least_64_of_256": binomial_upper(N, p, MIN_SEPARATING)}
            for p in (.10, .20, .25, .30, .40, .50)],
        "native_quota_probability": [
            {"assumed_native_acceptance_rate": p,
             "probability_at_least_256_of_2048": binomial_upper(SCREEN_BUDGET, p, N)}
            for p in (.10, .125, .15, .20)],
        "paired_scenarios": [paired_power(w, l) for w,l in ((.10,.05),(.15,.05),(.20,.05),(.30,.05))],
        "smaller_sample_reference": paired_power(.15, .05, n=128),
        "design_rationale": "N256 gives about88% favorable-direction power for robust wins/losses .15/.05; N128 gives about49.5%. This is a hypothetical planning alternative, not a forecast or a +10-point tested threshold.",
        "start_rationale": "64/256 is an operational25% minimum potential coverage. It is not a power threshold and does not ensure any particular pair has64 separated families.",
        "deterministic_resolution_fact": "For the same target vector, |E_i-E_j| <= ||forecast_i-forecast_j||_infinity. A forecast gap<=.022 therefore cannot yield a guarded max-error advantage>.022, irrespective of the target values.",
        "cost": {"unit": "sequence-forwards across both dtypes, not wall time",
                 "native": SCREEN_BUDGET*2, "calibration": N*8*2*4,
                 "targets_if_start_passes": N*8*2*4,
                 "maximum": SCREEN_BUDGET*2+N*16*2*4,
                 "calibration_stop": SCREEN_BUDGET*2+N*8*2*4,
                 "transfer_forward_recipe": "4 per dtype: recipient-native, donor-native, identity, patch"},
        "limits": [
            "The new centered-contrast question does not rescue or reinterpret round004's failed absolute-adequacy accounts.",
            "The .020 meaningful advantage and .002 numerical guard are distinct choices; .022 is not an absolute prediction-tolerance claim.",
            "The exact tests compare meaningful-win versus meaningful-loss probabilities, not mean loss or absolute adequacy.",
            "Neutral families remain in the reported denominator; only non-neutral families are binomial trials.",
            "Power concerns one hypothetical pair and all256 complete outcomes; it is not a probability of identifying a unique winner among all3 candidates.",
            "No independence among pairwise tests is assumed for the Bonferroni error bound.",
            "Start depends on the same calibration used to predict outcomes. Do not multiply stage probabilities or interpret the displayed test power as conditional on a passed start gate.",
            "The64-family coverage rule may be distributed over different pairs and does not establish their predictive accuracy.",
            "Pure interaction can make all three main forecasts coincide; the descriptive full-cell predictor cannot override a failed start.",
            "Centering removes shared-stem offsets by definition and cannot establish what explains that discarded variation.",
            "Recency changes here are specific matched token-edit bundles, not isolated manipulation of an abstract scalar recency variable."
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = planning_report()
    path = Path(__file__).with_name("planning.json")
    if args.check:
        assert_table_matches(json.loads(path.read_text()), output)
        print("PASS: hypothetical planning table")
    else:
        path.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
        print(path)


if __name__ == "__main__":
    main()
