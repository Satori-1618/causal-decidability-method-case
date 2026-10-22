"""Hypothetical precision/power and cost table. No measured outcomes are read."""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

from analysis import (N, MIN_SEPARATING, ALPHA, ADEQUACY, binomial_mass,
                      binomial_tail, require)

SCREEN_BUDGET = 2048


@lru_cache(maxsize=None)
def adequacy_cutoffs(n):
    """Equivalent exact-test cutoffs; no binomial-rate approximation."""
    require(type(n) is int and n > 0, "Positive denominator required")
    adequate = next((k for k in range(n+1) if binomial_tail(n, ADEQUACY, k) < ALPHA), None)
    excluded = max((k for k in range(n+1) if binomial_tail(n, ADEQUACY, k, False) < ALPHA), default=None)
    return {"families": n, "minimum_definite_hits_for_lower_above_90pct": adequate,
            "maximum_possible_hits_for_upper_below_90pct": excluded}


def adequacy_power(n, true_rate):
    require(0 <= true_rate <= 1, "Invalid hypothetical hit rate")
    cut = adequacy_cutoffs(n)
    k_a = cut["minimum_definite_hits_for_lower_above_90pct"]
    k_e = cut["maximum_possible_hits_for_upper_below_90pct"]
    return {"assumed_hit_rate": true_rate,
            "adequacy_probability": 0. if k_a is None else binomial_tail(n, true_rate, k_a),
            "exclusion_probability": 0. if k_e is None else binomial_tail(n, true_rate, k_e, False),
            "assumption": "IID separated-family hits; definite=possible in this idealized power calculation."}


@lru_cache(maxsize=None)
def minimum_wins(discordants):
    require(type(discordants) is int and discordants >= 0, "Invalid discordant count")
    return next((k for k in range(discordants+1)
                 if binomial_tail(discordants, .5, k) <= ALPHA), discordants+1)


def paired_power(gain, loss, n=N):
    require(0 <= gain <= 1 and 0 <= loss <= 1 and gain+loss <= 1,
            "Hypothetical gain/loss probabilities are invalid")
    discordance = gain+loss
    if not discordance:
        probability = 0.
    else:
        probability = math.fsum(binomial_mass(n, m, discordance)
                                * binomial_tail(m, gain/discordance, minimum_wins(m))
                                for m in range(n+1))
    return {"families": n, "assumed_robust_gain_probability": gain,
            "assumed_robust_loss_probability": loss,
            "assumed_robust_improvement": gain-loss,
            "one_sided_exact_paired_test_power": min(1., probability),
            "assumption": "IID robust discordant outcomes for all planned target families. No start/technical failures included."}


def planning_report():
    return {
        "status": "hypothetical_planning_only_no_measured_outcomes",
        "constants": {"native_candidate_budget": SCREEN_BUDGET, "accepted_families": N,
                      "minimum_new_forecast_separating_families": MIN_SEPARATING,
                      "adequacy_target": ADEQUACY, "each_candidate_tail_alpha": ALPHA,
                      "paired_test_alpha": ALPHA, "global_error_budget": 5*ALPHA},
        "new_forecast_start_probability": [
            {"assumed_new_forecast_separation_rate": p,
             "probability_at_least_128_of_256": binomial_tail(N, p, MIN_SEPARATING)}
            for p in (.35, .4, .5, .6, .7, .8, .9)],
        "native_quota_probability": [
            {"assumed_acceptance_rate": p,
             "probability_at_least_256_of_2048": binomial_tail(SCREEN_BUDGET, p, N)}
            for p in (.10, .125, .15, .20)],
        "adequacy": [dict(adequacy_cutoffs(n), scenarios=[adequacy_power(n, p)
                          for p in (.60, .75, .85, .90, .95, .97, .98)])
                     for n in (128, 192, 256)],
        "paired_averaging": [paired_power(g, l) for g, l in ((.15, .05), (.20, .05), (.30, .05))],
        "smaller_design_comparison": {"families": 128, "gain": .15, "loss": .05,
                                      "power": paired_power(.15, .05, n=128)["one_sided_exact_paired_test_power"]},
        "design_rationale": "256 target families give about 89% idealized power for a +10-point robust averaging gain (.15 gains/.05 losses). This is a planning alternative, not a tested +10-point threshold.",
        "cost": {"unit": "sequence forwards, both dtypes; no wall-time or cache-saving claim",
                 "screening": SCREEN_BUDGET*2,
                 "calibration": N*8*2*4,
                 "targets_if_start_passes": N*8*2*4,
                 "maximum": SCREEN_BUDGET*2+N*16*2*4,
                 "if_calibration_start_fails": SCREEN_BUDGET*2+N*8*2*4,
                 "per_transfer_per_dtype": "recipient native, donor native, self-control, patched = 4"},
        "limits": [
            "Hypothetical rates are design assumptions, not forecasts from earlier results.",
            "New averaged-forecast separation is different from the earlier diagonal-anchor screen endpoint.",
            "Under an ideal balance-only table, new forecast distance is half the balance gap; old screen yield cannot simply be reused.",
            "Binomial adequacy power conditions on a given number of separated families and assumes independent family sampling.",
            "With only 128 separated families, true 95% accuracy has low power to establish population adequacy above 90%.",
            "Paired power is for a full hypothetical set of 256 outcomes; it does not condition on passing a calibration gate correlated with those outcomes.",
            "Do not multiply start, quota and test powers without a joint data-generating model. No joint-power guarantee is made.",
            "Uncertain numerical-band cases can reduce robust comparison power.",
            "The future-confirmation planning gate uses 90% observed definite accuracy, not a population-adequacy conclusion."
        ]}


def compare_planning(actual, expected, path="root"):
    """Allow tiny libm drift in probabilities; preserve structure/counts exactly."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), path+": keys differ")
        for key in expected:
            compare_planning(actual[key], expected[key], path+"."+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path+": lengths differ")
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_planning(a, e, path+"["+str(i)+"]")
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0., abs_tol=1e-12), path+": number differs")
    else:
        require(type(actual) is type(expected) and actual == expected, path+": value differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = planning_report()
    path = Path(__file__).with_name("planning.json")
    if args.check:
        compare_planning(result, json.loads(path.read_text()))
        print("PASS: frozen hypothetical planning table")
    else:
        path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
        print(path)


if __name__ == "__main__":
    main()
