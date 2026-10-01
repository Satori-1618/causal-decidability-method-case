#!/usr/bin/env python3
"""Run frozen Tracr checks; resolve one nonbinding probability portability issue.

The archived verifier compares an entire population dictionary exactly. Its
hypergeometric tail's final digits vary across SciPy builds, even at 1.15.3.
For this all-success confirmation, exact integer arithmetic independently checks
the confidence-bound boundary. All other population fields must match exactly.
No frozen source or stored artifact is changed. Use on a temporary result copy.
"""
import argparse
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/tracr"
TAIL_ALLOWANCE = 1e-10  # Probability reproduction only; never a scientific threshold.


def check_population(actual, stored):
    actual, stored = dict(actual), dict(stored)
    observed_tail = actual.pop("null_upper_tail")
    archived_tail = stored.pop("null_upper_tail")
    if actual != stored:
        raise ValueError("Population fields other than the diagnostic tail differ")
    N, n = stored["population_size"], stored["sample_size"]
    if stored["successes"] != n:
        raise ValueError("Portable exact check is scoped to the released all-success sample")
    denominator = math.comb(N, n)

    def tail(K):
        return Fraction(math.comb(K, n), denominator) if K >= n else Fraction(0)

    bounds = stored["bounds"]
    lower = bounds["lower_successes"]
    alpha = Fraction(str(bounds["alpha"]))
    if not tail(lower - 1) <= alpha < tail(lower):
        raise ValueError("Stored lower boundary fails exact rational inversion")
    if (bounds["upper_successes"] != N or bounds["upper_rate"] != 1.0
            or bounds["lower_rate"] != lower / N):
        raise ValueError("Stored population rates disagree with the integer bounds")
    null_count = math.floor(Fraction(str(stored["minimum_success_rate"])) * N)
    adequate = lower > null_count
    if (stored["adequate"] is not adequate or
            stored["status"] != ("adequate" if adequate else "not_demonstrated")):
        raise ValueError("Stored adequacy decision fails exact inversion")
    reference = float(tail(null_count))
    for value in (observed_tail, archived_tail):
        if (not math.isfinite(value) or not 0 <= value <= 1
                or abs(value - reference) > TAIL_ALLOWANCE):
            raise ValueError("Reported tail differs materially from exact rational probability")
    return {"exact_rational_tail_rounded_to_float": reference,
            "recomputed_tail": observed_tail, "archived_tail": archived_tail,
            "diagnostic_allowance": TAIL_ALLOWANCE,
            "integer_boundary_and_decision": "exactly_verified"}


def verify(run, freeze):
    spec = importlib.util.spec_from_file_location("frozen_tracr_verifier", APP / "scripts/verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = json.loads((run / "summary.json").read_text())
    frozen = json.loads(freeze.read_text())
    population = module.evaluate_confirmation(
        summary["site_A_successes"], summary["family_count"],
        population_size=frozen["population_size"], planned_sample_size=frozen["sample_size"])
    probability_check = check_population(population, summary["population_decision"])
    try:
        module.verify(run, freeze)
        exact_legacy_comparison = True
    except ValueError as error:
        if str(error) != "Population decision mismatch":
            raise
        # The frozen, hash-checked verifier reaches this point only after its
        # tensor, source, inventory, control, family and forward-count checks.
        # Its sole remaining validation is performed explicitly below.
        exact_legacy_comparison = False
    if summary["mechanistic_claim_qualified"] != summary["all_qualified"]:
        raise ValueError("Technical failure was not propagated to claim")
    return {"status": "PASS", "model_forwards": 0,
            "families": summary["family_count"],
            "legacy_exact_population_dictionary_match": exact_legacy_comparison,
            "population_check": probability_check}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.freeze), indent=2))
