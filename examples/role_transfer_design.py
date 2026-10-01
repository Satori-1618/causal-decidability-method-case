"""A prospective three-role example, including strong gain-flexible alternatives.

Run with python3 -I -S examples/role_transfer_design.py [--json]. The default
command is entirely synthetic and standard-library-only. --exercise-analysis
also runs the actual statistical analyzer on generated families; requires NumPy.
Neither command runs an LLM or estimates the chance of success on GPT-2.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "applications/makelov-2311.17030/src")]
from causal_decidability.design import signatures
from role_design import (DIAGNOSTICS, RECIPIENTS, ROLES, WORDINGS, directions,
                         expected_cell_ids, expected_structure, explanation_example, grid)
from role_geometry import closest_line_fit, rms_distance

NOTICE = "CONSTRUCTED DESIGN CHECK — NOT LLM RESULTS"
GAINS = {q: .5 for q in ROLES}


def synthetic_endpoints(scale=4.):
    """Stipulated informative native logits, not calibrated neural measurements."""
    return {rid: {q: [scale if i == context["binding"][j] else 0. for i in range(3)]
                  for j, q in enumerate(ROLES)} for rid, context in RECIPIENTS.items()}


def demonstration():
    cells = grid()
    rows = directions(synthetic_endpoints())
    by_id = {r["cell_id"]: r for r in rows}
    predictions = {r["cell_id"]: tuple(.5*x for x in r["role_direction"]) for r in rows}
    gaps = {}
    for rival in ("name", "position"):
        gaps[rival] = min(closest_line_fit(predictions[c], by_id[c][rival+"_direction"])["distance"]
                          for c in DIAGNOSTICS[rival])
    switch_predictions = [predictions[c] for c in DIAGNOSTICS["switch"]]
    center = tuple(sum(v)/2 for v in zip(*switch_predictions))
    gaps["switch"] = min(rms_distance(v, center) for v in switch_predictions)
    gaps["no_op"] = min(rms_distance(predictions[c], (0., 0., 0.))
                        for c in DIAGNOSTICS["no_op"])
    targets = {h: [c["targets"][h] for c in cells] for h in ("role", "name", "position", "no_op")}
    # Both alternate donor queries change the question, but request different answers.
    switch_cells = [c for c in cells if c["cell_id"] in DIAGNOSTICS["switch"]]
    return {"notice": NOTICE, "example": explanation_example(),
            "cells_per_wording": {"raw": len(cells), "primary": len(rows),
                                  "same_query_diagnostics": len(cells)-len(rows)},
            "target_signature_groups": signatures(targets),
            "changed_query_witness": {"changed": [True, True],
                                      "role_targets": [c["targets"]["role"] for c in switch_cells]},
            "stipulated_native_logit_gap": 4., "stipulated_role_gain": .5,
            "oracle_diagnostic_gaps_nat": gaps,
            "all_gaps_exceed_proposed_0_52": all(g > .52 for g in gaps.values()),
            "positive_path_possible_in_this_constructed_world": (
                all(g > .52 for g in gaps.values())
                and len(signatures(targets)) == 4
                and len({c["targets"]["role"] for c in switch_cells}) == 2),
            "scope": "Only this explicit response menu is checked. A lexical algorithm implementing "
                     "the same role mapping remains equivalent. Actual GPT-2 competence, geometry, "
                     "gains, numeric stability, power and confirmation remain unmeasured."}


def constructed_records(world, n=128, seed=39001):
    """Independent synthetic family draws for software testing, never empirical data."""
    if world not in ("role", "position_variable_gain", "no_effect"):
        raise ValueError("unknown constructed world")
    rng = random.Random(seed)
    records = []
    for i in range(n):
        scale = rng.uniform(3.8, 4.2)
        wordings = {}
        for wording in WORDINGS:
            rows = directions(synthetic_endpoints(scale))
            for index, row in enumerate(rows):
                if world == "role":
                    row["observed"] = [.5*x for x in row["role_direction"]]
                elif world == "position_variable_gain":
                    # Context- and cell-dependent amplitude and sign: the oracle
                    # must not lose merely because a global gain would be too weak.
                    gain = (-1. if index % 3 == 0 else 1.) * rng.uniform(.2, 1.6)
                    row["observed"] = [gain*x for x in row["position_direction"]]
                else:
                    row["observed"] = [0., 0., 0.]
            wordings[wording] = {"cells": rows, "controls": {"passed": True}}
        records.append({"case_id": f"synthetic-{seed}-{i}", "precisions": {
            p: {"wordings": copy.deepcopy(wordings), "controls": {"passed": True}}
            for p in ("float32", "float64")}})
    return records


def exercise_analysis():
    from role_response import analyze_records
    output = {}
    for world in ("role", "position_variable_gain", "no_effect"):
        result = analyze_records(
            constructed_records(world), GAINS, expected_cell_ids=expected_cell_ids(),
            expected_diagnostics=DIAGNOSTICS, expected_structure=expected_structure(),
            bootstrap_draws=2000, evidence_kind="synthetic")
        # Keep the demo output compact; raw constructed inputs are regenerated
        # exactly by constructed_records(world, n=128, seed=39001).
        output[world] = {k: v for k, v in result.items() if k != "records"}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--exercise-analysis", action="store_true")
    args = parser.parse_args()
    result = demonstration()
    if args.exercise_analysis:
        result["constructed_analysis"] = exercise_analysis()
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print(NOTICE)
    print("Recipient:", result["example"]["recipient"])
    for row in result["example"]["rows"]:
        print("Donor:", row["donor"])
        print("  Answer / slot:", row["donor_answer"], row["donor_mention_position"])
        print("  Recipient targets:", row["predicted_targets"])
    print("108 raw cells / 72 cross-query cells per wording; every cell stays in its family.")
    print("Synthetic distances from the strongest declared rival sets:")
    for name, gap in result["oracle_diagnostic_gaps_nat"].items():
        print(f"  {name}: {gap:.3f} nat")
    print("Positive prediction is possible in the constructed example; no model success is claimed.")
    if args.exercise_analysis:
        for world, value in result["constructed_analysis"].items():
            print(world, value.get("status", value.get("outcome", "inspect --json")))
    print(result["scope"])


if __name__ == "__main__":
    main()
