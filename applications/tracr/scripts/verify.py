#!/usr/bin/env python3
"""Records-only verification: no model imports, compilation or forwards."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracr_demo.design import CANDIDATES, Tolerance, cumulative_retention, select_discriminator
from tracr_demo.experiment import (NUMERIC_CAP, check_arrays, dump, load_family, max_abs,
                                  read, sha, source_hashes)
from tracr_demo.statistics import evaluate_confirmation


def verify(run, freeze=None):
    summary, manifest = read(run / "summary.json"), read(run / "manifest.json")
    if source_hashes() != manifest["source_hashes"]:
        raise ValueError("Current executable source differs from recorded run")
    if set(summary["hashes"]) != {"manifest.json", "records.jsonl", "tensors.npz", "trajectories.json"}:
        raise ValueError("Missing or extra artifact hash binding")
    for name, digest in summary["hashes"].items():
        if sha(run / name) != digest:
            raise ValueError(f"Changed artifact: {name}")
    if manifest["phase"] == "confirmation":
        if freeze is None or sha(freeze) != manifest["freeze_sha256"]:
            raise ValueError("Confirmation needs its matching frozen contract")
        frozen = read(freeze)
        for field in ("source_hashes", "upstream", "model", "families", "numeric_allowance", "scientific_tolerance"):
            if manifest[field] != frozen[field]:
                raise ValueError(f"Freeze mismatch: {field}")
        if len(manifest["families"]) != frozen["sample_size"]:
            raise ValueError("Incomplete confirmation sample")
    rows = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]
    stored_trajectories = read(run / "trajectories.json")
    tensors = np.load(run / "tensors.npz", allow_pickle=False)
    row_map, measured, seen_keys, record_ids = {}, {}, set(), set()
    families = [load_family(v) for v in manifest["families"]]
    if (len({f.family_id for f in families}) != len(families)
            or len({f.signature for f in families}) != len(families)):
        raise ValueError("Duplicate sampling unit")
    family_cases = {case.case_id: case for f in families
                    for case in (f.shared, *f.discriminators, *f.followups)}
    sites = ("site_A", "site_B") if manifest["phase"] == "development" else ("site_A",)
    expected_trajectories = {(f.family_id, site) for f in families for site in sites}
    trajectory_keys = [(t["family_id"], t["site"]) for t in stored_trajectories]
    if len(set(trajectory_keys)) != len(trajectory_keys) or set(trajectory_keys) != expected_trajectories:
        raise ValueError("Missing, duplicate or extra trajectory")
    for row in rows:
        key = (row["case"]["case_id"], row["site"], row["dtype"])
        if (key in row_map or row["record_id"] in record_ids or row["site"] not in sites
                or row["dtype"] not in ("float32", "float64")):
            raise ValueError("Duplicate record or unexpected precision")
        record_ids.add(row["record_id"])
        case = family_cases[key[0]]
        if json.loads(json.dumps(case.to_dict())) != row["case"]:
            raise ValueError("Record case differs from manifest")
        if row["coordinates"] != manifest["model"]["sites"][row["site"]]["indices"]:
            raise ValueError("Changed intervention coordinates")
        arrays = {name: tensors[f"{row['record_id']}__{name}"] for name in row["arrays"]}
        seen_keys.update(f"{row['record_id']}__{name}" for name in row["arrays"])
        recalculated = check_arrays(arrays, dtype=row["dtype"], target=case.target_query + 1,
                                    coordinates=row["coordinates"], recipient=case.recipient, donor=case.donor,
                                    source_position=case.source_query + 1,
                                    site_layer=manifest["model"]["sites"][row["site"]]["layer"],
                                    site_timing=manifest["model"]["sites"][row["site"]]["timing"])
        if recalculated != row["checks"]:
            raise ValueError("Stored qualification does not follow from tensors")
        actual = arrays["patch_scores"][0, case.target_query + 1].tolist()
        if actual != row["target_scores"]:
            raise ValueError("Stored readout differs from full tensor")
        if row["state_dtype"] != str(arrays["before"].dtype) or row["score_dtype"] != str(arrays["patch_scores"].dtype):
            raise ValueError("Incorrect dtype provenance")
        row_map[key] = (row, arrays)
    if seen_keys != set(tensors.files) or len(rows) != summary["record_count"]:
        raise ValueError("Unexpected/missing tensor or record")
    conditions = {(case_id, site) for case_id, site, dtype in row_map}
    if set(row_map) != {(case_id, site, dtype) for case_id, site in conditions
                        for dtype in ("float32", "float64")}:
        raise ValueError("Each condition requires exactly one fp32 and one fp64 record")
    precision_max, all_qualified = 0., True
    for case_id, site, dtype in row_map:
        if dtype != "float64":
            continue
        high, hi = row_map[(case_id, site, "float64")]
        low, lo = row_map[(case_id, site, "float32")]
        deviation = max(max_abs(lo[name] - hi[name])
                        for name in ("patch_scores", "native_scores", "donor_scores"))
        precision_max = max(precision_max, deviation)
        qualified = high["checks"]["passed"] and low["checks"]["passed"]
        qualified &= deviation <= manifest["numeric_allowance"]
        all_qualified &= qualified
        for row in (high, low):
            if (deviation != row["paired_precision_error"] or
                    row["precision_passed"] != (deviation <= manifest["numeric_allowance"])):
                raise ValueError("Incorrect paired precision calculation")
        measured[(case_id, site)] = (qualified, low, high)
    tolerance = Tolerance(manifest["scientific_tolerance"], manifest["numeric_allowance"])
    consumed, outcomes = set(), []
    for family in families:
        sites = ("site_A", "site_B") if manifest["phase"] == "development" else ("site_A",)
        for site in sites:
            stored = next(t for t in stored_trajectories if t["family_id"] == family.family_id and t["site"] == site)
            cases, observations = [family.shared], {"float32": [], "float64": []}
            expected = "address" if site == "site_A" else "donor_answer"
            technical_failure = False
            used_record_ids = []
            for stage in range(3):
                case = cases[-1]
                qualified, lo, hi = measured[(case.case_id, site)]
                used_record_ids.extend([lo["record_id"], hi["record_id"]])
                consumed.add((case.case_id, site))
                if not qualified:
                    technical_failure = True
                    break
                for dtype, row in (("float32", lo), ("float64", hi)):
                    observations[dtype].append(row["target_scores"])
                if stage == 0:
                    tr = cumulative_retention(cases, observations["float64"], tolerance, controls_passed=True)
                    selection = select_discriminator(family, tr[-1]["retained_after"], tolerance)
                    if json.loads(json.dumps(selection)) != stored.get("selection"):
                        raise ValueError("Choice did not follow the frozen prediction-only rule")
                    if selection["selected_case_id"] is None:
                        break
                    cases.append(next(c for c in family.discriminators if c.case_id == selection["selected_case_id"]))
                elif stage == 1:
                    cases.append(family.followup_for(case.case_id))
            passed = False
            expected_status = "qualification_failed" if technical_failure else "no_discriminator"
            if not technical_failure and len(observations["float64"]) == 3:
                trajectories = {dtype: cumulative_retention(cases, obs, tolerance, controls_passed=True)
                                for dtype, obs in observations.items()}
                for dtype, field in (("float64", "trajectory"), ("float32", "float32_trajectory")):
                    if json.loads(json.dumps(trajectories[dtype])) != stored[field]:
                        raise ValueError("Stored trajectory does not follow from measurements")
                pattern = [sorted(CANDIDATES), [expected], [expected]]
                passed = all([s["retained_after"] for s in tr] == pattern
                             and not any(s["native_null_compatible"] for s in tr)
                             for tr in trajectories.values())
                expected_status = trajectories["float64"][-1]["outcome"]
                if stored["expected_control_label_for_scoring_only"] != expected:
                    raise ValueError("Incorrect ground-truth scoring label")
            if passed != stored["success"]:
                raise ValueError("Incorrect compound family outcome")
            if stored["status"] != expected_status or stored["records"] != used_record_ids:
                raise ValueError("Incorrect trajectory status or record links")
            outcomes.append((site, passed))
    if consumed != set(measured):
        raise ValueError("Unaccounted measurements or adaptive extra look")
    successes = sum(passed for site, passed in outcomes if site == "site_A")
    if successes != summary["site_A_successes"] or all_qualified != summary["all_qualified"]:
        raise ValueError("Summary does not follow from whole families")
    if precision_max != summary["paired_precision_max_error"]:
        raise ValueError("Precision maximum mismatch")
    site_b = sum(passed for site, passed in outcomes if site == "site_B")
    if site_b != summary["site_B_positive_controls"] or len(families) != summary["family_count"]:
        raise ValueError("Incorrect positive-control or family count")
    native_inputs = {(tuple(row["case"][role]), row["dtype"])
                     for row in rows for role in ("recipient", "donor")}
    expected_forwards = {"native": len(native_inputs), "noop": len(rows),
                         "self_patch": len(rows), "patch": len(rows)}
    if summary["forward_counts"] != expected_forwards:
        raise ValueError("Incorrect forward budget")
    if manifest["phase"] == "development":
        proposed = max(64 * np.finfo("float32").eps, 4 * precision_max)
        eligible = bool(all_qualified and proposed <= NUMERIC_CAP
                        and successes == len(families) and site_b == len(families))
        if (summary["proposed_numeric_allowance"] != proposed
                or summary["confirmation_start_eligible"] != eligible):
            raise ValueError("Incorrect development start gate")
    if manifest["phase"] == "confirmation":
        result = evaluate_confirmation(successes, len(families), population_size=frozen["population_size"],
                                       planned_sample_size=frozen["sample_size"])
        if result != summary["population_decision"]:
            raise ValueError("Population decision mismatch")
        if summary["mechanistic_claim_qualified"] != all_qualified:
            raise ValueError("Technical failure was not propagated to claim")
    result = {"status": "PASS", "model_forwards": 0, "families": len(families),
              "records": len(rows), "site_A_successes": successes,
              "paired_precision_max_error": precision_max,
              "summary_sha256": sha(run / "summary.json"),
              "checks": ["artifact hashes", "freeze bindings", "actual inserted tensors",
                         "full-score precision", "controls", "prediction-only choice",
                         "shared production classifier", "whole-family adequacy"]}
    dump(run / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.freeze), indent=2))
