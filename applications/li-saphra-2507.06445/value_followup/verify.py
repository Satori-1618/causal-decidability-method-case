"""Independent, model-free audit of the bounded value-transfer development.

Uses only Python's standard library. It imports no design, runtime, producer or
analyzer. Saved whole-layer control assertions are not full tensor snapshots;
their truth cannot be independently reconstructed by this records-only check.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ANCHORS = ("neg_20_0", "pos_28_0")
CELLS = tuple(f"{s}_{p}_{r}" for s in ("neg", "pos") for p in (20, 28) for r in (0, 1))
TARGETS = tuple(c for c in CELLS if c not in ANCHORS)
CANDIDATES = ("H_state", "H_position")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def number(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "Expected a finite number")
    return float(value)


def close(left, right, message, *, atol=1e-11, rtol=1e-11):
    require(math.isclose(number(left), number(right), abs_tol=atol, rel_tol=rtol), message)


def float32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def phase(text):
    return "development" if int(text_hash("value-prefix-v1:" + text), 16) % 5 == 0 else "confirmation"


def walk(text):
    depth, lowest = 0, 0
    for char in text:
        require(char in "()", "Invalid bracket character")
        depth += 1 if char == "(" else -1
        lowest = min(lowest, depth)
    return depth, lowest


def shape(text):
    require(isinstance(text, str) and len(text) == 32 and text.count("(") == 16
            and text.count(")") == 16, "Expected length 32 and 16 brackets of each kind")


def validate_input(family, index, exclusions):
    require(family["phase"] == "development", "Unexpected input phase")
    require(family["family_id"] == text_hash(f"development:25100201:{index}")[:20],
            "Input family identifier does not match fixed preparation")
    recipient = family["recipient"]
    shape(recipient)
    require(phase(recipient) == "development" and walk(recipient)[1] < 0,
            "Recipient outside declared population/hash partition")
    require(recipient not in exclusions["recipient_strings"], "Recipient is not fresh")
    require(len(family["donors"]) == 8, "Expected eight donor cells")
    donors, seen = {}, {}
    for donor in family["donors"]:
        name = donor["cell"]
        require(name in CELLS and name not in donors, "Duplicate or unknown donor cell")
        sign, position, replica = name.split("_")
        position, replica = int(position), int(replica)
        balance = -2 if sign == "neg" else 2
        require((donor["position"], donor["replica"], donor["balance"]) ==
                (position, replica, balance), "Donor metadata disagrees with its cell")
        text = donor["string"]
        shape(text)
        prefix = text[:position]
        require(text[position - 1] == ")" and walk(prefix) == (balance, -4),
                "Donor prefix does not meet balance/minimum/token constraints")
        require(phase(prefix) == "development" and text_hash(prefix) == donor["prefix_sha256"],
                "Donor prefix hash or phase mismatch")
        require(prefix not in exclusions["prefixes"][str(position)], "Donor prefix is not fresh")
        key = (balance, position)
        require(prefix not in seen.setdefault(key, set()), "Repeated prefix within a donor cell")
        seen[key].add(prefix)
        donors[name] = donor
    require(set(donors) == set(CELLS), "Incomplete donor grid")
    return donors


def forecast(anchor_values):
    a, d = anchor_values[ANCHORS[0]], anchor_values[ANCHORS[1]]
    return {
        "H_state": {cell: a if cell.split("_")[0] == "neg" else d for cell in TARGETS},
        "H_position": {cell: a if int(cell.split("_")[1]) == 20 else d for cell in TARGETS},
    }


def check_snapshot(snapshot, dtype, recipient, recipient_position, donor):
    require(snapshot["dtype"] == "torch." + dtype, "Snapshot dtype mismatch")
    require(snapshot["recipient_string"] == recipient and snapshot["donor_string"] == donor["string"],
            "Snapshot strings differ from prepared sources")
    require(snapshot["recipient_position"] == recipient_position
            and snapshot["donor_position"] == donor["position"], "Snapshot positions differ from sources")
    require(recipient[recipient_position - 1] == ")", "Recipient target is not the matching closing token")
    arrays = {}
    for field in ("v_r", "v_d", "h_r", "h_patch_intended", "h_patch_delivered",
                  "requested_node_delta", "delivered_node_delta"):
        require(isinstance(snapshot[field], list) and len(snapshot[field]) == 16,
                "Expected sixteen-dimensional head snapshots: " + field)
        arrays[field] = [number(x) for x in snapshot[field]]
    cast = float32 if dtype == "float32" else float
    a = number(snapshot["a_r"])
    require(0 <= a <= 1, "Invalid recipient attention coefficient")
    for field in ("native_margin", "patched_margin", "donor_native_margin", "margin_change"):
        number(snapshot[field])
    if dtype == "float32":
        require(all(cast(x) == x for values in arrays.values() for x in values),
                "Saved float32 tensor contains non-float32 values")
        require(cast(a) == a, "Float32 attention coefficient is not representable")
    differences = [cast(d - r) for d, r in zip(arrays["v_d"], arrays["v_r"])]
    requested = [cast(a * v) for v in differences]
    intended = [cast(h + delta) for h, delta in zip(arrays["h_r"], requested)]
    delivered = [cast(h - before) for h, before in zip(intended, arrays["h_r"])]
    require(arrays["requested_node_delta"] == requested, "Requested value-contribution formula mismatch")
    require(arrays["h_patch_intended"] == intended, "Intended post-dtype node formula mismatch")
    require(arrays["h_patch_delivered"] == intended, "Delivered node differs from intended node")
    require(arrays["delivered_node_delta"] == delivered, "Delivered node delta mismatch")
    reference = [h + a * (d - r) for h, d, r in zip(arrays["h_r"], arrays["v_d"], arrays["v_r"])]
    roundoff = max(abs(x - y) for x, y in zip(intended, reference))
    close(snapshot["construction_roundoff_linf"], roundoff, "Incorrect recorded construction roundoff")
    norm_tol = 2e-6 if dtype == "float32" else 1e-11
    close(snapshot["value_difference_l2"], math.sqrt(math.fsum(v*v for v in differences)),
          "Value-difference norm mismatch", atol=norm_tol, rtol=norm_tol)
    close(snapshot["node_delta_l2"], math.sqrt(math.fsum(v*v for v in delivered)),
          "Node-delta norm mismatch", atol=norm_tol, rtol=norm_tol)
    require(snapshot["margin_change"] == cast(snapshot["patched_margin"] - snapshot["native_margin"]),
            "Saved margin change is inconsistent with execution dtype")
    return roundoff


def family_result(row):
    values = {c: number(row["cells"][c]["float64"]["patched_margin"]) for c in CELLS}
    expected = forecast({a: values[a] for a in ANCHORS})
    require(row["predictions"] == expected, "Stored predictions differ from fixed anchor rules")
    precision, discrepancies, errors = 0.0, {}, {}
    for candidate in CANDIDATES:
        error_values = []
        for cell in TARGETS:
            anchor = ANCHORS[0] if ((candidate == "H_state" and cell.startswith("neg_"))
                                    or (candidate == "H_position" and cell.split("_")[1] == "20")) else ANCHORS[1]
            signed64 = values[cell] - values[anchor]
            signed32 = row["cells"][cell]["float32"]["patched_margin"] - row["cells"][anchor]["float32"]["patched_margin"]
            discrepancy = abs(signed32 - signed64)
            discrepancies[candidate + ":" + cell] = discrepancy
            precision = max(precision, discrepancy)
            error_values.append(abs(signed64))
        errors[candidate] = max(error_values)
    require(precision <= .001, "Prediction-error precision allowance exceeded")
    require(row["prediction_error_dtype_discrepancies"] == discrepancies,
            "Stored prediction-error dtype discrepancies differ")
    pairs = {f"{s}_{p}": abs(values[f"{s}_{p}_1"] - values[f"{s}_{p}_0"])
             for s in ("neg", "pos") for p in (20, 28)}
    means = {f"{s}_{p}": (values[f"{s}_{p}_0"] + values[f"{s}_{p}_1"]) / 2
             for s in ("neg", "pos") for p in (20, 28)}
    gap = abs(values[ANCHORS[1]] - values[ANCHORS[0]])
    return {"family_id": row["family_id"], "eligible": gap > .202, "anchor_gap": gap,
            "max_prediction_error": errors, "max_error_precision_difference": precision,
            "same_label_prefix_differences": pairs, "max_same_label_prefix_difference": max(pairs.values()),
            "balance_contrast": ((means["pos_20"] - means["neg_20"]) + (means["pos_28"] - means["neg_28"])) / 2,
            "position_contrast": ((means["neg_28"] - means["neg_20"]) + (means["pos_28"] - means["pos_20"])) / 2,
            "interaction": means["pos_28"] - means["pos_20"] - means["neg_28"] + means["neg_20"],
            "max_native_margin_change": max(abs(row["cells"][c]["float64"]["margin_change"]) for c in CELLS),
            "recipient_attention": row["cells"][ANCHORS[0]]["float64"]["a_r"]}


def summarize(families):
    eligible = [f for f in families if f["eligible"]]
    candidates = {}
    for name in CANDIDATES:
        errors = [f["max_prediction_error"][name] for f in eligible]
        candidates[name] = {"eligible_families": len(errors), "definite_hits": sum(e <= .099 for e in errors),
                            "possible_hits": sum(e <= .101 for e in errors),
                            "mean_max_error": statistics.mean(errors) if errors else None,
                            "max_error": max(errors) if errors else None}
    descriptions = {}
    for key in ("anchor_gap", "max_same_label_prefix_difference", "balance_contrast", "position_contrast",
                "interaction", "max_native_margin_change", "recipient_attention"):
        vals = [f[key] for f in families]
        descriptions[key] = {"min": min(vals), "median": statistics.median(vals),
                             "mean": statistics.mean(vals), "max": max(vals)}
    return {"status": "development_only", "families": len(families), "eligible_families": len(eligible),
            "thresholds": {"prediction_tolerance": .10, "numerical_allowance": .001, "strict_anchor_gap": .202},
            "candidates": candidates,
            "confirmation_start_gate": len(eligible) >= 16 and any(10*c["definite_hits"] >= 9*len(eligible) for c in candidates.values()),
            "same_label_control_exceeds_tolerance": sum(f["max_same_label_prefix_difference"] > .10 for f in families),
            "same_label_control_exceeds_tolerance_eligible": sum(f["max_same_label_prefix_difference"] > .10 for f in eligible),
            "descriptive": descriptions,
            "max_error_precision_difference": max(f["max_error_precision_difference"] for f in families),
            "family_results": families}


def compare(actual, expected, location="report"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), "Expected mapping at " + location)
        for key, value in expected.items():
            require(key in actual, "Missing field: " + location + "." + key)
            compare(actual[key], value, location + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "List mismatch at " + location)
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, location + f"[{i}]")
    elif isinstance(expected, float):
        close(actual, expected, "Numerical report mismatch at " + location)
    else:
        require(actual == expected, "Report mismatch at " + location)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def audit(run, cases_path, report=None, root=ROOT):
    run, cases_path, root = Path(run), Path(cases_path), Path(root)
    manifest = json.loads((run / "manifest.json").read_text())
    require(manifest["status"] == "completed" and manifest["phase"] == "development",
            "Only completed development runs support this audit")
    require(manifest["task"] == {"model_id": "a9g0io1r", "n_layer": 2, "n_head": 4, "head": 1},
            "Unexpected checkpoint/task")
    require(manifest["numerical_allowance_on_prediction_error"] == .001, "Numerical allowance changed")
    require(manifest["input_cases_sha256"] == digest(cases_path), "Input cases hash mismatch")
    required_outputs = {"cases.jsonl", "controls.json", "anchor_forecasts.jsonl", "anchor_receipt.json", "recipient_selection.json"}
    require(required_outputs <= set(manifest["output_hashes"]), "Manifest omits a required output")
    for name, expected in manifest["output_hashes"].items():
        require(Path(name).name == name, "Output path must be local to run")
        require(digest(run / name) == expected, "Altered output artifact: " + name)
    for name, expected in manifest["source_hashes"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "Source path escapes repository")
        require(digest(path) == expected, "Current source differs from run: " + name)
    receipt = json.loads((run / "anchor_receipt.json").read_text())
    forecast_hash = digest(run / "anchor_forecasts.jsonl")
    require(receipt["anchor_forecasts_sha256"] == forecast_hash == manifest["anchor_forecasts_sha256"],
            "Forecast commitment hashes differ")
    require(receipt["source_hashes"] == manifest["source_hashes"], "Forecast source bindings differ")
    times = [datetime.fromisoformat(x) for x in (manifest["started_at"], receipt["written_before_target_execution_at"],
                                                manifest["target_measurement_started_at"], manifest["finished_at"])]
    require(all(t.tzinfo is not None for t in times) and times[0] <= times[1] < times[2] <= times[3],
            "Local forecast/target timestamp ordering is invalid")
    preparation = json.loads((cases_path.parent / "preparation.json").read_text())
    exclusions_path = cases_path.parent / "exclusions.json"
    exclusions = json.loads(exclusions_path.read_text())
    require(preparation["cases_sha256"] == digest(cases_path)
            and preparation["exclusions_sha256"] == digest(exclusions_path), "Input preparation hashes differ")
    require((preparation["phase"], preparation["seed"], preparation["families"]) == ("development", 25100201, 32),
            "Input preparation settings differ")
    inputs, outcomes, anchors = read_jsonl(cases_path), read_jsonl(run / "cases.jsonl"), read_jsonl(run / "anchor_forecasts.jsonl")
    selections = json.loads((run / "recipient_selection.json").read_text())
    require(len(inputs) == len(outcomes) == len(anchors) == len(selections) == 32, "Expected thirty-two families")
    require(len({f["family_id"] for f in inputs}) == 32, "Duplicate family identifiers")
    prefixes, recipients, prefix_snapshots = Counter(), Counter(), {}
    computed, max_roundoff = [], {"float32": 0., "float64": 0.}
    for index, (family, outcome, anchor, selection) in enumerate(zip(inputs, outcomes, anchors, selections)):
        donors = validate_input(family, index, exclusions)
        recipients[family["recipient"]] += 1
        require(outcome["family_id"] == anchor["family_id"] == family["family_id"], "Family ordering mismatch")
        require(outcome["phase"] == "development" and outcome["recipient"] == anchor["recipient"] == family["recipient"],
                "Outcome input binding differs")
        j = outcome["recipient_position"]
        require(type(j) is int and 1 <= j <= 32 and j == anchor["recipient_position"], "Invalid recipient position")
        require(selection["family_id"] == family["family_id"] and selection["recipient"] == family["recipient"],
                "Selection record does not match input family")
        attention = selection["native_eos_attention"]
        require(isinstance(attention, list) and len(attention) == 42, "Expected complete native EOS attention row")
        attention = [number(a) for a in attention]
        require(min(attention) >= 0 and all(a == 0 for a in attention[34:]), "Invalid native EOS attention support")
        close(math.fsum(attention), 1.0, "Native EOS attention is not normalized", atol=1e-12, rtol=0)
        closing = [k + 1 for k, char in enumerate(family["recipient"]) if char == ")"]
        selected = max(closing, key=lambda k: attention[k])
        require(j == selected == selection["recipient_position"], "Recipient selection is not first highest-attention closing token")
        require(set(outcome["cells"]) == set(CELLS), "Measured cell inventory mismatch")
        require(set(anchor["anchors"]) == set(ANCHORS), "Anchor cell inventory mismatch")
        for name in ANCHORS:
            require(outcome["cells"][name] == anchor["anchors"][name], "Anchor changed after forecast commitment")
        require(outcome["predictions"] == anchor["predictions"], "Forecast changed after target measurement")
        base = {}
        for name in CELLS:
            donor, cell = donors[name], outcome["cells"][name]
            require(cell["donor_metadata"] == donor, "Measured donor metadata differs from input")
            prefixes[donor["prefix_sha256"]] += 1
            for tag in ("float32", "float64"):
                snapshot = cell[tag]
                max_roundoff[tag] = max(max_roundoff[tag], check_snapshot(snapshot, tag, family["recipient"], j, donor))
                recipient_state = {k: snapshot[k] for k in ("a_r", "v_r", "h_r", "native_margin")}
                require(base.setdefault(tag, recipient_state) == recipient_state, "Recipient state changed within family")
                if tag == "float64":
                    require(snapshot["a_r"] == attention[j] and snapshot["native_margin"] == selection["native_margin"],
                            "Native snapshot disagrees with saved recipient selection")
                key = (tag, donor["position"], donor["prefix_sha256"])
                require(prefix_snapshots.setdefault(key, snapshot["v_d"]) == snapshot["v_d"],
                        "Same donor prefix produced inconsistent saved value vectors")
        computed.append(family_result(outcome))
    compare(preparation, {"unique_donor_prefixes": len(prefixes), "donor_prefix_draws": sum(prefixes.values()),
                          "max_prefix_reuse_across_families": max(prefixes.values()),
                          "unique_recipients": len(recipients), "max_recipient_reuse": max(recipients.values())}, "preparation")
    result = summarize(computed)
    controls = json.loads((run / "controls.json").read_text())
    for stage in ("anchors", "remaining_targets"):
        for tag in ("float32", "float64"):
            c = controls[stage][tag]
            tolerance = 1e-5 if tag == "float32" else 1e-10
            require(c["identity_tolerance"] == tolerance, "Identity tolerance changed")
            for name in ("identity_max_margin_error", "identity_max_node_error", "self_same_position_max_margin_error"):
                require(0 <= number(c[name]) <= tolerance, "Reported identity check failed: " + name)
            for name in ("inserted_node_exactly_intended_after_dtype", "all_target_layer_attention_weights_unchanged",
                         "all_target_layer_value_projections_unchanged", "nontarget_head_and_query_preprojection_exactly_unchanged"):
                require(c[name] is True, "Reported operator check did not pass: " + name)
    require(controls["precision_gate_passed"] is True and controls["prediction_error_dtype_allowance"] == .001,
            "Reported precision gate failed or changed")
    close(controls["maximum_prediction_error_dtype_discrepancy"], result["max_error_precision_difference"],
          "Global precision diagnostic differs from raw margins")
    if report is not None:
        saved = json.loads(Path(report).read_text())
        compare(saved, result)
        compare(saved["provenance"], {n: digest(run / n) for n in ("manifest.json", "cases.jsonl", "anchor_forecasts.jsonl")}, "provenance")
    return {"verification": "PASS", "audit_kind": "independent_code_path_records_only_not_external_replication",
            "manifest_sha256": digest(run / "manifest.json"), "inputs_sha256": digest(cases_path),
            "verified_source_files": len(manifest["source_hashes"]), "verified_output_files": len(manifest["output_hashes"]),
            "snapshots_checked": 32 * 8 * 2, "maximum_recomputed_construction_roundoff": max_roundoff,
            "local_forecast_ordering_checked": True, "analysis_report_checked": report is not None,
            "recomputed": result,
            "limits": ["Local timestamps/hashes do not establish an external preregistration timestamp.",
                       "No model forward or independent reexecution of saved tensor controls was performed.",
                       "Highest-attention recipient selection is verified from the saved EOS row, not independently rerun on the model.",
                       "Full attention/value arrays are absent: whole-layer unchangedness is a checked producer assertion, not an independently reconstructed fact.",
                       "This verifies development arithmetic and provenance, not a semantic mechanism or confirmation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.cases, args.report)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
