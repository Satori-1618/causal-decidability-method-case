"""Independent records-only audit of the prospective native-margin screen.

No producer, analyzer, design or model-runtime imports. Reuses only the previous
independently implemented stdlib artifact verifier's tensor arithmetic checks.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
_spec = importlib.util.spec_from_file_location("independent_value_audit", HERE.parent / "verify.py")
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
require, number, close = old.require, old.number, old.close
digest, read_jsonl = old.digest, old.read_jsonl
ANCHORS, CELLS, TARGETS = old.ANCHORS, old.CELLS, old.TARGETS


def exact_interval(k, n=32):
    """Invert binomial CDFs; four simultaneous one-sided tails of 0.0125."""
    require(type(k) is int and type(n) is int and 0 <= k <= n and n > 0,
            "Invalid binomial count")
    alpha = .0125

    def cdf(p, through):
        return math.fsum(math.comb(n, j) * p**j * (1-p)**(n-j)
                         for j in range(through + 1))

    def inverse(through, target):
        low, high = 0., 1.
        for _ in range(80):
            mid = (low + high) / 2
            if cdf(mid, through) > target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    return [0. if k == 0 else inverse(k-1, 1-alpha),
            1. if k == n else inverse(k, alpha)]


def primary(accepted, rejected):
    groups = {name: {"separating": k, "families": 32, "rate": k/32,
                     "simultaneous_interval": exact_interval(k)}
              for name, k in (("accepted", accepted), ("rejected", rejected))}
    a, r = (groups[s]["simultaneous_interval"] for s in ("accepted", "rejected"))
    bounds = [a[0]-r[1], a[1]-r[0]]
    return {"strata": groups, "enrichment": (accepted-rejected)/32,
            "simultaneous_interval": bounds, "minimum_uplift": .25,
            "decision": ("supports_at_least_25pp_enrichment" if bounds[0] > .25
                         else "excludes_25pp_enrichment" if bounds[1] < .25
                         else "unresolved_at_25pp"),
            "directional_status": ("positive_enrichment" if bounds[0] > 0
                                   else "negative_enrichment" if bounds[1] < 0
                                   else "unresolved")}


def recount(rows):
    require(len(rows) == 64 and len({r["family_id"] for r in rows}) == 64,
            "Expected 64 unique intervention families")
    groups = {s: [r for r in rows if r["stratum"] == s] for s in ("accepted", "rejected")}
    require(all(len(g) == 32 for g in groups.values()), "Expected 32 families per stratum")
    records, counts, secondary = [], {}, []
    for name, group in groups.items():
        counts[name] = 0
        for row in group:
            require(set(row["cells"]) == set(CELLS if name == "accepted" else ANCHORS),
                    "Measured-cell inventory differs from protocol")
            gaps = {tag: abs(number(row["cells"][ANCHORS[1]][tag]["patched_margin"])
                             - number(row["cells"][ANCHORS[0]][tag]["patched_margin"]))
                    for tag in ("float32", "float64")}
            discrepancy = abs(gaps["float32"]-gaps["float64"])
            require(discrepancy <= .001 and (gaps["float32"] > .202) == (gaps["float64"] > .202),
                    "Unresolved anchor-gap numerical classification")
            native = number(row["cells"][ANCHORS[0]]["float64"]["native_margin"])
            require((native < 8) == (name == "accepted"), "Native margin/stratum mismatch")
            eligible = gaps["float64"] > .202
            counts[name] += eligible
            record = {"family_id": row["family_id"], "stratum": name,
                      "native_margin": native, "anchor_gap": gaps["float64"],
                      "separating": eligible, "anchor_gap_dtype_difference": discrepancy}
            if name == "accepted":
                record["secondary"] = old.family_result(row)
                if eligible:
                    secondary.append(record["secondary"])
            records.append(record)
    candidates = {}
    for candidate in ("H_state", "H_position"):
        errors = [r["max_prediction_error"][candidate] for r in secondary]
        candidates[candidate] = {"eligible_families": len(errors),
                                 "definite_hits": sum(e <= .099 for e in errors),
                                 "possible_hits": sum(e <= .101 for e in errors)}
    gate = len(secondary) >= 16 and any(10*c["definite_hits"] >= 9*len(secondary)
                                     for c in candidates.values())
    return {"primary": primary(counts["accepted"], counts["rejected"]),
            "secondary": {"status": "development_only", "accepted_families": 32,
                          "eligible_families": len(secondary), "candidates": candidates,
                          "confirmation_start_gate": gate},
            "family_results": records}


def check_input_donors(family, exclusions):
    require(len(family["donors"]) == 8, "Expected eight donor cells")
    donors, seen = {}, {}
    for donor in family["donors"]:
        cell = donor["cell"]
        require(cell in CELLS and cell not in donors, "Unknown/duplicate donor cell")
        sign, pos, rep = cell.split("_")
        pos, rep = int(pos), int(rep)
        depth = -2 if sign == "neg" else 2
        require((donor["position"], donor["balance"], donor["replica"]) == (pos, depth, rep),
                "Incorrect donor metadata")
        text = donor["string"]
        old.shape(text)
        prefix = text[:pos]
        require(text[pos-1] == ")" and old.walk(prefix) == (depth, -4),
                "Donor token/balance/minimum constraint failed")
        require(old.phase(prefix) == "development" and old.text_hash(prefix) == donor["prefix_sha256"],
                "Donor phase/hash failed")
        require(prefix not in exclusions["prefixes"][str(pos)], "Donor prefix is not fresh")
        require(text not in exclusions["recipient_strings"], "Donor full string is not fresh")
        key = depth, pos
        require(prefix not in seen.setdefault(key, set()), "Same prefix reused within cell")
        seen[key].add(prefix)
        donors[cell] = donor
    return donors


def check_chronology(manifest, selected_receipt, anchor_receipt):
    fields = [manifest["started_at"], selected_receipt["written_before_transfers_at"],
              manifest["anchors_measurement_started_at"], anchor_receipt["written_before_target_execution_at"],
              manifest["target_measurement_started_at"], manifest["finished_at"]]
    t = [datetime.fromisoformat(s) for s in fields]
    require(all(x.tzinfo is not None for x in t)
            and t[0] <= t[1] < t[2] <= t[3] < t[4] <= t[5],
            "Screen/anchor/forecast/target chronology is invalid")


def check_selection_row(selection):
    text = selection["recipient"]
    attention = [number(a) for a in selection["native_eos_attention"]]
    require(len(attention) == 42 and min(attention) >= 0 and all(a == 0 for a in attention[34:]),
            "Invalid complete native EOS attention row")
    close(math.fsum(attention), 1., "Attention row is not normalized", atol=1e-12, rtol=0)
    closing = [j+1 for j, char in enumerate(text) if char == ")"]
    pos = max(closing, key=lambda j: attention[j])
    require(selection["recipient_position"] == pos, "Recipient position is not first maximum closing token")
    return pos, attention


def check_screen(candidates, screening, exclusions):
    require(len(candidates) == len(screening) == 1024, "Screen must preserve all 1024 candidates")
    indices = {"accepted": [], "rejected": []}
    maximum = 0.
    banned = set(exclusions["recipient_strings"])
    for i, (candidate, observed) in enumerate(zip(candidates, screening)):
        require(candidate["candidate_id"] == old.text_hash(f"screen-002:25100202:{i}")[:20]
                and candidate["candidate_index"] == i, "Candidate order/identity differs")
        for key in ("candidate_id", "candidate_index", "recipient"):
            require(observed[key] == candidate[key], "Screen output/input binding differs")
        text = candidate["recipient"]
        old.shape(text)
        require(text not in banned and old.phase(text) == "development" and old.walk(text)[1] < 0,
                "Candidate is outside frozen fresh population")
        a, b = number(observed["margin_float32"]), number(observed["margin_float64"])
        maximum = max(maximum, abs(a-b))
        require(abs(a-b) <= .001 and (a < 8) == (b < 8), "Native screen precision gate failed")
        require(observed["screen_accept_float32"] is (a < 8)
                and observed["screen_accept_float64"] is (b < 8), "Incorrect screen label")
        group = "accepted" if b < 8 else "rejected"
        require(observed["stratum"] == group, "Wrong signed-margin stratum")
        indices[group].append(i)
    require(all(len(x) >= 32 for x in indices.values()), "Screen has insufficient stratum yield")
    selected = {s: v[:32] for s, v in indices.items()}
    chosen = set(selected["accepted"] + selected["rejected"])
    for i, row in enumerate(screening):
        require(row["selected"] is (i in chosen), "Selection is not first 32 per stratum")
    return selected, {s: len(v) for s, v in indices.items()}, maximum


def check_exclusions(exclusions, root):
    """Extend the committed historical ban list; raw cache files are optional.

    This validates continuity from the bound historical list, not independent
    completeness of that list against unavailable original public data.
    """
    app = root / "applications/li-saphra-2507.06445"
    historical = json.loads((app/"value_followup/inputs/development_001/exclusions.json").read_text())
    old_input = "value_followup/inputs/development_001/cases.jsonl"
    require(exclusions["source_hashes"] == {**historical["source_hashes"], old_input: digest(app/old_input)},
            "Exclusion source history changed")
    for name, expected in historical["source_hashes"].items():
        path = app/"native_followup"/name
        if path.exists():
            require(digest(path) == expected, "Available historical source hash mismatch: " + name)
    strings = set(historical["recipient_strings"])
    new_strings = set()
    for family in read_jsonl(app/old_input):
        new_strings.add(family["recipient"])
        new_strings.update(d["string"] for d in family["donors"])
    strings.update(new_strings)
    require(set(exclusions["recipient_strings"]) == strings, "Prior full-string bans are incomplete or changed")
    for pos in (20, 28):
        expected = set(historical["prefixes"][str(pos)]) | {s[:pos] for s in new_strings if len(s) >= pos}
        require(set(exclusions["prefixes"][str(pos)]) == expected,
                "Cross-role/cross-position prefix bans are incomplete")


def audit(run, inputs, report=None, root=ROOT):
    run, inputs, root = Path(run), Path(inputs), Path(root)
    manifest = json.loads((run / "manifest.json").read_text())
    require(manifest["status"] == "completed" and manifest["round"] == "screen_002",
            "Only a completed prospective screen supports this audit")
    require(manifest["task"] == {"model_id": "a9g0io1r", "n_layer": 2, "n_head": 4, "head": 1},
            "Unexpected checkpoint or head")
    require((manifest["native_margin_threshold"], manifest["anchor_gap_threshold"],
             manifest["prediction_error_allowance"]) == (8., .202, .001), "Frozen thresholds changed")
    required = {"screening.jsonl", "selected_families.jsonl", "screening_receipt.json",
                "recipient_selection.json", "anchor_cases.jsonl", "anchor_forecasts.jsonl",
                "anchor_receipt.json", "cases.jsonl", "controls.json"}
    require(required <= set(manifest["output_hashes"]), "Missing bound run artifact")
    for name, expected in manifest["output_hashes"].items():
        require(Path(name).name == name and digest(run/name) == expected, "Changed output artifact: " + name)
    for name, expected in manifest["source_hashes"].items():
        path = (root/name).resolve()
        require(path.is_relative_to(root.resolve()) and digest(path) == expected,
                "Current source differs from recorded run: " + name)
    require({"applications/li-saphra-2507.06445/value_followup/inputs/development_001/"+name
             for name in ("cases.jsonl", "exclusions.json")} <= set(manifest["source_hashes"]),
            "Historical cases and ban list must both be source-bound")
    preparation = json.loads((inputs/"preparation.json").read_text())
    require(manifest["preparation"] == preparation, "Preparation is not the bound input manifest")
    require((preparation["round"], preparation["phase"], preparation["screen_seed"],
             preparation["donor_seed"], preparation["candidate_pool_size"], preparation["per_stratum"])
            == ("screen_002", "development", 25100202, 25100203, 1024, 32), "Preparation settings changed")
    require(set(preparation["files"]) == {"candidates.jsonl", "donor_families.jsonl", "exclusions.json"},
            "Missing prepared input binding")
    for name, expected in preparation["files"].items():
        require(digest(inputs/name) == expected, "Prepared input changed: " + name)
    for name, expected in preparation["sources"].items():
        source = HERE/name if name == "prepare_screen.py" else HERE.parent/name
        require(digest(source) == expected, "Preparation source changed: " + name)
    exclusions = json.loads((inputs/"exclusions.json").read_text())
    check_exclusions(exclusions, root)
    candidates, templates = read_jsonl(inputs/"candidates.jsonl"), read_jsonl(inputs/"donor_families.jsonl")
    require(len(templates) == 64, "Expected 64 frozen donor templates")
    template_donors, prefix_counts = [], Counter()
    for i, template in enumerate(templates):
        require(template["phase"] == "development"
                and template["family_id"] == old.text_hash(f"development:25100203:{i}")[:20],
                "Donor template identity or phase changed")
        donors = check_input_donors(template, exclusions)
        template_donors.append(donors)
        prefix_counts.update(d["prefix_sha256"] for d in donors.values())
    screening = read_jsonl(run/"screening.jsonl")
    selected_indices, pool_counts, max_native_discrepancy = check_screen(candidates, screening, exclusions)
    candidate_counts = Counter(c["recipient"] for c in candidates)
    old.compare(preparation, {"unique_candidate_strings": len(candidate_counts),
                             "max_candidate_reuse": max(candidate_counts.values()),
                             "unique_donor_prefixes": len(prefix_counts),
                             "donor_prefix_draws": sum(prefix_counts.values()),
                             "maximum_donor_prefix_reuse": max(prefix_counts.values())})
    screen_receipt = json.loads((run/"screening_receipt.json").read_text())
    for file in ("screening.jsonl", "selected_families.jsonl", "recipient_selection.json"):
        require(screen_receipt[file.replace(".jsonl", "").replace(".json", "")+"_sha256"] == digest(run/file),
                "Screen receipt hash differs: " + file)
    require(screen_receipt["source_hashes"] == manifest["source_hashes"], "Screen receipt sources differ")
    anchor_receipt = json.loads((run/"anchor_receipt.json").read_text())
    for file in ("anchor_forecasts.jsonl", "anchor_cases.jsonl"):
        require(anchor_receipt[file.replace(".jsonl", "")+"_sha256"] == digest(run/file),
                "Anchor receipt hash differs: " + file)
    check_chronology(manifest, screen_receipt, anchor_receipt)
    families, rows = read_jsonl(run/"selected_families.jsonl"), read_jsonl(run/"cases.jsonl")
    anchor_rows = read_jsonl(run/"anchor_cases.jsonl")
    forecast_rows = read_jsonl(run/"anchor_forecasts.jsonl")
    selections = json.loads((run/"recipient_selection.json").read_text())
    require(len(families) == len(rows) == len(anchor_rows) == len(selections) == 64
            and len(forecast_rows) == 32, "Output family counts differ")
    order = [(s, i) for s in ("accepted", "rejected") for i in selected_indices[s]]
    prefix_values, max_roundoff, snapshots = {}, {"float32": 0., "float64": 0.}, 0
    max_signed_gap_discrepancy = 0.
    for k, ((stratum, index), family, row, anchors, selection) in enumerate(zip(order, families, rows, anchor_rows, selections)):
        source, template, observed = candidates[index], templates[k], screening[index]
        for item in (family, row, anchors):
            require(item["family_id"] == item["candidate_id"] == source["candidate_id"]
                    and item["candidate_index"] == index and item["stratum"] == stratum
                    and item["recipient"] == source["recipient"]
                    and item["donor_template_id"] == template["family_id"], "Selected donor assignment/input binding differs")
            for tag in ("float32", "float64"):
                require(item["margin_"+tag] == observed["margin_"+tag], "Selected native score changed")
        require(family["donors"] == template["donors"], "Donors changed after outcome-blind assignment")
        require(selection["family_id"] == source["candidate_id"] and selection["recipient"] == source["recipient"],
                "Recipient selection source mismatch")
        pos, attention = check_selection_row(selection)
        require(all(item["recipient_position"] == pos for item in (family, row, anchors, observed)),
                "Recipient intervention position changed")
        close(selection["native_margin"], observed["margin_float64"], "Saved native selection margin mismatch")
        require(set(anchors["cells"]) == set(ANCHORS), "Anchor receipt contains wrong cells")
        require(set(row["cells"]) == set(CELLS if stratum == "accepted" else ANCHORS), "Wrong final cell inventory")
        for name in ANCHORS:
            require(row["cells"][name] == anchors["cells"][name], "Anchor record changed after measurement")
        predicted = old.forecast({name: anchors["cells"][name]["float64"]["patched_margin"] for name in ANCHORS})
        if stratum == "accepted":
            f = forecast_rows[k]
            require(f["family_id"] == row["family_id"] and f["predictions"] == row["predictions"] == predicted,
                    "Anchor forecast formulas/binding changed")
            require(f["anchors"] == anchors["cells"], "Forecast anchor tensors changed")
        baseline = {}
        for name, cell in row["cells"].items():
            donor = template_donors[k][name]
            require(cell["donor_metadata"] == donor, "Cell donor metadata differs from prepared template")
            for tag in ("float32", "float64"):
                snapshot = cell[tag]
                max_roundoff[tag] = max(max_roundoff[tag], old.check_snapshot(snapshot, tag, family["recipient"], pos, donor))
                snapshots += 1
                state = {key: snapshot[key] for key in ("a_r", "v_r", "h_r", "native_margin")}
                require(baseline.setdefault(tag, state) == state, "Recipient state varies within family")
                close(snapshot["native_margin"], observed["margin_"+tag], "Native transfer/screen score differs", atol=1e-5 if tag == "float32" else 1e-10, rtol=0)
                if tag == "float64":
                    require(snapshot["a_r"] == attention[pos], "Transfer attention coefficient differs from saved row")
                key = tag, donor["position"], donor["prefix_sha256"]
                require(prefix_values.setdefault(key, snapshot["v_d"]) == snapshot["v_d"], "Identical prefix has inconsistent value")
        differences = {tag: row["cells"][ANCHORS[1]][tag]["patched_margin"]-row["cells"][ANCHORS[0]][tag]["patched_margin"]
                       for tag in ("float32", "float64")}
        max_signed_gap_discrepancy = max(max_signed_gap_discrepancy, abs(differences["float32"]-differences["float64"]))
        for item in (row, anchors):
            for tag in ("float32", "float64"):
                require(item["anchor_gap_"+tag] == abs(differences[tag]), "Recorded anchor gap is incorrect")
            require(item["anchor_separating"] is (abs(differences["float64"]) > .202), "Incorrect recorded separation label")
    require(max_signed_gap_discrepancy <= .001, "Signed anchor contrast precision gate failed")
    result = recount(rows)
    controls = json.loads((run/"controls.json").read_text())
    old.compare(controls["native_screen"], {"max_dtype_margin_difference": max_native_discrepancy,
                "dtype_classification_mismatches": 0, "cases_per_dtype": 1024,
                "selected_counts": {"accepted": 32, "rejected": 32}, "pool_counts": pool_counts})
    old.compare(controls["anchor_precision"], {"maximum_signed_contrast_dtype_difference": max_signed_gap_discrepancy,
                                             "separation_boundary_straddles": 0})
    maximum = max(r["secondary"]["max_error_precision_difference"] for r in result["family_results"] if "secondary" in r)
    old.compare(controls["target_precision"], {"maximum_prediction_error_dtype_difference": maximum, "gate_passed": True})
    for stage in ("anchors", "accepted_targets"):
        for tag in ("float32", "float64"):
            c = controls[stage][tag]
            tolerance = 1e-5 if tag == "float32" else 1e-10
            require(c["identity_tolerance"] == tolerance, "Identity tolerance changed")
            for name in ("identity_max_margin_error", "identity_max_node_error", "self_same_position_max_margin_error"):
                require(0 <= number(c[name]) <= tolerance, "Reported identity check failed")
            for name in ("inserted_node_exactly_intended_after_dtype", "all_target_layer_attention_weights_unchanged",
                         "all_target_layer_value_projections_unchanged", "nontarget_head_and_query_preprojection_exactly_unchanged"):
                require(c[name] is True, "Reported intervention control failed")
    if report is not None:
        saved = json.loads(Path(report).read_text())
        old.compare(saved, result)
        old.compare(saved["provenance"], {"git_head": manifest["git_head"], "manifest_sha256": digest(run/"manifest.json"),
                    "cases_sha256": digest(run/"cases.jsonl"), "analysis_sha256": digest(HERE/"analyze_screen.py")})
    return {"verification": "PASS", "scope": "independent_code_path_saved_records_not_model_replication",
            "manifest_sha256": digest(run/"manifest.json"), "verified_source_files": len(manifest["source_hashes"]),
            "verified_output_files": len(manifest["output_hashes"]), "native_candidates_checked": 1024,
            "snapshots_checked": snapshots, "maximum_construction_roundoff": max_roundoff,
            "local_forecast_ordering_checked": True, "analysis_report_checked": report is not None,
            "recomputed": result,
            "limits": ["Local receipts and hashes do not establish externally witnessed preregistration.",
                       "Freshness is validated against the bound historical ban list; unavailable public sources are not reparsed.",
                       "Whole-layer unchangedness is producer-reported; full-layer tensors are not archived.",
                       "No model inference was independently repeated; selection is checked against saved native rows.",
                       "The primary screen and secondary mechanism-development conclusions remain separate."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.inputs, args.report)
    encoded = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
