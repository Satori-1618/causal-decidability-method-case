"""Gate table of the Round 1 development pilot, from its stored files only.

Standard library plus the analyzer and the design module (numpy only for the optional
audit of full-vocabulary logits). DEVELOPMENT DATA: every anchor here is descriptive; the
frozen anchors and delta come from splits A and B, never from the pilot.

Per cell, s_min follows the declared rule (upper order statistic at 0.99 of S over the
qualifying cases' no-patch runs, floor 0.10; with at most 100 runs that is their
maximum). The unresolved rate for the N rule is pooled over the qualifying conflict
cases of all cells, each resolved under its own cell's pilot s_min.
"""
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import mixing_round1_analysis as ra
import mixing_round1_design as design

KEYS = ("i_P", "i_L", "i_R")


def _load(path):
    return json.loads(Path(path).read_text())


def _jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rate(k, n):
    return {"k": k, "n": n, "rate": (k / n) if n else None}


def _quantiles(values):
    values = sorted(values)
    if not values:
        return None

    def q(level):
        position = level * (len(values) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return {"n": len(values), "min": values[0], "q10": q(0.1), "q25": q(0.25), "median": q(0.5),
            "q75": q(0.75), "q90": q(0.9), "max": values[-1], "mean": statistics.fmean(values)}


def _argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def _position_class(index, cell, w):
    if abs(index - cell["i_P"]) <= w:
        return "P-window"
    if index == cell["i_L"]:
        return "i_L"
    if index == cell["i_R"]:
        return "i_R"
    if index == cell["i_N"]:
        return "i_N"
    return "other"


def _mean_q(cases):
    resolved = [c["q"] for c in cases if c["resolved"]]
    if not resolved:
        return None
    return [math.fsum(q[i] for q in resolved) / len(resolved) for i in range(3)]


def _label_shares(cases):
    resolved = [c for c in cases if c["resolved"]]
    counts = Counter()
    for c in resolved:
        for label in c["labels"]:
            counts[label] += 1 / len(c["labels"])
    return {label: (counts[label] / len(resolved) if resolved else None) for label in ra.LABELS}


def _argmax_shares(logit_lists, cell, w):
    counts = Counter(_position_class(_argmax(x), cell, w) for x in logit_lists)
    total = len(logit_lists)
    return {k: counts[k] / total for k in ("P-window", "i_L", "i_R", "i_N", "other")} if total else None


def _validity(runs, cell, w, target=None):
    """How often the in-context entity readout names the entity the patched model
    generates. ``runs`` are readouts with 'entity_logits', 'patched_generation' and
    'answer_form_logits'. Diagnostic only."""
    total = len(runs)
    named = [r for r in runs if r["patched_generation"]["entity_index"] is not None]
    agree = sum(_argmax(r["entity_logits"]) == r["patched_generation"]["entity_index"] for r in named)
    forms = [r for r in runs if all(x is not None for x in r["answer_form_logits"]["capitalized"])]
    form_agree = sum(_argmax(r["entity_logits"]) == _argmax(r["answer_form_logits"]["capitalized"])
                     for r in forms)
    classes = Counter(_position_class(r["patched_generation"]["entity_index"], cell, w)
                      if r["patched_generation"]["entity_index"] is not None else "not an in-context entity"
                      for r in runs)
    result = {"runs": total,
              "generation_names_an_in_context_entity": _rate(len(named), total),
              "readout_argmax_equals_generated_entity": _rate(agree, len(named)),
              "readout_argmax_equals_capitalized_form_argmax": _rate(form_agree, len(forms)),
              "generated_entity_position_shares": {k: v / total for k, v in classes.items()} if total else None}
    if target is not None:
        result["generation_names_common_target"] = _rate(
            sum(r["patched_generation"]["entity_index"] == t for r, t in zip(runs, target)), total)
    return result


def summarize(output, pilot):
    output = Path(output)
    manifest = _load(output / "manifest.json")
    records = _jsonl(output / "records.jsonl")
    fp32 = {r["case_id"]: r for r in _jsonl(output / "fp32_reference.jsonl")}
    timings = _load(output / "timings.json") if (output / "timings.json").exists() else {}
    n, w = pilot["n"], ra.CONTRACT["w"]
    cells = {key: cell for key, cell in pilot["cells"]}
    gates = {}

    # Gate 1: model and tokenizer files
    hashes = manifest["model"]["hashes"]
    gates["1_model_hashes"] = {"passed": hashes["passed"], "problems": hashes["problems"],
                               "files": len(hashes["files"])}

    # Gate 3 (tokens) and technical checks
    technical_failures = [(r["case_id"], r["technical"]["failures"]) for r in records
                          if not r["technical"]["passed"]]
    measured = [r for r in records if "matrix" in r]
    gate3 = manifest["gate3_tokens"]
    alignment_failures = [f for f in technical_failures if any("differ from the design" in x for x in f[1])]
    gates["3_tokens"] = {"passed": gate3["passed"] and not alignment_failures,
                         "pools_kept": gate3["pools_kept"], "pools_dropped": gate3["pools_dropped"],
                         "dropped_prefix": gate3["dropped_prefix"],
                         "alignment_failures": alignment_failures}

    # Gate 4: hooks and shapes, from the per-run checks
    hook_checks = Counter()
    hook_failed = []
    for r in measured:
        for name, ok in r["technical"]["checks"].items():
            if name.endswith("hook"):
                hook_checks[name] += 1
                if not ok:
                    hook_failed.append((r["case_id"], name))
    design_failed = [(r["case_id"], name) for r in measured
                     for name, ok in r["technical"]["checks"].items()
                     if name.endswith("design_indices") and not ok]
    shapes = sorted({json.dumps(r["hook"]["shapes"]) for r in measured})
    gates["4_hooks"] = {"passed": not hook_failed, "checks": dict(hook_checks), "failed": hook_failed,
                        "conflict_hook_shapes_seen": [json.loads(s) for s in shapes][:5],
                        "design_index_failures": design_failed,
                        "technical_failures": technical_failures}

    # Gate 5: identity self-patch
    identities = [r["identity"] for r in measured if r.get("identity")]
    worst_entity = max(i["max_abs_entity_logit_difference"] for i in identities)
    worst_full = max(i["max_abs_logit_difference"] for i in identities)
    gates["5_identity"] = {
        "passed": (worst_entity <= pilot["identity_tolerance"]
                   and all(i["same_entity_argmax"] for i in identities)
                   and all(i["same_generation"] for i in identities)),
        "families": len(identities), "max_abs_entity_logit_difference": worst_entity,
        "max_abs_full_vocab_logit_difference": worst_full,
        "same_argmax": sum(i["same_entity_argmax"] for i in identities),
        "same_generation": sum(i["same_generation"] for i in identities),
        "tolerance": pilot["identity_tolerance"]}

    # Gate 2: native competence per position group and yield
    recipient_by_iN, donor_by_iP, agreement_by_target = {}, {}, {}
    readout_match = {"recipient": [0, 0], "donor": [0, 0]}
    for r in measured:
        rec, don = r["native"]["recipient"], r["native"]["donor"]
        a = recipient_by_iN.setdefault(r["cell"]["i_N"], [0, 0])
        a[0] += rec["correct"]; a[1] += 1
        b = donor_by_iP.setdefault(r["cell"]["i_P"], [0, 0])
        b[0] += don["correct"]; b[1] += 1
        for role, x in (("recipient", rec), ("donor", don)):
            readout_match[role][0] += x["readout_matches_generation"]
            readout_match[role][1] += 1
        for ag in r["agreement"]:
            c = agreement_by_target.setdefault(f"{ag['target']}={ag['j']}", [0, 0])
            c[0] += ag["donor_native"]["correct"]; c[1] += 1
    yield_by_cell = {}
    for key in cells:
        group = [r for r in records if r["cell_key"] == key]
        yield_by_cell[key] = _rate(sum(r["qualifies"] for r in group), len(group))
    overall_yield = _rate(sum(r["qualifies"] for r in records), len(records))
    gates["2_native_and_yield"] = {
        "passed": overall_yield["rate"] is not None and overall_yield["rate"] >= pilot["yield_floor"]
                  and all(v["rate"] >= pilot["yield_floor"] for v in yield_by_cell.values()),
        "recipient_correct_by_i_N": {str(k): _rate(*v) for k, v in sorted(recipient_by_iN.items())},
        "donor_correct_by_queried_position": {str(k): _rate(*v) for k, v in sorted(donor_by_iP.items())},
        "agreement_donor_correct_by_target": {k: _rate(*v) for k, v in sorted(agreement_by_target.items())},
        "readout_matches_generation": {k: _rate(*v) for k, v in readout_match.items()},
        "yield_by_cell": yield_by_cell, "yield": overall_yield, "floor": pilot["yield_floor"]}

    # Per-cell measures on qualifying families
    per_cell, pooled_unresolved, pooled_qualifying = {}, 0, 0
    agreement_all = {"runs": 0, "transfer": 0, "resolved": 0}
    for key, cell in cells.items():
        group = [r for r in measured if r["cell_key"] == key and r["qualifies"]]
        if not group:
            per_cell[key] = {"cell": cell, "qualifying": 0}
            continue
        nopatch_S = [ra.q_map(ra.softmax(r["nopatch"]["entity_logits"]), cell, w)[0] for r in group]
        s_min = ra.anchored_s_min(nopatch_S, pilot["s_min_quantile"], pilot["s_min_floor"])
        conflict = [ra.case_measures(r["entity_logits"], cell, w, s_min) for r in group]
        layer19 = [ra.case_measures(r["diagnostic"]["readout"]["entity_logits"], cell, w, s_min)
                   for r in group if r.get("diagnostic")]
        unresolved = sum(not c["resolved"] for c in conflict)
        pooled_unresolved += unresolved
        pooled_qualifying += len(group)
        agreement = [(k, ag["j"], ag["entity_logits"]) for k, r in enumerate(group) for ag in r["agreement"]]
        rates = ra.agreement_anchor(agreement, cell, w, s_min)
        agreement_all["runs"] += rates["agreement_total"]
        agreement_all["resolved"] += rates["agreement_resolved"]
        agreement_all["transfer"] += round(rates["agreement_transfer_rate"] * rates["agreement_total"])
        try:
            anchor = ra.anchors([r["entity_logits"] for r in group], agreement, cell, w, s_min)
            anchor = {k: anchor[k] for k in ("T_W", "T_A", "d", "q_bar", "T_A_by_target",
                                             "m_resolved", "m_total")}
        except ValueError as error:
            anchor = {"error": str(error)}
        per_cell[key] = {
            "cell": cell, "generated": sum(r["cell_key"] == key for r in records),
            "qualifying": len(group), "s_min": s_min,
            "s_min_is_floor": s_min == pilot["s_min_floor"],
            "nopatch_S": _quantiles(nopatch_S),
            "conflict_S": _quantiles([c["S"] for c in conflict]),
            "conflict_T": _quantiles([c["T"] for c in conflict if c["resolved"]]),
            "unresolved": _rate(unresolved, len(group)),
            "agreement": {k: rates[k] for k in ("agreement_total", "agreement_resolved",
                                                "agreement_resolution_rate", "agreement_transfer_rate",
                                                "agreement_runs_by_target", "agreement_resolved_by_target",
                                                "T_A_by_target")},
            "anchors_descriptive": anchor,
            "entity_mass_full_vocab": {
                "conflict_median": statistics.median(r["entity_mass_full_vocab"] for r in group),
                "nopatch_median": statistics.median(r["nopatch"]["entity_mass_full_vocab"] for r in group)},
            "layer18": {"mean_q": _mean_q(conflict), "label_shares_resolved": _label_shares(conflict),
                        "argmax_position_shares": _argmax_shares([r["entity_logits"] for r in group], cell, w),
                        "resolved": sum(c["resolved"] for c in conflict)},
            "layer19_diagnostic": {"mean_q": _mean_q(layer19), "label_shares_resolved": _label_shares(layer19),
                                   "argmax_position_shares": _argmax_shares(
                                       [r["diagnostic"]["readout"]["entity_logits"] for r in group], cell, w),
                                   "resolved": sum(c["resolved"] for c in layer19)},
        }

    # Readout validity (diagnostic): does the in-context readout name the generated entity?
    validity = {"conflict_layer18": [], "conflict_layer19": [], "agreement_layer18": []}
    for key, cell in cells.items():
        group = [r for r in measured if r["cell_key"] == key and r["qualifies"]]
        if not group:
            continue
        per_cell[key]["readout_validity"] = {
            "conflict_layer18": _validity([r["readout"] for r in group], cell, w),
            "conflict_layer19": _validity([r["diagnostic"]["readout"] for r in group if r.get("diagnostic")], cell, w),
            "agreement_layer18": _validity([a["readout"] for r in group for a in r["agreement"]], cell, w,
                                           target=[a["j"] for r in group for a in r["agreement"]]),
        }
    pooled_validity = {}
    for name in validity:
        rows = [v["readout_validity"][name] for v in per_cell.values() if "readout_validity" in v]
        if not rows:
            continue

        def pool(field):
            k = sum(r[field]["k"] for r in rows)
            n_ = sum(r[field]["n"] for r in rows)
            return _rate(k, n_)

        pooled_validity[name] = {field: pool(field) for field in (
            "generation_names_an_in_context_entity", "readout_argmax_equals_generated_entity",
            "readout_argmax_equals_capitalized_form_argmax")}
        if name == "agreement_layer18":
            pooled_validity[name]["generation_names_common_target"] = pool("generation_names_common_target")
    native_forms = [r["native"]["recipient"]["readout"] for r in measured if r["qualifies"]]
    native_forms = [x for x in native_forms if all(v is not None for v in x["answer_form_logits"]["capitalized"])]
    pooled_validity["native_recipient"] = {
        "readout_argmax_equals_capitalized_form_argmax": _rate(
            sum(_argmax(x["entity_logits"]) == _argmax(x["answer_form_logits"]["capitalized"]) for x in native_forms),
            len(native_forms))}

    unresolved_rate = pooled_unresolved / pooled_qualifying if pooled_qualifying else None
    gates["8_support"] = {
        "pooled_unresolved": _rate(pooled_unresolved, pooled_qualifying),
        "resolution_rate": (1 - unresolved_rate) if unresolved_rate is not None else None,
        "floor": pilot["resolution_rate_floor"],
        "passed": unresolved_rate is not None and 1 - unresolved_rate >= pilot["resolution_rate_floor"],
        "by_cell": {k: v.get("unresolved") for k, v in per_cell.items()}}
    transfer = agreement_all["transfer"] / agreement_all["runs"] if agreement_all["runs"] else None
    resolution = agreement_all["resolved"] / agreement_all["runs"] if agreement_all["runs"] else None
    gates["6_agreement"] = {
        "transfer_rate": transfer, "resolution_rate": resolution, "runs": agreement_all["runs"],
        "transfer_floor": pilot["agreement_transfer_floor"],
        "resolution_floor": pilot["agreement_resolution_floor"],
        "transfer_passed": transfer is not None and transfer >= pilot["agreement_transfer_floor"],
        "resolution_passed": resolution is not None and resolution >= pilot["agreement_resolution_floor"],
        "by_cell": {k: v.get("agreement") for k, v in per_cell.items()}}
    gates["6_agreement"]["passed"] = (gates["6_agreement"]["transfer_passed"]
                                      and gates["6_agreement"]["resolution_passed"])

    # Gate 7: bfloat16 against float32 on the same prompts
    comparisons, comparisons_unembed = [], []
    for r in measured:
        ref = fp32.get(r["case_id"])
        if ref is None:
            continue
        cell = cells[r["cell_key"]]
        s_min = per_cell.get(r["cell_key"], {}).get("s_min", pilot["s_min_floor"])
        for variant, logits, sink in (("model", r["entity_logits"], comparisons),
                                      ("fp32_unembed", r["readout"]["entity_logits_fp32_unembed"],
                                       comparisons_unembed)):
            a = ra.case_measures(logits, cell, w, s_min)
            b = ra.case_measures(ref["conflict"], cell, w, s_min)
            sink.append({"case_id": r["case_id"], "qualifies": r["qualifies"],
                         "T_bf16": a["T"], "T_fp32": b["T"],
                         "abs_T_difference": abs(a["T"] - b["T"]) if a["T"] is not None and b["T"] is not None else None,
                         "same_resolution": a["resolved"] == b["resolved"],
                         "same_labels": a["labels"] == b["labels"],
                         "max_abs_entity_logit_difference": max(abs(x - y) for x, y in zip(logits, ref["conflict"])),
                         "hook_ok_fp32": ref["hook_ok"]})

    def dtype_gate(rows):
        if not rows:
            return None
        worst = max(x["abs_T_difference"] for x in rows)
        return {"cases": len(rows), "max_abs_T_difference": worst,
                "median_abs_T_difference": statistics.median(x["abs_T_difference"] for x in rows),
                "same_resolution": sum(x["same_resolution"] for x in rows),
                "same_labels": sum(x["same_labels"] for x in rows),
                "max_abs_entity_logit_difference": max(x["max_abs_entity_logit_difference"] for x in rows),
                "passed": worst <= pilot["dtype_tolerance_T"] and all(x["same_resolution"] and x["same_labels"] for x in rows)}

    gates["7_dtype"] = {"primary_bf16_logits": dtype_gate(comparisons),
                        "diagnostic_fp32_unembedding": dtype_gate(comparisons_unembed),
                        "tolerance": pilot["dtype_tolerance_T"],
                        "note": "resolution and labels use each cell's pilot s_min; all sampled families, qualifying or not",
                        "rows": comparisons}
    gates["7_dtype"]["passed"] = bool(gates["7_dtype"]["primary_bf16_logits"]
                                      and gates["7_dtype"]["primary_bf16_logits"]["passed"])

    # Gate 9 (descriptive only): separation per cell
    d_by_cell = {k: v.get("anchors_descriptive", {}).get("d") for k, v in per_cell.items()}
    finite = [(k, d) for k, d in d_by_cell.items() if isinstance(d, float)]
    likely = ra.select_cell(finite, pilot["d_min"]) if finite else None
    gates["9_separation_descriptive"] = {"d_by_cell": d_by_cell, "d_min": pilot["d_min"],
                                         "likely_selection": likely,
                                         "note": "pilot, descriptive; selection happens on split A"}

    # N rule
    n_rule = design.n_rule(unresolved_rate) if unresolved_rate is not None else None

    # Runtime
    runtimes = [r["runtime_seconds"] for r in measured]
    mean_runtime = statistics.fmean(runtimes) if runtimes else None
    y = overall_yield["rate"] or None

    def projection(qualifying):
        if not (mean_runtime and y):
            return None
        generated = math.ceil(qualifying / y)
        return {"qualifying": qualifying, "generated_expected": generated,
                "hours_upper": generated * mean_runtime / 3600}

    runtime = {"per_family_seconds": _quantiles(runtimes), "timings": timings,
               "projection_note": ("pilot families include the identity self-patch and the layer-19 "
                                   "diagnostic, so projections are upper bounds; generated = qualifying / pilot yield"),
               "split_A": projection(200), "split_B": projection(200),
               "confirmation": projection(n_rule["N"]) if n_rule else None}

    # Audit sample: recompute the readout from the stored full-vocabulary logits
    audit = {"checked": False}
    npz = output / "audit_full_logits.npz"
    if npz.exists():
        try:
            import numpy as np
            data = np.load(npz)
            worst = 0.0
            for case_id in data.files:
                logits = data[case_id].astype("float64")
                r = next(x for x in measured if x["case_id"] == case_id)
                top = logits.max()
                lse = float(top) + math.log(float(np.exp(logits - top).sum()))
                worst = max(worst, float(abs(lse - r["readout"]["logsumexp_full"])))
            audit = {"checked": True, "cases": list(data.files),
                     "max_abs_logsumexp_difference": worst, "passed": worst <= 1e-3}
        except ImportError:
            audit = {"checked": False, "reason": "numpy unavailable"}

    top_tokens = Counter(r["native"]["recipient"]["readout"]["top_token"] for r in measured)
    return {
        "label": "DEVELOPMENT PILOT: descriptive; not used to estimate frozen anchors or delta",
        "families": len(records), "measured": len(measured),
        "gates": gates, "per_cell": per_cell,
        "unresolved_rate_for_N_rule": unresolved_rate, "n_rule": n_rule,
        "readout_validity_diagnostic": pooled_validity,
        "runtime": runtime, "audit_full_logits": audit,
        "nopatch_top_full_vocab_tokens": top_tokens.most_common(10),
        "device": manifest["environment"]["device"],
    }
