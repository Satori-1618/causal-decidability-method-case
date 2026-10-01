"""Gate table of a Round 1 development pilot under protocol v2, from its stored files.

Standard library plus the analyzer (numpy only for the audit of full-vocabulary logits).
DEVELOPMENT DATA: every anchor here is descriptive; the frozen anchors and delta come
from splits A and B, and the pilot's unresolved rate gives only a provisional planning N.

Decisions use the primary answer-form readout only (``answer_logits``,
``answer_mass_full_vocab``) through the analyzer; a case is resolved iff S >= s_min and
the answer-token mass >= 0.5. The paper's in-context readout is summarised separately,
with the analyzer's descriptive helpers, and feeds no gate. Per cell, s_min follows the
declared rule on all no-patch runs of the cell's qualifying families.

The first pilot (protocol v1, results/pilot/) was summarised by this module at commit
3cdaffd; that version is kept in the repository's history.
"""
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import mixing_round1_analysis as ra

RUN_TYPES = ("no_patch_recipient", "native_conflict_donor", "native_agreement_donors",
             "conflict_layer18", "conflict_layer19_diagnostic", "agreement_layer18")


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
    if index is None:
        return "not an in-context entity"
    if abs(index - cell["i_P"]) <= w:
        return "P-window"
    for key in ("i_L", "i_R", "i_N"):
        if index == cell[key]:
            return key
    return "other"


def _shares(counter, total):
    return {k: v / total for k, v in sorted(counter.items())} if total else None


def _label_shares(cases):
    resolved = [c for c in cases if c["resolved"]]
    counts = Counter()
    for c in resolved:
        for label in c["labels"]:
            counts[label] += 1 / len(c["labels"])
    return {label: (counts[label] / len(resolved) if resolved else None) for label in ra.LABELS}


def _mean_q(cases):
    resolved = [c["q"] for c in cases if c["resolved"]]
    if not resolved:
        return None
    return [math.fsum(q[i] for q in resolved) / len(resolved) for i in range(3)]


def _paper(readout):
    return readout["descriptive"]["paper_readout"]


def summarize(output, pilot):
    output = Path(output)
    manifest = _load(output / "manifest.json")
    records = _jsonl(output / "records.jsonl")
    reference = {r["case_id"]: r for r in _jsonl(output / "gate7_cpu_reference.jsonl")}
    timings = _load(output / "timings.json") if (output / "timings.json").exists() else {}
    w = ra.CONTRACT["w"]
    floor = ra.CONTRACT["answer_mass_floor"]
    cells = {key: cell for key, cell in pilot["cells"]}
    measured = [r for r in records if "matrix" in r]
    qualifying = [r for r in measured if r["qualifies"]]
    gates = {}

    # Gate 1: model and tokenizer files
    hashes = manifest["model"]["hashes"]
    gates["1_model_hashes"] = {"passed": hashes["passed"], "problems": hashes["problems"],
                               "files": len(hashes["files"])}

    # Gate 3: tokens, pools and alignment
    technical_failures = [(r["case_id"], r["technical"]["failures"]) for r in records
                          if not r["technical"]["passed"]]
    alignment_failures = [f for f in technical_failures if any("differ from the design" in x for x in f[1])]
    gate3 = manifest["gate3_tokens"]
    gates["3_tokens"] = {"passed": gate3["passed"] and not alignment_failures,
                         "pools_equal_the_lock": gate3["pools_equal_the_lock"],
                         "pools_kept": gate3["pools_kept"], "pools_dropped": gate3["pools_dropped"],
                         "dropped_prefix": gate3["dropped_prefix"],
                         "alignment_failures": alignment_failures}

    # Gate 4: hooks and shapes
    hook_checks, hook_failed = Counter(), []
    for r in measured:
        for name, ok in r["technical"]["checks"].items():
            if name.endswith("hook"):
                hook_checks[name] += 1
                if not ok:
                    hook_failed.append((r["case_id"], name))
    design_failed = [(r["case_id"], name) for r in measured
                     for name, ok in r["technical"]["checks"].items()
                     if name.endswith("design_indices") and not ok]
    gates["4_hooks"] = {"passed": not hook_failed and not design_failed and not technical_failures,
                        "checks": dict(hook_checks), "failed": hook_failed,
                        "conflict_hook_shapes_seen": sorted({json.dumps(r["hook"]["shapes"]) for r in measured}),
                        "design_index_failures": design_failed,
                        "technical_failures": technical_failures}

    # Gate 5: identity self-patch (execution device and dtype)
    identities = [r["identity"] for r in measured if r.get("identity")]
    gates["5_identity"] = {
        "passed": bool(identities) and (
            max(i["max_abs_answer_logit_difference"] for i in identities) <= pilot["identity_tolerance"]
            and all(i["same_answer_argmax"] for i in identities)
            and all(i["same_generation"] for i in identities)),
        "families": len(identities),
        "max_abs_answer_logit_difference": max((i["max_abs_answer_logit_difference"] for i in identities), default=None),
        "max_abs_full_vocab_logit_difference": max((i["max_abs_logit_difference"] for i in identities), default=None),
        "same_argmax": sum(i["same_answer_argmax"] for i in identities),
        "same_generation": sum(i["same_generation"] for i in identities),
        "tolerance": pilot["identity_tolerance"]}

    # Gate 2: native competence per position group, first-token criterion, yield
    def tally(pairs):
        table = {}
        for key, ok in pairs:
            row = table.setdefault(str(key), [0, 0])
            row[0] += bool(ok)
            row[1] += 1
        return {k: _rate(*v) for k, v in sorted(table.items())}

    native_runs = [(r, role, r["native"][role]) for r in measured for role in ("recipient", "donor")]
    agreement_donors = [(a["target"], a["j"], a["donor_native"]) for r in measured for a in r["agreement"]]
    yield_by_cell = {key: _rate(sum(r["qualifies"] for r in records if r["cell_key"] == key),
                                sum(r["cell_key"] == key for r in records)) for key in cells}
    overall = _rate(len(qualifying), len(records))
    gates["2_native_and_yield"] = {
        "passed": overall["rate"] is not None and overall["rate"] >= pilot["yield_floor"]
                  and all(v["rate"] is not None and v["rate"] >= pilot["yield_floor"] for v in yield_by_cell.values()),
        "recipient_correct_by_i_N": tally((r["cell"]["i_N"], x["correct"]) for r, role, x in native_runs if role == "recipient"),
        "donor_correct_by_queried_position": tally((r["cell"]["i_P"], x["correct"]) for r, role, x in native_runs if role == "donor"),
        "first_token_is_answer_form": tally((role, x["first_token_is_answer_form"]) for r, role, x in native_runs),
        "readout_matches_generation": tally((role, x["readout_matches_generation"]) for r, role, x in native_runs),
        "agreement_donor_correct_by_target": tally((f"{t}={j}", x["correct"]) for t, j, x in agreement_donors),
        "yield_by_cell": yield_by_cell, "yield": overall, "floor": pilot["yield_floor"]}

    # Per cell: s_min, primary measures, anchors, unresolved, diagnostics
    per_cell, pooled = {}, {"unresolved": 0, "qualifying": 0, "by_support": 0, "by_mass": 0}
    agreement_totals = Counter()
    for key, cell in cells.items():
        group = [r for r in qualifying if r["cell_key"] == key]
        entry = {"cell": cell, "generated": sum(r["cell_key"] == key for r in records), "qualifying": len(group)}
        per_cell[key] = entry
        if not group:
            continue
        s_min = ra.anchored_s_min(ra.nopatch_supports([r["nopatch"] for r in group], cell, w),
                                  pilot["s_min_quantile"], pilot["s_min_floor"])
        conflict = [ra.case_measures(r, cell, w, s_min) for r in group]
        layer19 = [ra.case_measures(r["diagnostic"]["readout"], cell, w, s_min) for r in group if r.get("diagnostic")]
        agreement = [(k, a["j"], a) for k, r in enumerate(group) for a in r["agreement"]]
        rates = ra.agreement_anchor(agreement, cell, w, s_min)
        agreement_totals["runs"] += rates["agreement_total"]
        agreement_totals["resolved"] += rates["agreement_resolved"]
        agreement_totals["transfer"] += round(rates["agreement_transfer_rate"] * rates["agreement_total"])
        try:
            anchor = ra.anchors(group, agreement, cell, w, s_min)
            anchor = {k: anchor[k] for k in ("T_W", "T_A", "d", "q_bar", "T_A_by_target", "m_resolved", "m_total")}
        except ValueError as error:
            anchor = {"error": str(error)}
        unresolved = sum(not c["resolved"] for c in conflict)
        by_support = sum(not c["S_ok"] for c in conflict)
        by_mass = sum(c["S_ok"] and not c["mass_ok"] for c in conflict)
        pooled["unresolved"] += unresolved
        pooled["qualifying"] += len(group)
        pooled["by_support"] += by_support
        pooled["by_mass"] += by_mass

        # the paper readout, DESCRIPTIVE ONLY (resolution by S alone, its own s_min)
        paper_nopatch = [ra.q_map(ra.softmax(_paper(r["nopatch"])["entity_logits"]), cell, w)[0] for r in group]
        paper_s_min = ra.anchored_s_min(paper_nopatch, pilot["s_min_quantile"], pilot["s_min_floor"])
        paper_conflict = [ra.describe_logits(r["descriptive"]["paper_readout"]["entity_logits"], cell, w, paper_s_min)
                          for r in group]
        try:
            paper_anchor = ra.describe_anchors(
                [r["descriptive"]["paper_readout"]["entity_logits"] for r in group],
                [(k, a["j"], _paper(a["readout"])["entity_logits"]) for k, r in enumerate(group) for a in r["agreement"]],
                cell, w, paper_s_min)
            paper_anchor = {k: paper_anchor[k] for k in ("T_W", "T_A", "d", "q_bar", "T_A_by_target",
                                                         "agreement_transfer_rate", "agreement_resolution_rate")}
        except ValueError as error:
            paper_anchor = {"error": str(error)}

        def generated_shares(readouts):
            return _shares(Counter(_position_class(x["patched_generation"]["entity_index"], cell, w) for x in readouts),
                           len(readouts))

        entry.update({
            "s_min": s_min, "s_min_is_floor": s_min == pilot["s_min_floor"],
            "nopatch_S": _quantiles(ra.nopatch_supports([r["nopatch"] for r in group], cell, w)),
            "conflict_S": _quantiles([c["S"] for c in conflict]),
            "conflict_answer_mass": _quantiles([c["answer_mass"] for c in conflict]),
            "conflict_T_resolved": _quantiles([c["T"] for c in conflict if c["resolved"]]),
            "unresolved": _rate(unresolved, len(group)),
            "unresolved_by_support": by_support, "unresolved_by_answer_mass_only": by_mass,
            "agreement": {k: rates[k] for k in ("agreement_total", "agreement_resolved", "agreement_resolution_rate",
                                                "agreement_transfer_rate", "agreement_runs_by_target",
                                                "agreement_resolved_by_target", "T_A_by_target")},
            "anchors_descriptive": anchor,
            "label_shares_resolved": _label_shares(conflict),
            "layer18_vs_19_diagnostic": {
                "mean_q_18": _mean_q(conflict), "mean_q_19": _mean_q(layer19),
                "generated_entity_position_18": generated_shares([r["readout"] for r in group]),
                "generated_entity_position_19": generated_shares([r["diagnostic"]["readout"] for r in group
                                                                  if r.get("diagnostic")])},
            "paper_readout_descriptive_only": {
                "s_min_by_its_own_no_patch_runs": paper_s_min,
                "unresolved_by_S": _rate(sum(not c["resolved"] for c in paper_conflict), len(group)),
                "anchors": paper_anchor, "label_shares_resolved": _label_shares(paper_conflict)},
        })

    unresolved_rate = pooled["unresolved"] / pooled["qualifying"] if pooled["qualifying"] else None
    gates["8_support"] = {
        "pooled_unresolved": _rate(pooled["unresolved"], pooled["qualifying"]),
        "pooled_unresolved_by_support": pooled["by_support"],
        "pooled_unresolved_by_answer_mass_only": pooled["by_mass"],
        "resolution_rate": (1 - unresolved_rate) if unresolved_rate is not None else None,
        "floor": pilot["resolution_rate_floor"],
        "passed": unresolved_rate is not None and 1 - unresolved_rate >= pilot["resolution_rate_floor"],
        "by_cell": {k: v.get("unresolved") for k, v in per_cell.items()},
        "s_min_by_cell": {k: v.get("s_min") for k, v in per_cell.items()}}
    runs = agreement_totals["runs"]
    transfer = agreement_totals["transfer"] / runs if runs else None
    resolution = agreement_totals["resolved"] / runs if runs else None
    gates["6_agreement"] = {
        "transfer_rate": transfer, "resolution_rate": resolution, "runs": runs,
        "transfer_floor": pilot["agreement_transfer_floor"], "resolution_floor": pilot["agreement_resolution_floor"],
        "transfer_passed": transfer is not None and transfer >= pilot["agreement_transfer_floor"],
        "resolution_passed": resolution is not None and resolution >= pilot["agreement_resolution_floor"],
        "by_cell": {k: v.get("agreement") for k, v in per_cell.items()}}
    gates["6_agreement"]["passed"] = gates["6_agreement"]["transfer_passed"] and gates["6_agreement"]["resolution_passed"]

    # Gate 7: MPS float32 against CPU float32 on the declared families
    rows = []
    declared = pilot["gate7_reference"]["families"]
    by_id = {r["case_id"]: r for r in measured}
    for r in records:
        if r["draw_index"] not in declared:
            continue
        ref = reference.get(r["case_id"])
        if r["case_id"] not in by_id or ref is None:
            rows.append({"case_id": r["case_id"], "missing": True})
            continue
        cell = cells[r["cell_key"]]
        s_min = per_cell[r["cell_key"]].get("s_min", pilot["s_min_floor"])
        a = ra.case_measures(r, cell, w, s_min)
        b = ra.case_measures(ref["conflict"], cell, w, s_min)
        rows.append({"case_id": r["case_id"], "cell_key": r["cell_key"], "qualifies": r["qualifies"],
                     "T_mps": a["T"], "T_cpu": b["T"], "abs_T_difference": abs(a["T"] - b["T"]),
                     "same_resolution": a["resolved"] == b["resolved"], "same_labels": a["labels"] == b["labels"],
                     "answer_mass_mps": a["answer_mass"], "answer_mass_cpu": b["answer_mass"],
                     "max_abs_answer_logit_difference": max(abs(x - y) for x, y in zip(r["answer_logits"],
                                                                                      ref["conflict"]["answer_logits"])),
                     "hook_ok_cpu": ref["hook_ok"]})
    complete = [x for x in rows if not x.get("missing")]
    per_cell_count = Counter(x["cell_key"] for x in complete)
    gates["7_dtype_device"] = {
        "comparison": "MPS float32 (execution) against CPU float32 (reference), conflict patch, primary readout",
        "declared_families": declared, "compared": len(complete), "missing": [x["case_id"] for x in rows if x.get("missing")],
        "per_cell": dict(per_cell_count),
        "max_abs_T_difference": max((x["abs_T_difference"] for x in complete), default=None),
        "median_abs_T_difference": statistics.median(x["abs_T_difference"] for x in complete) if complete else None,
        "same_resolution": sum(x["same_resolution"] for x in complete),
        "same_labels": sum(x["same_labels"] for x in complete),
        "max_abs_answer_logit_difference": max((x["max_abs_answer_logit_difference"] for x in complete), default=None),
        "tolerance": pilot["gate7_tolerance_T"], "rows": rows}
    gates["7_dtype_device"]["passed"] = bool(
        complete and len(complete) == len(declared)
        and gates["7_dtype_device"]["max_abs_T_difference"] <= pilot["gate7_tolerance_T"]
        and all(x["same_resolution"] and x["same_labels"] and x["hook_ok_cpu"] for x in complete))

    # Gate 9 (descriptive): separation per cell, likely selection and its margin
    d_by_cell = {k: v.get("anchors_descriptive", {}).get("d") for k, v in per_cell.items()}
    finite = [(k, d) for k, d in d_by_cell.items() if isinstance(d, float)]
    likely = ra.select_cell(finite, pilot["d_min"]) if finite else None
    ordered = sorted((d for _, d in finite), reverse=True)
    gates["9_separation_descriptive"] = {
        "d_by_cell": d_by_cell, "d_min": pilot["d_min"], "likely_selection": likely,
        "margin_over_second": (ordered[0] - ordered[1]) if len(ordered) > 1 else None,
        "note": "pilot, descriptive; selection happens on split A"}

    planning = ra.n_rule(unresolved_rate) if unresolved_rate is not None else None

    # Answer-token mass per run type (qualifying families)
    def masses(name):
        if name == "no_patch_recipient":
            return [r["native"]["recipient"]["readout"]["answer_mass_full_vocab"] for r in qualifying]
        if name == "native_conflict_donor":
            return [r["native"]["donor"]["readout"]["answer_mass_full_vocab"] for r in qualifying]
        if name == "native_agreement_donors":
            return [a["donor_native"]["readout"]["answer_mass_full_vocab"] for r in qualifying for a in r["agreement"]]
        if name == "conflict_layer18":
            return [r["answer_mass_full_vocab"] for r in qualifying]
        if name == "conflict_layer19_diagnostic":
            return [r["diagnostic"]["readout"]["answer_mass_full_vocab"] for r in qualifying if r.get("diagnostic")]
        return [a["answer_mass_full_vocab"] for r in qualifying for a in r["agreement"]]

    answer_mass = {name: {**(_quantiles(masses(name)) or {}),
                          "below_floor": sum(m < floor for m in masses(name))} for name in RUN_TYPES}

    # Readout validity: does each readout's argmax name the generated entity?
    def validity(readouts):
        named = [x for x in readouts if x["patched_generation"]["entity_index"] is not None]
        return {"runs": len(readouts), "generation_names_an_in_context_entity": _rate(len(named), len(readouts)),
                "primary_argmax_equals_generated": _rate(
                    sum(_argmax(x["answer_logits"]) == x["patched_generation"]["entity_index"] for x in named), len(named)),
                "paper_argmax_equals_generated_descriptive": _rate(
                    sum(_argmax(_paper(x)["entity_logits"]) == x["patched_generation"]["entity_index"] for x in named),
                    len(named))}

    native_named = [x for r in qualifying for x in (r["native"]["recipient"], r["native"]["donor"])]
    readout_validity = {
        "conflict_layer18": validity([r["readout"] for r in qualifying]),
        "conflict_layer19_diagnostic": validity([r["diagnostic"]["readout"] for r in qualifying if r.get("diagnostic")]),
        "agreement_layer18": validity([a["readout"] for r in qualifying for a in r["agreement"]]),
        "native_recipient_and_donor": {
            "primary_argmax_equals_generated": _rate(sum(x["readout_argmax_entity"] == x["first_word"] for x in native_named),
                                                     len(native_named)),
            "paper_argmax_equals_generated_descriptive": _rate(
                sum(x["descriptive_paper_argmax_entity"] == x["first_word"] for x in native_named), len(native_named))},
        "paper_readout_mass_descriptive": {
            "conflict_layer18": _quantiles([r["descriptive"]["paper_readout"]["entity_mass_full_vocab"] for r in qualifying]),
            "no_patch_recipient": _quantiles([_paper(r["nopatch"])["entity_mass_full_vocab"] for r in qualifying])},
    }

    # Runtime and projections
    runtimes = [r["runtime_seconds"] for r in measured]
    mean_runtime = statistics.fmean(runtimes) if runtimes else None
    y = overall["rate"] or None

    def projection(count):
        if not (mean_runtime and y):
            return None
        generated = math.ceil(count / y)
        return {"qualifying": count, "generated_expected": generated, "hours_upper": generated * mean_runtime / 3600}

    runtime = {"per_family_seconds": _quantiles(runtimes), "timings": timings,
               "projection_note": ("pilot families include the identity self-patch, the layer-19 patch and "
                                   "generation under every patch, so projections are upper bounds; "
                                   "generated = qualifying / pilot yield"),
               "split_A": projection(200), "split_B": projection(200),
               "confirmation_at_planning_N": projection(planning["N"]) if planning else None}

    # Audit sample: recompute the primary readout from the stored full-vocabulary logits
    audit = {"checked": False}
    npz = output / "audit_full_logits.npz"
    if npz.exists():
        try:
            import numpy as np
            data = np.load(npz)
            answer_ids = manifest["entity_pools"]["answer_form_ids"]
            worst_lse = worst_mass = worst_logit = 0.0
            for case_id in data.files:
                logits = data[case_id].astype("float64")
                r = by_id[case_id]
                top = float(logits.max())
                lse = top + math.log(float(np.exp(logits - top).sum()))
                ids = [answer_ids[g[1]] for g in r["matrix"]]
                answer = [float(logits[i]) for i in ids]
                mass = sum(math.exp(a - lse) for a in answer)
                worst_lse = max(worst_lse, abs(lse - r["readout"]["logsumexp_full"]))
                worst_mass = max(worst_mass, abs(mass - r["answer_mass_full_vocab"]))
                worst_logit = max(worst_logit, max(abs(a - b) for a, b in zip(answer, r["answer_logits"])))
            audit = {"checked": True, "cases": list(data.files), "max_abs_logsumexp_difference": worst_lse,
                     "max_abs_answer_mass_difference": worst_mass, "max_abs_answer_logit_difference": worst_logit,
                     "tolerances": {"logsumexp": 1e-3, "answer_mass": 1e-5, "answer_logit": 1e-6},
                     "passed": worst_lse <= 1e-3 and worst_mass <= 1e-5 and worst_logit <= 1e-6}
        except ImportError:
            audit = {"checked": False, "reason": "numpy unavailable"}

    return {
        "label": "PILOT/DEVELOPMENT, DESCRIPTIVE (protocol v2): not used to estimate frozen anchors or delta",
        "protocol": "v2", "families": len(records), "measured": len(measured), "qualifying": len(qualifying),
        "gates": gates, "per_cell": per_cell,
        "unresolved_rate_for_planning_N": unresolved_rate,
        "planning_N": planning,
        "planning_N_note": ("provisional; the final N and the adequacy label are set at the freeze by the "
                            "same rule applied to split B's unresolved rate in the selected cell"),
        "answer_mass_by_run_type": answer_mass,
        "readout_validity": readout_validity,
        "runtime": runtime, "audit_full_logits": audit,
        "no_patch_top_tokens": Counter(r["native"]["recipient"]["readout"]["top_token"] for r in measured).most_common(10),
        "device": manifest["environment"]["device"],
    }
