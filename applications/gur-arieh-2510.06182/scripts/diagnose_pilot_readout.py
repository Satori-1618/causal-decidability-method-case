"""Readout-validity DIAGNOSTIC on the stored pilot-1 records (development data, descriptive).

Pilot-1 (protocol v1) producer of results/pilot/diagnostic_answer_form.json. It reads the
v1 record fields and uses only the analyzer's descriptive helpers (``describe_logits``,
``describe_anchors``), which keep the v1 semantics (resolution by S alone), so it
reproduces its stored output byte for byte under the protocol-v2 analyzer.

It changes nothing in the declared statistic or in the code paths used for decisions; it
reads results/pilot/records.jsonl and the pinned tokenizer (offline) and writes
results/pilot/diagnostic_answer_form.json.

The declared readout scores the in-context entity tokens (' country', with a leading
space), as upstream. The answer-form readout scores the token form the model emits first
at the answer position. That form is determined here, not assumed:

1. For every unpatched native generation in the pilot (recipient, conflict donor and
   agreement donors), the first generated token is compared with the tokenization of
   the correct entity's candidate surface forms.
2. The form that the first tokens match is the capitalised form without a leading
   space ('Country'). Each entity is checked for being one token in that form.
3. The answer-form readout uses those single tokens' logits, stored by the runner for
   every run as ``answer_form_logits['capitalized']``; it is evaluated only on families
   whose n entities all have a single-token answer form.

For both readouts and each run type it reports the absolute full-vocabulary mass on the
n scored tokens, how often the readout's argmax names the entity that greedy generation
produces, and, per cell, T_W, T_A, d, the unresolved rate (s_min by the declared rule
applied to that readout's own no-patch runs) and the label shares.

    python scripts/diagnose_pilot_readout.py --results results/pilot
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
APPLICATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APPLICATION / "src"))
import mixing_round1_analysis as ra  # noqa: E402

GENRE = re.compile(r"performed (\S+) music on the")
READOUTS = ("declared_in_context", "answer_form")


def answer_form(entity):
    return entity[:1].upper() + entity[1:]


def rate(k, n):
    return {"k": k, "n": n, "rate": k / n if n else None}


def mass(logits, lse_full):
    return math.fsum(math.exp(x - lse_full) for x in logits)


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def scored(readout, which):
    if which == "declared_in_context":
        return readout["entity_logits"]
    return readout["answer_form_logits"]["capitalized"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    results = Path(args.results)
    manifest = json.loads((results / "manifest.json").read_text())
    records = [json.loads(line) for line in (results / "records.jsonl").read_text().splitlines()]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"]["snapshot"], local_files_only=True)
    genres = manifest["single_token_pools"]["Genre"]
    forms = {}
    for g in genres:
        forms[g] = {label: tokenizer.tokenize(text) for label, text in (
            ("in_context", " " + g), ("answer_form", answer_form(g)), ("lowercase", g))}
    single = {g: len(forms[g]["answer_form"]) == 1 for g in genres}

    # 1. Which form does the model emit first?
    first = Counter()
    for r in records:
        if "matrix" not in r:
            continue
        runs = [r["native"]["recipient"], r["native"]["donor"]] + [a["donor_native"] for a in r["agreement"]]
        for x in runs:
            token = x["generation_ids"][0]
            ids = tokenizer.encode(answer_form(x["answer"]), add_special_tokens=False)
            if len(ids) == 1 and token == ids[0]:
                first["single-token answer form of the correct entity"] += 1
            elif token == ids[0]:
                first["first piece of a multi-token answer form of the correct entity"] += 1
            elif token == tokenizer.encode(" " + x["answer"], add_special_tokens=False)[0]:
                first["in-context form of the correct entity"] += 1
            else:
                first["other token: " + tokenizer.convert_ids_to_tokens(token)] += 1

    qualifying = [r for r in records if r.get("qualifies")]
    subset = [r for r in qualifying if all(single[g[1]] for g in r["matrix"])]
    cells = {key: cell for key, cell in manifest["pilot"]["cells"]}
    w = ra.CONTRACT["w"]

    # 2. Mass and argmax agreement with generation, per run type, on the subset
    def native_rows(r):
        rows = []
        rec = r["native"]["recipient"]
        rows.append(("no_patch_recipient", rec["readout"], [g[1] for g in r["matrix"]], rec["first_word"]))
        don = r["native"]["donor"]
        rows.append(("native_conflict_donor", don["readout"], [g[1] for g in r["donor_matrix"]], don["first_word"]))
        for a in r["agreement"]:
            d = a["donor_native"]
            order = GENRE.findall(d["raw_prompt"])
            rows.append(("native_agreement_donor", d["readout"], order, d["first_word"]))
        return rows

    def patched_rows(r):
        entities = [g[1] for g in r["matrix"]]
        rows = [("conflict_layer18", r["readout"], entities, r["readout"]["patched_generation"]["first_word"])]
        if r.get("diagnostic"):
            d = r["diagnostic"]["readout"]
            rows.append(("conflict_layer19_diagnostic", d, entities, d["patched_generation"]["first_word"]))
        for a in r["agreement"]:
            rows.append(("agreement_layer18", a["readout"], entities, a["readout"]["patched_generation"]["first_word"]))
        return rows

    by_type = {}
    for r in subset:
        for name, readout, entities, word in native_rows(r) + patched_rows(r):
            if len(entities) != len(readout["entity_logits"]):
                raise SystemExit(f"{r['case_id']}: entity order could not be recovered")
            slot = by_type.setdefault(name, {x: {"mass": [], "agree": 0} for x in READOUTS} | {"runs": 0})
            slot["runs"] += 1
            for which in READOUTS:
                logits = scored(readout, which)
                slot[which]["mass"].append(mass(logits, readout["logsumexp_full"]))
                slot[which]["agree"] += entities[argmax(logits)].lower() == word
    run_types = {}
    for name, slot in by_type.items():
        run_types[name] = {"runs": slot["runs"]}
        for which in READOUTS:
            values = slot[which]["mass"]
            run_types[name][which] = {
                "full_vocab_mass_on_n_scored_tokens": {"median": median(values), "min": min(values),
                                                       "max": max(values)},
                "argmax_names_generated_entity": rate(slot[which]["agree"], slot["runs"])}

    # 3. Per cell, descriptively: anchors, unresolved rate and label shares, both readouts
    per_cell = {}
    for key, cell in cells.items():
        group = [r for r in subset if r["cell_key"] == key]
        entry = {"cell": cell, "families": len(group)}
        for which in READOUTS:
            nopatch = [scored(r["native"]["recipient"]["readout"], which) for r in group]
            s_min = ra.anchored_s_min([ra.q_map(ra.softmax(x), cell, w)[0] for x in nopatch], 0.99, 0.10)
            conflict = [ra.describe_logits(scored(r["readout"], which), cell, w, s_min) for r in group]
            agreement = [(k, a["j"], scored(a["readout"], which)) for k, r in enumerate(group) for a in r["agreement"]]
            try:
                anchor = ra.describe_anchors([scored(r["readout"], which) for r in group], agreement, cell, w, s_min)
                anchor = {k: anchor[k] for k in ("T_W", "T_A", "d", "q_bar", "T_A_by_target",
                                                 "agreement_resolution_rate", "agreement_transfer_rate")}
            except ValueError as error:
                anchor = {"error": str(error)}
            resolved = [c for c in conflict if c["resolved"]]
            shares = Counter()
            for c in resolved:
                for label in c["labels"]:
                    shares[label] += 1 / len(c["labels"])
            entry[which] = {"s_min": s_min, "unresolved": rate(len(group) - len(resolved), len(group)),
                            "anchors": anchor,
                            "label_shares_resolved": {l: (shares[l] / len(resolved) if resolved else None)
                                                      for l in ra.LABELS}}
        per_cell[key] = entry

    out = {
        "label": "DIAGNOSTIC, pilot/development, descriptive; decides nothing and changes no declared value",
        "answer_form_rule": "capitalised first letter, no leading space (entity[:1].upper() + entity[1:])",
        "first_generated_token_of_unpatched_native_runs": dict(first),
        "tokenization": forms,
        "single_token_answer_form": single,
        "entities_without_single_token_answer_form": [g for g in genres if not single[g]],
        "qualifying_families": len(qualifying),
        "families_with_all_answer_forms_single_token": len(subset),
        "subset_rule": "qualifying families whose seven genres all have a single-token answer form",
        "run_types": run_types,
        "per_cell": per_cell,
    }
    path = results / "diagnostic_answer_form.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"first": dict(first), "subset": len(subset),
                      "run_types": {k: {w_: v[w_]["argmax_names_generated_entity"]["rate"] for w_ in READOUTS}
                                    for k, v in run_types.items()}}, indent=2))


if __name__ == "__main__":
    main()
