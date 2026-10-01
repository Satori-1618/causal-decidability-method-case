"""Reproduce the retrospective Li/Saphra ablation audit; Python standard library only."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
from statistics import mean, median

from countermodels import report as countermodel_report

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
N = 1000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    lock = json.loads((ROOT / "SOURCE_LOCK.json").read_text())
    for name, expected in lock["files"].items():
        p = UPSTREAM / name
        if p.stat().st_size != expected["bytes"] or sha(p) != expected["sha256"]:
            raise ValueError(f"Upstream source mismatch: {name}")
    return lock


def read_csv(name):
    with (UPSTREAM / name).open(newline="") as f:
        return list(csv.DictReader(f))


def count(value):
    """Recover integer correct counts, allowing only CSV float serialization error."""
    exact = Decimal(value) * N
    integer = int(exact.to_integral_value())
    if not 0 <= integer <= N or abs(exact - integer) > Decimal("0.00000001"):
        raise ValueError(f"Not an accuracy count out of {N}: {value}")
    return integer


def close(a, b):
    if abs(float(a) - float(b)) > 1e-12:
        raise ValueError(f"Inconsistent redundant field: {a} vs {b}")


def kind(sign, violation):
    return ("both" if sign and violation else "sign-matching" if sign else
            "violation-detecting" if violation else "neither")


def state(change, band=10):
    return "improved" if change > band else "damaged" if change < -band else "within_band"


def load_records():
    hp_rows = read_csv("data/transformer_head_properties.csv")
    hp = {r["id"]: r for r in hp_rows}
    if len(hp) != len(hp_rows):
        raise ValueError("Duplicate model IDs")
    raw = read_csv("heldout/results/mean_ablation_dyck_single.csv")
    records, seen = [], set()
    expected = {(r["id"], l, h) for r in hp_rows
                for l in range(1, int(r["n_layer"]) + 1)
                for h in range(1, int(r["n_head"]) + 1)}
    for r in raw:
        model, l, h = r["id"], int(r["layer"]), int(r["head"])
        key = (model, l, h)
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate or impossible head: {key}")
        seen.add(key)
        p = hp[model]
        for field in ("n_layer", "n_head", "wd"):
            close(r[field], p[field])
        sign, viol = float(r["sign_score"]), float(r["viol_score"])
        close(sign, p[f"cp5_sign_head_l{l}_h{h}_ood"])
        close(viol, p[f"cp5_neg_head_l{l}_h{h}_ood"])
        label = kind(sign >= .8, viol >= .8)
        if label != r["head_type"]:
            raise ValueError("Head classification mismatch")
        native = count(p["cp5_ood_acc"])
        if native != count(r["ood_acc"]) or native != count(r["csv_ood_acc"]):
            raise ValueError("Native baseline differs across producers")
        uniform = count(p[f"cp5_l{l}_h{h}_ood"])
        mean_count = count(r["ood_acc_single_mean"])
        du, dm = uniform-native, mean_count-native
        close(r["delta_uniform_single"], du/N)
        close(r["delta_mean_single"], dm/N)
        records.append({
            "model_id": model, "n_layer": int(p["n_layer"]), "n_head": int(p["n_head"]),
            "weight_decay": float(p["wd"]), "initialization_seed": int(p["rdm_seed"]),
            "shuffle_seed": int(p["shuffle_seed"]), "layer": l, "head": h,
            "head_type": label, "sign_score": sign, "violation_score": viol,
            "native_correct": native, "uniform_correct": uniform, "mean_correct": mean_count,
            "uniform_change_count": du, "mean_change_count": dm,
            "native_id_correct": count(p["cp5_indist_acc"]),
            "uniform_id_correct": count(p[f"cp5_l{l}_h{h}_indist"]),
            "uniform_state_10_cases": state(du), "mean_state_10_cases": state(dm),
            "removal_effect_identified_lower_pp": -native/10,
            "removal_effect_identified_upper_pp": (N-native)/10,
            "zero_reference_measured": False,
        })
    if seen != expected:
        raise ValueError("Missing actual heads")
    whole = read_csv("heldout/results/mean_ablation_dyck.csv")
    if len(whole) != len(hp) or {r["id"] for r in whole} != set(hp):
        raise ValueError("All-head model coverage mismatch")
    all_records = []
    for r in whole:
        p = hp[r["id"]]
        baseline = count(p["cp5_ood_acc"])
        if baseline != count(r["ood_acc"]) or baseline != count(r["csv_ood_acc"]):
            raise ValueError("All-head baseline mismatch")
        own = [x for x in records if x["model_id"] == r["id"]]
        label = kind(any(x["sign_score"] >= .8 for x in own),
                     any(x["violation_score"] >= .8 for x in own))
        if label != r["head_type"]:
            raise ValueError("All-head classification mismatch")
        du = count(p["cp5_full_ablate_ood"])-baseline
        dm = count(r["ood_acc_mean_ablated"])-baseline
        close(r["delta_uniform"], du/N)
        close(r["delta_mean"], dm/N)
        all_records.append({"model_id": r["id"], "n_layer": int(p["n_layer"]),
                            "head_type": label, "uniform_change_count": du,
                            "mean_change_count": dm})
    return sorted(records, key=lambda x: (x["model_id"], x["layer"], x["head"])), all_records


def group_summary(rows):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model_id"]].append(r)
    out = {"heads_or_allhead_records": len(rows), "models": len(by_model)}
    for operator in ("uniform", "mean"):
        field = operator + "_change_count"
        changes = [r[field] for r in rows]
        out[operator] = {
            "head_weighted_mean_pp": mean(changes)/10,
            "model_weighted_mean_pp": mean(mean(r[field] for r in group) for group in by_model.values())/10,
            "median_pp": median(changes)/10, "range_pp": [min(changes)/10, max(changes)/10],
            "positive_any": sum(x > 0 for x in changes), "negative_any": sum(x < 0 for x in changes),
            "zero": sum(x == 0 for x in changes),
            "improved_over_1pp": sum(x > 10 for x in changes),
            "damaged_over_1pp": sum(x < -10 for x in changes),
        }
    out["opposite_sign_over_1pp"] = sum(
        (r["uniform_change_count"] > 10 and r["mean_change_count"] < -10) or
        (r["uniform_change_count"] < -10 and r["mean_change_count"] > 10) for r in rows)
    out["both_improved_over_1pp"] = sum(r["uniform_change_count"] > 10 and r["mean_change_count"] > 10 for r in rows)
    return out


def make_results():
    lock = verify_sources()
    records, all_heads = load_records()
    primary = [r for r in records if r["n_layer"] >= 2]
    target = [r for r in primary if r["head_type"] == "sign-matching"]
    summaries = {k: group_summary([r for r in primary if r["head_type"] == k])
                 for k in sorted({r["head_type"] for r in primary})}
    example = next(r for r in target if r["uniform_change_count"] > 10 and r["mean_change_count"] > 10)
    proof = countermodel_report([example[k] for k in ("native_correct", "uniform_correct", "mean_correct")])
    # This reports all outcomes; selection is a declared post-hoc illustration.
    example = {**example, "selection": "Lexicographically first model/head with both changes >1pp; post-hoc illustration only."}
    ids = {r["model_id"] for r in target}
    contrasts = []
    for model in sorted(ids):
        ts = [r for r in target if r["model_id"] == model]
        controls = [r for r in primary if r["model_id"] == model and r["head_type"] == "neither"]
        if controls:
            contrasts.append({"model_id": model, "target_heads": len(ts), "control_heads": len(controls),
                              **{op + "_difference_pp": (mean(r[op+"_change_count"] for r in ts)-mean(r[op+"_change_count"] for r in controls))/10
                                 for op in ("uniform", "mean")}})
    summary = {
        "status": "retrospective_public_artifact_audit_not_fresh_confirmation",
        "upstream_revision": lock["revision"],
        "integrity": {"models": len(all_heads), "heads": len(records),
                      "primary_models": len({r["model_id"] for r in primary}), "primary_heads": len(primary),
                      "missing_heads": 0, "cross_producer_baseline_count_mismatches": 0,
                      "nominal_ood_cases": N,
                      "seed_pairs": len({(r["initialization_seed"], r["shuffle_seed"]) for r in primary})},
        "single_head_by_type": summaries,
        "all_heads_by_type": {k: group_summary([r for r in all_heads if r["n_layer"] >= 2 and r["head_type"] == k])
                              for k in sorted({r["head_type"] for r in all_heads if r["n_layer"] >= 2})},
        "target_id_uniform_mean_change_pp": mean(r["uniform_id_correct"]-r["native_id_correct"] for r in target)/10,
        "same_model_neither_controls": {
            "models": len(contrasts), "control_heads": sum(r["control_heads"] for r in contrasts),
            **{op + "_model_weighted_difference_pp": mean(r[op+"_difference_pp"] for r in contrasts) for op in ("uniform", "mean")},
            "scope": "Descriptive controls share model, not layer, norm, or functional importance; no exchangeability claim.",
        },
        "example": example,
        "identification": {
            "relative_to_native_replacement_effect": "Observed; sign-matching cohort improves under both specified replacements.",
            "uniform_only_artifact": "Mean replacement also improves; phenomenon is not restricted to uniform attention.",
            "native_removal_vs_replacement_contribution": "multiple_candidates_remain",
            "why": "No zero-output reference exists in the published intervention tables; every observed effect is removal plus replacement contribution.",
            "scope": "Compatibility is a demonstrated nonidentification result, not adequate fit of a neural mechanism.",
            "native_removal_effect_bounds": "For accuracy a_N, a_0-a_N is in [-a_N,1-a_N]; observed replacement accuracies do not tighten it without additional assumptions.",
        },
        "resolution": {
            "accuracy_count_grid": 1/N,
            "reporting_band_pp": 1,
            "population_inference": "Not estimated: fixed model sweep, reused seeds and test set; no heads-as-iid interval.",
            "precision_fidelity": "Producer code inspected and recorded baselines agree; no new cross-dtype or tensor intervention audit.",
            "zero_reference": "Missing observation, not a low-power test. More repeats of existing operators cannot distinguish the constructed witnesses.",
        },
    }
    return records, contrasts, summary, proof


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def encoded_results():
    records, contrasts, summary, proof = make_results()
    outputs = {"heads.csv": csv_text(records), "same_model_controls.csv": csv_text(contrasts),
               "summary.json": json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n",
               "countermodels.json": json.dumps(proof, indent=2, sort_keys=True, allow_nan=False)+"\n"}
    method_files = [ROOT.parents[1]/"examples/causal_preflight.py", ROOT.parents[1]/"src/causal_decidability/design.py"]
    producers = [ROOT/"analyze.py", ROOT/"countermodels.py", ROOT/"PROTOCOL.md", ROOT/"SOURCE_LOCK.json"]+method_files
    manifest = {"status": "retrospective", "input_and_producer_sha256": {str(p.relative_to(ROOT.parents[1])): sha(p) for p in producers},
                "output_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in outputs.items()},
                "source_hashes": json.loads((ROOT/"SOURCE_LOCK.json").read_text())["files"]}
    outputs["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True)+"\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only reconstruction and exact stored-artifact verification")
    args = parser.parse_args()
    outputs = encoded_results()
    directory = ROOT / "results"
    if args.check:
        for name, text in outputs.items():
            if (directory/name).read_bytes() != text.encode():
                raise SystemExit(f"Stored output mismatch: {name}")
        print("PASS: all source hashes, data joins, exact countermodels, method preflight, and stored outputs.")
    else:
        directory.mkdir(exist_ok=True)
        for name, text in outputs.items():
            destination = directory/name
            if destination.exists() and destination.read_bytes() != text.encode():
                raise SystemExit(f"Refusing to overwrite changed result: {name}")
            temp = directory/(name+".tmp")
            temp.write_text(text)
            temp.replace(destination)
        print("Wrote deterministic retrospective audit results.")


if __name__ == "__main__":
    main()
