"""Verify the archived Q1 result using Python's standard library only.

No model, tensor archive, network request, or output file is needed. This checks
recorded evidence; it does not replay hooks or authenticate the time of the run.
The historical scorer and verifier remain unchanged.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

APP_ROOT = Path(__file__).resolve().parents[1]
RUN = "results/makelov_read_source_q1/"
FREEZE = "829228ce8f39e55ac5ffa14a76a4e196652cb53b"
DIRECTION_SHA = "a3d41db3ce884de11a35e643d28474ebb1158eb2f597414a7b676fe94706e475"
ARMS = {"baseline", "identity", "full", "read_row", "read_null"}
PATCH_ARMS = ARMS - {"baseline"}
# Bytes of the archived application at 6388102, not newly calculated expectations.
PINNED = {
    "PREREG_READ_SOURCE_Q1.md": "f5bb62caaea253e7789f919b20a9bb78c68fc4e14e6219e260110f0f415f8a72",
    "scripts/score_read_source_q1.py": "a9a67d9d4aba55e0e9580a036189f407be4b9c7c8328455988495173204b036b",
    "src/makelov_read_source.py": "71d73bcef3071080ab3c2b9df632a487dbc0d655f50649c389a8fcf6b076204a",
    "scripts/run_makelov_read_source.py": "af7d9dd561fa8c21631e04fccad6b7952a4d066f8c1659769ed353ecf7996188",
    "results/makelov_read_source_001/cases.json": "32ea3f441c863f6684b6eac4911c5c45e055cff3c0ecab7bcd60670796b103ec",
    "artifacts/makelov_source/source_manifest.json": "137c8cd9a764069672d3f914a9d9eba10b939f90c098d060ac71d74726a732d4",
    RUN + "cases.json": "386cd48b89b316b73cac383a209d9cddc539c4cc3339931a4c43cb6d23aa66f2",
    RUN + "manifest.json": "043a2f02e99b5d22c5ff2b507e6e2721146511eb73f2e9ed6738303552b450c3",
    RUN + "records.jsonl": "eb4428e174b67dd65d3a9948a550273dc6fefe6adbfb4c014baadf82542f0ef1",
    RUN + "summary.json": "6dd5dc2610f8eecc44f1fb6d4958fc25ac5a4d42e8ea04a06279744241c08b9c",
    RUN + "score.json": "17b1139642730088192440aba4270ce9f3e152620b51d5ecdbf9ca44e1aa3f8c",
    RUN + "reference_cpu32.json": "d3c401afd8d98f5fb033c5fc1641cef8678b43fb81d8d43e17177e7e34dc1a6f",
    RUN + "reference_cpu64.json": "0f69d4ba84fb236acdcecff1d4e442d25601f03cb444e21990b2eafe0923e968",
}


class RecordCheckError(ValueError):
    """Archived evidence is missing, changed, or internally inconsistent."""


def require(condition, message):
    if not condition:
        raise RecordCheckError(message)


def close(found, expected, name):
    require(math.isfinite(found) and math.isfinite(expected)
            and math.isclose(found, expected, rel_tol=0, abs_tol=1e-12),
            f"{name}: found {found!r}, expected {expected!r}")


def directed_losses(row):
    m = row["margins"]
    return {
        "A_visible_read": (abs(m["read_row"] - m["full"])
                           + abs(m["read_null"] - m["baseline"])) / 2,
        "B_null_read": (abs(m["read_row"] - m["baseline"])
                        + abs(m["read_null"] - m["full"])) / 2,
    }


def audit_records(records, cases, expected_pairs):
    """Check individual fields independently of the file-hash gate.

    The two reciprocal swaps remain one unit. Fidelity entries are saved scalar
    measurements, not tensors from which a live intervention can be reconstructed.
    """
    require(len(cases) == expected_pairs, "unexpected number of base cases")
    ids = [c["case_id"] for c in cases]
    require(len(set(ids)) == expected_pairs, "duplicated base case id")
    require(len(records) == 2 * expected_pairs, "unexpected directed-record count")
    prompts = [p for c in cases for p in c["prompts"]]
    require(len(set(prompts)) == 2 * expected_pairs, "duplicated or missing prompt")
    grouped = defaultdict(list)
    for r in records:
        require(r["case_id"] in ids, "unknown record case id")
        grouped[r["case_id"]].append(r)
    controls = []
    split_errors = []
    losses = {}
    for c in cases:
        digest = hashlib.sha256(json.dumps(c["prompts"]).encode()).hexdigest()[:16]
        require(c["case_id"] == digest, "case id does not hash its prompt pair")
        require(c["patterns"] == ["ABB", "BAB"], "case patterns differ")
        require(len(c["token_ids"]) == 2 and
                len(c["token_ids"][0]) == len(c["token_ids"][1]) == c["position"] + 1,
                "absolute position or token-length mismatch")
        rows = grouped[c["case_id"]]
        require(len(rows) == 2 and
                {r["receiver_pattern"] for r in rows} == {"ABB", "BAB"},
                "each base pair needs both reciprocal directions exactly once")
        for r in rows:
            require(r["donor_pattern"] == ("BAB" if r["receiver_pattern"] == "ABB" else "ABB"),
                    "donor is not the reciprocal prompt")
            require(r["position"] == c["position"], "record position mismatch")
            require(set(r["margins"]) == set(r["answer_logits"]) == ARMS,
                    "missing or unexpected measurement arm")
            for arm, logits in r["answer_logits"].items():
                require(len(logits) == 2 and all(math.isfinite(x) for x in logits),
                        "nonfinite or malformed answer logits")
                close(logits[0] - logits[1], r["margins"][arm], "logit-margin arithmetic")
            require(r["answer_logits"]["identity"] == r["answer_logits"]["baseline"],
                    "recorded identity control failed")
            require(set(r["fidelity"]) == PATCH_ARMS, "missing fidelity arm")
            for arm, control in r["fidelity"].items():
                values = [control[k] for k in ("insertion_error_max_per_item",
                          "rounding_budget_per_item", "actual_delta_l2_per_item")]
                require(all(math.isfinite(x) and x >= 0 for x in values),
                        "nonfinite or negative fidelity measurement")
                require(control["calls"] == 1 and control["passed"] is True
                        and control["other_positions_unchanged"] is True
                        and values[0] <= values[1], "recorded fidelity control failed")
                if arm == "identity":
                    require(values[0] == values[2] == 0, "identity delta is nonzero")
                controls.append(control)
            split = r["split_delta_max_error"]
            require(math.isfinite(split) and split >= 0, "invalid split-delta error")
            split_errors.append(split)
        per_direction = [directed_losses(r) for r in rows]
        losses[c["case_id"]] = {
            key: fmean(d[key] for d in per_direction) for key in per_direction[0]}
    return losses, {
        "base_pairs": expected_pairs,
        "directed_records": len(records),
        "unique_prompts": len(set(prompts)),
        "recorded_arm_checks": len(controls),
        "max_recorded_insertion_error": max(c["insertion_error_max_per_item"] for c in controls),
        "max_recorded_split_delta_error": max(split_errors),
        "scope": "Recorded scalar checks; not a replay of the hooks or saved activation tensors.",
    }


def primary_result(losses):
    differences = [v["A_visible_read"] - v["B_null_read"] for v in losses.values()]
    a, b = sum(x < 0 for x in differences), sum(x > 0 for x in differences)
    n = a + b
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n) if n else 1.0
    outcome = "no difference shown" if p > .05 else "B predicts better" if b > a else "A predicts better"
    return {
        "test": "exact two-sided sign test on base-pair loss differences",
        "wins": {"A_visible_read": a, "B_null_read": b},
        "ties": len(losses) - n, "p": p, "alpha": .05, "outcome": outcome,
    }, {
        "mean_loss": {k: fmean(v[k] for v in losses.values()) for k in next(iter(losses.values()))},
        "mean_difference_A_minus_B": fmean(differences),
    }


def check(application_root=APP_ROOT):
    """Return a serializable audit report; raise RecordCheckError on any failure."""
    root = Path(application_root)
    payloads = {}
    for relative, expected in PINNED.items():
        try:
            content = (root / relative).read_bytes()
        except OSError as exc:
            raise RecordCheckError(f"missing or unreadable required artifact: {relative}") from exc
        require(hashlib.sha256(content).hexdigest() == expected,
                f"frozen hash mismatch: {relative}")
        payloads[relative] = content
    load = lambda name: json.loads(payloads[name])
    manifest, summary, score, cases = [load(RUN + n + ".json")
                                     for n in ("manifest", "summary", "score", "cases")]
    records = [json.loads(line) for line in payloads[RUN + "records.jsonl"].splitlines()]
    require(manifest["git_head"] == score["run"]["git_head"] == FREEZE, "freeze commit mismatch")
    require(manifest["seed"] == score["run"]["seed"] == 20260922, "seed mismatch")
    require(manifest["n_base_pairs"] == score["base_pairs"] == 64, "sample-size mismatch")
    require(manifest["working_device"] == score["run"]["device"] == "cpu"
            and manifest["working_dtype"] == "float32", "working runtime mismatch")
    require(manifest["layer_zero_based"] == 8 and manifest["same_write_vector_all_arms"] is True
            and manifest["separate_component_normalization"] is False, "operator declaration mismatch")
    require(manifest["runtime"] == {"torch": "2.5.1", "transformers": "4.57.3",
                                   "transformer-lens": "2.17.0", "numpy": "1.26.4"},
            "recorded package versions differ")
    require(manifest["source"] == load("artifacts/makelov_source/source_manifest.json"),
            "upstream source manifest mismatch")
    for path, digest in manifest["files"].items():
        require(PINNED.get(path) == digest, f"manifest source-code hash mismatch: {path}")
    for name, digest in summary["input_hashes"].items():
        expected = DIRECTION_SHA if name == "directions.npz" else PINNED.get(RUN + name)
        require(expected is not None and digest == expected, f"summary input pin differs: {name}")
    # Bindings are checked; directions.npz itself is deliberately not read.
    require(manifest["directions_sha256"] == DIRECTION_SHA, "direction pin mismatch")
    require(manifest["cases_sha256"] == PINNED[RUN + "cases.json"], "case pin mismatch")
    require(score["prereg_sha256"] == PINNED["PREREG_READ_SOURCE_Q1.md"]
            and score["scorer_sha256"] == PINNED["scripts/score_read_source_q1.py"]
            and score["run"]["records_sha256"] == PINNED[RUN + "records.jsonl"],
            "score hash bindings differ")
    pilot = load("results/makelov_read_source_001/cases.json")
    require(not ({c["case_id"] for c in cases} & {c["case_id"] for c in pilot}), "pilot case overlap")
    require(not ({p for c in cases for p in c["prompts"]}
                 & {p for c in pilot for p in c["prompts"]}), "pilot prompt overlap")
    losses, controls = audit_records(records, cases, expected_pairs=64)
    primary, reported = primary_result(losses)
    for k in ("wins", "ties", "p", "alpha"):
        require(primary[k] == score["primary"][k], f"recomputed primary differs: {k}")
    require(primary["outcome"] == score["outcome"], "recomputed outcome differs")
    for key, value in reported["mean_loss"].items():
        close(value, score["reported_not_judged"]["mean_loss"][key], key)
    close(reported["mean_difference_A_minus_B"],
          score["reported_not_judged"]["mean_difference_A_minus_B"], "mean advantage")
    for name, value in (("A_visible_read_MAE", reported["mean_loss"]["A_visible_read"]),
                        ("B_null_read_MAE", reported["mean_loss"]["B_null_read"]),
                        ("paired_MAE_A_minus_B_positive_favors_B", reported["mean_difference_A_minus_B"])):
        close(value, summary[name]["mean"], name)
    require(summary["identity_controls_passed"] is True and summary["fidelity_controls_passed"] is True,
            "summary control flags failed")
    require(manifest["reference_case_ids"] == [c["case_id"] for c in cases[:4]],
            "reference subset differs from the first four declared pairs")
    working = {(r["case_id"], r["receiver_pattern"]): r for r in records}
    precision = {}
    for name in ("reference_cpu32.json", "reference_cpu64.json"):
        reference = load(RUN + name)
        _, reference_controls = audit_records(reference, cases[:4], expected_pairs=4)
        changes = []
        for r in reference:
            old = directed_losses(working[(r["case_id"], r["receiver_pattern"])])
            new = directed_losses(r)
            changes.append((old["A_visible_read"] - old["B_null_read"])
                           - (new["A_visible_read"] - new["B_null_read"]))
        precision[name] = {
            "base_pairs": 4,
            "max_directed_difference_in_loss_advantage": max(abs(x) for x in changes),
            "mean_directed_difference_in_loss_advantage": fmean(changes),
            "recorded_arm_checks": reference_controls["recorded_arm_checks"],
        }
    return {
        "status": "archived_records_verified",
        "pinned_files_verified": len(PINNED),
        "primary": primary,
        "reported_not_judged": reported,
        "controls": controls,
        "pilot_overlap": {"case_ids": 0, "prompts": 0},
        "saved_reference_comparison": precision,
        "scope": {
            "verified": ["frozen file bytes and recorded bindings", "case and reciprocal-pair accounting",
                         "logit arithmetic and recorded scalar controls", "primary result and mean losses",
                         "stored four-pair numerical reference comparison"],
            "not_verified": ["live model or hook execution", "original activation tensors or direction payload",
                             "upstream files and model weights", "tokenization or regeneration of the prompts",
                             "external timestamp or chronology of freeze and run"],
            "numerical_limit": "Four predeclared pairs only; float64 reuses processed float32 weights. No universal error bound.",
            "claim_limit": "Q1 compares prediction loss. No adequacy threshold, unique mechanism, or semantic identification.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, default=APP_ROOT)
    parser.add_argument("--json", action="store_true", help="print the full audit report")
    args = parser.parse_args()
    try:
        report = check(args.application_root)
    except (RecordCheckError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"FAILED: {exc}\n")
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        p = report["primary"]
        print(f"PASS: {report['pinned_files_verified']} frozen files; 64 base pairs / 128 directed records.")
        print(f"Q1: {p['outcome']}; B wins {p['wins']['B_null_read']}/64; exact sign p = {p['p']:.4g}.")
        print("Mean loss (reported, not judged): " + ", ".join(
            f"{k} {v:.4f} nats" for k, v in report["reported_not_judged"]["mean_loss"].items()))
        print("Recorded controls and four-pair numerical references checked. No model or hook replay.")
        print("This confirms relative prediction performance, not adequacy or a unique mechanism.")


if __name__ == "__main__":
    main()
