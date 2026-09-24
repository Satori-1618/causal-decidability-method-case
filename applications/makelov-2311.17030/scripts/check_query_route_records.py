"""Reproduce a query-route result from records only (stdlib; no model loading).

Required result files and frozen code are verified, raw answer logits are checked,
and response-profile counts are calculated independently of the producer. The
shipped analyzer separately reproduces the CP intervals and detailed summary.
Missing original source/model files and directions.npz are explicitly reported;
their absence does not prevent this records-only check.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

APPLICATION = Path(__file__).resolve().parents[1]
REPOSITORY = APPLICATION.parents[1]
sys.path.insert(0, str(APPLICATION / "src"))
from query_route_analysis import analyze_records  # noqa: E402

CELLS = tuple("ABCDEFG")
PRECISIONS = ("float32", "float64")
MLP = "blocks.8.mlp.hook_post"
QUERIES = {"blocks.9.attn.hook_q": [6, 9], "blocks.10.attn.hook_q": [0]}
PROFILE_NAMES = ("transfer", "joint_dependence", "preservation")
REQUIRED_FILES = {"manifest.json", "cases.json", "records.jsonl", "summary.json",
                  "RUN_STARTED.json", "measurements_float32.jsonl",
                  "measurements_float64.jsonl"}
CODE_PATHS = {"scripts/run_query_route.py", "src/makelov_read_source.py",
              "src/query_route.py", "src/query_route_analysis.py",
              "src/query_route_sampling.py"}


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finite(value, label, nonnegative=False):
    require(not isinstance(value, bool) and isinstance(value, (int, float)),
            f"{label}: expected a number")
    require(math.isfinite(value) and (not nonnegative or value >= 0),
            f"{label}: expected a finite {'nonnegative ' if nonnegative else ''}number")
    return value


def vector(values, label, length=2, nonnegative=False):
    require(isinstance(values, list) and len(values) == length, f"{label}: invalid vector length")
    return [finite(v, label, nonnegative) for v in values]


def hexhash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def relative_path(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe artifact path: {name}")
    return root / path


def check_hash(path, expected, label):
    require(hexhash(expected), f"{label}: invalid sha256")
    require(path.is_file(), f"missing required file: {label}")
    require(sha256(path) == expected, f"hash mismatch: {label}")


def audit_precision(value, case, precision):
    label = f"{case['case_id']}.{precision}"
    cells, logits = value["cells"], value["answer_logits"]
    require(set(cells) == set(logits) == set(CELLS), f"{label}: require A-G logits and margins")
    for cell in CELLS:
        vector(cells[cell], f"{label}.{cell} margins")
        require(isinstance(logits[cell], list) and len(logits[cell]) == 2,
                f"{label}.{cell}: require two directional logit rows")
        for direction in range(2):
            row = vector(logits[cell][direction], f"{label}.{cell} logits")
            margin = finite(row[0] - row[1], f"{label}.{cell} recomputed margin")
            require(margin == cells[cell][direction], f"{label}.{cell}: margin disagrees with logits")

    controls = value["controls"]
    require(controls["passed"] is True, f"{label}: controls failed")
    scale = max(1., max(abs(x) for cell in logits.values() for row in cell for x in row))
    epsilon = 2.0**(-23 if precision == "float32" else -52)
    tolerance = 64 * epsilon * scale
    require(controls["answer_logit_identity_tolerance"] == tolerance,
            f"{label}: identity tolerance does not follow dtype/scale rule")
    require(controls["identity_tolerance"] == 2*tolerance,
            f"{label}: incorrect margin identity tolerance")
    for a, b in (("C", "A"), ("F", "B"), ("G", "A")):
        name = f"{a}_vs_{b}"
        error = max(abs(x-y) for row_a, row_b in zip(logits[a], logits[b])
                    for x, y in zip(row_a, row_b))
        require(error <= tolerance, f"{label}: failed {name} identity")
        require(controls["identity_errors"][name] == error,
                f"{label}: incorrect identity error for {name}")
        require(controls["identity_exact"][name] is (logits[a] == logits[b]),
                f"{label}: incorrect exact-identity flag for {name}")

    source_hashes = value["query_source_hashes"]
    require(set(source_hashes) == {"q0", "q1"}, f"{label}: incomplete query provenance")
    for source in source_hashes.values():
        require(set(source) == set(QUERIES) and all(hexhash(h) for h in source.values()),
                f"{label}: invalid query-source hashes")
    expected_sites = {
        "B": [MLP], "C": list(QUERIES), "D": [MLP, *QUERIES],
        "E": list(QUERIES), "F": [MLP, *QUERIES], "G": [MLP],
    }
    audits, events = controls["fidelity"], controls["hook_events"]
    require(set(audits) == set(events) == set(expected_sites), f"{label}: wrong audited arms")
    for arm, sites in expected_sites.items():
        require(events[arm] == sites and set(audits[arm]) == set(sites),
                f"{label}.{arm}: hook order/sites differ from contract")
        for site in sites:
            audit = audits[arm][site]
            require(type(audit["calls"]) is int and audit["calls"] == 1
                    and audit["passed"] is True, f"{label}.{arm}.{site}: failed/count audit")
            if site == MLP:
                require(audit["other_positions_unchanged"] is True,
                        f"{label}.{arm}: MLP write was not isolated")
                errors = vector(audit["insertion_error_max_per_item"], label, nonnegative=True)
                budgets = vector(audit["rounding_budget_per_item"], label, nonnegative=True)
                changes = vector(audit["actual_delta_l2_per_item"], label, nonnegative=True)
                if arm == "G":
                    require(changes == [0, 0], f"{label}: zero-delta MLP control changed tensor")
            else:
                require(audit["position"] == case["position"] and audit["heads"] == QUERIES[site],
                        f"{label}.{arm}.{site}: wrong position/heads")
                require(audit["other_slices_unchanged"] is True
                        and audit["inserted_equals_cast_source"] is True,
                        f"{label}.{arm}.{site}: insertion not isolated/exact after cast")
                source = "q0" if arm in ("C", "D") else "q1"
                require(audit["source_sha256"] == source_hashes[source][site]
                        and audit["inserted_sha256"] == source_hashes[source][site],
                        f"{label}.{arm}.{site}: wrong cached query provenance")
                errors = vector(audit["insertion_error_per_item"], label, nonnegative=True)
                budgets = vector(audit["dtype_budget_per_item"], label, nonnegative=True)
                vector(audit["actual_change_l2_per_item"], label, nonnegative=True)
                per_head = audit["actual_change_l2_per_item_head"]
                require(isinstance(per_head, list) and len(per_head) == 2,
                        f"{label}: wrong directional head-norm shape")
                for row in per_head:
                    vector(row, label, len(QUERIES[site]), nonnegative=True)
            require(all(e <= b for e, b in zip(errors, budgets)),
                    f"{label}.{arm}.{site}: fidelity error exceeds recorded budget")
    require(audits["B"][MLP] == audits["D"][MLP] == audits["F"][MLP],
            f"{label}: original patch differs across B/D/F")
    require(value["model_forward_calls"] == 7, f"{label}: expected seven model forwards")
    changes = value["natural_query_change_l2"]
    require(set(changes) == set(QUERIES), f"{label}: missing query-change records")
    for site, heads in QUERIES.items():
        require(isinstance(changes[site], list) and len(changes[site]) == 2,
                f"{label}: query-change direction count")
        for row in changes[site]:
            vector(row, label, len(heads), nonnegative=True)


def independent_counts(records):
    """Deliberately do not call the producer's profile or resolution helpers."""
    counts = dict.fromkeys(PROFILE_NAMES, 0)
    resolved_count = 0
    for record in records:
        wins = dict.fromkeys(PROFILE_NAMES, True)
        resolved = True
        for direction in range(2):
            contrasts = {}
            for precision in PRECISIONS:
                cells = record["precisions"][precision]["cells"]
                t = cells["B"][direction] - cells["A"][direction]
                r = cells["D"][direction] - cells["C"][direction]
                s = cells["E"][direction] - cells["C"][direction]
                for v in (t, r, s):
                    finite(v, "recomputed contrast")
                contrasts[precision] = (t, r, s)
                tolerance = .25 * abs(t)
                wins["transfer"] &= abs(r) <= tolerance and abs(s-t) <= tolerance
                wins["joint_dependence"] &= abs(r) <= tolerance and abs(s) <= tolerance
                wins["preservation"] &= abs(r-t) <= tolerance and abs(s) <= tolerance
            t32, t64 = contrasts["float32"][0], contrasts["float64"][0]
            scale = min(abs(t32), abs(t64))
            same_sign = (t32 > 0 and t64 > 0) or (t32 < 0 and t64 < 0)
            discrepancy = max(abs(a-b) for a, b in zip(contrasts["float32"], contrasts["float64"]))
            resolved &= scale > 0 and same_sign and discrepancy <= .025 * scale
        resolved_count += resolved
        for name in PROFILE_NAMES:
            counts[name] += bool(resolved and wins[name])
    return counts, resolved_count


def verify(results, *, repository_root=REPOSITORY, source_root=None, model_snapshot=None):
    results, repository_root = Path(results), Path(repository_root)
    application = repository_root / "applications/makelov-2311.17030"
    require(not (results / "FAILED.json").exists(), "run has FAILED.json")
    hashes = load(results / "artifact_hashes.json")
    require(REQUIRED_FILES <= set(hashes), "artifact manifest omits a required result file")
    optional_verified, optional_unavailable = [], []
    for name, digest in hashes.items():
        require(Path(name).name == name, f"result artifact is not a basename: {name}")
        path = results / name
        if not path.exists() and name == "directions.npz":
            optional_unavailable.append("directions.npz")
        else:
            check_hash(path, digest, name)
    manifest, cases = load(results / "manifest.json"), load(results / "cases.json")
    summary, started = load(results / "summary.json"), load(results / "RUN_STARTED.json")
    manifest_hash = sha256(results / "manifest.json")
    require(summary["manifest_sha256"] == started["manifest_sha256"] == manifest_hash,
            "freeze hash disagrees between manifest, start and summary")
    check_hash(results / "cases.json", manifest["cases_sha256"], "frozen cases")
    check_hash(results / "records.jsonl", summary["records_sha256"], "summary records")
    if (results / "directions.npz").exists():
        check_hash(results / "directions.npz", manifest["directions_sha256"], "directions.npz")
        optional_verified.append("directions.npz")
    elif "directions.npz" not in optional_unavailable:
        optional_unavailable.append("directions.npz")

    prefix = "applications/makelov-2311.17030/"
    require(set(manifest["code_files_sha256"]) == {prefix+p for p in CODE_PATHS},
            "frozen code file set differs from declared instrument")
    for name, digest in manifest["code_files_sha256"].items():
        check_hash(relative_path(repository_root, name), digest, name)
    source_root = Path(source_root) if source_root else application / "artifacts/makelov_source"
    source_manifest = source_root / "source_manifest.json"
    if source_manifest.exists():
        require(load(source_manifest) == manifest["source"], "source manifest disagrees with frozen source")
    for name, digest in manifest["source"]["files"].items():
        path = relative_path(source_root, name)
        if path.exists():
            check_hash(path, digest, "source:"+name)
            optional_verified.append("source:"+name)
        else:
            optional_unavailable.append("source:"+name)
    if model_snapshot is None:
        hub = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home()/".cache/huggingface"))/"hub"))
        model_snapshot = hub / "models--gpt2/snapshots" / manifest["model_snapshot_revision"]
    for name, digest in manifest["model_files_sha256"].items():
        path = relative_path(Path(model_snapshot), name)
        if path.exists():
            check_hash(path, digest, "model:"+name)
            optional_verified.append("model:"+name)
        else:
            optional_unavailable.append("model:"+name)

    contract = manifest["scientific_contract"]
    for key, expected in (("kappa", .25), ("numerical_fraction", .025),
                          ("coverage", .8), ("alpha", .05), ("profiles", 3)):
        require(contract[key] == expected, f"unexpected scientific contract: {key}")
    require(manifest["query_heads"] == QUERIES and manifest["mlp_site"] == MLP
            and manifest["precisions"] == list(PRECISIONS), "manifest instrument differs")
    n = manifest["n_base_pairs"]
    require(type(n) is int and n > 0 and len(cases) == n, "case count differs from frozen n")
    sampling = manifest["sampling"]
    require(sampling["n_base_pair_draws"] == n and sampling["draw_cases_sha256"] == canonical_hash(cases),
            "sampling draw-count/hash mismatch")
    require(sampling["population_definition_sha256"] == canonical_hash(sampling["sampling_definition"]),
            "sampling-population definition hash mismatch")
    excluded = set(sampling["sampling_definition"]["excluded_prompts"])
    ids = [case["case_id"] for case in cases]
    require(all(isinstance(x, str) and x for x in ids) and len(set(ids)) == n,
            "draw IDs must be unique")
    for i, case in enumerate(cases):
        require(case["unique_draw_id"] == case["case_id"] and case["draw_index"] == i,
                "case order/draw ID mismatch")
        require(case["patterns"] == ["ABB", "BAB"] and len(case["prompts"]) == 2,
                "case is not a reciprocal pair")
        require(not any(p in excluded for p in case["prompts"]), "case is in frozen exclusions")
        require(len(case["token_ids"]) == 2 and len(case["token_ids"][0]) == len(case["token_ids"][1])
                and case["position"] == len(case["token_ids"][0])-1, "case position/alignment mismatch")
    records = jsonl(results / "records.jsonl")
    require([r["pair_id"] for r in records] == ids, "records omit/reorder/duplicate frozen draws")
    for precision in PRECISIONS:
        measurements = jsonl(results / f"measurements_{precision}.jsonl")
        require([r["pair_id"] for r in measurements] == ids, f"{precision}: measurements differ from cases")
        for case, record, measured in zip(cases, records, measurements):
            expected = dict(measured)
            del expected["pair_id"]
            require(record["precisions"][precision] == expected,
                    f"{case['case_id']}: records differ from measurements_{precision}")
            audit_precision(expected, case, precision)

    counts, resolved = independent_counts(records)
    reproduced = analyze_records(records)
    for name in PROFILE_NAMES:
        require(counts[name] == summary["profiles"][name]["successes"],
                f"independent count differs from summary: {name}")
    require(resolved == summary["n_resolved_pairs"], "independent resolution count differs")
    # Python/libm versions can round lgamma differently. This tolerance applies
    # only to reproducing analytic CP bounds, never to scientific cell predicates,
    # counts, status decisions, raw records or control identities.
    cp_tolerance, cp_max_error = 1e-12, 0.0
    for key, value in reproduced.items():
        if key == "profiles":
            require(set(summary[key]) == set(value), "summary profile names differ")
            for name, profile in value.items():
                stored = summary[key][name]
                require({k: v for k, v in profile.items() if k != "interval"}
                        == {k: v for k, v in stored.items() if k != "interval"},
                        f"analyzer reproduction differs from summary profile: {name}")
                bounds = vector(stored["interval"], f"{name} CP interval")
                error = max(abs(a-b) for a, b in zip(profile["interval"], bounds))
                cp_max_error = max(cp_max_error, error)
                require(error <= cp_tolerance, f"CP interval differs from reproduction: {name}")
            continue
        require(summary[key] == value, f"analyzer reproduction differs from summary: {key}")
    require(summary["model_forward_calls"] == 14*n and summary["stage"] == manifest["stage"]
            and summary["confirmation"] is (manifest["stage"] == "confirmation"),
            "summary stage/forward count differs")
    return {"verified": True, "mode": "records_only", "n_base_pairs": n,
            "independent_profile_counts": counts, "n_resolved_pairs": resolved,
            "profiles": summary["profiles"], "outcome": summary["outcome"],
            "cp_interval_reproduction_tolerance": cp_tolerance,
            "cp_interval_reproduction_max_abs_error": cp_max_error,
            "optional_original_inputs_verified": optional_verified,
            "optional_original_inputs_unavailable": optional_unavailable,
            "scope": ["No model was loaded or rerun. Counts independently recomputed; "
                      "CP intervals and detailed summary reproduced with the frozen analyzer.",
                      "Audits and query hashes are checked for internal consistency; "
                      "serialized records do not independently recover the activation tensors.",
                      "Query positions are serialized; the MLP absolute position is bound "
                      "by frozen code and case position, not a separate MLP audit field.",
                      "Artifact hashes establish byte consistency, not an independent "
                      "preregistration timestamp or truth of the recorded forwards."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.results)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
