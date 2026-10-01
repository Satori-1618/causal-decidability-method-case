"""Write FREEZE_DRAFT.md: the values a freeze WOULD fix, for the user's review. NOT A FREEZE.

Reads the committed split-A and split-B decisions (development data) and writes, in the
application directory, a draft freeze manifest: the selected cell, the split-B anchors,
d, s_min, delta, the final N and its adequacy label, the contract (kappa, coverage,
unresolved and resolution rules, alpha), the gates, the confirmation seed block and its
gate-7 cases, and the code, data and model hashes. The embedded manifest has the shape
the analyzer's confirmation step reads, and the script checks it with the analyzer's
manifest validation and the records-only checker's N rule. It writes no result file and
touches no seed.

    python scripts/draft_freeze_manifest.py
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

APPLICATION = Path(__file__).resolve().parents[1]
REPOSITORY = APPLICATION.parents[1]
sys.path.insert(0, str(APPLICATION / "src"))
import mixing_round1_analysis as ra  # noqa: E402

DRAFT = APPLICATION / "FREEZE_DRAFT.md"
DRAFT_JSON = APPLICATION / "FREEZE_DRAFT.json"
CODE = [
    "applications/makelov-2311.17030/src/query_route_analysis.py",
    "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py",
    "applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py",
    "applications/gur-arieh-2510.06182/src/mixing_runner.py",
    "applications/gur-arieh-2510.06182/src/mixing_prompts.py",
    "applications/gur-arieh-2510.06182/src/mixing_round1_design.py",
    "applications/gur-arieh-2510.06182/src/mixing_splits.py",
    "applications/gur-arieh-2510.06182/src/mixing_pilot_summary.py",
    "applications/gur-arieh-2510.06182/scripts/run_mixing_split.py",
    "applications/gur-arieh-2510.06182/scripts/run_mixing_confirmation.py",
    "applications/gur-arieh-2510.06182/scripts/draft_freeze_manifest.py",
    "applications/gur-arieh-2510.06182/scripts/run_mixing_pilot.py",
    "applications/gur-arieh-2510.06182/scripts/lock_sources.py",
]
DEVELOPMENT_FILES = {
    "SOURCE_LOCK.json": "source_lock_sha256",
    "PROPOSED_VALUES.json": "proposed_values_sha256",
    "SPLIT_A_B_PROTOCOL.md": "protocol_sha256",
}
# The checker is intentionally hardened after development.  Every other producer shared
# with split B must remain byte-identical so the confirmation cannot silently change the
# prompts, design, patching, measurements, or statistical contract used to fit its values.
DEVELOPMENT_PRODUCER_EXCEPTIONS = {
    "applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py",
}
DIRECT_DEVELOPMENT_DATA = (
    "results/split_A/manifest.json",
    "results/split_A/records.jsonl",
    "results/split_A/split_A_decision.json",
    "results/split_A/artifact_hashes.json",
    "results/split_B/manifest.json",
    "results/split_B/records.jsonl",
    "results/split_B/split_B_decision.json",
    "results/split_B/artifact_hashes.json",
)
EXPECTED_DEVELOPMENT = {
    "split_A": {
        "result_commit": "fa83ffffa08368f2dddab668feea2a4ffd74211f",
        "artifact_index_sha256": "2bab7554b8821769e3a4dc84e5bd55dec118f18be7abb6ae33eb5376b70c046d",
        "manifest_sha256": "5d972e6b8bba589590cf1765ebdf785cabf377abe409c8895381dc52751c719c",
        "records_sha256": "3d0e7dc0d4b13e30e20892fa0693bb53c89277d56252af9681effd89bd279b22",
        "decision_sha256": "030a11c071c3859b3c2f1027f13e10a21bb17993825c63ab63142c13c80e1db3",
    },
    "split_B": {
        "result_commit": "a24fd21681258538fc42b822941d06d7888451f5",
        "artifact_index_sha256": "bc7fa145a2cfbd622daa9cd42fc410a2dc192a64824f6dc136bb834bccd0fcab",
        "manifest_sha256": "387a0b6e0f485585b3dae14b48f6bea8d93743341418f195a5304c7840227c2e",
        "records_sha256": "ebfba9b6cb9d96b6f59dd347cd1a49351d8f9e7017387bf6a84e04d6170f44dc",
        "decision_sha256": "319d9820ac1622e7c80d4773feb4e3dc153aba248df0e9de3010ba0a213682b9",
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def git_blob(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPOSITORY)


def verify_result(stage):
    """Verify a committed development bundle before using any value from it."""
    relative = f"applications/gur-arieh-2510.06182/results/{stage}"
    directory = REPOSITORY / relative
    index = json.loads((directory / "artifact_hashes.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    data = index.get("data")
    producers = index.get("producers")
    if not isinstance(data, dict) or not isinstance(producers, dict):
        raise SystemExit(f"{stage}: artifact index is not schema v2")
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.name != "artifact_hashes.json"}
    if set(data) != actual:
        raise SystemExit(f"{stage}: artifact index does not exactly cover the result directory")
    for name, digest in data.items():
        if sha256(directory / name) != digest:
            raise SystemExit(f"{stage}: artifact hash mismatch for {name}")
    run_head = manifest.get("git_head")
    if manifest.get("git_dirty") is not False or index.get("git_head") != run_head:
        raise SystemExit(f"{stage}: run was dirty or index and manifest disagree on git head")
    if producers != manifest.get("producers_sha256"):
        raise SystemExit(f"{stage}: producer maps differ between index and manifest")
    for path, digest in producers.items():
        if hashlib.sha256(git_blob(run_head, path)).hexdigest() != digest:
            raise SystemExit(f"{stage}: producer hash does not match its recorded git tree: {path}")
    started = json.loads((directory / "RUN_STARTED.json").read_text())
    if started.get("manifest_sha256") != sha256(directory / "manifest.json"):
        raise SystemExit(f"{stage}: RUN_STARTED does not bind this manifest")
    result_commit = git("rev-list", "-1", "HEAD", "--", relative)
    if not result_commit:
        raise SystemExit(f"{stage}: result directory is not committed")
    for older, newer in ((run_head, result_commit), (result_commit, git("rev-parse", "HEAD"))):
        if subprocess.call(["git", "merge-base", "--is-ancestor", older, newer], cwd=REPOSITORY):
            raise SystemExit(f"{stage}: git lineage is not run-code -> result -> current HEAD")
    verified = {
        "run_code_commit": run_head,
        "result_commit": result_commit,
        "manifest_sha256": sha256(directory / "manifest.json"),
        "records_sha256": sha256(directory / "records.jsonl"),
        "decision_sha256": sha256(directory / f"{stage}_decision.json"),
        "artifact_index_sha256": sha256(directory / "artifact_hashes.json"),
        "manifest": manifest,
    }
    for key, expected in EXPECTED_DEVELOPMENT[stage].items():
        if verified.get(key) != expected:
            raise SystemExit(f"{stage}: reviewed {key} changed; expected {expected}, got {verified.get(key)}")
    return verified


def verify_confirmation_block_unused():
    confirmation = APPLICATION / "results/confirmation"
    if confirmation.exists():
        raise SystemExit("a confirmation result path already exists; refuse to draft")
    for path in (APPLICATION / "results").rglob("*.jsonl"):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                seed = json.loads(line).get("seed")
            except (json.JSONDecodeError, AttributeError) as error:
                raise SystemExit(f"cannot audit seeds in {path}:{line_number}: {error}") from error
            if type(seed) is int and 4_000_000 <= seed < 5_000_000:
                raise SystemExit(f"confirmation-block seed already present in {path}:{line_number}")


def frozen_model_from_lock(lock):
    """Return the compact model lock consumed by the confirmation runner."""
    try:
        model = lock["model"]
        files = {name: row["sha256"] for name, row in model["content_hashes_gate1"].items()}
        result = {"id": model["id"], "revision": model["revision"], "files_sha256": files}
    except (KeyError, TypeError, AttributeError) as error:
        raise SystemExit(f"SOURCE_LOCK.json has no complete gate-1 model lock: {error}") from error
    return result


def frozen_model_from_split(manifest_b):
    """Return split B's independently recorded model identity and content hashes."""
    try:
        model = manifest_b["model"]
        report = model["hashes"]
        if report.get("passed") is not True or report.get("problems") != []:
            raise SystemExit("split B did not record a passing model-hash gate")
        files = {name: row["sha256"] for name, row in report["files"].items()}
        result = {"id": model["id"], "revision": model["revision"], "files_sha256": files}
    except (KeyError, TypeError, AttributeError) as error:
        raise SystemExit(f"split B has no complete model-hash record: {error}") from error
    return result


def verify_development_surface(manifest_b):
    """Require all current inputs reused by confirmation to match verified split B."""
    for relative, manifest_key in DEVELOPMENT_FILES.items():
        expected = manifest_b.get(manifest_key)
        actual = sha256(APPLICATION / relative)
        if actual != expected:
            raise SystemExit(
                f"{relative} differs from the verified split-B manifest: "
                f"expected {expected}, got {actual}")

    producers = manifest_b.get("producers_sha256")
    if not isinstance(producers, dict) or not producers:
        raise SystemExit("split B has no producer hash map")
    for relative, expected in sorted(producers.items()):
        if relative in DEVELOPMENT_PRODUCER_EXCEPTIONS:
            continue
        path = REPOSITORY / relative
        if not path.is_file():
            raise SystemExit(f"missing shared split-B producer: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(
                f"shared development producer differs from split B: {relative}; "
                f"expected {expected}, got {actual}")

    lock = json.loads((APPLICATION / "SOURCE_LOCK.json").read_text())
    model = frozen_model_from_lock(lock)
    if model != frozen_model_from_split(manifest_b):
        raise SystemExit("the current source-lock model differs from the verified split-B model")
    if lock.get("round1_entity_pools") != manifest_b.get("entity_pools"):
        raise SystemExit("the current source-lock entity pools differ from verified split B")

    spec = manifest_b.get("spec")
    task = manifest_b.get("task_spec")
    upstream = manifest_b.get("upstream")
    environment = manifest_b.get("environment")
    if (not isinstance(spec, dict) or not isinstance(task, dict)
            or task.get("name") != spec.get("task")):
        raise SystemExit("split B has no self-consistent frozen task specification")
    if not isinstance(upstream, dict) or not isinstance(upstream.get("check"), dict):
        raise SystemExit("split B has no frozen upstream verification report")
    if not isinstance(environment, dict) or not isinstance(environment.get("packages"), dict):
        raise SystemExit("split B has no frozen development environment")
    return lock, model


def validate_generated_provenance(manifest, manifest_b, provenance_a, provenance_b, head):
    """Fail before writing if the draft does not carry the verified lineage verbatim."""
    expected = {"split_A": provenance_a, "split_B": provenance_b}
    if manifest.get("development_provenance") != expected:
        raise SystemExit("generated development_provenance differs from the verified results")
    if manifest.get("based_on_commit") != head:
        raise SystemExit("generated based_on_commit differs from the clean drafting commit")
    for key in ("task_spec", "entity_pools", "development_environment"):
        source_key = "environment" if key == "development_environment" else key
        if manifest.get(key) != manifest_b.get(source_key):
            raise SystemExit(f"generated {key} differs from verified split B")
    if manifest.get("upstream") != manifest_b.get("upstream", {}).get("check"):
        raise SystemExit("generated upstream report differs from verified split B")
    missing = set(DIRECT_DEVELOPMENT_DATA) - set(manifest.get("data_sha256", {}))
    if missing:
        raise SystemExit(f"generated data hash set omits direct development artifacts: {sorted(missing)}")


def main():
    if git("status", "--porcelain"):
        raise SystemExit("refuse to draft from a dirty worktree; commit and verify the tooling first")
    if DRAFT.exists() or DRAFT_JSON.exists():
        raise SystemExit("refuse to overwrite an existing freeze draft")
    verify_confirmation_block_unused()
    provenance_a = verify_result("split_A")
    provenance_b = verify_result("split_B")
    a = json.loads((APPLICATION / "results/split_A/split_A_decision.json").read_text())
    b = json.loads((APPLICATION / "results/split_B/split_B_decision.json").read_text())
    if a["status"] != "PROCEED" or b["status"] != "PROCEED":
        raise SystemExit("split A or split B stopped: there is nothing to draft")
    provenance_a.pop("manifest")
    manifest_b = provenance_b.pop("manifest")
    _lock, frozen_model = verify_development_surface(manifest_b)
    spec = manifest_b["spec"]
    selected = manifest_b["spec"].get("selected_on_split_A", {})
    if (selected.get("cell") != a.get("selected")
            or selected.get("decision_sha256") != provenance_a["decision_sha256"]):
        raise SystemExit("split B does not bind the committed split-A decision")
    if provenance_b["run_code_commit"] != provenance_a["result_commit"]:
        raise SystemExit("split B was not run from the committed split-A result tree")
    freeze = b["split_B"]["freeze"]
    cell_key = b["cell_key"]
    head = git("rev-parse", "HEAD")
    manifest = {
        "schema_version": 2, "application": "gur-arieh-2510.06182", "round": 1,
        "stage": "confirmation (DRAFT, NOT FROZEN)", "freeze_status": "DRAFT",
        "n_groups": spec["n"], "task": spec["task"],
        "t_entity": 2, "layer": spec["layer"], "patch_positions": [-1],
        "cell_key": cell_key, "cell": freeze["cell"], "N": freeze["N"], "N_rule": freeze["N_rule"],
        "rule": {**ra.CONTRACT, "s_min": freeze["s_min"], "d_min": freeze["d_min"],
                 "agreement_transfer_floor": freeze["agreement_transfer_floor"]},
        "anchors": freeze["anchors"], "mean_gate": freeze["mean_gate"],
        "development_gates": freeze["development_gates"],
        "execution_contract": {
            "device": spec["device"], "dtype": spec["dtype"], "attention": spec["attention"],
            "layer": spec["layer"], "patch_position": "last token",
            "identity_tolerance": spec["identity_tolerance"],
            "gate7": {"reference_device": spec["gate7_reference"]["device"],
                      "reference_dtype": spec["gate7_reference"]["dtype"],
                      "tolerance_T": spec["gate7_tolerance_T"],
                      "same_resolution_required": True, "same_labels_required": True},
        },
        "confirmation": {
            "authorized": False,
            "seed_base": 4_000_000,
            "seed_formula": "seed_base + draw_index",
            "case_id_prefix": "confirmation",
            "order": f"family i uses cell {cell_key} and seed 4,000,000 + i; "
                     "generation stops at the N-th qualifying family",
            "cap": 2 * freeze["N"],
            "gate7_draw_indices": list(range(32)),
            "audit_full_logit_draw_indices": [0, 1, 2, 3],
            "execution": "float32 on MPS, eager attention; primary answer-form readout; combined resolution gate"},
        "code_files_sha256": {path: sha256(REPOSITORY / path) for path in CODE},
        "data_sha256": {name: sha256(APPLICATION / name) for name in
                        (*DEVELOPMENT_FILES, *DIRECT_DEVELOPMENT_DATA)},
        "model": frozen_model,
        "task_spec": manifest_b["task_spec"],
        "entity_pools": manifest_b["entity_pools"],
        "upstream": manifest_b["upstream"]["check"],
        "development_environment": manifest_b["environment"],
        "development_provenance": {"split_A": provenance_a, "split_B": provenance_b},
        "based_on_commit": head,
    }
    validate_generated_provenance(manifest, manifest_b, provenance_a, provenance_b, head)
    ra._validated_manifest(manifest)
    spec_c = importlib.util.spec_from_file_location("checker", APPLICATION / "scripts/check_mixing_round1_records.py")
    checker = importlib.util.module_from_spec(spec_c)
    spec_c.loader.exec_module(checker)
    rule_N, powered, status = checker.rule_for(freeze["N_rule"]["unresolved_rate_B"])
    if (rule_N, powered, status) != (freeze["N"], freeze["N_rule"]["adequacy_powered"], "PROCEED"):
        raise SystemExit("the checker's N rule disagrees with the draft")
    anchors = freeze["anchors"]
    lines = [
        "# Freeze manifest: DRAFT, NOT FROZEN",
        "",
        "**This is a draft for the user's review. Nothing is frozen. No confirmation artifacts or "
        "4,000,000-series seeds exist in this repository snapshot; this does not establish that no "
        "external run occurred.** It lists the values a freeze "
        "would fix, all taken from split B (development data) except the cell, which split A "
        "selected. Generated by `scripts/draft_freeze_manifest.py`; the embedded manifest passes "
        "the analyzer's manifest validation and the records-only checker's N rule.",
        "",
        "| value | draft |",
        "|---|---|",
        f"| selected cell | {cell_key} {json.dumps(freeze['cell'])} (split A, margin {a['margin_over_second']:.4f} over the second) |",
        f"| T_W | {anchors['T_W']:.6f} |",
        f"| T_A | {anchors['T_A']:.6f} |",
        f"| d | {anchors['d']:.6f} (d_min {freeze['d_min']}) |",
        f"| mean q of split B (P/L/R) | {' / '.join(f'{x:.6f}' for x in anchors['q_bar_B'])} (m_B = {anchors['m_B']}) |",
        f"| s_min | {freeze['s_min']} ({freeze['s_min_source']}) |",
        f"| δ | {freeze['mean_gate']['delta']:.6f} (m = {freeze['mean_gate']['m']}, N = {freeze['mean_gate']['N']}, "
        f"{freeze['mean_gate']['resamples']} pairs, seed {freeze['mean_gate']['seed']}, false-INVALID rate "
        f"{freeze['mean_gate']['false_invalid_rate']}) |",
        f"| split B unresolved rate | {freeze['N_rule']['unresolved_rate_B']:.4f} |",
        f"| N | {freeze['N']} ({freeze['N_rule']['branch']}; adequacy power {freeze['N_rule']['adequacy_power']:.3f}, "
        f"exclusion power {freeze['N_rule']['exclusion_power']:.3f}) |",
        f"| adequacy label | {'powered' if freeze['N_rule']['adequacy_powered'] else 'NOT powered'} |",
        f"| κ, coverage, α | {ra.CONTRACT['kappa']}, {ra.CONTRACT['coverage']}, {ra.CONTRACT['alpha']} "
        f"(α/4 = {ra.CONTRACT['label_tail']} per tail, two profiles) |",
        f"| unresolved rule | {ra.CONTRACT['unresolved_rule']} |",
        f"| resolution rule | {ra.CONTRACT['resolution_rule']} (floor {ra.CONTRACT['answer_mass_floor']}) |",
        f"| development gates (split B) | resolution {freeze['development_gates']['resolution_rate_B']:.4f}, "
        f"agreement resolution {freeze['development_gates']['agreement_resolution_rate_B']:.4f}, "
        f"agreement transfer {freeze['development_gates']['agreement_transfer_rate_B']:.4f} |",
        "| confirmation seeds | 4,000,000 + i in the selected cell; stop at the N-th qualifying family; cap 2N |",
        "| confirmation gate-7 cases | draw indices 0–31 (the first 32 generated families), CPU float32 |",
        f"| runner | `src/mixing_runner.py` sha256 {manifest['code_files_sha256']['applications/gur-arieh-2510.06182/src/mixing_runner.py'][:16]}… |",
        f"| checker | `scripts/check_mixing_round1_records.py` sha256 {manifest['code_files_sha256']['applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py'][:16]}… |",
        f"| based on commit | `{manifest['based_on_commit']}` |",
        f"| split-A run/result commits | `{provenance_a['run_code_commit'][:12]}` / "
        f"`{provenance_a['result_commit'][:12]}` |",
        f"| split-B run/result commits | `{provenance_b['run_code_commit'][:12]}` / "
        f"`{provenance_b['result_commit'][:12]}` |",
        "",
        "## Embedded draft manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2, allow_nan=False),
        "```",
        "",
    ]
    with DRAFT_JSON.open("x") as handle:
        handle.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    with DRAFT.open("x") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {DRAFT.relative_to(APPLICATION)} and {DRAFT_JSON.relative_to(APPLICATION)} "
          "(DRAFT, NOT FROZEN)")


if __name__ == "__main__":
    main()
