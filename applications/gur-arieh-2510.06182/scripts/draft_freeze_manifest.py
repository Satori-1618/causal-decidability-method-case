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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def main():
    if git("status", "--porcelain"):
        raise SystemExit("refuse to draft from a dirty worktree; commit and verify the tooling first")
    a = json.loads((APPLICATION / "results/split_A/split_A_decision.json").read_text())
    b = json.loads((APPLICATION / "results/split_B/split_B_decision.json").read_text())
    if a["status"] != "PROCEED" or b["status"] != "PROCEED":
        raise SystemExit("split A or split B stopped: there is nothing to draft")
    spec = json.loads((APPLICATION / "results/split_B/manifest.json").read_text())["spec"]
    lock = json.loads((APPLICATION / "SOURCE_LOCK.json").read_text())
    freeze = b["split_B"]["freeze"]
    cell_key = b["cell_key"]
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
        "data_sha256": {name: sha256(APPLICATION / name) for name in (
            "SOURCE_LOCK.json", "PROPOSED_VALUES.json", "SPLIT_A_B_PROTOCOL.md",
            "results/split_A/split_A_decision.json", "results/split_B/split_B_decision.json",
            "results/split_B/artifact_hashes.json", "results/split_A/artifact_hashes.json")},
        "model": {"id": lock["model"]["id"], "revision": lock["model"]["revision"],
                  "files_sha256": {k: v["sha256"] for k, v in lock["model"]["content_hashes_gate1"].items()}},
        "entity_pools": "SOURCE_LOCK.json round1_entity_pools",
        "based_on_commit": git("rev-parse", "HEAD"),
        "split_B_result_commit": git("rev-list", "-1", "HEAD", "--",
                                      "applications/gur-arieh-2510.06182/results/split_B"),
    }
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
        "**This is a draft for the user's review. Nothing is frozen, no confirmation has run and "
        "the confirmation seed block (4,000,000 + i) is untouched.** It lists the values a freeze "
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
        "",
        "## Embedded draft manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2, allow_nan=False),
        "```",
        "",
    ]
    DRAFT_JSON.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    DRAFT.write_text("\n".join(lines))
    print(f"wrote {DRAFT.relative_to(APPLICATION)} and {DRAFT_JSON.relative_to(APPLICATION)} "
          "(DRAFT, NOT FROZEN)")


if __name__ == "__main__":
    main()
