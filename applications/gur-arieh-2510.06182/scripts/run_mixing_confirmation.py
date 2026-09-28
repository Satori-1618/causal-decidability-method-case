"""Run the frozen Round 1 confirmation once, offline, and fail closed.

The input is a reviewed JSON freeze manifest.  Before creating any result artifact this
script requires the final freeze markers, validates the statistical contract with the
Round 1 analyzer, and verifies every frozen code, data, and model-file sha256.  Model
execution then uses the already cached Gemma snapshot only: float32 on MPS with eager
attention, one frozen cell, seeds ``4_000_000 + draw_index``, and case ids
``confirmation-{draw_index:04d}``.  It stops at the N-th qualifying family or 2N draws.

The first four generated families retain full-vocabulary logits for an audit.  The first
32 are repeated on CPU float32 for gate 7.  Generic gates 1--5, gate 7, and the quota are
binding before ``analyze_confirmation`` is called.  A passing run is finally checked by
the independent records-only checker.

    python scripts/run_mixing_confirmation.py \
      --manifest FREEZE.json --upstream /path/to/mixing-mechs \
      --output results/confirmation
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

APPLICATION = Path(__file__).resolve().parents[1]
REPOSITORY = APPLICATION.parents[1]
sys.path.insert(0, str(APPLICATION / "src"))
import mixing_round1_analysis as ra  # noqa: E402

RUNNER_RELATIVE = "applications/gur-arieh-2510.06182/scripts/run_mixing_confirmation.py"
CHECKER_RELATIVE = "applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py"
SOURCE_LOCK_RELATIVE = "SOURCE_LOCK.json"
SPLIT_A_DECISION_RELATIVE = "results/split_A/split_A_decision.json"
SPLIT_B_DECISION_RELATIVE = "results/split_B/split_B_decision.json"
SEED_BASE = 4_000_000
CASE_ID_PREFIX = "confirmation"
GATE7_DRAW_INDICES = list(range(32))
AUDIT_DRAW_INDICES = [0, 1, 2, 3]
YIELD_FLOOR = 0.50
IDENTITY_TOLERANCE = 0.001
GATE7_T_TOLERANCE = 0.01
AUDIT_TOLERANCES = {"logsumexp": 1e-3, "answer_mass": 1e-5, "answer_logit": 1e-6}
RUNTIME_PACKAGES = ("torch", "transformers", "tokenizers", "numpy")
BINDING_GATES = (
    "1_model_hashes",
    "2_native_and_yield",
    "3_tokens",
    "4_hooks",
    "5_identity",
    "7_dtype_device",
)
REQUIRED_CODE = {
    RUNNER_RELATIVE,
    CHECKER_RELATIVE,
    "applications/makelov-2311.17030/src/query_route_analysis.py",
    "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py",
    "applications/gur-arieh-2510.06182/src/mixing_runner.py",
    "applications/gur-arieh-2510.06182/src/mixing_prompts.py",
    "applications/gur-arieh-2510.06182/src/mixing_round1_design.py",
    "applications/gur-arieh-2510.06182/src/mixing_splits.py",
}
REQUIRED_MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}


class ConfirmationError(ValueError):
    """The final freeze or a frozen artifact is invalid."""


def sha256(path, chunk_size=1 << 24):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def save_json(path, value, *, exclusive=True):
    with Path(path).open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def write_jsonl(path, values, *, exclusive=True):
    with Path(path).open("x" if exclusive else "w") as handle:
        for value in values:
            handle.write(json.dumps(value, allow_nan=False) + "\n")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def case_id(draw_index):
    if type(draw_index) is not int or draw_index < 0:
        raise ConfirmationError("draw_index must be a nonnegative integer")
    return f"{CASE_ID_PREFIX}-{draw_index:04d}"


def seed_for(draw_index):
    if type(draw_index) is not int or draw_index < 0:
        raise ConfirmationError("draw_index must be a nonnegative integer")
    return SEED_BASE + draw_index


def _require(condition, message):
    if not condition:
        raise ConfirmationError(message)


def _hash_map(value, label):
    _require(isinstance(value, dict) and value, f"manifest.{label} must be a nonempty mapping")
    for name, digest in value.items():
        _require(isinstance(name, str) and name, f"manifest.{label} has an invalid path")
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(c in "0123456789abcdef" for c in digest),
                 f"manifest.{label}[{name!r}] is not a lowercase sha256")
    return value


def _relative_path(root, name, label, *, basename=False):
    relative = Path(name)
    _require(not relative.is_absolute() and name not in ("", ".") and ".." not in relative.parts,
             f"unsafe {label} path: {name}")
    if basename:
        _require(relative.name == name, f"{label} path must be a basename: {name}")
    return Path(root) / relative


def _git(repository_root, *args, binary=False, check=True):
    completed = subprocess.run(
        ["git", *args], cwd=repository_root, capture_output=True,
        text=not binary, check=False)
    if check and completed.returncode:
        stderr = completed.stderr.decode(errors="replace") if binary else completed.stderr
        raise ConfirmationError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return completed


def verify_git_freeze(manifest_path, repository_root=REPOSITORY):
    """Bind the freeze bytes to HEAD and require a completely clean worktree."""
    repository_root = Path(repository_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    try:
        relative = manifest_path.relative_to(repository_root)
    except ValueError as error:
        raise ConfirmationError("freeze manifest must be inside the repository") from error
    top = _git(repository_root, "rev-parse", "--show-toplevel").stdout.strip()
    _require(Path(top).resolve() == repository_root,
             "repository_root is not the Git worktree root")
    relative_name = relative.as_posix()
    tracked = _git(repository_root, "ls-files", "--error-unmatch", "--", relative_name,
                   check=False)
    _require(tracked.returncode == 0, "freeze manifest must be tracked by Git at HEAD")
    head = _git(repository_root, "rev-parse", "HEAD").stdout.strip()
    committed = _git(repository_root, "show", f"HEAD:{relative_name}", binary=True).stdout
    disk = manifest_path.read_bytes()
    _require(disk == committed,
             "freeze manifest is not byte-identical to the version committed at HEAD")
    status = _git(repository_root, "status", "--porcelain", "--untracked-files=all",
                  "--ignore-submodules=none").stdout
    _require(not status, "confirmation requires a clean Git worktree")
    return {"git_head": head, "manifest_path": relative_name,
            "manifest_sha256": hashlib.sha256(disk).hexdigest(),
            "git_dirty": False, "manifest_bytes": disk}


def validate_frozen_manifest(manifest):
    """Validate final authorization plus the analyzer and execution contracts."""
    _require(isinstance(manifest, dict), "freeze manifest must be a JSON object")
    _require(manifest.get("stage") == "confirmation", "manifest.stage must equal 'confirmation'")
    _require(manifest.get("freeze_status") == "FROZEN",
             "manifest.freeze_status must equal 'FROZEN'")
    confirmation = manifest.get("confirmation")
    _require(isinstance(confirmation, dict), "manifest.confirmation missing")
    _require(confirmation.get("authorized") is True,
             "manifest.confirmation.authorized must be true")
    frozen = ra._validated_manifest(manifest)
    _require(type(confirmation.get("seed_base")) is int
             and confirmation["seed_base"] == SEED_BASE,
             "confirmation.seed_base must be the numeric value 4,000,000")
    _require(confirmation.get("seed_formula") == "seed_base + draw_index",
             "confirmation.seed_formula differs from the frozen formula")
    _require(confirmation.get("case_id_prefix") == CASE_ID_PREFIX,
             "confirmation.case_id_prefix must equal 'confirmation'")
    _require(type(confirmation.get("cap")) is int and confirmation["cap"] == 2 * frozen["N"],
             "confirmation.cap must equal 2N")
    _require(confirmation.get("gate7_draw_indices") == GATE7_DRAW_INDICES,
             "confirmation.gate7_draw_indices must be the first 32 draws")
    _require(confirmation.get("audit_full_logit_draw_indices") == AUDIT_DRAW_INDICES,
             "confirmation.audit_full_logit_draw_indices must be the first four draws")
    _require(manifest.get("application") == "gur-arieh-2510.06182" and manifest.get("round") == 1,
             "manifest names the wrong application or round")
    _require(type(manifest.get("t_entity")) is int and manifest["t_entity"] == 2
             and manifest.get("patch_positions") == [-1],
             "confirmation requires t_entity = 2 and the last-token patch only")
    _require(type(manifest.get("layer")) is int and manifest["layer"] >= 0,
             "manifest.layer must be a nonnegative integer")
    _require(isinstance(manifest.get("cell_key"), str) and manifest["cell_key"],
             "manifest.cell_key missing")
    execution = manifest.get("execution_contract")
    _require(isinstance(execution, dict), "manifest.execution_contract missing")
    _require(execution.get("device") == "mps" and execution.get("dtype") == "float32"
             and execution.get("attention") == "eager"
             and execution.get("layer") == manifest["layer"]
             and execution.get("patch_position") == "last token",
             "execution contract must freeze float32 MPS/eager at the declared layer and last token")
    _require(execution.get("identity_tolerance") == IDENTITY_TOLERANCE,
             "execution contract has the wrong identity tolerance")
    gate7 = execution.get("gate7")
    _require(isinstance(gate7, dict)
             and gate7.get("reference_device") == "cpu"
             and gate7.get("reference_dtype") == "float32"
             and gate7.get("tolerance_T") == GATE7_T_TOLERANCE
             and gate7.get("same_resolution_required") is True
             and gate7.get("same_labels_required") is True,
             "execution contract has the wrong gate-7 comparison")
    _require(isinstance(manifest.get("task_spec"), dict)
             and manifest["task_spec"].get("name") == manifest.get("task"),
             "manifest.task_spec missing or names a different task")
    _require(isinstance(manifest.get("entity_pools"), dict),
             "manifest.entity_pools missing")
    _require(isinstance(manifest.get("upstream"), dict), "manifest.upstream missing")
    _require(isinstance(manifest.get("development_environment"), dict),
             "manifest.development_environment missing")
    code = _hash_map(manifest.get("code_files_sha256"), "code_files_sha256")
    _require(REQUIRED_CODE <= set(code),
             "frozen code set omits the confirmation runner or one of its direct producers")
    _hash_map(manifest.get("data_sha256"), "data_sha256")
    model = manifest.get("model")
    _require(isinstance(model, dict), "manifest.model missing")
    _require(isinstance(model.get("id"), str) and model["id"]
             and isinstance(model.get("revision"), str) and model["revision"],
             "manifest.model id or revision missing")
    model_files = _hash_map(model.get("files_sha256"), "model.files_sha256")
    _require(REQUIRED_MODEL_FILES <= set(model_files),
             "frozen model hash set omits a required Gemma snapshot file")
    return frozen


def validate_runtime_environment(manifest, actual=None):
    """Require the scientifically relevant runtime versions used for development."""
    expected_environment = manifest.get("development_environment")
    _require(isinstance(expected_environment, dict),
             "manifest.development_environment missing")
    expected_packages = expected_environment.get("packages")
    _require(isinstance(expected_environment.get("python"), str)
             and isinstance(expected_packages, dict),
             "frozen development environment omits Python or package versions")
    for name in RUNTIME_PACKAGES:
        _require(isinstance(expected_packages.get(name), str) and expected_packages[name],
                 f"frozen development environment omits {name}")
    if actual is None:
        try:
            actual = {
                "python": platform.python_version(),
                "packages": {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES},
            }
        except importlib.metadata.PackageNotFoundError as error:
            raise ConfirmationError(f"required runtime package is not installed: {error}") from error
    _require(isinstance(actual, dict) and isinstance(actual.get("packages"), dict),
             "actual runtime report is malformed")
    expected = {"python": expected_environment["python"],
                "packages": {name: expected_packages[name] for name in RUNTIME_PACKAGES}}
    observed = {"python": actual.get("python"),
                "packages": {name: actual["packages"].get(name) for name in RUNTIME_PACKAGES}}
    _require(observed == expected,
             f"runtime versions differ from the frozen development environment: "
             f"expected {expected}, observed {observed}")
    return {"passed": True, **observed}


def _default_cache_root():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache/huggingface/hub"


def _snapshot_path(model, cache_root=None):
    cache = Path(cache_root) if cache_root is not None else _default_cache_root()
    return cache / f"models--{model['id'].replace('/', '--')}" / "snapshots" / model["revision"]


def _verify_one(path, expected, label):
    _require(Path(path).is_file(), f"missing frozen {label}: {path}")
    actual = sha256(path)
    _require(actual == expected, f"hash mismatch for frozen {label}: {path}")
    return actual


def validate_development_binding(manifest, application_root=APPLICATION):
    """Require every frozen empirical value to equal the hash-locked split decisions."""
    data = manifest["data_sha256"]
    required = {SPLIT_A_DECISION_RELATIVE, SPLIT_B_DECISION_RELATIVE}
    _require(required <= set(data),
             "frozen data set must include the split-A and split-B decisions")
    application_root = Path(application_root)
    split_a = load_json(application_root / SPLIT_A_DECISION_RELATIVE)
    split_b = load_json(application_root / SPLIT_B_DECISION_RELATIVE)
    _require(split_a.get("status") == "PROCEED" and split_a.get("stops") == []
             and split_b.get("status") == "PROCEED" and split_b.get("stops") == [],
             "split A or split B did not PROCEED")
    _require(manifest.get("cell_key") == "c4"
             and manifest["cell_key"] == split_a.get("selected") == split_b.get("cell_key"),
             "frozen cell_key is not c4 selected on split A and used on split B")
    split_b_result = split_b.get("split_B")
    freeze = split_b_result.get("freeze") if isinstance(split_b_result, dict) else None
    _require(isinstance(freeze, dict), "split-B decision has no freeze values")
    comparisons = {
        "cell": (manifest.get("cell"), freeze.get("cell")),
        "N": (manifest.get("N"), freeze.get("N")),
        "N_rule": (manifest.get("N_rule"), freeze.get("N_rule")),
        "rule.s_min": (manifest.get("rule", {}).get("s_min"), freeze.get("s_min")),
        "rule.d_min": (manifest.get("rule", {}).get("d_min"), freeze.get("d_min")),
        "rule.agreement_transfer_floor": (
            manifest.get("rule", {}).get("agreement_transfer_floor"),
            freeze.get("agreement_transfer_floor")),
        "anchors": (manifest.get("anchors"), freeze.get("anchors")),
        "mean_gate": (manifest.get("mean_gate"), freeze.get("mean_gate")),
        "development_gates": (manifest.get("development_gates"),
                              freeze.get("development_gates")),
    }
    for label, (actual, expected) in comparisons.items():
        _require(actual == expected, f"manifest {label} differs from the split-B freeze")
    return {"passed": True, "split_A_status": split_a["status"],
            "split_B_status": split_b["status"], "cell_key": manifest["cell_key"]}


def verify_frozen_hashes(manifest, *, repository_root=REPOSITORY,
                         application_root=APPLICATION, cache_root=None):
    """Verify every listed hash and the source-lock/model agreement."""
    repository_root, application_root = Path(repository_root), Path(application_root)
    code = manifest["code_files_sha256"]
    data = manifest["data_sha256"]
    model = manifest["model"]
    verified = {"code": {}, "data": {}, "model": {}}
    for name, expected in sorted(code.items()):
        path = _relative_path(repository_root, name, "code")
        verified["code"][name] = _verify_one(path, expected, "code file")
    for name, expected in sorted(data.items()):
        path = _relative_path(application_root, name, "data")
        verified["data"][name] = _verify_one(path, expected, "data file")

    development = validate_development_binding(manifest, application_root)

    _require(SOURCE_LOCK_RELATIVE in data,
             "frozen data set must include SOURCE_LOCK.json")
    source_lock_path = application_root / SOURCE_LOCK_RELATIVE
    source_lock = load_json(source_lock_path)
    locked_model = source_lock.get("model")
    _require(isinstance(locked_model, dict), "SOURCE_LOCK.json has no model lock")
    _require((model["id"], model["revision"])
             == (locked_model.get("id"), locked_model.get("revision")),
             "frozen model id or revision differs from SOURCE_LOCK.json")
    locked_files = locked_model.get("content_hashes_gate1")
    _require(isinstance(locked_files, dict) and locked_files,
             "SOURCE_LOCK.json has no gate-1 model hashes")
    locked_sha = {name: row.get("sha256") for name, row in locked_files.items()}
    _require(model["files_sha256"] == locked_sha,
             "frozen model hashes differ from SOURCE_LOCK.json")

    snapshot = _snapshot_path(model, cache_root)
    for name, expected in sorted(model["files_sha256"].items()):
        path = _relative_path(snapshot, name, "model", basename=True)
        verified["model"][name] = _verify_one(path, expected, "model file")
    return {
        "passed": True,
        "counts": {kind: len(rows) for kind, rows in verified.items()},
        "verified": verified,
        "snapshot": str(snapshot),
        "source_lock": source_lock,
        "development": development,
    }


def preflight(manifest_path, output=None, *, repository_root=REPOSITORY,
              application_root=APPLICATION, cache_root=None):
    """Read a final JSON freeze and verify all hashes without writing anything."""
    manifest_path = Path(manifest_path)
    _require(manifest_path.suffix.lower() == ".json" and manifest_path.is_file(),
             "--manifest must name an existing JSON file")
    if output is not None:
        output = Path(output)
        _require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
                 "output exists and is not an empty directory")
    git_freeze = verify_git_freeze(manifest_path, repository_root)
    manifest_bytes = git_freeze["manifest_bytes"]
    manifest = json.loads(manifest_bytes)
    frozen = validate_frozen_manifest(manifest)
    hashes = verify_frozen_hashes(
        manifest, repository_root=repository_root, application_root=application_root,
        cache_root=cache_root)
    final_git_freeze = verify_git_freeze(manifest_path, repository_root)
    _require(final_git_freeze == git_freeze,
             "Git HEAD or the committed freeze changed during preflight")
    return {"manifest": manifest, "manifest_bytes": manifest_bytes, "frozen": frozen,
            "hashes": hashes, "git_freeze": git_freeze, "manifest_path": manifest_path}


def write_artifact_hashes(output):
    """Write the checker's deliberately simple, flat artifact-hash mapping."""
    output = Path(output)
    values = {path.name: sha256(path) for path in sorted(output.iterdir())
              if path.is_file() and path.name != "artifact_hashes.json"}
    save_json(output / "artifact_hashes.json", values, exclusive=False)
    return values


def _rate(k, n):
    return {"k": k, "n": n, "rate": k / n if n else None}


def _tally(pairs):
    values = {}
    for key, passed in pairs:
        row = values.setdefault(str(key), [0, 0])
        row[0] += bool(passed)
        row[1] += 1
    return {key: _rate(*row) for key, row in sorted(values.items())}


def _full_logit_audit(npz_path, records, answer_ids, target_index):
    expected = [case_id(i) for i in AUDIT_DRAW_INDICES]
    result = {"checked": False, "expected_cases": expected, "tolerances": AUDIT_TOLERANCES}
    try:
        import numpy as np

        by_id = {record["case_id"]: record for record in records}
        with np.load(npz_path) as arrays:
            cases = list(arrays.files)
            if cases != expected or any(name not in by_id for name in expected):
                return {**result, "cases": cases, "passed": False,
                        "reason": "the stored audit cases differ from the frozen first four draws"}
            worst_lse = worst_mass = worst_logit = 0.0
            for name in expected:
                logits = arrays[name].astype("float64")
                record = by_id[name]
                ids = [answer_ids[group[target_index]] for group in record["matrix"]]
                top = float(logits.max())
                lse = top + math.log(float(np.exp(logits - top).sum()))
                answer = [float(logits[index]) for index in ids]
                mass = math.fsum(math.exp(value - lse) for value in answer)
                worst_lse = max(worst_lse, abs(lse - record["readout"]["logsumexp_full"]))
                worst_mass = max(worst_mass, abs(mass - record["answer_mass_full_vocab"]))
                worst_logit = max(worst_logit,
                                  max(abs(a - b) for a, b in zip(answer, record["answer_logits"])))
        passed = (worst_lse <= AUDIT_TOLERANCES["logsumexp"]
                  and worst_mass <= AUDIT_TOLERANCES["answer_mass"]
                  and worst_logit <= AUDIT_TOLERANCES["answer_logit"])
        return {**result, "checked": True, "cases": expected,
                "max_abs_logsumexp_difference": worst_lse,
                "max_abs_answer_mass_difference": worst_mass,
                "max_abs_answer_logit_difference": worst_logit, "passed": passed}
    except (KeyError, TypeError, ValueError, OSError) as error:
        return {**result, "passed": False, "reason": str(error)}


def generic_gate_table(manifest, records, references, gate3, hash_report, audit,
                       quota_counts, quota_met, execution):
    """Build the binding generic gate table from stored raw records and CPU repeats."""
    cell_key, cell = manifest["cell_key"], manifest["cell"]
    measured = [record for record in records if "matrix" in record]
    qualifying = [record for record in measured if record.get("qualifies") is True]
    gates = {
        "1_model_hashes": {"passed": hash_report.get("passed") is True,
                           "verified_counts": hash_report.get("counts")},
    }

    overall = _rate(len([r for r in records if r.get("qualifies") is True]), len(records))
    native = [(record, role, record["native"][role]) for record in measured
              for role in ("recipient", "donor")]
    agreement = [(row["target"], row["j"], row["donor_native"])
                 for record in measured for row in record.get("agreement", [])]
    gates["2_native_and_yield"] = {
        "passed": overall["rate"] is not None and overall["rate"] >= YIELD_FLOOR,
        "yield": overall, "floor": YIELD_FLOOR,
        "recipient_correct_by_i_N": _tally(
            (r["cell"]["i_N"], value.get("correct")) for r, role, value in native
            if role == "recipient"),
        "donor_correct_by_queried_position": _tally(
            (r["cell"]["i_P"], value.get("correct")) for r, role, value in native
            if role == "donor"),
        "first_token_is_answer_form": _tally(
            (role, value.get("first_token_is_answer_form")) for _, role, value in native),
        "readout_matches_generation": _tally(
            (role, value.get("readout_matches_generation")) for _, role, value in native),
        "agreement_donor_correct_by_target": _tally(
            (f"{target}={j}", value.get("correct")) for target, j, value in agreement),
    }

    failures = [(record.get("case_id"), record.get("technical", {}).get("failures", []))
                for record in records if record.get("technical", {}).get("passed") is not True]
    alignment = [(name, reasons) for name, reasons in failures
                 if any("differ from the design" in str(reason) for reason in reasons)]
    gates["3_tokens"] = {**gate3, "alignment_failures": alignment,
                         "passed": gate3.get("passed") is True and not alignment}

    hook_failed, design_failed, hook_counts = [], [], {}
    for record in measured:
        for name, passed in record["technical"]["checks"].items():
            if name.endswith("hook"):
                hook_counts[name] = hook_counts.get(name, 0) + 1
                if not passed:
                    hook_failed.append([record["case_id"], name])
            if name.endswith("design_indices") and not passed:
                design_failed.append([record["case_id"], name])
    gates["4_hooks"] = {
        "passed": not failures and not hook_failed and not design_failed and audit.get("passed") is True,
        "checks": hook_counts, "failed": hook_failed,
        "design_index_failures": design_failed, "technical_failures": failures,
        "full_logit_audit": audit,
    }

    identities = [record["identity"] for record in measured if record.get("identity")]
    maximum = max((row["max_abs_answer_logit_difference"] for row in identities), default=None)
    gates["5_identity"] = {
        "passed": len(identities) == len(measured) > 0 and maximum <= IDENTITY_TOLERANCE
                  and all(row["same_answer_argmax"] and row["same_generation"] for row in identities),
        "families": len(identities), "measured_families": len(measured),
        "max_abs_answer_logit_difference": maximum, "tolerance": IDENTITY_TOLERANCE,
        "same_argmax": sum(row["same_answer_argmax"] for row in identities),
        "same_generation": sum(row["same_generation"] for row in identities),
    }

    by_draw = {record.get("draw_index"): record for record in records}
    by_reference = {row.get("case_id"): row for row in references}
    rows = []
    for draw_index in GATE7_DRAW_INDICES:
        record = by_draw.get(draw_index)
        reference = by_reference.get(case_id(draw_index))
        if record is None or reference is None or "answer_logits" not in record:
            rows.append({"draw_index": draw_index, "case_id": case_id(draw_index), "missing": True})
            continue
        try:
            mps = ra.case_measures(record, cell, ra.CONTRACT["w"], manifest["rule"]["s_min"])
            cpu = ra.case_measures(reference["conflict"], cell, ra.CONTRACT["w"],
                                   manifest["rule"]["s_min"])
            rows.append({
                "draw_index": draw_index, "case_id": record["case_id"],
                "abs_T_difference": abs(mps["T"] - cpu["T"]),
                "same_resolution": mps["resolved"] == cpu["resolved"],
                "same_labels": mps["labels"] == cpu["labels"],
                "max_abs_answer_logit_difference": max(
                    abs(a - b) for a, b in zip(record["answer_logits"],
                                                reference["conflict"]["answer_logits"])),
                "hook_ok_cpu": reference.get("hook_ok") is True,
                "cpu_device": reference.get("device"), "cpu_dtype": reference.get("dtype"),
            })
        except (KeyError, TypeError, ValueError) as error:
            rows.append({"draw_index": draw_index, "case_id": record.get("case_id"),
                         "missing": True, "reason": str(error)})
    complete = [row for row in rows if not row.get("missing")]
    max_t = max((row["abs_T_difference"] for row in complete), default=None)
    execution_ok = (execution.get("main_device") == "mps"
                    and execution.get("main_dtype") == "torch.float32"
                    and execution.get("attention") == "eager")
    gate7_passed = (len(complete) == len(GATE7_DRAW_INDICES)
                    and max_t is not None and max_t <= GATE7_T_TOLERANCE
                    and execution_ok
                    and all(row["same_resolution"] and row["same_labels"]
                            and row["hook_ok_cpu"] and row["cpu_device"] == "cpu"
                            and row["cpu_dtype"] == "torch.float32" for row in complete))
    gates["7_dtype_device"] = {
        "passed": gate7_passed,
        "comparison": "MPS float32/eager against CPU float32/eager, conflict patch",
        "declared_draw_indices": GATE7_DRAW_INDICES, "compared": len(complete),
        "missing": [row["case_id"] for row in rows if row.get("missing")],
        "max_abs_T_difference": max_t, "tolerance": GATE7_T_TOLERANCE,
        "execution": execution, "rows": rows,
    }

    count = quota_counts.get(cell_key, {"generated": len(records), "qualifying": len(qualifying)})
    quota = {
        "passed": bool(quota_met and count["qualifying"] == manifest["N"]
                       and count["generated"] == len(records)
                       and len(records) <= manifest["confirmation"]["cap"]
                       and records and records[-1].get("qualifies") is True),
        "N": manifest["N"], "cap": manifest["confirmation"]["cap"],
        "counts": {cell_key: count}, "generator_reported_met": bool(quota_met),
    }
    stops = [f"gate {name} failed" for name in BINDING_GATES if gates[name]["passed"] is not True]
    if not quota["passed"]:
        stops.append("confirmation quota was not met within the frozen cap")
    return {
        "label": "CONFIRMATION: generic gates evaluated before the frozen analysis",
        "cell_key": cell_key, "families": len(records), "qualifying": len(qualifying),
        "gates": gates, "quota": quota,
        "status": "STOP" if stops else "PROCEED", "stops": stops,
    }


def _blank_gate_table(hash_report, gate3, reason, gate="3_tokens"):
    gates = {name: {"passed": None, "note": "not evaluated after an earlier STOP"}
             for name in BINDING_GATES}
    gates["1_model_hashes"] = {"passed": True, "verified_counts": hash_report["counts"]}
    gates[gate] = {**gate3, "passed": False, "reason": reason}
    return {
        "label": "CONFIRMATION: generic gates evaluated before the frozen analysis",
        "gates": gates, "quota": {"passed": None, "note": "not evaluated"},
        "status": "STOP", "stops": [f"gate {gate} failed: {reason}"],
    }


def _load_checker(repository_root=REPOSITORY):
    path = Path(repository_root) / CHECKER_RELATIVE
    spec = importlib.util.spec_from_file_location("mixing_confirmation_records_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze_and_check(manifest, records, output, *, repository_root=REPOSITORY):
    """Call the governed analyzer once, then invoke the independent records-only checker."""
    output = Path(output)
    summary = ra.analyze_confirmation(manifest, records)
    summary.update(
        manifest_sha256=sha256(output / "manifest.json"),
        records_sha256=sha256(output / "records.jsonl"),
        confirmation_gates_sha256=sha256(output / "confirmation_gates.json"),
        timings_sha256=sha256(output / "timings.json"),
    )
    save_json(output / "summary.json", summary)
    write_artifact_hashes(output)
    checker = _load_checker(repository_root)
    report = checker.verify(output, repository_root=repository_root)
    save_json(output / "checker_report.json", report)
    write_artifact_hashes(output)
    final_report = checker.verify(output, repository_root=repository_root)
    _require(final_report == report,
             "records-only checker report changed after indexing checker_report.json")
    return summary, final_report


def _finalize_stop(output, table, timings):
    output = Path(output)
    save_json(output / "timings.json", timings, exclusive=False)
    save_json(output / "confirmation_gates.json", table, exclusive=False)
    save_json(output / "STOP.json", {"status": "STOP", "level": "S1",
                                      "reasons": table["stops"], "stopped": now()})
    write_artifact_hashes(output)
    return {"status": "STOP", "reasons": table["stops"]}


def run_confirmation(manifest_path, upstream, output, *, cache_root=None,
                     repository_root=REPOSITORY, application_root=APPLICATION):
    """Execute a final confirmation.  This is the only function that loads the model."""
    output = Path(output).resolve()
    checked = preflight(manifest_path, output, repository_root=repository_root,
                        application_root=application_root, cache_root=cache_root)
    manifest, frozen, hash_report = checked["manifest"], checked["frozen"], checked["hashes"]

    # The external dependency is checked while the output is still untouched.
    from mixing_prompts import load_adapter
    adapter = load_adapter()
    upstream_report = adapter.check(upstream, hash_report["source_lock"])
    task = adapter.schema_spec(upstream, manifest["task"])
    _require(upstream_report == manifest["upstream"],
             "current upstream verification report differs from the frozen report")
    _require(task == manifest["task_spec"],
             "current upstream task specification differs from the frozen task_spec")
    runtime_report = validate_runtime_environment(manifest)
    final_git_freeze = verify_git_freeze(manifest_path, repository_root)
    _require(final_git_freeze == checked["git_freeze"],
             "Git HEAD or the committed freeze changed after preflight")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.json").open("xb") as handle:
        handle.write(checked["manifest_bytes"])
    manifest_hash = sha256(output / "manifest.json")
    save_json(output / "RUN_STARTED.json", {
        "manifest_sha256": manifest_hash, "started": now(),
        "git_head": checked["git_freeze"]["git_head"],
        "manifest_path": checked["git_freeze"]["manifest_path"],
        "git_dirty": False,
        "seed_base": SEED_BASE, "case_id_prefix": CASE_ID_PREFIX,
        "frozen_hashes_verified": hash_report["counts"], "upstream": upstream_report,
        "runtime_versions": runtime_report,
    })
    (output / "records.jsonl").open("x").close()
    timings = {"started": now(), "execution": "float32 MPS/eager; CPU float32/eager gate 7"}

    try:
        import torch
        import mixing_runner as mr
        from mixing_splits import generate_families
        from transformers import AutoTokenizer

        snapshot = Path(hash_report["snapshot"])
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        pools, dropped, context_ids, answer_ids = mr.round1_pools(tokenizer, task)
        pool_record = mr.pools_record(pools, dropped, context_ids, answer_ids, task)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True)
        gate3 = {
            "pools_equal_the_manifest": json.loads(json.dumps(pool_record)) == manifest["entity_pools"],
            "pools_equal_the_lock": json.loads(json.dumps(pool_record))
                                    == hash_report["source_lock"].get("round1_entity_pools"),
            "pools_kept": {category: len(values) for category, values in pools.items()},
            "pools_dropped": dropped, "dropped_prefix": rendered[:5],
        }
        gate3["passed"] = (gate3["pools_equal_the_manifest"] and gate3["pools_equal_the_lock"]
                           and rendered[:5] == "<bos>"
                           and all(len(values) >= frozen["n"] + 2 for values in pools.values()))
        if not gate3["passed"]:
            table = _blank_gate_table(hash_report, gate3, "token pools or chat prefix differ from the lock")
            return _finalize_stop(output, table, timings)
        if not torch.backends.mps.is_available():
            table = _blank_gate_table(hash_report, gate3, "MPS is unavailable", gate="7_dtype_device")
            return _finalize_stop(output, table, timings)

        t0 = time.perf_counter()
        model, _ = mr.load_model(snapshot, torch.float32, "mps", "eager")
        timings["load_fp32_mps_seconds"] = time.perf_counter() - t0
        main_dtype = str(next(model.parameters()).dtype)
        main_device = next(model.parameters()).device.type
        execution = {"main_device": main_device, "main_dtype": main_dtype, "attention": "eager"}
        if execution != {"main_device": "mps", "main_dtype": "torch.float32", "attention": "eager"}:
            del model
            torch.mps.empty_cache()
            table = _blank_gate_table(hash_report, gate3, "model did not load as float32 MPS/eager",
                                      gate="7_dtype_device")
            return _finalize_stop(output, table, timings)

        runner = mr.Runner(model, tokenizer, task, pools, context_ids, answer_ids,
                           manifest["layer"], diagnostic_layer=None)
        audit_arrays, records = {}, []

        def run_family(*, case_id, draw_index, seed, cell_key, cell):
            _require(case_id == globals()["case_id"](draw_index)
                     and seed == seed_for(draw_index), "generator violated the frozen coordinates")
            try:
                return runner.family(
                    case_id=case_id, draw_index=draw_index, seed=seed,
                    cell_key=cell_key, cell=cell, rng=random.Random(seed), n=frozen["n"],
                    audit=draw_index in AUDIT_DRAW_INDICES, identity=True)
            except mr.TechnicalError as error:
                return ({"case_id": case_id, "draw_index": draw_index, "seed": seed,
                         "cell_key": cell_key, "cell": dict(cell), "qualifies": False,
                         "technical": {"passed": False, "failures": [str(error)], "checks": {}}}, None)

        def on_record(record, full_logits):
            if full_logits is not None:
                audit_arrays[record["case_id"]] = full_logits.detach().cpu().numpy()
            with (output / "records.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
            records.append(record)
            print(f"{record['case_id']} qualifies={record['qualifies']} "
                  f"technical={record['technical']['passed']} "
                  f"{record.get('runtime_seconds', 0):.2f}s", flush=True)

        t0 = time.perf_counter()
        counts, met = generate_families(
            run_family, [[manifest["cell_key"], manifest["cell"]]],
            seed_base=SEED_BASE, quota=frozen["N"], cap=2 * frozen["N"], per_cell=False,
            prefix=CASE_ID_PREFIX, on_record=on_record)
        timings["families_fp32_mps_seconds"] = time.perf_counter() - t0
        timings["counts"], timings["quota_met"] = counts, met
        import numpy as np
        np.savez_compressed(output / "audit_full_logits.npz", **audit_arrays)
        try:
            timings["mps_driver_gib_after_families"] = torch.mps.driver_allocated_memory() / 2 ** 30
        except RuntimeError:
            timings["mps_driver_gib_after_families"] = None
        del runner, model
        torch.mps.empty_cache()

        t0 = time.perf_counter()
        cpu_model, _ = mr.load_model(snapshot, torch.float32, "cpu", "eager")
        timings["load_fp32_cpu_seconds"] = time.perf_counter() - t0
        cpu_runner = mr.Runner(cpu_model, tokenizer, task, pools, context_ids, answer_ids,
                               manifest["layer"], diagnostic_layer=None)
        references = []
        t0 = time.perf_counter()
        with (output / "gate7_cpu_reference.jsonl").open("x") as handle:
            for record in records:
                if record["draw_index"] in GATE7_DRAW_INDICES and "matrix" in record:
                    reference = cpu_runner.conflict_only(record)
                    references.append(reference)
                    handle.write(json.dumps(reference, allow_nan=False) + "\n")
                    handle.flush()
        timings["gate7_cpu_reference_seconds"] = time.perf_counter() - t0
        del cpu_runner, cpu_model

        audit = _full_logit_audit(output / "audit_full_logits.npz", records, answer_ids,
                                  manifest["t_entity"] - 1)
        table = generic_gate_table(manifest, records, references, gate3, hash_report, audit,
                                   counts, met, execution)
        timings["finished_model_work"] = now()
        save_json(output / "timings.json", timings)
        save_json(output / "confirmation_gates.json", table)
        if table["status"] != "PROCEED":
            save_json(output / "STOP.json", {"status": "STOP", "level": "S1",
                                              "reasons": table["stops"], "stopped": now()})
            write_artifact_hashes(output)
            return {"status": "STOP", "reasons": table["stops"]}

        summary, checker_report = analyze_and_check(
            manifest, records, output, repository_root=repository_root)
        return {"status": "COMPLETE", "run_status": summary["run_status"],
                "level": summary["level"], "statuses": summary["statuses"],
                "checker": checker_report}
    except Exception as error:
        timings["failed"] = now()
        timings["error_type"] = type(error).__name__
        save_json(output / "timings.json", timings, exclusive=False)
        save_json(output / "FAILED.json", {"status": "FAILED", "failed": now(),
                                            "error_type": type(error).__name__,
                                            "reason": str(error)}, exclusive=False)
        write_artifact_hashes(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--upstream", default=os.environ.get("MIXING_MECHS_UPSTREAM"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default=None,
                        help="Hugging Face hub cache root; defaults to HF_HUB_CACHE/HF_HOME")
    args = parser.parse_args()
    if not args.upstream:
        parser.exit(2, "pass --upstream or set MIXING_MECHS_UPSTREAM\n")
    try:
        result = run_confirmation(args.manifest, args.upstream, args.output, cache_root=args.cache)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"FAILED: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] == "STOP":
        parser.exit(1, "STOP: a binding generic gate or the quota failed\n")


if __name__ == "__main__":
    main()
