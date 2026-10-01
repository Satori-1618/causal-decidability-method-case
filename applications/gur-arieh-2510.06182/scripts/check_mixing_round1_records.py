"""Reproduce a Round 1 confirmation from the frozen manifest and raw records only.

Standard library; no model is loaded and no runner is imported. Everything is
recomputed here, separately from the analyzer: the softmax over the primary answer-form
logits, the q-map and T per case, resolution (S >= s_min and answer-token mass >= 0.5),
both profile statuses under the declared unresolved rule, the technical and design-index
checks, the mean-consistency gate, the level, whether the between-case sentence is
earned, and the final N with its adequacy label from the N rule at split B's unresolved
rate. The paper's in-context readout, kept under a record's ``descriptive`` field, is
never read. Only the Clopper-Pearson intervals come from the repository's existing
helper, and that helper's file must carry the hash the manifest froze. The result must
equal summary.json exactly (floats within 1e-12).

    python scripts/check_mixing_round1_records.py --results results/<confirmation>
"""
import argparse
import array
import ast
import hashlib
import json
import math
import statistics
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

APPLICATION = Path(__file__).resolve().parents[1]
REPOSITORY = APPLICATION.parents[1]
HELPER = "applications/makelov-2311.17030/src/query_route_analysis.py"
REQUIRED_CODE = {HELPER,
                 "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py",
                 "applications/gur-arieh-2510.06182/src/mixing_round1_design.py",
                 "applications/gur-arieh-2510.06182/src/mixing_prompts.py",
                 "applications/gur-arieh-2510.06182/src/mixing_runner.py",
                 "applications/gur-arieh-2510.06182/src/mixing_splits.py",
                 "applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py",
                 "applications/gur-arieh-2510.06182/scripts/run_mixing_confirmation.py"}
sys.path.insert(0, str(REPOSITORY / Path(HELPER).parent))
from query_route_analysis import clopper_pearson  # noqa: E402

REQUIRED_FILES = {"manifest.json", "records.jsonl", "summary.json", "RUN_STARTED.json",
                  "timings.json", "confirmation_gates.json", "gate7_cpu_reference.jsonl",
                  "audit_full_logits.npz"}
BINDING_GATES = ("1_model_hashes", "2_native_and_yield", "3_tokens", "4_hooks",
                 "5_identity", "7_dtype_device")
GATE7_DRAW_INDICES = tuple(range(32))
AUDIT_DRAW_INDICES = tuple(range(4))
YIELD_FLOOR = 0.50
AUDIT_TOLERANCES = {"logsumexp": 1e-3, "answer_mass": 1e-5, "answer_logit": 1e-6}
REQUIRED_TECHNICAL_CHECKS = {
    "conflict_design_indices", "agreement_i_P_design_indices",
    "agreement_i_L_design_indices", "agreement_i_R_design_indices",
    "dropped_prefix_is_bos", "single_bos", "conflict_hook",
    "agreement_i_P_hook", "agreement_i_L_hook", "agreement_i_R_hook",
    "identity_hook", "identity_generation_hook",
}
CONTRACT = {"w": 1, "kappa": 0.25, "coverage": 0.8, "alpha": 0.05, "family_size": 2,
            "label_tail": 0.0125, "resolution_rate_floor": 0.9,
            "agreement_resolution_floor": 0.9,
            "readout": "answer form: capitalised, no leading space, exactly one token",
            "answer_mass_floor": 0.5,
            "resolution_rule": "S >= s_min and answer-token mass >= answer_mass_floor",
            "unresolved_rule": "non-match for adequacy; match for exclusion"}
N_RULE = {"default_N": 200, "unresolved_threshold": 0.02, "larger_N": (300, 400, 500),
          "declared_power": 0.80, "adequacy_coverage": 0.90, "exclusion_coverage": 0.70}
LABELS = ("positional", "lexical", "reflexive")
TOLERANCE = 1e-12
CONFIRMATION_SEED_BASE = 4_000_000
SPLIT_A_DECISION = "results/split_A/split_A_decision.json"
SPLIT_B_DECISION = "results/split_B/split_B_decision.json"


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def number(value, label):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value), f"{label}: expected a finite number")
    return float(value)


def close(a, b, label):
    if a is None or b is None:
        require(a is None and b is None, f"{label}: summary differs from recomputation")
        return
    require(abs(a - b) <= TOLERANCE, f"{label}: summary differs from recomputation")


def _git(repository_root, *args, binary=False, check=True):
    completed = subprocess.run(
        ["git", *args], cwd=repository_root, capture_output=True,
        text=not binary, check=False)
    if check and completed.returncode:
        stderr = completed.stderr.decode(errors="replace") if binary else completed.stderr
        raise VerificationError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return completed


def verify_run_started(started, manifest_path, repository_root):
    """Validate every Git claim that remains observable after the run.

    The historical clean-worktree state cannot be reconstructed after result files are
    written.  The checker therefore requires the recorded false dirty bit and proves that
    the claimed commit exists and contains the exact frozen manifest copied into the run.
    The current checkout must still expose the same manifest bytes at the recorded path.
    """
    require(isinstance(started, dict), "RUN_STARTED.json must be an object")
    require(started.get("git_dirty") is False,
            "RUN_STARTED does not record a clean preflight worktree")
    head = started.get("git_head")
    relative_name = started.get("manifest_path")
    require(isinstance(head, str) and len(head) in (40, 64)
            and all(character in "0123456789abcdef" for character in head),
            "RUN_STARTED git_head is not a full lowercase object id")
    require(isinstance(relative_name, str) and relative_name not in ("", "."),
            "RUN_STARTED manifest_path is missing")
    relative = Path(relative_name)
    require(not relative.is_absolute() and ".." not in relative.parts
            and relative.as_posix() == relative_name,
            "RUN_STARTED manifest_path is unsafe or non-canonical")

    repository_root = Path(repository_root).resolve()
    top = _git(repository_root, "rev-parse", "--show-toplevel").stdout.strip()
    require(Path(top).resolve() == repository_root,
            "repository_root is not the Git worktree root")
    resolved_call = _git(repository_root, "rev-parse", "--verify", f"{head}^{{commit}}",
                         check=False)
    require(resolved_call.returncode == 0 and resolved_call.stdout.strip() == head,
            "RUN_STARTED git_head is not the claimed full commit id")
    tracked = _git(repository_root, "cat-file", "-e", f"{head}:{relative_name}", check=False)
    require(tracked.returncode == 0,
            "RUN_STARTED manifest_path is not tracked at the claimed commit")
    committed = _git(repository_root, "show", f"{head}:{relative_name}", binary=True).stdout
    frozen_bytes = Path(manifest_path).read_bytes()
    require(committed == frozen_bytes,
            "RUN_STARTED Git commit does not contain the frozen result manifest")
    current = repository_root / relative
    try:
        current.resolve().relative_to(repository_root)
    except ValueError as error:
        raise VerificationError("RUN_STARTED manifest_path escapes through a symlink") from error
    require(current.is_file() and current.read_bytes() == frozen_bytes,
            "current repository manifest differs from the frozen result manifest")


def _read_npy_float32(archive, member):
    """Read the runner's one-dimensional float32 NPY member using the stdlib only."""
    with archive.open(member) as handle:
        require(handle.read(6) == b"\x93NUMPY", f"audit member {member} has no NPY header")
        version = handle.read(2)
        require(len(version) == 2 and version[0] in (1, 2, 3),
                f"audit member {member} has an unsupported NPY version")
        width = 2 if version[0] == 1 else 4
        length_bytes = handle.read(width)
        require(len(length_bytes) == width, f"audit member {member} has a truncated header")
        header_length = struct.unpack("<H" if width == 2 else "<I", length_bytes)[0]
        require(0 < header_length <= 100_000,
                f"audit member {member} has an unreasonable NPY header")
        encoding = "utf-8" if version[0] == 3 else "latin1"
        header_bytes = handle.read(header_length)
        require(len(header_bytes) == header_length,
                f"audit member {member} has a truncated NPY header")
        try:
            header = ast.literal_eval(header_bytes.decode(encoding).strip())
        except (SyntaxError, ValueError, UnicodeDecodeError) as error:
            raise VerificationError(f"audit member {member} has an invalid NPY header") from error
        require(isinstance(header, dict) and header.get("descr") == "<f4"
                and header.get("fortran_order") is False,
                f"audit member {member} must be a little-endian C-order float32 array")
        shape = header.get("shape")
        require(isinstance(shape, tuple) and len(shape) == 1
                and type(shape[0]) is int and 0 < shape[0] <= 1_000_000,
                f"audit member {member} must be a bounded one-dimensional array")
        payload = handle.read()
        require(len(payload) == 4 * shape[0],
                f"audit member {member} payload length differs from its shape")
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    require(len(values) == shape[0] and all(math.isfinite(value) for value in values),
            f"audit member {member} contains a non-finite or malformed logit")
    return values


def verify_full_logit_audit(results, manifest, records, stored):
    """Recompute the first-four full-vocabulary audit without importing NumPy."""
    expected = [f"confirmation-{index:04d}" for index in AUDIT_DRAW_INDICES]
    pools = manifest.get("entity_pools")
    answer_ids = pools.get("answer_form_ids") if isinstance(pools, dict) else None
    require(isinstance(answer_ids, dict) and answer_ids
            and all(isinstance(name, str) and name and type(token) is int and token >= 0
                    for name, token in answer_ids.items())
            and len(set(answer_ids.values())) == len(answer_ids),
            "frozen answer-form ids are malformed or non-unique")
    t_entity = manifest.get("t_entity")
    require(type(t_entity) is int and t_entity >= 1,
            "manifest.t_entity must be a positive paper-based index")
    target_index = t_entity - 1
    by_id = {record.get("case_id"): record for record in records}
    worst_lse = worst_mass = worst_logit = 0.0
    path = Path(results) / "audit_full_logits.npz"
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            expected_members = [f"{name}.npy" for name in expected]
            require(members == expected_members,
                    "full-logit NPZ cases differ from the frozen first four draws")
            for name, member in zip(expected, members):
                logits = _read_npy_float32(archive, member)
                record = by_id.get(name)
                require(isinstance(record, dict), f"full-logit audit record missing for {name}")
                matrix = record.get("matrix")
                require(isinstance(matrix, list) and len(matrix) == manifest["n_groups"]
                        and all(isinstance(group, list) and len(group) > target_index
                                and isinstance(group[target_index], str) for group in matrix),
                        f"full-logit audit matrix is malformed for {name}")
                try:
                    ids = [answer_ids[group[target_index]] for group in matrix]
                except KeyError as error:
                    raise VerificationError(
                        f"full-logit audit matrix uses an unfrozen answer form for {name}") from error
                require(len(set(ids)) == len(ids) and max(ids) < len(logits),
                        f"full-logit audit answer ids are duplicated or out of range for {name}")
                readout = record.get("readout")
                require(isinstance(readout, dict)
                        and readout.get("answer_logits") == record.get("answer_logits")
                        and readout.get("answer_mass_full_vocab")
                            == record.get("answer_mass_full_vocab"),
                        f"stored primary and detailed readouts contradict each other for {name}")
                logged_lse = number(readout.get("logsumexp_full"),
                                    f"full-vocabulary logsumexp for {name}")
                top = max(logits)
                lse = top + math.log(math.fsum(math.exp(value - top) for value in logits))
                answer = [float(logits[index]) for index in ids]
                mass = math.fsum(math.exp(value - lse) for value in answer)
                worst_lse = max(worst_lse, abs(lse - logged_lse))
                worst_mass = max(
                    worst_mass, abs(mass - number(record.get("answer_mass_full_vocab"),
                                                  f"answer mass for {name}")))
                logged_answer = record.get("answer_logits")
                require(isinstance(logged_answer, list) and len(logged_answer) == len(answer),
                        f"answer logits are malformed for {name}")
                worst_logit = max(
                    worst_logit,
                    max(abs(actual - number(claimed, f"answer logit for {name}"))
                        for actual, claimed in zip(answer, logged_answer)))
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise VerificationError(f"cannot read full-logit audit NPZ: {error}") from error

    recomputed = {
        "checked": True,
        "expected_cases": expected,
        "tolerances": AUDIT_TOLERANCES,
        "cases": expected,
        "max_abs_logsumexp_difference": worst_lse,
        "max_abs_answer_mass_difference": worst_mass,
        "max_abs_answer_logit_difference": worst_logit,
        "passed": (worst_lse <= AUDIT_TOLERANCES["logsumexp"]
                   and worst_mass <= AUDIT_TOLERANCES["answer_mass"]
                   and worst_logit <= AUDIT_TOLERANCES["answer_logit"]),
    }
    require(recomputed["passed"], "independent full-logit audit failed")
    require(isinstance(stored, dict)
            and all(stored.get(key) == recomputed[key]
                    for key in ("checked", "expected_cases", "tolerances", "cases", "passed")),
            "full-logit gate metadata differs from the independent audit")
    for key in ("max_abs_logsumexp_difference", "max_abs_answer_mass_difference",
                "max_abs_answer_logit_difference"):
        claimed = number(stored.get(key), f"stored full-logit {key}")
        require(abs(claimed - recomputed[key]) <= 1e-10,
                f"stored full-logit {key} differs from the independent audit")
    return recomputed


def _rate(k, n):
    return {"k": k, "n": n, "rate": k / n if n else None}


def _tally(pairs):
    values = {}
    for key, passed in pairs:
        row = values.setdefault(str(key), [0, 0])
        row[0] += bool(passed)
        row[1] += 1
    return {key: _rate(*row) for key, row in sorted(values.items())}


def verify_development_binding(manifest, application_root):
    """Bind every fitted confirmation value to the frozen split-A/B decision files."""
    data = manifest.get("data_sha256")
    require(isinstance(data, dict) and {SPLIT_A_DECISION, SPLIT_B_DECISION} <= set(data),
            "frozen data set omits the split-A or split-B decision")
    decision_a = load(Path(application_root) / SPLIT_A_DECISION)
    decision_b = load(Path(application_root) / SPLIT_B_DECISION)
    require(decision_a.get("status") == "PROCEED" and decision_a.get("stops") == [],
            "the frozen split-A decision did not proceed")
    require(decision_b.get("status") == "PROCEED" and decision_b.get("stops") == [],
            "the frozen split-B decision did not proceed")
    cell_key = manifest.get("cell_key")
    require(decision_a.get("selected") == cell_key and decision_b.get("cell_key") == cell_key,
            "confirmation cell differs from the split-A/B decision")
    try:
        freeze = decision_b["split_B"]["freeze"]
    except (KeyError, TypeError) as error:
        raise VerificationError("split-B decision has no freeze block") from error
    expected = {
        "cell": freeze.get("cell"),
        "N": freeze.get("N"),
        "N_rule": freeze.get("N_rule"),
        "anchors": freeze.get("anchors"),
        "mean_gate": freeze.get("mean_gate"),
        "development_gates": freeze.get("development_gates"),
    }
    observed = {key: manifest.get(key) for key in expected}
    require(observed == expected,
            "confirmation cell, N, anchors or gates differ from the frozen split-B values")
    rule = manifest.get("rule")
    require(isinstance(rule, dict)
            and rule.get("s_min") == freeze.get("s_min")
            and rule.get("d_min") == freeze.get("d_min")
            and rule.get("agreement_transfer_floor") == freeze.get("agreement_transfer_floor"),
            "confirmation rule differs from the frozen split-B values")
    return freeze


# ---- independent recomputation ------------------------------------------------------

def distribution(logits):
    top = max(logits)
    weights = [math.exp(v - top) for v in logits]
    total = math.fsum(weights)
    return [v / total for v in weights]


def measure(p, mass, cell, w, s_min):
    n = len(p)
    span = [j for j in range(cell["i_P"] - w, cell["i_P"] + w + 1) if 0 <= j < n]
    positional = math.fsum(p[j] for j in span)
    parts = (positional, p[cell["i_L"]], p[cell["i_R"]])
    support = math.fsum(parts)
    q = tuple(x / support for x in parts) if support > 0 else None
    resolved = q is not None and support >= s_min and mass >= CONTRACT["answer_mass_floor"]
    excluded = set(span) | {cell["i_L"], cell["i_R"], cell["i_N"]}
    background = [p[j] for j in range(n) if j not in excluded]
    require(background, "no background entity for the argmax labels")
    scores = {"positional": positional - (2 * w + 1) * statistics.median(background),
              "lexical": p[cell["i_L"]], "reflexive": p[cell["i_R"]]}
    best = max(scores.values())
    return {"q": q, "T": max(q) if q is not None else None, "resolved": resolved,
            "labels": [l for l in LABELS if scores[l] == best] if resolved else []}


def technical_failure(record, cell, n):
    """Reason string if a qualifying record cannot be measured, else None. Reads the
    primary answer-form fields only."""
    if not isinstance(record.get("technical"), dict) or record["technical"].get("passed") is not True:
        return "technical checks did not pass"
    if record.get("design_indices") != cell:
        return "design indices differ from the frozen cell"
    logits = record.get("answer_logits")
    if not isinstance(logits, list) or len(logits) != n:
        return "wrong number of answer-form logits"
    values = [logits[i] for i in range(n)] + [record.get("answer_mass_full_vocab")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return "missing or non-finite primary readout"
    if not 0 <= record["answer_mass_full_vocab"] <= 1:
        return "answer-token mass outside [0, 1]"
    return None


def native_qualification(record, answer_ids):
    """Recompute the runner's qualification bit from the stored native observations."""
    native = record.get("native")
    require(isinstance(native, dict), "native competence record missing")
    passed = []
    for role in ("recipient", "donor"):
        row = native.get(role)
        require(isinstance(row, dict), f"native {role} record missing")
        answer, first_word = row.get("answer"), row.get("first_word")
        generation_ids = row.get("generation_ids")
        readout_entity = row.get("readout_argmax_entity")
        first_token = row.get("first_token_is_answer_form")
        correct = row.get("correct")
        agrees = row.get("readout_matches_generation")
        require(isinstance(answer, str) and isinstance(first_word, str)
                and isinstance(generation_ids, list) and generation_ids
                and all(type(token) is int for token in generation_ids)
                and isinstance(readout_entity, str) and answer in answer_ids
                and isinstance(first_token, bool) and isinstance(correct, bool)
                and isinstance(agrees, bool), f"native {role} competence fields are malformed")
        require(first_token is (generation_ids[0] == answer_ids[answer]),
                f"native {role} first-token claim is inconsistent with generation_ids")
        require(agrees is (first_word == readout_entity.lower()),
                f"native {role} readout/generation agreement is inconsistent")
        require(correct is (first_word == answer.lower() and first_token),
                f"native {role} correctness is inconsistent with its answer tokens")
        passed.extend((correct, agrees))
    return all(passed)


# ---- the N rule, reimplemented here -----------------------------------------------------

def binomial(k, n, p):
    if p in (0.0, 1.0):
        return float(k == (n if p == 1.0 else 0))
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    + k * math.log(p) + (n - k) * math.log1p(-p))


_THRESHOLDS = {}


def thresholds(N):
    """(smallest k declaring adequacy, largest k declaring exclusion) for the contract's
    intervals; both CP bounds increase with k."""
    if N not in _THRESHOLDS:
        def lower(k):
            return clopper_pearson(k, N, CONTRACT["alpha"], CONTRACT["family_size"])[0]

        def upper(k):
            return clopper_pearson(k, N, CONTRACT["alpha"], CONTRACT["family_size"])[1]

        lo, hi = 0, N + 1
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if lower(mid) > CONTRACT["coverage"] else (mid + 1, hi)
        k_adequate = lo
        lo, hi = 0, N + 1
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if upper(mid) >= CONTRACT["coverage"] else (mid + 1, hi)
        _THRESHOLDS[N] = (k_adequate, lo - 1)
    return _THRESHOLDS[N]


def rule_for(unresolved):
    """(N, adequacy powered, status) by the recorded N rule at an unresolved rate."""
    def powers(N):
        k_adequate, k_excluded = thresholds(N)
        match = N_RULE["adequacy_coverage"] * (1 - unresolved)
        exclusion = N_RULE["exclusion_coverage"] * (1 - unresolved) + unresolved
        return (math.fsum(binomial(k, N, match) for k in range(k_adequate, N + 1)),
                math.fsum(binomial(k, N, exclusion) for k in range(0, k_excluded + 1)))

    target = N_RULE["declared_power"]
    if unresolved <= N_RULE["unresolved_threshold"]:
        adequacy, _ = powers(N_RULE["default_N"])
        return N_RULE["default_N"], adequacy >= target, "PROCEED"
    for N in N_RULE["larger_N"]:
        if powers(N)[0] >= target:
            return N, True, "PROCEED"
    N = N_RULE["larger_N"][-1]
    return N, False, "PROCEED" if powers(N)[1] >= target else "STOP"


def recompute(manifest, records):
    rule, anchors, cell, n, N = (manifest["rule"], manifest["anchors"], manifest["cell"],
                                 manifest["n_groups"], manifest["N"])
    for key, value in CONTRACT.items():
        require(rule.get(key) == value, f"manifest rule {key} differs from the declared contract")
    w, s_min = rule["w"], number(rule["s_min"], "s_min")
    indices = [cell[k] for k in ("i_P", "i_L", "i_R", "i_N")]
    require(all(type(i) is int and 0 <= i < n for i in indices) and len(set(indices)) == 4
            and all(abs(i - cell["i_P"]) > w for i in indices[1:]), "frozen cell is not admissible")
    t_w, t_a, d = (number(anchors[k], k) for k in ("T_W", "T_A", "d"))
    q_bar_b = [number(x, "q_bar_B") for x in anchors["q_bar_B"]]
    require(len(q_bar_b) == 3 and abs(max(q_bar_b) - t_w) <= TOLERANCE, "T_W is not max(q_bar_B)")
    require(abs(d - (t_a - t_w)) <= TOLERANCE, "d is not T_A - T_W")
    require(d >= number(rule["d_min"], "d_min") > 0, "frozen separation below d_min")
    delta = number(manifest["mean_gate"]["delta"], "delta")
    require(delta > 0, "delta must be positive")
    gates = manifest["development_gates"]
    require(number(gates["resolution_rate_B"], "resolution rate") >= CONTRACT["resolution_rate_floor"],
            "frozen development resolution rate below 0.90")
    require(number(gates["agreement_resolution_rate_B"], "agreement resolution rate")
            >= CONTRACT["agreement_resolution_floor"],
            "frozen agreement-control resolution rate below 0.90")
    rule_N, powered, rule_status = rule_for(1 - number(gates["resolution_rate_B"], "resolution rate"))
    sizing = manifest.get("N_rule") or {}
    require(rule_status == "PROCEED" and N == rule_N and sizing.get("N") == rule_N
            and sizing.get("adequacy_powered") is powered,
            "frozen N or adequacy label differs from the N rule at split B's unresolved rate")
    require(number(gates["agreement_transfer_rate_B"], "transfer rate")
            >= number(rule["agreement_transfer_floor"], "transfer floor"),
            "frozen agreement transfer rate below its floor")

    confirmation = manifest.get("confirmation")
    require(isinstance(confirmation, dict), "manifest.confirmation missing")
    seed_base = confirmation.get("seed_base")
    prefix = confirmation.get("case_id_prefix")
    cap = confirmation.get("cap")
    require(type(seed_base) is int and seed_base == CONFIRMATION_SEED_BASE,
            "confirmation seed base differs from the declared 4,000,000 block")
    require(isinstance(prefix, str) and prefix == "confirmation",
            "confirmation case-id prefix differs from the declared prefix")
    require(type(cap) is int and cap == 2 * N,
            "confirmation generation cap must equal 2N")
    cell_key = manifest.get("cell_key")
    require(isinstance(cell_key, str) and cell_key,
            "manifest.cell_key missing")
    pools = manifest.get("entity_pools")
    answer_ids = pools.get("answer_form_ids") if isinstance(pools, dict) else None
    require(isinstance(answer_ids, dict) and answer_ids,
            "manifest.entity_pools.answer_form_ids missing")

    ids = [r.get("case_id") for r in records]
    require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids),
            "duplicate or missing case_id")
    require(all(type(r.get("draw_index")) is int for r in records)
            and [r.get("draw_index") for r in records] == list(range(len(records))),
            "records omit, duplicate or reorder a generated case")
    require(len(records) <= cap, "confirmation generated more than the frozen cap")
    for position, record in enumerate(records):
        require(record.get("case_id") == f"{prefix}-{position:04d}",
                "case_id differs from the frozen confirmation formula")
        require(type(record.get("seed")) is int
                and record.get("seed") == seed_base + position,
                "seed differs from the frozen confirmation formula")
        require(record.get("cell_key") == cell_key,
                "record cell key differs from the frozen cell")
        require(record.get("qualifies") is native_qualification(record, answer_ids),
                "qualifies differs from the stored native competence observations")
    qualifying = [r for r in records if r.get("qualifies") is True]
    require(all(isinstance(r.get("qualifies"), bool) for r in records), "qualifies must be boolean")
    require(len(qualifying) == N and records[-1]["qualifies"] is True,
            "qualifying cases differ from frozen N or generation did not stop at N")

    out = {"generated": len(records), "N": N, "adequacy_powered": powered}
    for record in qualifying:
        reason = technical_failure(record, cell, n)
        if reason:
            out.update(run_status="INVALID", invalid_kind="technical failure", level="S1",
                       statuses={"W_T": "INVALID", "A_T": "INVALID"}, earned=False)
            return out
    cases = [measure(distribution(r["answer_logits"]), r["answer_mass_full_vocab"], cell, w, s_min)
             for r in qualifying]
    band = CONTRACT["kappa"] * d
    resolved = [c for c in cases if c["resolved"]]
    u = N - len(resolved)
    k = {"W_T": sum(abs(c["T"] - t_w) <= band for c in resolved),
         "A_T": sum(abs(c["T"] - t_a) <= band for c in resolved)}
    require(not any(abs(c["T"] - t_w) <= band and abs(c["T"] - t_a) <= band for c in resolved),
            "a case fits both profiles")
    if resolved:
        mean = [math.fsum(c["q"][i] for c in resolved) / len(resolved) for i in range(3)]
        statistic = max(abs(a - b) for a, b in zip(q_bar_b, mean))
        passed = statistic <= delta
    else:
        statistic, passed = None, False
    intervals, statuses = {}, {}
    for profile in ("W_T", "A_T"):
        adequacy = clopper_pearson(k[profile], N, CONTRACT["alpha"], CONTRACT["family_size"])
        exclusion = clopper_pearson(k[profile] + u, N, CONTRACT["alpha"], CONTRACT["family_size"])
        intervals[profile] = (adequacy, exclusion)
        statuses[profile] = ("adequate" if adequacy[0] > CONTRACT["coverage"] else
                             "excluded" if exclusion[1] < CONTRACT["coverage"] else "undecided")
    counts = {l: sum(l in c["labels"] for c in resolved) for l in LABELS}
    upper = {l: clopper_pearson(counts[l] + u, N, CONTRACT["alpha"], CONTRACT["family_size"])[1]
             for l in LABELS}
    valid = passed
    if valid:
        excluded = [p for p in ("W_T", "A_T") if statuses[p] == "excluded"]
        level = ("S3 (both excluded)" if len(excluded) == 2 else
                 "S3 (A_T excluded)" if excluded == ["A_T"] else
                 "S3 (W_T excluded only)" if excluded == ["W_T"] else "S2")
    else:
        level = "S1"
    out.update(run_status="VALID" if valid else "INVALID",
               invalid_kind=None if valid else "stale anchor",
               level=level, u=u, k=k, intervals=intervals,
               statuses=statuses if valid else {"W_T": "INVALID", "A_T": "INVALID"},
               descriptive_statuses=statuses, statistic=statistic, passed=passed,
               counts=counts, upper=upper,
               earned=bool(valid and statuses["A_T"] == "adequate" and passed
                           and max(upper.values()) < CONTRACT["coverage"]))
    return out


def compare(r, summary):
    require(summary["run_status"] == r["run_status"], "summary run status differs from recomputation")
    require(summary["level"] == r["level"], "summary level differs from recomputation")
    require(summary["statuses"] == r["statuses"], "summary profile statuses differ from recomputation")
    require(summary["generated"] == r["generated"] and summary["N"] == r["N"], "summary yield differs")
    require(summary.get("adequacy_powered") is r["adequacy_powered"], "summary adequacy label differs")
    close(summary["yield"], r["N"] / r["generated"], "yield")
    require(summary["between_case"]["earned"] is r["earned"], "summary between-case decision differs")
    if r["invalid_kind"] == "technical failure":
        require(str(summary.get("invalid_reason", "")).startswith("technical failure"),
                "summary does not report the technical failure")
        return
    require((summary["invalid_reason"] is None) == (r["invalid_kind"] is None)
            and (r["invalid_kind"] is None or r["invalid_kind"] in summary["invalid_reason"]),
            "summary invalid reason differs")
    require(summary["u"] == r["u"], "summary unresolved count differs")
    for profile in ("W_T", "A_T"):
        stored = summary["profiles"][profile]
        require(stored["k_match"] == r["k"][profile]
                and stored["k_match_plus_unresolved"] == r["k"][profile] + r["u"],
                f"summary {profile} counts differ from recomputation")
        require(stored["status"] == r["descriptive_statuses"][profile],
                f"summary {profile} status differs from recomputation")
        adequacy, exclusion = r["intervals"][profile]
        for label, a, b in (("adequacy", stored["adequacy_interval"], adequacy),
                            ("exclusion", stored["exclusion_interval"], exclusion)):
            close(a[0], b[0], f"{profile} {label} interval")
            close(a[1], b[1], f"{profile} {label} interval")
    gate = summary["mean_gate"]
    require(gate["passed"] is r["passed"], "summary mean-consistency gate differs")
    close(gate["statistic"], r["statistic"], "mean-consistency statistic")
    between = summary["between_case"]
    require(between["label_counts_resolved"] == r["counts"], "summary label counts differ")
    for label in LABELS:
        close(between["U"][label], r["upper"][label], f"U[{label}]")


def verify_confirmation_gates(results, manifest, records, summary):
    """Rebuild the binding gates from raw records, CPU repeats and the NPZ audit."""
    require(summary.get("confirmation_gates_sha256") == sha256(results / "confirmation_gates.json"),
            "summary does not bind confirmation_gates.json")
    require(summary.get("timings_sha256") == sha256(results / "timings.json"),
            "summary does not bind timings.json")
    table = load(results / "confirmation_gates.json")
    require(table.get("status") == "PROCEED" and table.get("stops") == [],
            "binding confirmation gates did not proceed")
    require(table.get("cell_key") == manifest["cell_key"],
            "gate table names a different frozen cell")
    qualifying = [record for record in records if record.get("qualifies") is True]
    require(table.get("families") == len(records)
            and table.get("qualifying") == len(qualifying) == manifest["N"],
            "gate table family counts differ from the raw records")
    gates = table.get("gates")
    require(isinstance(gates, dict), "confirmation gate table is missing gates")
    for name in BINDING_GATES:
        require(isinstance(gates.get(name), dict) and gates[name].get("passed") is True,
                f"binding confirmation gate {name} did not pass")

    measured = [record for record in records if "matrix" in record]
    require(len(measured) == len(records),
            "a proceeding confirmation record lacks the raw family matrix")
    hook_failed, design_failed, hook_counts, technical_failures = [], [], {}, []
    identity_rows = []
    identity_tolerance = manifest["execution_contract"]["identity_tolerance"]
    for record in records:
        name = record.get("case_id", "<missing case_id>")
        technical = record.get("technical")
        require(isinstance(technical, dict), f"raw technical evidence is missing for {name}")
        checks, failures = technical.get("checks"), technical.get("failures")
        require(isinstance(checks, dict) and REQUIRED_TECHNICAL_CHECKS <= set(checks)
                and all(isinstance(key, str) and type(passed) is bool
                        for key, passed in checks.items())
                and isinstance(failures, list)
                and all(isinstance(failure, str) for failure in failures),
                f"raw technical evidence is malformed for {name}")
        require(technical.get("passed") is (not failures)
                and technical.get("passed") is True and all(checks.values()),
                f"raw technical passed/check/failure evidence contradicts itself for {name}")
        require(record.get("cell") == manifest["cell"]
                and record.get("design_indices") == manifest["cell"]
                and technical_failure(record, manifest["cell"], manifest["n_groups"]) is None,
                f"raw technical or design-index gate failed for {name}")
        for check_name, passed in checks.items():
            if check_name.endswith("hook"):
                hook_counts[check_name] = hook_counts.get(check_name, 0) + 1
                if not passed:
                    hook_failed.append([name, check_name])
            if check_name.endswith("design_indices") and not passed:
                design_failed.append([name, check_name])
        if technical.get("passed") is not True:
            technical_failures.append([name, failures])

        identity = record.get("identity")
        require(isinstance(identity, dict)
                and type(identity.get("same_answer_argmax")) is bool
                and type(identity.get("same_generation")) is bool,
                f"raw identity evidence is malformed for {name}")
        difference = number(identity.get("max_abs_answer_logit_difference"),
                            f"identity difference for {name}")
        require(identity["same_answer_argmax"] is True
                and identity["same_generation"] is True
                and difference <= identity_tolerance,
                f"raw identity gate failed for {name}")
        identity_rows.append((identity, difference))

    model_files = manifest.get("model", {}).get("files_sha256", {})
    require(isinstance(model_files, dict), "manifest model file hashes are malformed")
    expected_gate1 = {
        "passed": True,
        "verified_counts": {"code": len(manifest["code_files_sha256"]),
                            "data": len(manifest["data_sha256"]),
                            "model": len(model_files)},
    }
    require(gates["1_model_hashes"] == expected_gate1,
            "gate-1 evidence contradicts the frozen hash sets")

    native = [(record, role, record["native"][role]) for record in measured
              for role in ("recipient", "donor")]
    agreement = [(row["target"], row["j"], row["donor_native"])
                 for record in measured for row in record.get("agreement", [])]
    overall = _rate(len(qualifying), len(records))
    expected_gate2 = {
        "passed": overall["rate"] is not None and overall["rate"] >= YIELD_FLOOR,
        "yield": overall,
        "floor": YIELD_FLOOR,
        "recipient_correct_by_i_N": _tally(
            (record["cell"]["i_N"], row.get("correct"))
            for record, role, row in native if role == "recipient"),
        "donor_correct_by_queried_position": _tally(
            (record["cell"]["i_P"], row.get("correct"))
            for record, role, row in native if role == "donor"),
        "first_token_is_answer_form": _tally(
            (role, row.get("first_token_is_answer_form")) for _, role, row in native),
        "readout_matches_generation": _tally(
            (role, row.get("readout_matches_generation")) for _, role, row in native),
        "agreement_donor_correct_by_target": _tally(
            (f"{target}={index}", row.get("correct"))
            for target, index, row in agreement),
    }
    require(gates["2_native_and_yield"] == expected_gate2 and expected_gate2["passed"],
            "gate-2 evidence contradicts raw native competence or yield")

    pools = manifest.get("entity_pools", {})
    frozen_pools, frozen_dropped = pools.get("pools"), pools.get("dropped")
    require(isinstance(frozen_pools, dict) and isinstance(frozen_dropped, dict),
            "frozen token-pool evidence is incomplete")
    alignment = [[record["case_id"], failures]
                 for record in records
                 for failures in [record["technical"]["failures"]]
                 if any("differ from the design" in failure for failure in failures)]
    expected_gate3 = {
        "pools_equal_the_manifest": True,
        "pools_equal_the_lock": True,
        "pools_kept": {category: len(values) for category, values in frozen_pools.items()},
        "pools_dropped": frozen_dropped,
        "dropped_prefix": "<bos>",
        "passed": not alignment and all(len(values) >= manifest["n_groups"] + 2
                                         for values in frozen_pools.values()),
        "alignment_failures": alignment,
    }
    require(gates["3_tokens"] == expected_gate3 and expected_gate3["passed"],
            "gate-3 evidence contradicts the frozen pools or raw alignment checks")

    audit = verify_full_logit_audit(
        results, manifest, records, gates["4_hooks"].get("full_logit_audit"))
    expected_gate4 = {
        "passed": not technical_failures and not hook_failed and not design_failed
                  and audit["passed"],
        "checks": hook_counts,
        "failed": hook_failed,
        "design_index_failures": design_failed,
        "technical_failures": technical_failures,
    }
    gate4_without_audit = {key: value for key, value in gates["4_hooks"].items()
                           if key != "full_logit_audit"}
    require(gate4_without_audit == expected_gate4 and expected_gate4["passed"],
            "gate-4 evidence contradicts raw technical or hook evidence")

    maximum = max(difference for _, difference in identity_rows)
    expected_gate5 = {
        "passed": (len(identity_rows) == len(measured) > 0
                   and maximum <= identity_tolerance
                   and all(row["same_answer_argmax"] and row["same_generation"]
                           for row, _ in identity_rows)),
        "families": len(identity_rows),
        "measured_families": len(measured),
        "max_abs_answer_logit_difference": maximum,
        "tolerance": identity_tolerance,
        "same_argmax": sum(row["same_answer_argmax"] for row, _ in identity_rows),
        "same_generation": sum(row["same_generation"] for row, _ in identity_rows),
    }
    require(gates["5_identity"] == expected_gate5 and expected_gate5["passed"],
            "gate-5 evidence contradicts the raw identity records")

    references = jsonl(results / "gate7_cpu_reference.jsonl")
    expected_ids = [f"confirmation-{i:04d}" for i in GATE7_DRAW_INDICES]
    require([row.get("case_id") for row in references] == expected_ids,
            "gate-7 CPU rows differ from the frozen first 32 draws")
    by_id = {record["case_id"]: record for record in records}
    w, s_min, cell = (manifest["rule"]["w"], manifest["rule"]["s_min"], manifest["cell"])
    rows, differences = [], []
    for draw_index, reference in zip(GATE7_DRAW_INDICES, references):
        name = reference["case_id"]
        require(reference.get("device") == "cpu" and reference.get("dtype") == "torch.float32"
                and reference.get("hook_ok") is True,
                f"gate-7 CPU execution metadata failed for {name}")
        record, conflict = by_id.get(name), reference.get("conflict")
        require(isinstance(record, dict) and isinstance(conflict, dict),
                f"gate-7 raw data missing for {name}")
        require(technical_failure(record, cell, manifest["n_groups"]) is None,
                f"gate-7 MPS record is not measurable for {name}")
        cpu_logits = conflict.get("answer_logits")
        cpu_mass = conflict.get("answer_mass_full_vocab")
        require(isinstance(cpu_logits, list) and len(cpu_logits) == manifest["n_groups"]
                and all(not isinstance(value, bool) and isinstance(value, (int, float))
                        and math.isfinite(value) for value in cpu_logits)
                and not isinstance(cpu_mass, bool) and isinstance(cpu_mass, (int, float))
                and math.isfinite(cpu_mass) and 0 <= cpu_mass <= 1,
                f"gate-7 CPU readout is invalid for {name}")
        mps = measure(distribution(record["answer_logits"]), record["answer_mass_full_vocab"],
                      cell, w, s_min)
        cpu = measure(distribution(cpu_logits), cpu_mass, cell, w, s_min)
        require(mps["T"] is not None and cpu["T"] is not None,
                f"gate-7 T is undefined for {name}")
        difference = abs(mps["T"] - cpu["T"])
        differences.append(difference)
        rows.append({
            "draw_index": draw_index,
            "case_id": name,
            "abs_T_difference": difference,
            "same_resolution": mps["resolved"] == cpu["resolved"],
            "same_labels": mps["labels"] == cpu["labels"],
            "max_abs_answer_logit_difference": max(
                abs(a - b) for a, b in zip(record["answer_logits"], cpu_logits)),
            "hook_ok_cpu": True,
            "cpu_device": "cpu",
            "cpu_dtype": "torch.float32",
        })
    execution = manifest["execution_contract"]["gate7"]
    gate7_passed = (max(differences) <= execution["tolerance_T"]
                    and all(row["same_resolution"] and row["same_labels"]
                            for row in rows))
    expected_gate7 = {
        "passed": gate7_passed,
        "comparison": "MPS float32/eager against CPU float32/eager, conflict patch",
        "declared_draw_indices": list(GATE7_DRAW_INDICES),
        "compared": len(GATE7_DRAW_INDICES),
        "missing": [],
        "max_abs_T_difference": max(differences),
        "tolerance": execution["tolerance_T"],
        "execution": {"main_device": "mps", "main_dtype": "torch.float32",
                      "attention": "eager"},
        "rows": rows,
    }
    stored_gate7 = gates["7_dtype_device"]
    require(isinstance(stored_gate7, dict) and gate7_passed,
            "gate-7 table contradicts the independently replayed CPU references")
    stored_rows = stored_gate7.get("rows")
    require(isinstance(stored_rows, list) and len(stored_rows) == len(rows),
            "gate-7 stored rows differ from the independently replayed CPU references")
    for stored_row, expected_row in zip(stored_rows, rows):
        for key in ("abs_T_difference", "max_abs_answer_logit_difference"):
            close(stored_row.get(key), expected_row[key], f"gate-7 row {key}")
        require({key: value for key, value in stored_row.items()
                 if key not in ("abs_T_difference", "max_abs_answer_logit_difference")}
                == {key: value for key, value in expected_row.items()
                    if key not in ("abs_T_difference", "max_abs_answer_logit_difference")},
                "gate-7 stored row metadata differs from the CPU replay")
    close(stored_gate7.get("max_abs_T_difference"), expected_gate7["max_abs_T_difference"],
          "gate-7 maximum T difference")
    require({key: value for key, value in stored_gate7.items()
             if key not in ("rows", "max_abs_T_difference")}
            == {key: value for key, value in expected_gate7.items()
                if key not in ("rows", "max_abs_T_difference")},
            "gate-7 table metadata contradicts the independently replayed CPU references")

    expected_count = {"generated": len(records), "qualifying": len(qualifying)}
    expected_quota = {
        "passed": bool(len(records) <= manifest["confirmation"]["cap"]
                       and len(qualifying) == manifest["N"]
                       and records and records[-1].get("qualifies") is True),
        "N": manifest["N"],
        "cap": manifest["confirmation"]["cap"],
        "counts": {manifest["cell_key"]: expected_count},
        "generator_reported_met": True,
    }
    require(table.get("quota") == expected_quota and expected_quota["passed"],
            "confirmation quota table differs from the raw records or freeze")


def verify(results, *, repository_root=REPOSITORY):
    results, repository_root = Path(results), Path(repository_root)
    require(not (results / "FAILED.json").exists(), "run has FAILED.json")
    require(not (results / "STOP.json").exists(), "run has STOP.json")
    hashes = load(results / "artifact_hashes.json")
    require(REQUIRED_FILES <= set(hashes), "artifact manifest omits a required result file")
    actual = {path.name for path in results.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    require(set(hashes) == actual, "artifact manifest does not exactly cover the result directory")
    for name, digest in hashes.items():
        require(Path(name).name == name, f"result artifact is not a basename: {name}")
        require((results / name).is_file(), f"missing required file: {name}")
        require(sha256(results / name) == digest, f"hash mismatch: {name}")
    manifest, summary = load(results / "manifest.json"), load(results / "summary.json")
    require(manifest.get("stage") == "confirmation"
            and manifest.get("freeze_status") == "FROZEN"
            and isinstance(manifest.get("confirmation"), dict)
            and manifest["confirmation"].get("authorized") is True,
            "results do not use an authorized final confirmation manifest")
    started = load(results / "RUN_STARTED.json")
    manifest_hash = sha256(results / "manifest.json")
    require(summary["manifest_sha256"] == started["manifest_sha256"] == manifest_hash,
            "freeze hash disagrees between manifest, start and summary")
    verify_run_started(started, results / "manifest.json", repository_root)
    require(summary["records_sha256"] == sha256(results / "records.jsonl"),
            "summary does not describe these records")
    code = manifest["code_files_sha256"]
    require(REQUIRED_CODE <= set(code), "frozen code set omits the analyzer, checker or CP helper")
    for name, digest in code.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, f"unsafe code path: {name}")
        require((repository_root / path).is_file(), f"missing frozen code file: {name}")
        require(sha256(repository_root / path) == digest, f"hash mismatch: {name}")
    require(sha256(repository_root / HELPER) == code[HELPER],
            "the Clopper-Pearson helper used here differs from the frozen one")
    data = manifest.get("data_sha256")
    require(isinstance(data, dict) and data, "frozen data hash set is missing")
    application_root = repository_root / "applications/gur-arieh-2510.06182"
    for name, digest in data.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts,
                f"unsafe frozen data path: {name}")
        require((application_root / path).is_file(), f"missing frozen data file: {name}")
        require(sha256(application_root / path) == digest, f"hash mismatch: {name}")
    verify_development_binding(manifest, application_root)
    records = jsonl(results / "records.jsonl")
    recomputed = recompute(manifest, records)
    compare(recomputed, summary)
    verify_confirmation_gates(results, manifest, records, summary)
    report = {"verified": True, "mode": "records_only", "run_status": recomputed["run_status"],
              "level": recomputed["level"], "statuses": recomputed["statuses"],
              "u": recomputed.get("u"), "N": recomputed["N"],
              "yield": recomputed["N"] / recomputed["generated"],
              "adequacy_powered": recomputed["adequacy_powered"],
              "between_case_earned": recomputed["earned"]}
    checker_report = results / "checker_report.json"
    if checker_report.exists():
        require(load(checker_report) == report,
                "checker_report.json differs from the independent recomputation")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--repository-root", default=str(REPOSITORY),
                        help="repository containing the frozen code/data; defaults to this checkout")
    args = parser.parse_args()
    try:
        report = verify(args.results, repository_root=args.repository_root)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"FAILED: {error}\n")
    print(json.dumps(report, indent=2))
    print("PASS: statuses, gates, level and the between-case decision reproduce from records.")


if __name__ == "__main__":
    main()
