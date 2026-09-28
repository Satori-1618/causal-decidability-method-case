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
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

APPLICATION = Path(__file__).resolve().parents[1]
REPOSITORY = APPLICATION.parents[1]
HELPER = "applications/makelov-2311.17030/src/query_route_analysis.py"
REQUIRED_CODE = {HELPER,
                 "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py",
                 "applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py"}
sys.path.insert(0, str(REPOSITORY / Path(HELPER).parent))
from query_route_analysis import clopper_pearson  # noqa: E402

REQUIRED_FILES = {"manifest.json", "records.jsonl", "summary.json", "RUN_STARTED.json"}
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

    ids = [r.get("case_id") for r in records]
    require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids),
            "duplicate or missing case_id")
    require([r.get("draw_index") for r in records] == list(range(len(records))),
            "records omit, duplicate or reorder a generated case")
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


def verify(results, *, repository_root=REPOSITORY):
    results, repository_root = Path(results), Path(repository_root)
    require(not (results / "FAILED.json").exists(), "run has FAILED.json")
    hashes = load(results / "artifact_hashes.json")
    require(REQUIRED_FILES <= set(hashes), "artifact manifest omits a required result file")
    for name, digest in hashes.items():
        require(Path(name).name == name, f"result artifact is not a basename: {name}")
        require((results / name).is_file(), f"missing required file: {name}")
        require(sha256(results / name) == digest, f"hash mismatch: {name}")
    manifest, summary = load(results / "manifest.json"), load(results / "summary.json")
    started = load(results / "RUN_STARTED.json")
    manifest_hash = sha256(results / "manifest.json")
    require(summary["manifest_sha256"] == started["manifest_sha256"] == manifest_hash,
            "freeze hash disagrees between manifest, start and summary")
    require(summary["records_sha256"] == sha256(results / "records.jsonl"),
            "summary does not describe these records")
    code = manifest["code_files_sha256"]
    require(REQUIRED_CODE <= set(code), "frozen code set omits the analyzer, checker or CP helper")
    for name, digest in code.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, f"unsafe code path: {name}")
        require((repository_root / path).is_file(), f"missing frozen code file: {name}")
        require(sha256(repository_root / path) == digest, f"hash mismatch: {name}")
    require(sha256(REPOSITORY / HELPER) == code[HELPER],
            "the Clopper-Pearson helper used here differs from the frozen one")
    recomputed = recompute(manifest, jsonl(results / "records.jsonl"))
    compare(recomputed, summary)
    return {"verified": True, "mode": "records_only", "run_status": recomputed["run_status"],
            "level": recomputed["level"], "statuses": recomputed["statuses"],
            "u": recomputed.get("u"), "N": recomputed["N"], "yield": recomputed["N"] / recomputed["generated"],
            "adequacy_powered": recomputed["adequacy_powered"],
            "between_case_earned": recomputed["earned"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    try:
        report = verify(args.results)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"FAILED: {error}\n")
    print(json.dumps(report, indent=2))
    print("PASS: statuses, gates, level and the between-case decision reproduce from records.")


if __name__ == "__main__":
    main()
