"""Round 1 analyzer: does a single case look like the cell average? No model inference.

Per case k the entity readout gives logits over the n in-context entity tokens at the
last position; p_k is their softmax (the paper's per-case distribution). With the cell's
design indices (i_P, i_L, i_R) and the window w:

    S_k = P-window mass + p_k[i_L] + p_k[i_R],     P-window = i_P - w .. i_P + w
    q_k = (P-window mass, p_k[i_L], p_k[i_R]) / S_k
    T_k = max(q_k), between 1/3 and 1

Two concentration profiles, fixed on development split B for one cell:

    T_W = T(mean_k q_k)                    equal weights over resolved cases (q, not p)
    T_A = mean over k and j of T(q_k^agree(j)), same q-map
    d   = T_A - T_W, signed, at least d_min
    W_T-conform: resolved and |T_k - T_W| <= kappa * d
    A_T-conform: resolved and |T_k - T_A| <= kappa * d

They compare how concentrated a case is, never which target it favours. A case is
unresolved if S_k < s_min. Declared unresolved rule (new; the Makelov round 2/3A
analyzers counted unresolved units as non-successes for both bounds): unresolved cases
count as non-matches for adequacy and as matches for exclusion. Technical failures make
the run INVALID; they are never unresolved. Intervals are simultaneous Clopper-Pearson
over the two profiles, alpha/4 per tail, from the repository's existing helper.

Values the brief leaves open (s_min rule, d_min, delta procedure, gate floors) are
arguments without defaults here; the proposed values live in PROPOSED_VALUES.json and
await approval.
"""
import math
import random
import statistics
import sys
from collections.abc import Mapping
from pathlib import Path

_MAKELOV_SRC = Path(__file__).resolve().parents[2] / "makelov-2311.17030" / "src"
if str(_MAKELOV_SRC) not in sys.path:
    sys.path.insert(0, str(_MAKELOV_SRC))
from query_route_analysis import clopper_pearson  # noqa: E402  existing helper, reused

CP_HELPER_PATH = "applications/makelov-2311.17030/src/query_route_analysis.py"
ANALYZER_PATH = "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py"

# Fixed by the brief (Sec. 5.4, 5.5, 5.8, 5.11); not proposals.
CONTRACT = {
    "w": 1,
    "kappa": 0.25,
    "coverage": 0.8,
    "alpha": 0.05,
    "family_size": 2,
    "label_tail": 0.0125,
    "resolution_rate_floor": 0.9,
    "unresolved_rule": "non-match for adequacy; match for exclusion",
}
PROFILES = ("W_T", "A_T")
LABELS = ("positional", "lexical", "reflexive")
NOT_DECIDABLE = "NOT_DECIDABLE_WITH_CURRENT_INTERVENTIONS"
W_T_READING = ("More than 20% of cases are resolved and deviate from the cell-average "
               "concentration by more than kappa*d_c.")
W_T_CAVEAT = ("W_T is biased toward exclusion when case argmaxes vary; the bias is largest "
              "in the selected cell.")
BETWEEN_CASE_SENTENCE = "Cases differ in which candidate position they favour."


class TechnicalFailure(ValueError):
    """A technical failure: the run is INVALID, never a case left unresolved."""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _index(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer design index")
    return value


# ---- per-case readout -------------------------------------------------------------

def softmax(logits):
    """Softmax over the n entity logits (the paper's per-case distribution p_k)."""
    if isinstance(logits, (str, bytes, Mapping)) or not hasattr(logits, "__len__") or len(logits) < 2:
        raise ValueError("entity logits must be a sequence of at least two numbers")
    values = [_finite(v, "entity logit") for v in logits]
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    total = math.fsum(weights)
    return [w / total for w in weights]


def check_cell(cell, n, w):
    """Admissibility on design indices (0-based): all four distinct, each of i_L, i_R and
    i_N more than w positions from i_P. Upstream lets i_L and i_R collide."""
    if not isinstance(cell, Mapping):
        raise ValueError("cell must be a mapping of design indices")
    i_P, i_L, i_R, i_N = (_index(cell.get(k), k) for k in ("i_P", "i_L", "i_R", "i_N"))
    w = _index(w, "w")
    if w < 0 or n < 2 * w + 4:
        raise ValueError("need w >= 0 and n >= 2w + 4 for an admissible cell")
    if not all(0 <= i < n for i in (i_P, i_L, i_R, i_N)):
        raise ValueError("design indices must lie in 0..n-1")
    if len({i_P, i_L, i_R, i_N}) != 4:
        raise ValueError("inadmissible cell: design indices must be distinct")
    if any(abs(i - i_P) <= w for i in (i_L, i_R, i_N)):
        raise ValueError("inadmissible cell: i_L, i_R and i_N must lie more than w from i_P")
    return {"i_P": i_P, "i_L": i_L, "i_R": i_R, "i_N": i_N}


def window(i_P, w, n):
    return [j for j in range(i_P - w, i_P + w + 1) if 0 <= j < n]


def q_map(p, cell, w):
    """(S, q) for one distribution; q is None when S is zero."""
    n = len(p)
    positional = math.fsum(p[j] for j in window(cell["i_P"], w, n))
    parts = (positional, p[cell["i_L"]], p[cell["i_R"]])
    support = math.fsum(parts)
    if support <= 0:
        return support, None
    return support, tuple(x / support for x in parts)


def concentration(q):
    """T = max(q); lies in [1/3, 1] for a probability vector of length three."""
    return max(q)


def background_corrected_labels(p, cell, w):
    """Argmax labels with background-corrected positional mass (used only for the
    between-case check and descriptive shares). Background: the median mass of entities
    outside the window and not i_L, i_R or i_N. Ties are returned together."""
    n = len(p)
    inside = set(window(cell["i_P"], w, n)) | {cell["i_L"], cell["i_R"], cell["i_N"]}
    background = [p[j] for j in range(n) if j not in inside]
    if not background:
        raise ValueError("no background entity: n is too small for this cell and w")
    positional = math.fsum(p[j] for j in window(cell["i_P"], w, n))
    values = {
        "positional": positional - (2 * w + 1) * statistics.median(background),
        "lexical": p[cell["i_L"]],
        "reflexive": p[cell["i_R"]],
    }
    top = max(values.values())
    return [label for label in LABELS if values[label] == top], len(background)


def case_measures(logits, cell, w, s_min):
    p = softmax(logits)
    support, q = q_map(p, cell, w)
    resolved = q is not None and support >= s_min
    labels, background_size = background_corrected_labels(p, cell, w)
    return {
        "S": support,
        "q": list(q) if q is not None else None,
        "T": concentration(q) if q is not None else None,
        "resolved": resolved,
        "p_native": p[cell["i_N"]],
        "labels": labels if resolved else [],
        "background_entities": background_size,
    }


# ---- development: s_min, anchors, selection, delta ---------------------------------

def upper_order_statistic(values, level):
    """The smallest sample value v with at least a fraction ``level`` of values <= v."""
    values = sorted(_finite(v, "value") for v in values)
    if not values or not 0 < level <= 1:
        raise ValueError("need values and 0 < level <= 1")
    return values[max(0, math.ceil(level * len(values)) - 1)]


def anchored_s_min(nopatch_supports, quantile, floor):
    """s_min anchored to the S of no-patch runs: max(upper order statistic, floor).
    Both parameters are proposals (PROPOSED_VALUES.json); no default is assumed."""
    floor = _finite(floor, "floor")
    if floor < 0:
        raise ValueError("floor must be nonnegative")
    return max(upper_order_statistic(nopatch_supports, quantile), floor)


def anchors(conflict_logits, agreement_logits, cell, w, s_min):
    """T_W, T_A and d from one split. ``agreement_logits`` is a list of
    (case_index, j, logits) with j the common target."""
    conflict = [case_measures(x, cell, w, s_min) for x in conflict_logits]
    resolved = [c["q"] for c in conflict if c["resolved"]]
    if not resolved:
        raise ValueError("no resolved conflict case: anchors undefined")
    q_bar = [math.fsum(q[i] for q in resolved) / len(resolved) for i in range(3)]
    agreement, transfers = [], 0
    for _, j, logits in agreement_logits:
        measured = case_measures(logits, cell, w, s_min)
        p = softmax(logits)
        transfers += max(range(len(p)), key=p.__getitem__) == j
        if measured["resolved"]:
            agreement.append(measured["T"])
    if not agreement:
        raise ValueError("no resolved agreement case: T_A undefined")
    t_w = concentration(q_bar)
    t_a = math.fsum(agreement) / len(agreement)
    return {
        "T_W": t_w, "T_A": t_a, "d": t_a - t_w, "q_bar": q_bar,
        "m_resolved": len(resolved), "m_total": len(conflict),
        "resolution_rate": len(resolved) / len(conflict),
        "agreement_resolved": len(agreement), "agreement_total": len(agreement_logits),
        "agreement_transfer_rate": transfers / len(agreement_logits) if agreement_logits else None,
    }


def select_cell(estimates, d_min):
    """Largest d on split A among pre-declared candidates; ties go to the earlier
    candidate in declared order. ``estimates`` is an ordered list of (key, d)."""
    d_min = _finite(d_min, "d_min")
    best = None
    for key, d in estimates:
        d = _finite(d, f"d[{key}]")
        if best is None or d > best[1]:
            best = (key, d)
    if best is None or best[1] < d_min:
        return {"status": NOT_DECIDABLE, "level": "S1", "selected": None, "estimates": estimates}
    return {"status": "selected", "selected": best[0], "d_A": best[1], "estimates": estimates}


def delta_threshold(q_vectors, m, N, rate, resamples, seed):
    """Mean-consistency threshold under the no-drift null: draw two resamples, of sizes
    m and N, with replacement from split B's resolved q-vectors; delta is the (1 - rate)
    upper order statistic of the sup-norm difference of their means."""
    pool = [tuple(_finite(x, "q") for x in q) for q in q_vectors]
    if not pool or any(len(q) != 3 for q in pool):
        raise ValueError("need resolved three-component q-vectors from split B")
    m, N, resamples = (_index(v, name) for v, name in ((m, "m"), (N, "N"), (resamples, "resamples")))
    rate = _finite(rate, "rate")
    if m < 1 or N < 1 or resamples < 1 or not 0 < rate < 1:
        raise ValueError("need m, N, resamples >= 1 and 0 < rate < 1")
    rng = random.Random(seed)
    size = len(pool)
    differences = []
    for _ in range(resamples):
        sums = []
        for count in (m, N):
            s0 = s1 = s2 = 0.0
            for _ in range(count):
                x0, x1, x2 = pool[rng.randrange(size)]
                s0 += x0
                s1 += x1
                s2 += x2
            sums.append((s0 / count, s1 / count, s2 / count))
        differences.append(max(abs(a - b) for a, b in zip(*sums)))
    return upper_order_statistic(differences, 1 - rate)


def development_decision(split_a, split_b, candidates, *, n, s_min_quantile, s_min_floor,
                         d_min, agreement_transfer_floor, N, false_invalid_rate,
                         resamples, seed):
    """Gates on development data, then the values to freeze. Every STOP is S1.

    ``split_a`` maps each candidate key to {'conflict': [...], 'agreement': [...],
    'nopatch': [...]}; ``split_b`` holds the same for the selected cell only, keyed by
    candidate. ``candidates`` is the declared ordered list of (key, cell).
    """
    w = CONTRACT["w"]
    if not 1 <= len(candidates) <= 5:
        raise ValueError("between one and five pre-declared candidate cells")
    estimates, per_cell = [], {}
    for key, cell in candidates:
        cell = check_cell(cell, n, w)
        data = split_a[key]
        s_min = anchored_s_min([q_map(softmax(x), cell, w)[0] for x in data["nopatch"]],
                               s_min_quantile, s_min_floor)
        try:
            estimate = anchors(data["conflict"], data["agreement"], cell, w, s_min)
        except ValueError as error:
            estimate = {"d": -math.inf, "error": str(error)}
        per_cell[key] = {"cell": cell, "s_min": s_min, "split_A": estimate}
        estimates.append((key, estimate["d"] if math.isfinite(estimate["d"]) else -1.0))
    selection = select_cell(estimates, d_min)
    result = {"candidates": per_cell, "selection": selection}
    if selection["status"] == NOT_DECIDABLE:
        return {**result, "status": "STOP", "level": "S1", "reason": NOT_DECIDABLE}
    key = selection["selected"]
    cell, s_min = per_cell[key]["cell"], per_cell[key]["s_min"]
    b = split_b[key]
    measured = [case_measures(x, cell, w, s_min) for x in b["conflict"]]
    rate = sum(c["resolved"] for c in measured) / len(measured)
    result["resolution_rate_B"] = rate
    if rate < CONTRACT["resolution_rate_floor"]:
        return {**result, "status": "STOP", "level": "S1",
                "reason": "support resolution rate on split B below 0.90 (population STOP)"}
    anchor = anchors(b["conflict"], b["agreement"], cell, w, s_min)
    result["anchors"] = anchor
    if anchor["agreement_transfer_rate"] < agreement_transfer_floor:
        return {**result, "status": "STOP", "level": "S1",
                "reason": "agreement control did not move the answer at the declared rate"}
    if anchor["d"] < d_min:
        return {**result, "status": "STOP", "level": "S1",
                "reason": "separation d_c below d_min on split B"}
    resolved_q = [c["q"] for c in measured if c["resolved"]]
    delta = delta_threshold(resolved_q, len(resolved_q), N, false_invalid_rate, resamples, seed)
    result["freeze"] = {
        "cell": cell, "s_min": s_min, "d_min": d_min,
        "agreement_transfer_floor": agreement_transfer_floor,
        "anchors": {"T_W": anchor["T_W"], "T_A": anchor["T_A"], "d": anchor["d"],
                    "q_bar_B": anchor["q_bar"], "m_B": anchor["m_resolved"]},
        "mean_gate": {"delta": delta, "false_invalid_rate": false_invalid_rate,
                      "resamples": resamples, "seed": seed, "N": N},
        "development_gates": {"resolution_rate_B": rate,
                              "agreement_transfer_rate_B": anchor["agreement_transfer_rate"]},
    }
    return {**result, "status": "PROCEED"}


# ---- confirmation --------------------------------------------------------------------

def _validated_manifest(manifest):
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    rule = manifest.get("rule")
    if not isinstance(rule, Mapping):
        raise ValueError("manifest.rule missing")
    for key, value in CONTRACT.items():
        if rule.get(key) != value:
            raise ValueError(f"manifest.rule.{key} differs from the declared contract")
    n = _index(manifest.get("n_groups"), "n_groups")
    if n < 7:
        raise ValueError("n_groups must be at least 7")
    cell = check_cell(manifest.get("cell"), n, rule["w"])
    N = _index(manifest.get("N"), "N")
    if N < 1:
        raise ValueError("N must be positive")
    s_min = _finite(rule.get("s_min"), "s_min")
    d_min = _finite(rule.get("d_min"), "d_min")
    if not 0 < s_min < 1 or d_min <= 0:
        raise ValueError("require 0 < s_min < 1 and d_min > 0")
    a = manifest.get("anchors")
    if not isinstance(a, Mapping):
        raise ValueError("manifest.anchors missing")
    t_w, t_a, d = (_finite(a.get(k), k) for k in ("T_W", "T_A", "d"))
    q_bar = a.get("q_bar_B")
    if not isinstance(q_bar, list) or len(q_bar) != 3:
        raise ValueError("anchors.q_bar_B must hold three numbers")
    q_bar = [_finite(x, "q_bar_B") for x in q_bar]
    if abs(d - (t_a - t_w)) > 1e-12:
        raise ValueError("anchors.d must equal T_A - T_W")
    if d < d_min:
        raise ValueError("frozen separation d is below d_min")
    if abs(concentration(q_bar) - t_w) > 1e-12:
        raise ValueError("anchors.T_W must equal max(q_bar_B)")
    gate = manifest.get("mean_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("manifest.mean_gate missing")
    delta = _finite(gate.get("delta"), "delta")
    if delta <= 0:
        raise ValueError("delta must be positive")
    development = manifest.get("development_gates")
    if not isinstance(development, Mapping):
        raise ValueError("manifest.development_gates missing")
    if _finite(development.get("resolution_rate_B"), "resolution_rate_B") < CONTRACT["resolution_rate_floor"]:
        raise ValueError("frozen development resolution rate is below 0.90")
    if (_finite(development.get("agreement_transfer_rate_B"), "agreement_transfer_rate_B")
            < _finite(rule.get("agreement_transfer_floor"), "agreement_transfer_floor")):
        raise ValueError("frozen agreement-control transfer rate is below its floor")
    return {"n": n, "cell": cell, "N": N, "s_min": s_min, "d_min": d_min, "T_W": t_w,
            "T_A": t_a, "d": d, "q_bar_B": q_bar, "delta": delta, "rule": dict(rule)}


def _validated_records(records, frozen):
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be a list of case records")
    records = list(records)
    if not records:
        raise ValueError("no records")
    seen, qualifying = set(), []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("each record must be a mapping")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("case_id must be unique and nonempty (missing or duplicated record)")
        seen.add(case_id)
        if record.get("draw_index") != position:
            raise ValueError("draw_index must run 0, 1, 2, ... in order (missing, duplicated or reordered record)")
        if not isinstance(record.get("qualifies"), bool):
            raise ValueError(f"{case_id}: qualifies must be true or false")
        if record["qualifies"]:
            qualifying.append(record)
    if len(qualifying) != frozen["N"]:
        raise ValueError(f"expected N = {frozen['N']} qualifying cases, found {len(qualifying)}")
    if not records[-1]["qualifies"]:
        raise ValueError("generation must stop at the N-th qualifying case")
    return records, qualifying


def _technical(record, frozen):
    """Raise TechnicalFailure for a qualifying record that cannot be measured."""
    case_id = record["case_id"]
    technical = record.get("technical")
    if not isinstance(technical, Mapping) or technical.get("passed") is not True:
        raise TechnicalFailure(f"{case_id}: technical checks did not pass")
    if record.get("design_indices") != frozen["cell"]:
        raise TechnicalFailure(f"{case_id}: design indices differ from the frozen cell")
    logits = record.get("entity_logits")
    if not isinstance(logits, list) or len(logits) != frozen["n"]:
        raise TechnicalFailure(f"{case_id}: expected {frozen['n']} entity logits")
    try:
        values = [_finite(v, "logit") for v in logits]
        mass = _finite(record.get("entity_mass_full_vocab"), "entity_mass_full_vocab")
    except ValueError as error:
        raise TechnicalFailure(f"{case_id}: {error}") from error
    if not 0 <= mass <= 1:
        raise TechnicalFailure(f"{case_id}: entity mass outside [0, 1]")
    return values, mass


def _status(k_adequacy, k_exclusion, N):
    lower = clopper_pearson(k_adequacy, N, CONTRACT["alpha"], CONTRACT["family_size"])
    upper = clopper_pearson(k_exclusion, N, CONTRACT["alpha"], CONTRACT["family_size"])
    status = ("adequate" if lower[0] > CONTRACT["coverage"] else
              "excluded" if upper[1] < CONTRACT["coverage"] else "undecided")
    return {"k_match": k_adequacy, "k_match_plus_unresolved": k_exclusion, "N": N,
            "adequacy_interval": list(lower), "exclusion_interval": list(upper),
            "lower_bound_for_adequacy": lower[0], "upper_bound_for_exclusion": upper[1],
            "status": status}


def level(run_status, statuses):
    if run_status != "VALID":
        return "S1"
    w_out, a_out = statuses["W_T"] == "excluded", statuses["A_T"] == "excluded"
    if w_out and a_out:
        return "S3 (both excluded)"
    if a_out:
        return "S3 (A_T excluded)"
    if w_out:
        return "S3 (W_T excluded only)"
    return "S2"


def analyze_confirmation(manifest, records):
    """Statuses, gates, level and the between-case check for one frozen confirmation.

    Manifest (frozen before confirmation): ``n_groups``, ``cell`` (0-based i_P, i_L,
    i_R, i_N), ``N``, ``rule`` (the CONTRACT plus s_min, d_min and
    agreement_transfer_floor), ``anchors`` (T_W, T_A, d, q_bar_B, m_B), ``mean_gate``
    (delta and how it was drawn) and ``development_gates`` (resolution_rate_B,
    agreement_transfer_rate_B).

    Records, one per generated base context, in generation order::

        {'case_id': unique, 'draw_index': 0, 1, 2, ..., 'qualifies': bool,
         # only for qualifying cases (native checks passed):
         'design_indices': {...}, 'technical': {'passed': bool, ...},
         'entity_logits': [n numbers at the last position], 'entity_mass_full_vocab': x}

    Generation stops at the N-th qualifying case, so the yield is N / len(records).
    """
    frozen = _validated_manifest(manifest)
    records, qualifying = _validated_records(records, frozen)
    N, cell, w = frozen["N"], frozen["cell"], CONTRACT["w"]
    base = {"contract": dict(CONTRACT), "N": N, "generated": len(records),
            "yield": N / len(records), "cell": cell}
    try:
        measured = []
        for record in qualifying:
            logits, mass = _technical(record, frozen)
            case = case_measures(logits, cell, w, frozen["s_min"])
            case.update(case_id=record["case_id"], entity_mass_full_vocab=mass)
            measured.append(case)
    except TechnicalFailure as failure:
        return {**base, "run_status": "INVALID", "invalid_reason": f"technical failure: {failure}",
                "level": "S1", "statuses": {p: "INVALID" for p in PROFILES},
                "between_case": {"earned": False, "reason": "run INVALID"}}

    band = CONTRACT["kappa"] * frozen["d"]
    for case in measured:
        case["W_T"] = case["resolved"] and abs(case["T"] - frozen["T_W"]) <= band
        case["A_T"] = case["resolved"] and abs(case["T"] - frozen["T_A"]) <= band
        if case["W_T"] and case["A_T"]:
            raise ValueError("a case conforms to both profiles; kappa must stay below 0.5")
    resolved = [c for c in measured if c["resolved"]]
    u = N - len(resolved)

    if resolved:
        q_bar = [math.fsum(c["q"][i] for c in resolved) / len(resolved) for i in range(3)]
        statistic = max(abs(a - b) for a, b in zip(frozen["q_bar_B"], q_bar))
        gate = {"statistic": statistic, "delta": frozen["delta"], "q_bar_confirmation": q_bar,
                "passed": statistic <= frozen["delta"]}
    else:
        gate = {"statistic": None, "delta": frozen["delta"], "q_bar_confirmation": None,
                "passed": False, "note": "no resolved case: the gate cannot be evaluated"}
    run_status = "VALID" if gate["passed"] else "INVALID"

    profiles = {}
    for profile in PROFILES:
        k = sum(c[profile] for c in measured)
        profiles[profile] = _status(k, k + u, N)
    statuses = ({p: profiles[p]["status"] for p in PROFILES} if run_status == "VALID"
                else {p: "INVALID" for p in PROFILES})

    counts = {label: sum(label in c["labels"] for c in resolved) for label in LABELS}
    upper = {label: clopper_pearson(counts[label] + u, N, CONTRACT["alpha"],
                                    CONTRACT["family_size"])[1] for label in LABELS}
    conditions = {"A_T_adequate": run_status == "VALID" and statuses["A_T"] == "adequate",
                  "mean_gate_passed": gate["passed"],
                  "max_U_below_coverage": max(upper.values()) < CONTRACT["coverage"]}
    shares = {
        "resolved": len(resolved), "unresolved": u,
        "above_T_W": sum(c["T"] > frozen["T_W"] for c in resolved),
        "below_T_W": sum(c["T"] < frozen["T_W"] for c in resolved),
        "deviating_above": sum(c["T"] - frozen["T_W"] > band for c in resolved),
        "deviating_below": sum(frozen["T_W"] - c["T"] > band for c in resolved),
    }
    return {
        **base,
        "run_status": run_status,
        "invalid_reason": None if run_status == "VALID" else "stale anchor (mean-consistency gate)",
        "level": level(run_status, statuses),
        "u": u,
        "statuses": statuses,
        "profiles": profiles,
        "band_half_width": band,
        "anchors": {k: frozen[k] for k in ("T_W", "T_A", "d", "q_bar_B")},
        "mean_gate": gate,
        "between_case": {
            "sentence": BETWEEN_CASE_SENTENCE,
            "label_counts_resolved": counts,
            "ties_resolved": sum(len(c["labels"]) > 1 for c in resolved),
            "U": upper, "max_U": max(upper.values()),
            "conditions": conditions,
            "earned": all(conditions.values()),
        },
        "shares_vs_T_W": shares,
        "w_t_reading": W_T_READING,
        "w_t_caveat": W_T_CAVEAT,
        "descriptive": {
            "p_native_median": statistics.median(c["p_native"] for c in measured),
            "entity_mass_full_vocab_median": statistics.median(c["entity_mass_full_vocab"] for c in measured),
            "S_median": statistics.median(c["S"] for c in measured),
            "background_entities": measured[0]["background_entities"],
        },
        "cases": [{k: c[k] for k in ("case_id", "S", "T", "resolved", "labels", "W_T", "A_T")}
                  for c in measured],
    }


def post_hoc_strictness(summary, kappas=(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)):
    """Post hoc description of how strict kappa was: match counts only, no status."""
    anchors_ = summary["anchors"]
    rows = {}
    for kappa in kappas:
        if not 0 <= kappa < 0.5:
            raise ValueError("kappa must lie in [0, 0.5)")
        band = kappa * anchors_["d"]
        rows[f"{kappa:.2f}"] = {
            profile: sum(c["resolved"] and abs(c["T"] - anchors_[key]) <= band
                         for c in summary["cases"])
            for profile, key in (("W_T", "T_W"), ("A_T", "T_A"))}
    return {"status": "post hoc descriptive; decides nothing", "u": summary["u"],
            "N": summary["N"], "matches_by_kappa": rows}
