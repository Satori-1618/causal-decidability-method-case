"""Round 1 analyzer: does a single case look like the cell average? No model inference.

Protocol v2 (approved by the user on 28 September 2026, after the first pilot).

**Primary readout: the answer form.** For each run the runner records the logits, at the
last position, of the n in-context entities in their answer form (capitalised, no
leading space, exactly one token: 'Country') and the full-vocabulary probability on
those n tokens (the answer-token mass). p_k is the softmax over the n answer-form
logits. The paper's in-context readout (' country') is recorded under a separate
``descriptive`` field; no function in this module reads that field, so it cannot enter
qualification, cell selection, a gate or a decision.

With the cell's design indices (i_P, i_L, i_R) and the window w:

    S_k = P-window mass + p_k[i_L] + p_k[i_R],     P-window = i_P - w .. i_P + w
    q_k = (P-window mass, p_k[i_L], p_k[i_R]) / S_k
    T_k = max(q_k), between 1/3 and 1

A case is **resolved** iff S_k >= s_min AND its answer-token mass >= 0.5. Too little mass
is "unresolved"; a missing or non-finite measurement is a technical failure (INVALID).

Two concentration profiles, fixed on development split B for one cell:

    T_W = T(mean_k q_k)                    equal weights over resolved cases (q, not p)
    T_A = mean over j in {P, L, R} of (mean over resolved agreement runs with common
          target j of T(q)), same q-map, so P, L and R weigh equally
    d   = T_A - T_W, signed, at least d_min
    W_T-conform: resolved and |T_k - T_W| <= kappa * d
    A_T-conform: resolved and |T_k - T_A| <= kappa * d

They compare how concentrated a case is, never which target it favours. Declared
unresolved rule (new; the Makelov round 2/3A analyzers counted unresolved units as
non-successes for both bounds): unresolved cases count as non-matches for adequacy and
as matches for exclusion. Technical failures make the run INVALID; they are never
unresolved. Intervals are simultaneous Clopper-Pearson over the two profiles, alpha/4
per tail, from the repository's existing helper. At least 90% of agreement-control runs
must be resolved, else STOP.

**N.** The final N, and whether adequacy counts as powered, are set at the freeze by the
recorded N rule applied to split B's unresolved rate in the selected cell (``n_rule``);
a pilot's rate gives only a provisional planning N.
"""
import functools
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

# Fixed by the brief (Sec. 5.4, 5.5, 5.8, 5.11) and by the user's corrections of
# 2026-09-28 (agreement-resolution floor; protocol v2: readout and mass floor).
CONTRACT = {
    "w": 1,
    "kappa": 0.25,
    "coverage": 0.8,
    "alpha": 0.05,
    "family_size": 2,
    "label_tail": 0.0125,
    "resolution_rate_floor": 0.9,
    "agreement_resolution_floor": 0.9,
    "readout": "answer form: capitalised, no leading space, exactly one token",
    "answer_mass_floor": 0.5,
    "resolution_rule": "S >= s_min and answer-token mass >= answer_mass_floor",
    "unresolved_rule": "non-match for adequacy; match for exclusion",
}
PRIMARY_FIELDS = ("answer_logits", "answer_mass_full_vocab")
PROFILES = ("W_T", "A_T")
LABELS = ("positional", "lexical", "reflexive")
NOT_DECIDABLE = "NOT_DECIDABLE_WITH_CURRENT_INTERVENTIONS"
W_T_READING = ("More than 20% of cases are resolved and deviate from the cell-average "
               "concentration by more than kappa*d_c.")
W_T_CAVEAT = ("W_T is biased toward exclusion when case argmaxes vary; the bias is largest "
              "in the selected cell.")
BETWEEN_CASE_SENTENCE = "Cases differ in which candidate position they favour."

# The N rule, recorded on 2026-09-28 before the first pilot. The final N is set at the
# freeze from split B's unresolved rate in the selected cell.
N_RULE = {
    "default_N": 200,
    "unresolved_threshold": 0.02,
    "larger_N": (300, 400, 500),
    "declared_power": 0.80,
    "adequacy_coverage": 0.90,
    "exclusion_coverage": 0.70,
}


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
    """Softmax over the n entity logits (the per-case distribution p_k)."""
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


def _shape(logits, cell, w, s_min):
    p = softmax(logits)
    support, q = q_map(p, cell, w)
    labels, background_size = background_corrected_labels(p, cell, w)
    return {
        "S": support,
        "q": list(q) if q is not None else None,
        "T": concentration(q) if q is not None else None,
        "S_ok": q is not None and support >= s_min,
        "p_native": p[cell["i_N"]],
        "argmax": max(range(len(p)), key=p.__getitem__),
        "all_labels": labels,
        "background_entities": background_size,
    }


def describe_logits(logits, cell, w, s_min):
    """DESCRIPTIVE ONLY (e.g. the paper's in-context readout): S, q, T and labels of a
    bare logit vector, 'resolved' by S alone. Never used by any decision below."""
    shape = _shape(logits, cell, w, s_min)
    return {"S": shape["S"], "q": shape["q"], "T": shape["T"], "resolved": shape["S_ok"],
            "p_native": shape["p_native"], "labels": shape["all_labels"] if shape["S_ok"] else [],
            "background_entities": shape["background_entities"]}


def primary_readout(item, n=None):
    """The primary measurement of one run: (answer-form logits, answer-token mass).

    Reads only ``answer_logits`` and ``answer_mass_full_vocab``. A missing, malformed or
    non-finite value raises TechnicalFailure: the run is INVALID, never unresolved."""
    if not isinstance(item, Mapping):
        raise TechnicalFailure("measurement must be a mapping with the primary readout fields")
    logits, mass = item.get("answer_logits"), item.get("answer_mass_full_vocab")
    if not isinstance(logits, list) or len(logits) < 2 or (n is not None and len(logits) != n):
        raise TechnicalFailure(f"expected {n or 'at least two'} answer-form logits")
    try:
        values = [_finite(v, "answer logit") for v in logits]
        mass = _finite(mass, "answer_mass_full_vocab")
    except ValueError as error:
        raise TechnicalFailure(str(error)) from error
    if not 0 <= mass <= 1:
        raise TechnicalFailure("answer-token mass outside [0, 1]")
    return values, mass


def case_measures(item, cell, w, s_min):
    """Primary per-case measures: resolved iff S >= s_min and answer mass >= 0.5."""
    logits, mass = primary_readout(item)
    shape = _shape(logits, cell, w, s_min)
    mass_ok = mass >= CONTRACT["answer_mass_floor"]
    resolved = shape["S_ok"] and mass_ok
    return {
        "S": shape["S"], "q": shape["q"], "T": shape["T"],
        "answer_mass": mass, "S_ok": shape["S_ok"], "mass_ok": mass_ok,
        "resolved": resolved, "p_native": shape["p_native"], "argmax": shape["argmax"],
        "labels": shape["all_labels"] if resolved else [],
        "background_entities": shape["background_entities"],
    }


# ---- development: s_min, anchors, selection, delta, N ---------------------------------

def upper_order_statistic(values, level):
    """The smallest sample value v with at least a fraction ``level`` of values <= v."""
    values = sorted(_finite(v, "value") for v in values)
    if not values or not 0 < level <= 1:
        raise ValueError("need values and 0 < level <= 1")
    return values[max(0, math.ceil(level * len(values)) - 1)]


def anchored_s_min(nopatch_supports, quantile, floor):
    """s_min anchored to the S of no-patch runs: max(upper order statistic, floor)."""
    floor = _finite(floor, "floor")
    if floor < 0:
        raise ValueError("floor must be nonnegative")
    return max(upper_order_statistic(nopatch_supports, quantile), floor)


def nopatch_supports(items, cell, w):
    """S of the primary readout of no-patch runs (all qualifying families of a cell)."""
    return [q_map(softmax(primary_readout(item)[0]), cell, w)[0] for item in items]


AGREEMENT_TARGETS = ("i_P", "i_L", "i_R")


def _agreement(agreement, cell, measure):
    targets = {cell[key]: key for key in AGREEMENT_TARGETS}
    per_target = {key: [] for key in AGREEMENT_TARGETS}
    runs = {key: 0 for key in AGREEMENT_TARGETS}
    transfers = resolved = 0
    for _, j, item in agreement:
        if j not in targets:
            raise ValueError(f"agreement target {j} is not one of the cell's i_P, i_L, i_R")
        key = targets[j]
        runs[key] += 1
        measured = measure(item)
        transfers += measured["argmax"] == j
        if measured["resolved"]:
            resolved += 1
            per_target[key].append(measured["T"])
    total = len(agreement)
    means = {key: (math.fsum(v) / len(v) if v else None) for key, v in per_target.items()}
    missing = [key for key, mean in means.items() if mean is None]
    return {
        "T_A": (None if missing else math.fsum(means.values()) / len(means)),
        "T_A_by_target": means,
        "missing_targets": missing,
        "agreement_runs_by_target": runs,
        "agreement_resolved_by_target": {key: len(v) for key, v in per_target.items()},
        "agreement_resolved": resolved, "agreement_total": total,
        "agreement_resolution_rate": resolved / total if total else None,
        "agreement_transfer_rate": transfers / total if total else None,
    }


def _anchors(conflict, agreement, cell, measure):
    measured = [measure(x) for x in conflict]
    resolved = [c["q"] for c in measured if c["resolved"]]
    if not resolved:
        raise ValueError("no resolved conflict case: anchors undefined")
    q_bar = [math.fsum(q[i] for q in resolved) / len(resolved) for i in range(3)]
    rates = _agreement(agreement, cell, measure)
    if rates["T_A"] is None:
        raise ValueError("no resolved agreement run for target(s) "
                         f"{', '.join(rates['missing_targets'])}: T_A undefined")
    t_w = concentration(q_bar)
    return {
        "T_W": t_w, "T_A": rates["T_A"], "d": rates["T_A"] - t_w, "q_bar": q_bar,
        "m_resolved": len(resolved), "m_total": len(measured),
        "resolution_rate": len(resolved) / len(measured),
        **{k: v for k, v in rates.items() if k not in ("T_A", "missing_targets")},
    }


def _primary(cell, w, s_min):
    def measure(item):
        return case_measures(item, cell, w, s_min)
    return measure


def _descriptive(cell, w, s_min):
    def measure(logits):
        shape = _shape(logits, cell, w, s_min)
        return {**shape, "resolved": shape["S_ok"]}
    return measure


def agreement_anchor(agreement, cell, w, s_min):
    """T_A with P, L and R weighted equally, plus the agreement-control rates (primary
    readout). ``agreement`` is a list of (case_index, j, measurement), j the common
    target; a run transfers if the argmax over the n answer-form logits is j and is
    resolved under the combined gate."""
    return _agreement(agreement, cell, _primary(cell, w, s_min))


def anchors(conflict, agreement, cell, w, s_min):
    """T_W, T_A and d from one split, primary readout and combined resolution gate."""
    return _anchors(conflict, agreement, cell, _primary(cell, w, s_min))


def describe_anchors(conflict_logits, agreement_logits, cell, w, s_min):
    """DESCRIPTIVE ONLY: the same anchors from bare logit vectors, resolution by S alone
    (for the paper's in-context readout). Never used by any decision."""
    return _anchors(conflict_logits, agreement_logits, cell, _descriptive(cell, w, s_min))


def describe_agreement_anchor(agreement_logits, cell, w, s_min):
    """DESCRIPTIVE ONLY counterpart of ``agreement_anchor`` for bare logit vectors."""
    return _agreement(agreement_logits, cell, _descriptive(cell, w, s_min))


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


def _pmf(k, n, p):
    if p in (0.0, 1.0):
        return float(k == (n if p == 1.0 else 0))
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    + k * math.log(p) + (n - k) * math.log1p(-p))


@functools.lru_cache(maxsize=None)
def _thresholds(N, alpha, family_size, coverage):
    """Smallest k whose CP lower bound exceeds coverage, and largest k whose CP upper
    bound is below it (bisection; both bounds increase with k)."""
    def first(predicate):
        lo, hi = 0, N + 1
        while lo < hi:
            mid = (lo + hi) // 2
            if predicate(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    k_adequate = first(lambda k: clopper_pearson(k, N, alpha, family_size)[0] > coverage)
    k_excluded = first(lambda k: not clopper_pearson(k, N, alpha, family_size)[1] < coverage) - 1
    return k_adequate, k_excluded


def power(N, true_coverage, unresolved_rate=0.0, alpha=0.05, family_size=2, coverage=0.8):
    """Exact power of the declared rule for one profile, assuming unresolved cases occur
    independently of conformity: adequacy counts them as non-matches, exclusion as
    matches."""
    match = true_coverage * (1 - unresolved_rate)
    k_adequate, k_excluded = _thresholds(N, alpha, family_size, coverage)
    adequate = math.fsum(_pmf(k, N, match) for k in range(k_adequate, N + 1))
    excluded = math.fsum(_pmf(k, N, match + unresolved_rate) for k in range(0, k_excluded + 1))
    return {"adequate": adequate, "excluded": excluded}


def n_rule(unresolved_rate, rule=N_RULE):
    """The recorded N rule.

    N = 200 if the unresolved rate is at most 2%; otherwise the smallest N in
    (300, 400, 500) whose A_T-adequacy power (90% of resolved cases conform) reaches
    0.80; if none does, N = 500 with adequacy labelled not powered, and exclusion power
    (70% conform) must still reach 0.80, else STOP. A pilot's rate gives a provisional
    planning N; the final N comes from split B's rate in the selected cell, at the freeze.
    Exclusion power below 0.80 in another branch is flagged, not acted on.
    """
    u = _finite(unresolved_rate, "unresolved_rate")
    if not 0 <= u <= 1:
        raise ValueError("unresolved rate must lie in [0, 1]")
    target = rule["declared_power"]

    def powers(N):
        return {"N": N,
                "adequacy_power": power(N, rule["adequacy_coverage"], u)["adequate"],
                "exclusion_power": power(N, rule["exclusion_coverage"], u)["excluded"]}

    table = [powers(N) for N in (rule["default_N"], *rule["larger_N"])]
    if u <= rule["unresolved_threshold"]:
        chosen, branch = table[0], "unresolved rate at most 2%: N = 200"
        adequacy_powered = chosen["adequacy_power"] >= target
    else:
        chosen = next((row for row in table[1:] if row["adequacy_power"] >= target), None)
        if chosen is not None:
            branch, adequacy_powered = "smallest N in (300, 400, 500) with adequacy power >= 0.80", True
        else:
            chosen, branch, adequacy_powered = table[-1], "N = 500, adequacy not powered", False
    exclusion_ok = chosen["exclusion_power"] >= target
    stop = branch.startswith("N = 500") and not exclusion_ok
    return {
        "unresolved_rate": u, "N": chosen["N"], "branch": branch,
        "adequacy_power": chosen["adequacy_power"], "adequacy_powered": adequacy_powered,
        "exclusion_power": chosen["exclusion_power"], "exclusion_reaches_declared_power": exclusion_ok,
        "status": "STOP" if stop else "PROCEED",
        "flag": (None if exclusion_ok or stop else
                 "exclusion power below 0.80 at the chosen N; the rule text makes this a STOP "
                 "only when N = 500 and adequacy is not powered; the user decides"),
        "table": table,
    }


def development_decision(split_a, split_b, candidates, *, n, s_min_quantile, s_min_floor,
                         d_min, agreement_transfer_floor, false_invalid_rate, resamples, seed):
    """Gates on development data, then the values to freeze. Every STOP is S1.

    ``split_a`` maps each candidate key to {'conflict': [...], 'agreement': [...],
    'nopatch': [...]}, each item a primary measurement ({'answer_logits': [...],
    'answer_mass_full_vocab': x}; agreement items as (case_index, j, measurement));
    ``split_b`` holds the same for the selected cell. s_min comes from all no-patch runs
    of the cell's qualifying families. The final N and whether adequacy counts as
    powered come from the N rule applied to split B's unresolved rate.
    """
    w = CONTRACT["w"]
    if not 1 <= len(candidates) <= 5:
        raise ValueError("between one and five pre-declared candidate cells")
    estimates, per_cell = [], {}
    for key, cell in candidates:
        cell = check_cell(cell, n, w)
        data = split_a[key]
        s_min = anchored_s_min(nopatch_supports(data["nopatch"], cell, w), s_min_quantile, s_min_floor)
        try:
            estimate = anchors(data["conflict"], data["agreement"], cell, w, s_min)
        except TechnicalFailure:
            raise
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
    if anchor["agreement_resolution_rate"] < CONTRACT["agreement_resolution_floor"]:
        return {**result, "status": "STOP", "level": "S1",
                "reason": "fewer than 90% of agreement-control runs on split B are resolved"}
    if anchor["agreement_transfer_rate"] < agreement_transfer_floor:
        return {**result, "status": "STOP", "level": "S1",
                "reason": "agreement control did not move the answer at the declared rate"}
    if anchor["d"] < d_min:
        return {**result, "status": "STOP", "level": "S1",
                "reason": "separation d_c below d_min on split B"}
    sizing = n_rule(1 - rate)
    result["N_rule"] = sizing
    if sizing["status"] == "STOP":
        return {**result, "status": "STOP", "level": "S1",
                "reason": "power: even N = 500 does not give exclusion power 0.80 (N rule)"}
    N = sizing["N"]
    resolved_q = [c["q"] for c in measured if c["resolved"]]
    delta = delta_threshold(resolved_q, len(resolved_q), N, false_invalid_rate, resamples, seed)
    result["freeze"] = {
        "cell": cell, "s_min": s_min, "d_min": d_min, "N": N,
        "N_rule": {"unresolved_rate_B": 1 - rate, "N": N, "branch": sizing["branch"],
                   "adequacy_powered": sizing["adequacy_powered"],
                   "adequacy_power": sizing["adequacy_power"],
                   "exclusion_power": sizing["exclusion_power"]},
        "agreement_transfer_floor": agreement_transfer_floor,
        "anchors": {"T_W": anchor["T_W"], "T_A": anchor["T_A"], "d": anchor["d"],
                    "q_bar_B": anchor["q_bar"], "m_B": anchor["m_resolved"]},
        "mean_gate": {"delta": delta, "false_invalid_rate": false_invalid_rate,
                      "resamples": resamples, "seed": seed, "N": N},
        "development_gates": {"resolution_rate_B": rate,
                              "agreement_resolution_rate_B": anchor["agreement_resolution_rate"],
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
    resolution_b = _finite(development.get("resolution_rate_B"), "resolution_rate_B")
    if resolution_b < CONTRACT["resolution_rate_floor"]:
        raise ValueError("frozen development resolution rate is below 0.90")
    if (_finite(development.get("agreement_resolution_rate_B"), "agreement_resolution_rate_B")
            < CONTRACT["agreement_resolution_floor"]):
        raise ValueError("frozen agreement-control resolution rate is below 0.90")
    if (_finite(development.get("agreement_transfer_rate_B"), "agreement_transfer_rate_B")
            < _finite(rule.get("agreement_transfer_floor"), "agreement_transfer_floor")):
        raise ValueError("frozen agreement-control transfer rate is below its floor")
    sizing = manifest.get("N_rule")
    if not isinstance(sizing, Mapping):
        raise ValueError("manifest.N_rule missing")
    expected = n_rule(1 - resolution_b)
    if expected["status"] != "PROCEED":
        raise ValueError("the N rule stops at split B's unresolved rate")
    if (N != expected["N"] or sizing.get("N") != N
            or sizing.get("adequacy_powered") is not expected["adequacy_powered"]):
        raise ValueError("frozen N or adequacy label differs from the N rule at split B's unresolved rate")
    return {"n": n, "cell": cell, "N": N, "s_min": s_min, "d_min": d_min, "T_W": t_w,
            "T_A": t_a, "d": d, "q_bar_B": q_bar, "delta": delta, "rule": dict(rule),
            "adequacy_powered": expected["adequacy_powered"]}


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
    """Raise TechnicalFailure for a qualifying record that cannot be measured. Reads the
    primary readout only."""
    case_id = record["case_id"]
    technical = record.get("technical")
    if not isinstance(technical, Mapping) or technical.get("passed") is not True:
        raise TechnicalFailure(f"{case_id}: technical checks did not pass")
    if record.get("design_indices") != frozen["cell"]:
        raise TechnicalFailure(f"{case_id}: design indices differ from the frozen cell")
    try:
        primary_readout(record, frozen["n"])
    except TechnicalFailure as error:
        raise TechnicalFailure(f"{case_id}: {error}") from error
    return {field: record[field] for field in PRIMARY_FIELDS}


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
    i_R, i_N), ``N``, ``N_rule`` (N and the adequacy label from split B's unresolved
    rate), ``rule`` (the CONTRACT plus s_min, d_min and agreement_transfer_floor),
    ``anchors`` (T_W, T_A, d, q_bar_B, m_B), ``mean_gate`` (delta and how it was drawn)
    and ``development_gates`` (resolution_rate_B, agreement_resolution_rate_B,
    agreement_transfer_rate_B).

    Records, one per generated base context, in generation order::

        {'case_id': unique, 'draw_index': 0, 1, 2, ..., 'qualifies': bool,
         # only for qualifying cases (native checks passed):
         'design_indices': {...}, 'technical': {'passed': bool, ...},
         'answer_logits': [n numbers], 'answer_mass_full_vocab': x,
         'descriptive': {...}}   # never read here

    Generation stops at the N-th qualifying case, so the yield is N / len(records).
    """
    frozen = _validated_manifest(manifest)
    records, qualifying = _validated_records(records, frozen)
    N, cell, w = frozen["N"], frozen["cell"], CONTRACT["w"]
    base = {"contract": dict(CONTRACT), "N": N, "generated": len(records),
            "yield": N / len(records), "cell": cell,
            "adequacy_powered": frozen["adequacy_powered"]}
    try:
        measured = []
        for record in qualifying:
            primary = _technical(record, frozen)
            case = case_measures(primary, cell, w, frozen["s_min"])
            case.update(case_id=record["case_id"])
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
        "unresolved_by_support": sum(not c["S_ok"] for c in measured),
        "unresolved_by_answer_mass": sum(c["S_ok"] and not c["mass_ok"] for c in measured),
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
            "answer_mass_median": statistics.median(c["answer_mass"] for c in measured),
            "S_median": statistics.median(c["S"] for c in measured),
            "background_entities": measured[0]["background_entities"],
        },
        "cases": [{k: c[k] for k in ("case_id", "S", "T", "answer_mass", "resolved", "labels", "W_T", "A_T")}
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
