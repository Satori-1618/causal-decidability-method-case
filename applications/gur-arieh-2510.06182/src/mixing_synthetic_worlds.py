"""Six synthetic worlds that verify the Round 1 analyzer. SYNTHETIC: not evidence.

Each world builds development splits A and B and a confirmation set from designed
per-case entity distributions (no model), runs the development gates and, if they pass,
the confirmation analysis. Targets cycle in a fixed order and jitter comes in
antithetic pairs, so split B and the confirmation share their mean unless a world
changes it on purpose (the stale anchor).

VERIFICATION_PARAMETERS serve analyzer verification only. The values proposed for the
real run are in PROPOSED_VALUES.json and await the user's approval.
"""
import math
import random

from mixing_round1_analysis import CONTRACT, analyze_confirmation, development_decision

N_GROUPS = 7
CELL = {"i_P": 3, "i_L": 1, "i_R": 5, "i_N": 0}
VERIFICATION_PARAMETERS = {
    "s_min_quantile": 0.99, "s_min_floor": 0.10, "d_min": 0.20,
    "agreement_transfer_floor": 0.90, "false_invalid_rate": 0.05,
    "resamples": 2000, "seed": 20260928, "N": 200,
}
SIZES = {"A": 60, "B": 200}
LOW_SUPPORT = 0.03

W_LIKE = (0.45, 0.30, 0.25)
HALF_W_LIKE = (0.50, 0.30, 0.20)
CONCENTRATED = ((0.92, 0.04, 0.04), (0.04, 0.92, 0.04), (0.04, 0.04, 0.92))
OUTSIDE_PAIR = ((0.75, 0.10, 0.15), (0.15, 0.10, 0.75))


def logits(q, support, cell=CELL, n=N_GROUPS, spread=0.15):
    """Entity logits whose softmax puts q*support on (P-window, i_L, i_R); the
    positional share is spread over the window; the remainder goes mostly to i_N."""
    p = [0.0] * n
    i_P = cell["i_P"]
    p[i_P] += q[0] * support * (1 - 2 * spread)
    p[i_P - 1] += q[0] * support * spread
    p[i_P + 1] += q[0] * support * spread
    p[cell["i_L"]] += q[1] * support
    p[cell["i_R"]] += q[2] * support
    inside = {i_P - 1, i_P, i_P + 1, cell["i_L"], cell["i_R"], cell["i_N"]}
    others = [j for j in range(n) if j not in inside]
    rest = 1 - support
    p[cell["i_N"]] += 0.9 * rest
    for j in others:
        p[j] += 0.1 * rest / len(others)
    p = [max(x, 1e-9) for x in p]
    total = math.fsum(p)
    return [math.log(x / total) for x in p]


def agreement_logits(j, cell=CELL, n=N_GROUPS):
    """All three pointers on entity j: most mass at j."""
    p = [0.02 / (n - 2)] * n
    p[j] = 0.95
    p[cell["i_N"]] = 0.03
    total = math.fsum(p)
    return [math.log(x / total) for x in p]


def nopatch_logits(cell=CELL, n=N_GROUPS):
    p = [0.03 / (n - 1)] * n
    p[cell["i_N"]] = 0.97
    return [math.log(x) for x in p]


def _jitter(base, eps, rng, count):
    """Antithetic pairs around base, kept positive and renormalized."""
    out = []
    while len(out) < count:
        e = [rng.uniform(-eps, eps) for _ in range(3)]
        for sign in (1, -1):
            q = [max(b + sign * x, 1e-3) for b, x in zip(base, e)]
            total = sum(q)
            out.append(tuple(v / total for v in q))
    return out[:count]


def conflict_cases(kind, count, rng):
    """(q, S) per case for one world's conflict condition."""
    supports = [0.6, 0.7, 0.8, 0.9]
    if kind == "w_inside":
        qs = _jitter(W_LIKE, 0.02, rng, count)
    elif kind == "w_outside":
        a, b = (_jitter(base, 0.02, rng, count) for base in OUTSIDE_PAIR)
        qs = [(a if i % 2 == 0 else b)[i // 2] for i in range(count)]
    elif kind in ("heterogeneous", "below_resolution"):
        pools = [_jitter(base, 0.01, rng, count) for base in CONCENTRATED]
        qs = [pools[i % 3][i // 3] for i in range(count)]
    elif kind == "half":
        w_like = _jitter(HALF_W_LIKE, 0.02, rng, count)
        pools = [_jitter(base, 0.01, rng, count) for base in CONCENTRATED]
        qs = [w_like[i // 2] if i % 2 == 0 else pools[(i // 2) % 3][i // 6] for i in range(count)]
    elif kind == "all_lexical":
        qs = _jitter(CONCENTRATED[1], 0.01, rng, count)
    elif kind == "unresolved_heavy":
        qs = _jitter(W_LIKE, 0.02, rng, count)
    else:
        raise ValueError(f"unknown world kind {kind}")
    cases = []
    for i, q in enumerate(qs):
        support = supports[i % len(supports)]
        if kind == "below_resolution" and i % 10 in (0, 3, 6):
            support = LOW_SUPPORT
        if kind == "unresolved_heavy" and i % 20 < 7:
            support = LOW_SUPPORT
        cases.append((q, support))
    return cases


def development_split(kind, count, rng, cell=CELL):
    conflict = [logits(q, s, cell) for q, s in conflict_cases(kind, count, rng)]
    agreement = [(k, j, agreement_logits(j, cell)) for k in range(count)
                 for j in (cell["i_P"], cell["i_L"], cell["i_R"])]
    return {"conflict": conflict, "agreement": agreement,
            "nopatch": [nopatch_logits(cell) for _ in range(count)]}


def confirmation_records(kind, N, rng, world, cell=CELL):
    """Records in generation order; every fifth generated context fails the native
    checks and does not qualify; generation stops at the N-th qualifying case."""
    cases = conflict_cases(kind, N, rng)
    records, used = [], 0
    while used < N:
        index = len(records)
        qualifies = index % 5 != 4
        record = {"case_id": f"{world}-{index:04d}", "draw_index": index, "qualifies": qualifies,
                  "native": {"recipient_correct": qualifies, "donor_correct": True,
                             "readout_matches_generation": True}}
        if qualifies:
            q, support = cases[used]
            record.update(design_indices=dict(cell), technical={"passed": True},
                          entity_logits=logits(q, support, cell), entity_mass_full_vocab=0.9)
            used += 1
        records.append(record)
    return records


def manifest_from(freeze, N):
    return {
        "schema_version": 1, "application": "gur-arieh-2510.06182", "round": 1,
        "stage": "confirmation", "synthetic": True, "n_groups": N_GROUPS,
        "task": "synthetic", "t_entity": 2, "patch_positions": [-1],
        "cell": freeze["cell"], "N": N,
        "rule": {**CONTRACT, "s_min": freeze["s_min"], "d_min": freeze["d_min"],
                 "agreement_transfer_floor": freeze["agreement_transfer_floor"]},
        "anchors": freeze["anchors"], "mean_gate": freeze["mean_gate"],
        "development_gates": freeze["development_gates"],
    }


WORLDS = {
    # name: (development kind, confirmation kind)
    "1a_w_band_inside": ("w_inside", "w_inside"),
    "1b_w_band_outside": ("w_outside", "w_outside"),
    "2_heterogeneous_concentrated": ("heterogeneous", "heterogeneous"),
    "3_half_and_half": ("half", "half"),
    "4_below_resolution": ("below_resolution", None),
    "5_stale_anchor": ("heterogeneous", "all_lexical"),
    "6_unresolved_heavy": ("w_inside", "unresolved_heavy"),
}


def run_world(name, parameters=VERIFICATION_PARAMETERS):
    development_kind, confirmation_kind = WORLDS[name]
    rng = random.Random(f"{name}:{parameters['seed']}")
    split_a = {"c1": development_split(development_kind, SIZES["A"], rng)}
    split_b = {"c1": development_split(development_kind, SIZES["B"], rng)}
    decision = development_decision(
        split_a, split_b, [("c1", CELL)], n=N_GROUPS,
        s_min_quantile=parameters["s_min_quantile"], s_min_floor=parameters["s_min_floor"],
        d_min=parameters["d_min"], agreement_transfer_floor=parameters["agreement_transfer_floor"],
        N=parameters["N"], false_invalid_rate=parameters["false_invalid_rate"],
        resamples=parameters["resamples"], seed=parameters["seed"])
    result = {"world": name, "development": decision}
    if decision["status"] != "PROCEED":
        return result
    manifest = manifest_from(decision["freeze"], parameters["N"])
    records = confirmation_records(confirmation_kind, parameters["N"], rng, name)
    result.update(manifest=manifest, records=records,
                  summary=analyze_confirmation(manifest, records))
    return result


def outcome(result):
    """A compact, platform-stable view of one world's result."""
    decision = result["development"]
    if decision["status"] != "PROCEED":
        return {"development": decision["status"], "level": decision["level"],
                "reason": decision["reason"]}
    s = result["summary"]
    return {
        "development": "PROCEED",
        "run_status": s["run_status"], "invalid_reason": s["invalid_reason"],
        "level": s["level"], "statuses": s["statuses"],
        "descriptive_statuses_ignoring_gate": {p: s["profiles"][p]["status"] for p in s["profiles"]},
        "k_W": s["profiles"]["W_T"]["k_match"], "k_A": s["profiles"]["A_T"]["k_match"],
        "u": s["u"], "N": s["N"], "generated": s["generated"],
        "between_case_earned": s["between_case"]["earned"],
        "T_W": round(s["anchors"]["T_W"], 4), "T_A": round(s["anchors"]["T_A"], 4),
        "d": round(s["anchors"]["d"], 4),
        "mean_gate_passed": s["mean_gate"]["passed"],
        "max_U": round(s["between_case"]["max_U"], 4),
    }


def run_all():
    return {name: outcome(run_world(name)) for name in WORLDS}
