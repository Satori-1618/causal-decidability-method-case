"""Round 1 design at the level of binding matrices, and planning arithmetic. No model.

A binding matrix G is a list of n groups; each group is a tuple of m entities, one per
category. For the proposed task (music, t_entity = 2) a group is (Musician, Genre,
Instrument): the target is entity 2 and the two query entities are 1 and 3, as in the
paper's App. A.1. Indices are 0-based here; the paper counts from 1.

``design_indices`` is the design-index checker. It recomputes, from the recipient G,
the donor G' and the two queries, the indices each candidate points to, by the paper's
definitions (Sec. 3.2):

- i_P: the donor's queried group position in G';
- i_L: the recipient group holding the donor's query entities;
- i_R: the recipient group holding the donor's answer entity (None if absent);
- i_N: the recipient's own queried group.

``target_rebind`` builds the conflict case of one frozen cell, like upstream's
``ppkn_simpler_counterfactual_template_split_key_loc`` but with the cell fixed and
i_L != i_R enforced. ``agreement_control`` is this application's own three-way
construction (not the paper's App. D.1): keep group j fixed, derange the other groups
with their bindings intact, and ask the donor about group j, so that P, L and R all
point to j. Nothing here imports upstream code.
"""
import functools
import math
import random
import sys
from pathlib import Path

import mixing_round1_analysis as _analysis
from mixing_round1_analysis import check_cell

_MAKELOV_SRC = Path(__file__).resolve().parents[2] / "makelov-2311.17030" / "src"
if str(_MAKELOV_SRC) not in sys.path:
    sys.path.insert(0, str(_MAKELOV_SRC))
from query_route_analysis import clopper_pearson  # noqa: E402

TARGET = 1          # t_entity = 2 (0-based position 1 inside a group)
QUERY = (0, 2)      # q_entity = 1 and 3


def keys(group):
    return tuple(group[i] for i in QUERY)


def _position(groups, predicate, label):
    hits = [i for i, g in enumerate(groups) if predicate(g)]
    if len(hits) > 1:
        raise ValueError(f"{label}: entity occurs in more than one group")
    return hits[0] if hits else None


def check_matrix(G):
    if len(G) < 2 or len({len(g) for g in G}) != 1:
        raise ValueError("binding matrix needs at least two groups of equal size")
    for c in range(len(G[0])):
        column = [g[c] for g in G]
        if len(set(column)) != len(column):
            raise ValueError("entities within a category must be distinct")


def design_indices(recipient, donor, recipient_query, donor_query):
    """The design-index checker: which recipient positions each candidate points to."""
    check_matrix(recipient)
    check_matrix(donor)
    i_N = _position(recipient, lambda g: keys(g) == tuple(recipient_query), "recipient query")
    i_P = _position(donor, lambda g: keys(g) == tuple(donor_query), "donor query")
    if i_N is None or i_P is None:
        raise ValueError("a query does not name one bound group")
    i_L = _position(recipient, lambda g: keys(g) == tuple(donor_query), "lexical")
    if i_L is None:
        raise ValueError("the donor's query entities are not bound together in the recipient")
    answer = donor[i_P][TARGET]
    i_R = _position(recipient, lambda g: g[TARGET] == answer, "reflexive")
    return {"i_P": i_P, "i_L": i_L, "i_R": i_R, "i_N": i_N}


def target_rebind(G, cell, n, w=1):
    """Donor G' and both queries for the conflict case of one admissible cell."""
    cell = check_cell(cell, n, w)
    if len(G) != n:
        raise ValueError("binding matrix size differs from n")
    i_P, i_L, i_R, i_N = (cell[k] for k in ("i_P", "i_L", "i_R", "i_N"))
    donor = [list(g) for g in G]
    donor[i_P][TARGET], donor[i_R][TARGET] = G[i_R][TARGET], G[i_P][TARGET]
    for q in QUERY:
        donor[i_P][q], donor[i_L][q] = G[i_L][q], G[i_P][q]
    donor = [tuple(g) for g in donor]
    return {"recipient": [tuple(g) for g in G], "donor": donor,
            "recipient_query": keys(G[i_N]), "donor_query": keys(donor[i_P])}


def derangement(indices, rng):
    """A uniformly drawn permutation of ``indices`` without fixed points."""
    indices = list(indices)
    if len(indices) < 2:
        raise ValueError("a derangement needs at least two elements")
    while True:
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(indices, shuffled)):
            return dict(zip(indices, shuffled))


def agreement_control(G, j, i_N, rng):
    """Keep group j, derange the others with bindings intact, ask the donor about j."""
    n = len(G)
    if not 0 <= j < n or not 0 <= i_N < n or j == i_N:
        raise ValueError("need 0 <= j, i_N < n and j != i_N")
    moves = derangement([i for i in range(n) if i != j], rng)
    donor = [G[j] if i == j else G[moves[i]] for i in range(n)]
    return {"recipient": [tuple(g) for g in G], "donor": [tuple(g) for g in donor],
            "recipient_query": keys(G[i_N]), "donor_query": keys(G[j])}


def random_matrix(n, pools, rng):
    """n groups with distinct entities per category, drawn from the given pools."""
    columns = [rng.sample(pool, n) for pool in pools]
    return [tuple(column[i] for column in columns) for i in range(n)]


def upstream_draw_shares(n, w=1):
    """Exact shares of upstream's main-template draws (training.py:1226-1249) that
    collide (i_L = i_R) or are inadmissible under the window w."""
    total = collide = inadmissible = 0
    for i_N in range(n):
        for i_P in (i for i in range(n) if i != i_N):
            rest = [i for i in range(n) if i not in (i_N, i_P)]
            for i_R in rest:
                for i_L in rest:
                    total += 1
                    collide += i_L == i_R
                    try:
                        check_cell({"i_P": i_P, "i_L": i_L, "i_R": i_R, "i_N": i_N}, n, w)
                    except ValueError:
                        inadmissible += 1
    return {"collision": collide / total, "inadmissible": inadmissible / total}


# ---- planning arithmetic ---------------------------------------------------------------

def _pmf(k, n, p):
    if p in (0.0, 1.0):
        return float(k == (n if p == 1.0 else 0))
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    + k * math.log(p) + (n - k) * math.log1p(-p))


@functools.lru_cache(maxsize=None)
def _decisive_counts(N, alpha, family_size, coverage):
    """Counts k out of N whose interval declares adequacy (lower bound above coverage)
    or exclusion (upper bound below coverage). Cached: one CP evaluation per k and N."""
    adequate, excluded = [], []
    for k in range(N + 1):
        lower, upper = clopper_pearson(k, N, alpha, family_size)
        if lower > coverage:
            adequate.append(k)
        if upper < coverage:
            excluded.append(k)
    return tuple(adequate), tuple(excluded)


def power_by_scan(N, true_coverage, unresolved_rate=0.0, alpha=0.05, family_size=2, coverage=0.8):
    """The same exact power as ``power``, found by scanning every count k (no
    monotonicity assumption); kept as a cross-check of the bisection in the analyzer."""
    match = true_coverage * (1 - unresolved_rate)
    adequate_counts, excluded_counts = _decisive_counts(N, alpha, family_size, coverage)
    adequate = sum(_pmf(k, N, match) for k in adequate_counts)
    excluded = sum(_pmf(k, N, match + unresolved_rate) for k in excluded_counts)
    return {"adequate": adequate, "excluded": excluded}


# The N rule and exact power live in the analyzer (the freeze applies them); re-exported.
power = _analysis.power
N_RULE = _analysis.N_RULE
n_rule = _analysis.n_rule


def yield_gate(qualifying, generated, floor):
    if generated <= 0:
        raise ValueError("no generated cases")
    rate = qualifying / generated
    return {"yield": rate, "passed": rate >= floor}


def dtype_gate(pairs, tolerance):
    """``pairs``: per sampled case, (T, resolved, labels) in the run dtype and in fp32."""
    worst = max(abs(a[0] - b[0]) for a, b in pairs)
    same = all(a[1] == b[1] and list(a[2]) == list(b[2]) for a, b in pairs)
    return {"max_abs_T_difference": worst, "same_resolution_and_labels": same,
            "passed": worst <= tolerance and same}


def seeded(seed):
    return random.Random(seed)
