"""Round 0, RETROSPECTIVE: which of the paper's designs separate which candidates.

No model, no new data. Each candidate's prediction is written per condition with one of
four types:

- ``point``: one entity;
- ``set``: several entities, any of which is compatible (the positional window);
- ``undefined``: the candidate's mechanism produces no output in this condition. The
  prediction is **not evaluable**: it gives no support and no separation, and the
  candidate stays in the compatible set, untested. It is never a number.
- ``outside_scope``: the readout cannot represent the prediction, or the condition is
  not one the candidate makes a declared prediction for.

Entities are named by their 1-based group index in the recipient, as in the paper, or by
``ABSENT`` for a token that does not occur in the recipient. Two evaluable candidates
are separated in a condition when their predicted entity sets are disjoint. Identical
predictions form an exact equivalence group, computed with the repository's
``causal_decidability.signatures`` on indicator rows.
"""
from itertools import combinations

from causal_decidability import signatures

POINT, SET, UNDEFINED, OUTSIDE = "point", "set", "undefined", "outside_scope"
EVALUABLE = (POINT, SET)
ABSENT = "ABSENT"

CANDIDATES = {
    "P": "positional: retrieves the entity in the group at the donor's query position i_P",
    "L": "lexical: retrieves the entity bound to the donor's query entity, at i_L",
    "R": "reflexive pointer: follows a pointer to the donor's target entity, at i_R if present",
    "A": "completed-answer copy: the patched vector carries the donor's retrieved answer itself",
    "N": "no effect: the recipient's own answer, at i_N",
}


def point(entity):
    return {"type": POINT, "entities": [entity]}


def entity_set(entities):
    return {"type": SET, "entities": list(entities)}


def undefined(reason):
    return {"type": UNDEFINED, "reason": reason}


def outside(reason):
    return {"type": OUTSIDE, "reason": reason}


def window(i_P, w, n):
    """1-based indices within w of i_P, clipped to 1..n."""
    return [j for j in range(i_P - w, i_P + w + 1) if 1 <= j <= n]


def target_rebind(i_P, i_L, i_R, i_N, n, w=None):
    """Sec. 3.2 at layer l: the donor's answer occurs in the recipient, at i_R.

    ``w=None`` writes P as the paper declares it (a point at i_P, Sec. 3.1-3.2);
    an integer w writes P as the Round 1 window i_P +/- w (the paper reports the
    positional effect spread near i_P, Sec. 3.3).
    """
    for index in (i_P, i_L, i_R, i_N):
        if not 1 <= index <= n:
            raise ValueError("design indices must lie in 1..n")
    positional = point(i_P) if w is None else entity_set(window(i_P, w, n))
    return {"P": positional, "L": point(i_L), "R": point(i_R), "A": point(i_R), "N": point(i_N)}


def absent_answer(i_P, i_L, i_N, n, readout_sees_absent, w=None):
    """Sec. 3.4 at layer l: the donor's answer does not occur in the recipient."""
    base = target_rebind(i_P, i_L, i_L, i_N, n, w)
    return {
        "P": base["P"], "L": base["L"], "N": base["N"],
        "R": undefined("the pointer targets a token absent from the recipient; the "
                       "mechanism yields no output (Sec. 3.4)"),
        "A": point(ABSENT) if readout_sees_absent else outside(
            "the in-context entity readout (tasks/dist.py:358) cannot score an absent token"),
    }


def next_layer():
    """Sec. 3.4 repeated at layer l+1: a different intervention."""
    reason = "the candidates are statements about the signal at layer l; l+1 is another intervention"
    return {name: outside(reason) for name in CANDIDATES}


def analyse(predictions):
    """Equivalence groups and pairwise separation among evaluable predictions."""
    if set(predictions) != set(CANDIDATES):
        raise ValueError("every candidate needs a declared prediction")
    for name, prediction in predictions.items():
        if prediction["type"] not in (POINT, SET, UNDEFINED, OUTSIDE):
            raise ValueError(f"{name}: unknown prediction type")
        if prediction["type"] in EVALUABLE and not prediction["entities"]:
            raise ValueError(f"{name}: an evaluable prediction needs at least one entity")
    evaluable = {name: frozenset(p["entities"]) for name, p in predictions.items()
                 if p["type"] in EVALUABLE}
    labels = sorted({e for s in evaluable.values() for e in s}, key=str)
    rows = {name: tuple(1.0 if e in s else 0.0 for e in labels) for name, s in evaluable.items()}
    groups = signatures(rows) if rows else []
    separated, overlapping = [], []
    for a, b in combinations(sorted(evaluable), 2):
        (separated if evaluable[a].isdisjoint(evaluable[b]) else overlapping).append([a, b])
    return {
        "predictions": predictions,
        "evaluable": sorted(evaluable),
        "undefined_not_evaluable": sorted(n for n, p in predictions.items() if p["type"] == UNDEFINED),
        "outside_scope": sorted(n for n, p in predictions.items() if p["type"] == OUTSIDE),
        "equivalence_groups": groups,
        "separated_pairs": separated,
        "not_separated_pairs": overlapping,
    }


# The paper's Figure 1 example after patching: n = 4, i_P = 2, i_L = 1, i_R = 3; the
# recipient asks about group 4 (Sec. 3.2).
FIGURE1 = {"i_P": 2, "i_L": 1, "i_R": 3, "i_N": 4, "n": 4}
# An example that is admissible under the Round 1 window w = 1 (n = 7, middle i_P).
ADMISSIBLE_N7 = {"i_P": 4, "i_L": 2, "i_R": 6, "i_N": 1, "n": 7}


def reconstruct():
    """All Round 0 conditions and the retrospective reading of the paper's outcomes."""
    f, a = FIGURE1, ADMISSIBLE_N7
    conditions = {
        "sec3.2_layer_l_point_P_figure1": target_rebind(**f),
        "sec3.2_layer_l_collision_iL_eq_iR": target_rebind(f["i_P"], f["i_L"], f["i_L"], f["i_N"], f["n"]),
        "sec3.2_layer_l_window_w1_figure1": target_rebind(**f, w=1),
        "sec3.2_layer_l_window_w1_admissible_n7": target_rebind(**a, w=1),
        "sec3.4_layer_l_readout_sees_absent": absent_answer(f["i_P"], f["i_L"], f["i_N"], f["n"], True),
        "sec3.4_layer_l_in_context_readout": absent_answer(f["i_P"], f["i_L"], f["i_N"], f["n"], False),
        "sec3.4_layer_l_plus_1": next_layer(),
    }
    analysed = {name: analyse(p) for name, p in conditions.items()}
    return {
        "label": "RETROSPECTIVE",
        "scope": "Reconstruction from the paper's text and the released code; no model run and "
                 "no new data. Readings of the paper's outcomes are not frozen decisions.",
        "candidates": CANDIDATES,
        "conditions": analysed,
        "reported_outcomes": {
            "sec3.2_layer_l": "Averaged over cases, outcomes carry positional, lexical and reflexive "
                              "labels, and 'mixed' outcomes lie near i_P (Sec. 3.3, Figures 2, 3).",
            "sec3.4_layer_l": "The model did not answer with the absent donor answer (Sec. 3.4, Figure 4).",
            "sec3.4_layer_l_plus_1": "The model answered with the absent donor answer (Sec. 3.4, Figure 4); "
                                     "hence the paper's readout for this design could represent the absent token.",
        },
        "reading": {
            "sec3.2": "P, L and N are separated; R and A form one equivalence group, so Sec. 3.2 "
                      "cannot tell a reflexive pointer from a copy of the finished answer. With "
                      "i_L = i_R, L joins that group; with the window w = 1 at the Figure 1 indices, "
                      "P overlaps L, R and A. Admissibility is therefore checked on design indices.",
            "sec3.4": "With the donor answer absent, A predicts the absent token and R is not "
                      "evaluable. The layer-l outcome excludes a strong answer copy, provided the "
                      "readout sees the absent token, which the layer l+1 outcome indicates for the "
                      "paper's readout; the released default readout does not. An answer-copy "
                      "contribution below that resolution remains possible. R stays compatible and "
                      "is not positively identified.",
            "layers": "Layers l and l+1 are different interventions; the l+1 result speaks to another "
                      "alternative (suppression of absent entities), not to these candidates.",
            "credit": "The Sec. 3.4 design is the authors' own; this reconstruction only states "
                      "which candidates it separates.",
        },
        "compatible_after_round0": {
            "separated_by_design_no_decision": ["P", "L", "N"],
            "compatible_untested": ["R"],
            "strong_form_excluded_retrospectively": ["A"],
        },
        "still_open_handed_to_round1": "The paper's mixture model describes average distributions per "
                                       "cell. Whether single cases look like that average was not checked.",
    }
