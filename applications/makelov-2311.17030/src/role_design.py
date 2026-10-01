"""Prospective 3B design: explicit rivals, not observed model behavior.

The primary readout is the absolute cross-query patch response (patched minus
native recipient), in three pairwise name margins. Same-query cells remain in
the raw grid, but never dilute this primary comparison. No model is loaded.
"""
from itertools import product

from role_geometry import pairwise_margins, validate_logits

ROLES = ("giver", "receiver", "observer")
WORDINGS = ("fit", "heldout")  # Wording labels, NOT data-split membership.
DONORS = {
    "d0": {"binding": (0, 1, 2), "form": "gave"},
    "d1": {"binding": (1, 0, 2), "form": "received"},
}
RECIPIENTS = {
    f"b{i}_{form}": {"binding": (i, (i + 1) % 3, (i + 2) % 3), "form": form}
    for i in range(3) for form in ("gave", "watched")
}
DIAGNOSTICS = {
    "name": ("d1/b1_gave/observer/receiver",),
    "position": ("d1/b1_gave/observer/receiver",),
    "switch": ("d0/b1_gave/observer/giver", "d0/b1_gave/observer/receiver"),
    "no_op": ("d0/b1_gave/observer/giver", "d0/b1_gave/observer/receiver"),
}


def mentions(context):
    g, r, o = context["binding"]
    return {"gave": (g, r, o), "received": (r, g, o), "watched": (o, r, g)}[context["form"]]


def answer(context, query):
    return context["binding"][ROLES.index(query)]


def render(context, query, names=("Alice", "Bob", "Carol"), item="book", wording="fit"):
    """Illustrative frozen draft strings; native competence has not been measured."""
    if len(names) != 3 or len(set(names)) != 3 or not all(isinstance(n, str) and n for n in names):
        raise ValueError("three distinct nonempty names are required")
    if query not in ROLES or wording not in WORDINGS:
        raise ValueError("unknown query or wording")
    g, r, o = (names[i] for i in context["binding"])
    if wording == "fit":
        stories = {
            "gave": f"{g} gave a {item} to {r} while {o} watched.",
            "received": f"{r} received a {item} from {g} while {o} watched.",
            "watched": f"{o} watched as {r} received a {item} from {g}.",
        }
        questions = {
            "giver": f"The person who gave the {item} was",
            "receiver": f"The person who received the {item} was",
            "observer": "The person who watched the exchange was",
        }
    else:
        stories = {
            "gave": f"{g} handed a {item} to {r}, with {o} looking on.",
            "received": f"{r} accepted a {item} from {g}, with {o} looking on.",
            "watched": f"{o} looked on while {r} accepted a {item} from {g}.",
        }
        questions = {
            "giver": "The sender in this exchange was",
            "receiver": "The recipient in this exchange was",
            "observer": "The witness in this exchange was",
        }
    # A common prefix keeps the first name in a leading-space token context,
    # matching the subsequent name mentions and answer-token convention.
    return "Then, " + stories[context["form"]] + " " + questions[query]


def grid():
    """108 raw cells/wording; each family has both wordings and precisions."""
    cells = []
    for (did, donor), (rid, recipient), qr, qd in product(
            DONORS.items(), RECIPIENTS.items(), ROLES, ROLES):
        native = answer(recipient, qr)
        name = answer(donor, qd)
        slot = mentions(donor).index(name)
        cells.append({
            "cell_id": f"{did}/{rid}/{qr}/{qd}", "donor_id": did, "recipient_id": rid,
            "recipient_query": qr, "donor_query": qd, "group": f"{did}/{rid}/{qr}",
            "targets": {"role": answer(recipient, qd), "name": name,
                        "position": mentions(recipient)[slot], "no_op": native},
            "donor_answer": name, "donor_position": slot,
            "primary": qd != qr,
            "exact_self": donor == recipient and qd == qr,
        })
    return cells


def primary_cells():
    return [c for c in grid() if c["primary"]]


def expected_cell_ids():
    return [c["cell_id"] for c in primary_cells()]


def expected_structure():
    """Bind query labels and switch groups independently of observed records."""
    return {c["cell_id"]: {k: c[k] for k in ("group", "recipient_query", "donor_query")}
            for c in primary_cells()}


def directions(endpoints):
    """Build endpoint-relative rivals from independently measured native logits.

    endpoints[recipient_id][query] is the three-name logit vector in a FIXED
    name order. This function receives no patch outcomes. The oracle amplitude
    is allowed to vary independently for every absolute cell, including sign.
    """
    if set(endpoints) != set(RECIPIENTS):
        raise ValueError("all six recipient contexts are required")
    parsed = {}
    for rid, queries in endpoints.items():
        if set(queries) != set(ROLES):
            raise ValueError("all three native query endpoints are required")
        parsed[rid] = {q: pairwise_margins(validate_logits(z)) for q, z in queries.items()}
    result = []
    for cell in primary_cells():
        rid, qr = cell["recipient_id"], cell["recipient_query"]
        native = parsed[rid][qr]
        row = {k: cell[k] for k in ("cell_id", "group", "recipient_query", "donor_query")}
        for candidate in ("role", "name", "position"):
            target = cell["targets"][candidate]
            target_role = ROLES[RECIPIENTS[rid]["binding"].index(target)]
            row[candidate + "_direction"] = [a-b for a, b in zip(parsed[rid][target_role], native)]
        row["diagnostics"] = [k for k, ids in DIAGNOSTICS.items() if cell["cell_id"] in ids]
        result.append(row)
    return result


def explanation_example():
    """Same donor answer AND mention position; different recipient role targets."""
    ids = ("d0/b1_gave/observer/giver", "d1/b1_gave/observer/receiver")
    selected = [c for c in grid() if c["cell_id"] in ids]
    names = ("Alice", "Bob", "Carol")
    return {"recipient": render(RECIPIENTS["b1_gave"], "observer"),
            "native_answer": "Alice",
            "rows": [{"donor": render(DONORS[c["donor_id"]], c["donor_query"]),
                      "donor_answer": names[c["donor_answer"]],
                      "donor_mention_position": c["donor_position"] + 1,
                      "predicted_targets": {k: names[v] for k, v in c["targets"].items()}}
                     for c in selected]}
