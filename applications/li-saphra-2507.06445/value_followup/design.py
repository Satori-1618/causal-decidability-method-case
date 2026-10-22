"""Input-only design for a bounded semantic interchange comparison.

Families are sampled with replacement from fixed phase-specific pools. The two
donor prefixes within each cell differ; prefixes may repeat between families.
Suffixes never qualify a donor prefix as fresh.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

LENGTH = 32
POSITIONS = (20, 28)
BALANCES = (-2, 2)
MINIMUM = -4
ANCHORS = ("neg_20_0", "pos_28_0")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def phase_of(text):
    return "development" if int(sha("value-prefix-v1:" + text), 16) % 5 == 0 else "confirmation"


def prefix_state(text):
    depth = low = 0
    for symbol in text:
        depth += 1 if symbol == "(" else -1
        low = min(low, depth)
    return depth, low


@lru_cache(None)
def completion_count(remaining, depth, touched, target):
    if depth < MINIMUM or abs(target - depth) > remaining:
        return 0
    if remaining == 0:
        return int(depth == target and touched)
    return sum(completion_count(remaining - 1, depth + step,
                               touched or depth + step == MINIMUM, target)
               for step in (1, -1))


def draw_prefix(rng, position, balance):
    # Last symbol is ')', so sample a path ending one unit above its balance.
    depth, touched, out = 0, False, []
    for remaining in range(position - 1, 0, -1):
        counts = [completion_count(remaining - 1, depth + step,
                                   touched or depth + step == MINIMUM, balance + 1)
                  for step in (1, -1)]
        draw = rng.randrange(sum(counts))
        step = 1 if draw < counts[0] else -1
        depth += step
        touched |= depth == MINIMUM
        out.append("(" if step == 1 else ")")
    prefix = "".join(out) + ")"
    assert prefix_state(prefix) == (balance, MINIMUM)
    return prefix


def exclusion_inputs(native_root):
    """Reuse only already-open public/previous-run input surfaces."""
    import csv
    texts = set()
    sources = {}
    for name in ("indist", "ood"):
        path = native_root / "cache/data/model_preds" / (name + "_data_preds.csv")
        sources[str(path.relative_to(native_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as f:
            texts.update(row["string"] for row in csv.DictReader(f))
    path = native_root / "frozen/confirmation_001/cases.jsonl"
    sources[str(path.relative_to(native_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    texts.update(json.loads(line)["string"] for line in path.read_text().splitlines())
    return {"recipient_strings": sorted(texts),
            "prefixes": {str(p): sorted({s[:p] for s in texts if len(s) >= p}) for p in POSITIONS},
            "source_hashes": sources}


def generate(seed, n, phase, exclusions):
    if phase not in ("development", "confirmation") or n < 1:
        raise ValueError("Invalid generation request")
    banned_inputs = set(exclusions["recipient_strings"])
    banned_prefixes = {p: set(exclusions["prefixes"][str(p)]) for p in POSITIONS}
    rng = random.Random(seed)
    families = []
    for number in range(n):
        while True:
            chars = list("(" * 16 + ")" * 16)
            rng.shuffle(chars)
            recipient = "".join(chars)
            if (recipient not in banned_inputs and phase_of(recipient) == phase
                    and prefix_state(recipient)[1] < 0):
                break
        donors = []
        for balance in BALANCES:
            for position in POSITIONS:
                selected = set()
                for replica in range(2):
                    for _ in range(100000):
                        prefix = draw_prefix(rng, position, balance)
                        if (prefix not in selected and prefix not in banned_prefixes[position]
                                and phase_of(prefix) == phase):
                            break
                    else:
                        raise RuntimeError("No eligible donor prefix; do not change constraints silently")
                    selected.add(prefix)
                    suffix = list("(" * (16 - prefix.count("(")) + ")" * (16 - prefix.count(")")))
                    rng.shuffle(suffix)
                    text = prefix + "".join(suffix)
                    name = ("neg" if balance < 0 else "pos") + f"_{position}_{replica}"
                    donors.append({"cell": name, "balance": balance, "position": position,
                                   "replica": replica, "string": text, "prefix_sha256": sha(prefix)})
        family = {"family_id": sha(f"{phase}:{seed}:{number}")[:20],
                  "phase": phase, "recipient": recipient, "donors": donors}
        validate_family(family)
        families.append(family)
    return families


def validate_family(family):
    recipient = family["recipient"]
    if len(recipient) != LENGTH or recipient.count("(") != 16 or set(recipient) != {"(", ")"}:
        raise ValueError("Invalid recipient")
    if phase_of(recipient) != family["phase"] or prefix_state(recipient)[1] >= 0:
        raise ValueError("Recipient outside declared phase/population")
    expected = {(d, p, r) for d in BALANCES for p in POSITIONS for r in range(2)}
    found, prefixes = set(), {}
    for donor in family["donors"]:
        d, p, r = donor["balance"], donor["position"], donor["replica"]
        key = (d, p, r)
        if key not in expected or key in found:
            raise ValueError("Invalid or duplicate donor cell")
        found.add(key)
        text = donor["string"]
        prefix = text[:p]
        if (len(text) != LENGTH or text.count("(") != 16 or set(text) != {"(", ")"}
                or text[p-1] != ")" or prefix_state(prefix) != (d, MINIMUM)
                or phase_of(prefix) != family["phase"] or sha(prefix) != donor["prefix_sha256"]):
            raise ValueError("Invalid donor or prefix freshness")
        cell = ("neg" if d < 0 else "pos") + f"_{p}_{r}"
        if cell != donor["cell"] or prefix in prefixes.setdefault((d,p), set()):
            raise ValueError("Cell mismatch or repeated within-cell prefix")
        prefixes[(d,p)].add(prefix)
    if found != expected:
        raise ValueError("Incomplete family")


def predictions(anchor_margins):
    a, d = (anchor_margins[name] for name in ANCHORS)
    result = {"H_state": {}, "H_position": {}}
    for balance in BALANCES:
        for position in POSITIONS:
            for replica in range(2):
                name = ("neg" if balance < 0 else "pos") + f"_{position}_{replica}"
                if name in ANCHORS:
                    continue
                result["H_state"][name] = a if balance < 0 else d
                result["H_position"][name] = a if position == POSITIONS[0] else d
    return result
