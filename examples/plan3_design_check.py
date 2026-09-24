"""Plan-3 structural check only: stipulated answer targets, no neural measurements.

Uses the shipped exact-signature implementation. Names 0 and 1 are categorical
identities, not measured logits; only equality of predictions is evaluated.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_decidability.design import restrict, signatures


def part_a():
    predictions = {name: [] for name in (
        "identity", "position", "subject_position_inhibition", "recipient_role", "no_op")}
    for identity in (0, 1):
        for position in (0, 1):
            targets = {"identity": identity, "position": position,
                       "subject_position_inhibition": position,
                       "recipient_role": 0, "no_op": 0}
            for name, value in targets.items():
                predictions[name].append(value)
    groups = signatures(predictions)
    assert groups == [["identity"], ["no_op", "recipient_role"],
                      ["position", "subject_position_inhibition"]]
    return groups


def part_b(donor_form="received_from"):
    predictions = {name: [] for name in (
        "relation", "identity", "position", "inverse_position", "no_op", "flip")}
    cells = []
    numerical = {name: [] for name in (
        "relation", "conditional_flip", "identity", "position", "inhibition", "no_op")}
    # Donor role bindings: name 0 gives to name 1; their mention order varies by form.
    donor_mentions = [1, 0] if donor_form == "received_from" else [0, 1]
    for giver in (0, 1):
        recipient = 1 - giver
        for form in ("gave_to", "received_from"):
            mentions = [giver, recipient] if form == "gave_to" else [recipient, giver]
            for recipient_query in ("giver", "recipient"):
                original = giver if recipient_query == "giver" else recipient
                for donor_query in ("giver", "recipient"):
                    donor_answer = 0 if donor_query == "giver" else 1
                    donor_position = donor_mentions.index(donor_answer)
                    targets = {"relation": giver if donor_query == "giver" else recipient,
                               "identity": donor_answer,
                               "position": mentions[donor_position],
                               "inverse_position": mentions[1 - donor_position],
                               "no_op": original, "flip": 1 - original}
                    for name, value in targets.items():
                        predictions[name].append(value)
                    # A constructed, informative native endpoint: name 0 -> +1,
                    # name 1 -> -1. Gain .5 is illustrative, not model-calibrated.
                    native = 1 - 2 * original
                    for name in ("relation", "identity", "position"):
                        endpoint = 1 - 2 * targets[name]
                        numerical[name].append(native + .5 * (endpoint - native))
                    conditional_target = (original if donor_query == recipient_query
                                          else 1 - original)
                    numerical["conditional_flip"].append(
                        native + .5 * ((1 - 2 * conditional_target) - native))
                    numerical["inhibition"].append(
                        native - .5 * ((1 - 2 * targets["position"]) - native))
                    numerical["no_op"].append(native)
                    # When the story/form/query exactly match, all proper models
                    # respect zero donor-minus-recipient delta. The flat foils do not.
                    if giver == 0 and form == donor_form and donor_query == recipient_query:
                        assert all(v[-1] == native for v in numerical.values())
                        assert targets["inverse_position"] != original
                        assert targets["flip"] != original
                    cells.append({"form": form, "giver": giver,
                                  "donor_query": donor_query,
                                  "recipient_query": recipient_query})
    groups = signatures(predictions)
    assert len(cells) == 16 and len(groups) == 6
    same_query = [i for i, c in enumerate(cells)
                  if c["donor_query"] == c["recipient_query"]]
    same_groups = signatures(restrict(predictions, same_query))
    assert any(set(g) >= {"relation", "no_op"} for g in same_groups)
    gave_only = [i for i, c in enumerate(cells) if c["form"] == "gave_to"]
    form_groups = signatures(restrict(predictions, gave_only))
    position_rival = "inverse_position" if donor_form == "received_from" else "position"
    assert any(set(g) >= {"relation", position_rival} for g in form_groups)
    quantitative_groups = signatures(numerical)
    assert len(quantitative_groups) == 5
    assert any(set(g) == {"conditional_flip", "relation"} for g in quantitative_groups)
    return groups, same_groups, form_groups, quantitative_groups


def main():
    print("STRUCTURAL DEMONSTRATION — not model results or empirical identification")
    print("3A groups:", part_a())
    for form in ("received_from", "gave_to"):
        full, same, limited, quantitative = part_b(form)
        print(f"3B donor form {form}: 16 cells, groups = {full}")
        print("  Without cross-query cells:", same)
        print("  Without recipient form variation:", limited)
        print("  Constructed delta-aware numerical groups:", quantitative)
        print("  Literal inverse-position/always-flip foils violate self-identity.")
    print("PASS: declared distinctions and equivalences reproduced.")
    print("Fitted weak/no-effect numerical candidates can still coincide; assess them separately.")


if __name__ == "__main__":
    main()
