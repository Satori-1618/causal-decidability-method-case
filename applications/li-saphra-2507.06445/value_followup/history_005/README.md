# Which part of bracket history predicts the transferred effect?

**Prepared for independent review. No model measurements; execution blocked.**

The [previous round](../averaged_anchors_004/RESULT.md) excluded its frozen
balance-only predictor at the 90% adequacy target. This new round tests a
narrower question: **when current balance and read position are fixed, do fresh
history-edit effects follow last-minimum recency or the final bracket pattern?**

Every quartet shares an 18-bracket stem that ends at balance −4:

| Last visit to the minimum | Final pattern `()()` | Final pattern `(())` |
|---|---|---|
| 6 steps before reading | stem + `(())((` + `()()` | stem + `(())((` + `(())` |
| 10 steps before reading | stem + `(()()(` + `()()` | stem + `(()()(` + `(())` |

All four prefixes end at **position 28, balance −2, minimum −4, symbol `)`**.
The row change swaps brackets at positions 22/23; the column change swaps them
at 26/27. Each quartet shares its first 21 brackets and future suffix.
These are precisely controlled edits, not pure interventions on abstract
“recency”: other descriptions of the same local edit remain possible.

For each of **256 recipient–donor families**, two quartets calibrate three rules:
effects track the row, track the column, or show no within-quartet difference.
Their predictions are saved before measuring two new quartets. The recipient
and value-transfer operation stay fixed within a family.
The zero-difference rule allows a shared transfer effect; it does not mean
that the intervention is internally or behaviorally ineffective.

**What is compared:** the pattern of differences within each quartet, after
subtracting that quartet's mean. This removes shared offsets algebraically.
It deliberately leaves absolute transferred margins and shared history effects
outside the claim. The previous 0.10-nat / 90% adequacy test is unchanged; this
round has a different question and no absolute-adequacy claim.

The primary comparison asks which rule more often improves the worst fresh-cell
prediction error by more than **0.02 nat**, with a separate **0.002-nat numerical
guard**. Three corrected paired tests can support one direction, opposing
directions across pairs, or leave the comparison unresolved. A better rule is
not automatically accurate enough or the model's unique mechanism.

- [Full frozen protocol](PROTOCOL.md): question, predictions, start/stops and claims.
- [Machine-readable plan](plan.json) · [power and cost calculation](planning.json).
- [Sampling manifest](inputs/preparation.json): seeds, exact construction and freshness.
- [Review handoff](REVIEW.md) · [pending execution release](EXECUTION_RELEASE.json).

From this directory, these commands use no model:

```bash
python3 -B -S verify_freeze.py
python3 -B -S run.py --validate-only
python3 -B -S -m unittest discover -p 'test_*.py'
```

The maximum authorized design, **after a separate release**, is 36,864 sequence
forwards. Screening shortfall, inadequate forecast separation or technical
failure stops the round without replacements, retuning or retries.
