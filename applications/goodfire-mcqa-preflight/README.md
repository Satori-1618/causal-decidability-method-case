# A simple external application: Goodfire's multiple-choice experiments

**Question:** does an activation patch transfer the position of an answer, or its
letter? We applied the existing preflight to real input tables from
[Goodfire's CausaLab](https://github.com/goodfire-ai/causalab/tree/8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e).
This is a retrospective design audit: no model was run and no mechanism was identified.

## The example, using actual published inputs

The recipient prompt in the first row of both tables is:

```text
The shoe is blue. What color is the shoe?
P. blue
Y. orange
Answer:
```

Goodfire provides two kinds of donor. One moves the options, including their letters;
the other replaces the letters while keeping the options in place:

| Donor options (same question) | Transfer answer **position** | Transfer answer **letter** | Preserve recipient's **task answer** |
|---|---|---|---|
| Y. orange / P. blue | **Y**: donor is correct at position 2; recipient's second letter is Y | **P**: donor's correct letter | **P** |
| O. blue / Q. orange | **P**: donor is correct at position 1; recipient's first letter is P | **O**: donor's correct letter | **P** |

These predictions follow from the [published causal equations](https://github.com/goodfire-ai/causalab/blob/8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e/causalab/tasks/MCQA/causal_models.py).
They are hypothetical task-level rules, not measured neural outputs.

**What the preflight adds:** if the first type produces P, letter transfer and
preserving the task answer remain tied. More cases of that same type cannot break
this tie. The second type supplies a separating prediction: O versus P. Each
Goodfire design already separates its intended rule from the other two by
construction. The added use here is to expose ambiguity among the non-target
rules and plan a joint test of **one fixed patch**, not to discover a flaw in
Goodfire's design.

## Result on the complete published input tables

| Input menu | Rows checked | Remaining identical prediction patterns |
|---|---:|---|
| Position-test pairs only | 192 | Letter transfer = preserve task answer |
| Letter-test pairs only | 192 | Position transfer = preserve task answer |
| Both input families, as a proposed shared menu | 384 | All three declared rules have different patterns |

Each source table contains 128 train rows and 64 test rows. The same grouping holds
on each 64-row test subset. These are counts of design rows, **not an independent
sample size or a new confirmatory result**. The predictions are derived from the
input variables and checked against Goodfire's target labels. The script then calls
the same category preflight used in the introductory example.

**A further rival remains tied.** In all 384 input rows the donor and recipient
have the same object and correct color. A rule that transfers the donor's correct
color and looks up its letter in the recipient's options therefore predicts exactly
the same answer as `preserve_task_answer`, even across both designs. The three-rule
exports above do not identify a unique mechanism. To separate these two rules,
add a donor with a different correct color that is still among the recipient's
options: for the recipient above, a donor stating “The shoe is orange” predicts
Y under color transfer and P under task-answer preservation. This is a proposed
additional condition, not a published model result or sufficient evidence by itself.

## What a researcher should do next

Evaluate **one fixed learned patch** across both input families and any added
color-changing condition. Keep the model, layer, token position, subspace and
scoring rule fixed. Record the actual unpatched and
patched answers for every case, not only whether the intended target was matched.
Goodfire fits separate patches for the two target variables, so its two existing
scores cannot simply be pooled into this common-intervention comparison.

The third column is deliberately **not called “patch has no effect.”** That rival
must predict the measured unpatched answer. The published base accuracies are
87.5% and 92.1875%, so correct task answers cannot silently substitute for those
baselines. Before an empirical null comparison, obtain the actual answers per case.

Likewise, `symbol_transfer` predicts the donor's **correct task letter**, not its
observed response. Published donor accuracies are 82.8125% in the position design
and 92.1875% in the letter design. Testing “copy the donor's actual answer” requires
those per-case donor responses; it is a distinct candidate when the donor is wrong.

The second preflight check, **measurement resolution**, remains unassessed. Choice
labels give no logit-distance, and this audit has no matched precision calibration
or per-case outputs for the proposed common intervention. Do not turn a label
difference into a numerical effect size or sample-size recommendation.

## Reproduce in one command

From the method repository root, with Python 3.9+ and no extra packages:

```bash
python3 applications/goodfire-mcqa-preflight/run.py
```

The first run fetches under 0.5 MB of source files from the pinned commit and checks
their hashes. Later runs use the verified cache. No upstream code is executed.
See [summary.json](results/summary.json), the exported [position-test predictions](results/pointer_predictions.json),
and [letter-test predictions](results/symbol_predictions.json). Either export also
runs directly through `examples/causal_preflight.py --config PATH --structure-only`.

Sources: [position experiment](https://github.com/goodfire-ai/causalab/blob/8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e/demos/papers/mcqa_pointer.md),
[letter experiment](https://github.com/goodfire-ai/causalab/blob/8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e/demos/papers/mcqa_symbol.md),
[file hashes and URLs](artifacts/upstream/source_manifest.json).

Goodfire already provides both designs and discusses their target variables. This
application makes the remaining ties explicit; it does not claim that Goodfire
missed the distinction. The demonstrated capability is a reproducible design
diagnosis on an external repo. Forecasting success on a fresh run remains untested.
