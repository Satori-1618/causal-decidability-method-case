# A design analysis of CausaLab's multiple-choice experiments

**What this is.** [CausaLab](https://github.com/goodfire-ai/causalab/tree/8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e)
is Goodfire's open-source library for causal analysis of networks. Its two MCQA
demonstrations ask which layers of Qwen2.5-1.5B-Instruct carry the correct option's
**position** (`answer_position`) or its **letter** (`answer`) at the answer slot. They
use distributed alignment search (DAS), following Geiger et al.
([arXiv 2303.02536](https://arxiv.org/abs/2303.02536)) and Wiegreffe et al.
([arXiv 2407.15018](https://arxiv.org/abs/2407.15018)).

CausaLab already supports comparing causal hypotheses on the same base/counterfactual
pairs. Its [hypothesis analysis guide](https://github.com/goodfire-ai/causalab/blob/8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e/docs/hypothesis_analysis.md)
compares a target with alternatives and filters `distinguishing` pairs. Its published
results already report base accuracy, counterfactual accuracy and the rate at which the
unpatched model gives the target label (0.0 in both designs). Each published design
already separates its target variable from the other one by construction. The
position guide itself notes that the counterfactual prompt's own answer is the base
prompt's letter, not the label.

**What this audit adds** is a narrow design analysis of the published input tables, with
no model run:
- In each design, the rule that is *not* the target ties with "the base answer is
  preserved" in every row.
- A color-transfer rule ties with "preserved" in all published rows.
- Adding a proposed third design to both published ones separates all four rules.
- It is not a new neural test and makes no claim about Qwen's internals.

**Low-cost next step in CausaLab's own terms:**
1. Apply one fixed fitted rotation, for example the pointer fit at its chosen layer, to
   *both* designs' test pairs.
2. Save `top_k` (k = 1) as the guide recommends.
3. Call `compare_saved_outputs` with the target and the two other rules as alternatives.

That shows which rule the fixed patch follows on pairs where the rules disagree.

## The example, using actual published inputs

The recipient prompt in the first row of both tables is:

```text
The shoe is blue. What color is the shoe?
P. blue
Y. orange
Answer:
```

The two published designs use two kinds of counterfactual (donor). One moves the options,
including their letters; the other replaces the letters while keeping the options in place:

| Donor options (same question) | Transfer answer **position** | Transfer answer **letter** | Preserve recipient's **task answer** |
|---|---|---|---|
| Y. orange / P. blue | **Y**: donor is correct at position 2; recipient's second letter is Y | **P**: donor's correct letter | **P** |
| O. blue / Q. orange | **P**: donor is correct at position 1; recipient's first letter is P | **O**: donor's correct letter | **P** |

These predictions follow from the [published causal equations](https://github.com/goodfire-ai/causalab/blob/8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e/causalab/tasks/MCQA/causal_models.py).
They are hypothetical task-level rules, not measured neural outputs.

**What the preflight adds:** if the first type produces P, letter transfer and
preserving the task answer remain tied. More cases of that same type cannot break
this tie. The second type supplies a separating prediction: O versus P. Each
published design already separates its intended rule from the other two by
construction. The added use here is to make the remaining ties among the non-target
rules explicit and to plan a joint test of **one fixed patch**, not to find a flaw in
either design.

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

With all four rules, `run.py` writes [four_rule_extension.json](results/rederivation_20261022/four_rule_extension.json):

| Input menu | Rows | Rules still tied |
|---|---:|---|
| position design only | 192 | letter = color = preserved |
| letter design only | 192 | position = color = preserved |
| proposed color design only | 192 | position = letter = color |
| both published designs | 384 | color = preserved |
| all three designs | 576 | none |

The proposed color design keeps the recipient's options and makes the other color
correct. It is a design proposal, not published data.

## What a researcher should do next

Evaluate **one fixed learned patch** across both input families and any added
color-changing condition. Keep the model, layer, token position, subspace and
scoring rule fixed. Record the actual unpatched and
patched answers for every case, not only whether the intended target was matched.
The demonstrations fit separate DAS rotations for the two target variables, so those
two scores cannot simply be pooled into a common-intervention comparison. One published
measurement already applies a fixed intervention to both designs: the full-vector patch
at each layer (`full_patch` in each design's `iia_plotted.json`). At layer 21 it matches
the position target in 0.547 of held-out pairs and the letter target in 0.0625. At
layer 26 the values are 0.016 and 0.922. These are the authors' numbers, scored only
against each design's own label; they are copied unchanged into
[four_rule_extension.json](results/rederivation_20261022/four_rule_extension.json).

The third column is deliberately **not called “patch has no effect.”** That rival
must predict the measured unpatched answer. The published base accuracies on the 64
held-out pairs are 87.5% and 92.1875%, so correct task answers cannot silently substitute for those
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
python3 applications/goodfire-mcqa-preflight/run.py                  # derive and compare; writes nothing
python3 applications/goodfire-mcqa-preflight/run.py --write NEW_DIR  # write a new derivation
```

The default run downloads about 0.5 MB (503,367 bytes) of pinned source files into memory and checks
their hashes, including the two documentation files cited above
([reference manifest](reference_manifest.json)). It derives every export and compares it
with the committed files, and exits non-zero on any difference. `--write` never writes
into `results/` itself and refuses to replace a differing file. No upstream code is
executed.
See [summary.json](results/summary.json), the exported [position-test predictions](results/pointer_predictions.json),
[letter-test predictions](results/symbol_predictions.json) and the
[four-rule extension](results/rederivation_20261022/four_rule_extension.json). Either three-rule export also
runs directly through `examples/causal_preflight.py --config PATH --structure-only`.

*Note (22 October 2026):* the historical `results/summary.json` records a hash for
`examples/causal_preflight.py` that does not match that file at the pinned commit. The
historical file is kept unchanged. A re-derivation with the current code is in
[`results/rederivation_20261022/`](results/rederivation_20261022/). It records the actual
code hashes, reproduces the three historical prediction exports byte for byte, and adds the
four-rule extension. Apart from the recorded code hashes, its summary matches the
historical one.

Sources: [position experiment](https://github.com/goodfire-ai/causalab/blob/8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e/demos/papers/mcqa_pointer.md),
[letter experiment](https://github.com/goodfire-ai/causalab/blob/8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e/demos/papers/mcqa_symbol.md),
[file hashes and URLs](artifacts/upstream/source_manifest.json).

CausaLab already provides both designs, discusses their target variables and supports
hypothesis comparison. This application makes the remaining ties explicit; it does not
claim that the authors missed the distinction. The demonstrated capability is a reproducible design
diagnosis on an external repo. Forecasting success on a fresh run remains untested.
