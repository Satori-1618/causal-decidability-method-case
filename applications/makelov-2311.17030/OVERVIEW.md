# Narrowing down what a working patch explains

A patch can have exactly the expected effect and still fit several explanations. The goal
here is therefore not to identify the mechanism in one step. The goal is to narrow down,
round by round, the set of explanations that fit the data, and to record at each step what
has been decided and what remains open. Only explicitly written-down candidates are
compared; an explanation nobody formulated is neither tested nor excluded.

Every round follows the same steps:

1. Write the competing explanations down as executable predictions.
2. Find a condition in which they predict different outcomes.
3. Check that the difference is measurable, and fix the decision rule before the data.
4. Decide on fresh cases.

Whatever remains open becomes the question of the next round.

**Starting point.** Makelov, Lange and Nanda (2023) show that a patch along a learned
direction in layer MLP8 of GPT-2 Small shifts which name the model prefers as its answer.
Part of the direction lies in the null space of the MLP output; the artificial patch can
read information there and write it into a visible direction. Open question: from which
component does the patch read its signal?

## Round 1: read source

64 fresh sentence pairs. Decision rule fixed in advance, without a public timestamp.

| Step | What happened |
|---|---|
| **Open** | A: the visible component supplies the signal. B: the null component supplies it. Both reproduce the full patch, which alone cannot separate them. |
| **Added test** | Always write along the same direction, but read only from the visible or only from the null component. Here A and B predict opposite outcomes. |
| **Result** | B predicts better in 64 of 64 pairs (exact sign test, p = 1.1 × 10⁻¹⁹). Reading from the null component reaches a median 84% of the full effect, reading from the visible component 16%; these are ratios of measured effects, not identified shares of a mechanism. |
| **Narrowed** | The signal comes mostly from the null component. B is relatively better; A is worse but not refuted. This agrees with the authors' explanation and quantifies it. |
| **Still open** | Whether B is accurate enough. How the written change reaches the output. |

Evidence: [preregistration](PREREG_READ_SOURCE_Q1.md), [result](../../docs/CONFIRMED_CASE.md).

*Next question: what role do selected attention queries play on the way to the output?*

## Round 2: route through queries

192 fresh sentence pairs. Decision rule fixed publicly before the results.

| Step | What happened |
|---|---|
| **Open** | Three profiles for the queries of heads 9.6, 9.9 and 10.0: (a) resetting them removes the effect and the changed queries alone reproduce it; (b) resetting them removes the effect but the changed queries alone do little; (c) the effect survives the reset and the changed queries alone do little. |
| **Added test** | Reset the queries while the patch is active. Conversely, insert the changed queries alone, without the patch; only this reversal separates (a) from (b). |
| **Result** | (a) fits 0, (b) 0 and (c) 8 of 192 pairs; 80% was required at a tolerance of ±25% of the local effect. On average, about 73% of the effect remains after the reset; the queries alone produce about 28%. |
| **Narrowed** | All three simple profiles are excluded. The response lies between them. The reversal measures what the queries do alone, but in these data it does not change the exclusion already implied by the reset alone (0 and 17 of 192 pairs fit the two reset-only profiles). |
| **Still open** | A partial, roughly additive route; other heads and paths. Neither involvement nor non-involvement of the queries follows from these exclusions. |

Evidence: [protocol](QUERY_ROUTE_PROTOCOL.md), [result](../../docs/ROUND2_RESULT.md).

*Next question: does the read signal carry the answer's name or its position in the sentence?*

## Round 3A: name or position

512 fresh sentence families. Decision rule fixed publicly before the results.

| Step | What happened |
|---|---|
| **Open** | Only the position of the answer matters, or only its name. With a single sentence arrangement, both predict the same change. |
| **Added test** | In the donor sentence, cross answer name and position independently while holding the recipient fixed; repeat with the recipient's order reversed. |
| **Result** | "Position only" fits 301 and "name only" 0 of 512 families; 80% was required at ±0.25 nat. Position has a strong effect: −1.28 nat on average. |
| **Narrowed** | Both profiles are excluded. Position dominates, but the name assignment shifts the effect by more than 0.25 nat in 41% of families. |
| **Still open** | A positional process whose strength depends on the name. Neither a person nor a role has been shown. |

Evidence: [protocol](DONOR_FACTOR_PROTOCOL.md), [sample-size amendment](DONOR_FACTOR_512_AMENDMENT.md),
[result](../../docs/ROUND3A_CONFIRMATION.md).

*Next question: can a role question (who gives, who receives, who watches) be tested with this model at all?*

## Round 3B: qualifying a role task

32 sentence families, unmodified model runs only. Decision rule fixed publicly before the results.

| Step | What happened |
|---|---|
| **Open** | A task with three roles could separate a role rule from name and position rules, but only if the model solves the task itself. |
| **Added test** | Before any patch, check: does GPT-2 answer the role questions at least 90% correctly? Are the candidates' predictions far enough apart? |
| **Result** | All 18 task groups missed (6 to 75%; 36% overall against 33% chance). Sufficient separation in 0 of 32 families. No patch was run. |
| **Narrowed** | This task cannot test role transfer in GPT-2 Small. An unsupported role interpretation is thereby prevented. |
| **Still open** | Whether GPT-2 represents roles; this stop says nothing about that. |

Evidence: [protocol](ROLE_BASELINE_PROTOCOL.md), [result](../../docs/ROUND3B_STAGE_A_RESULT.md).

## After four rounds

**Narrowed:**

- The patch reads its signal mostly from the null component (relative comparison; measured effect ratios of 84% against 16%, not mechanism shares).
- Three simple routes through the queries, and pure position or name invariance, do not describe the effect.
- The planned role task cannot be tested with GPT-2 Small.

**Explicitly open:**

- Whether the better explanation is accurate enough.
- The complete path of the effect to the output.
- Whether the unmodified model uses this information in the same way, and whether it represents roles.

**Checkable.** Every decision can be recomputed from the included raw records without a
model; see [verification](VERIFICATION.md). Rounds 2 to 3B were fixed publicly before their
results were published. The statements hold only for this model, this site, this direction
and this sentence pattern. The [case study](CASE_STUDY.md) gives the details, and the
[method guide](../../docs/USING_THE_METHOD.md#case-wise-profiles-with-a-population-coverage-requirement)
explains how to apply the same decision rules to other data.
