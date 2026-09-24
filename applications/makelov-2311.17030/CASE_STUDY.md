# What a successful patch tells us — and what it leaves open

**A vector changes the model's answer scores. What does that tell us about the
information it transfers?** This application of causal decidability turns that question
into explicit rival predictions, adds interventions where those predictions differ,
and records both the distinctions obtained and the questions still unanswered.

Start here for the current Makelov branch. The [general method](../../README.md#the-method)
is already defined; this page explains what applying it contributes. The original
[application README](README.md) is a hash-frozen historical archive, not the current
round index. [Verify the results](VERIFICATION.md) without downloading a model.

## Before the additional tests

Makelov, Lange and Nanda showed that successful subspace patching can be misleading.
Their published GPT-2 Small direction changes name preferences in an indirect-object
task. But a patch both **reads** information and **writes** a change: the computation
created by the intervention need not be the computation the unmodified model uses.

At MLP8, part of the direction lies in the output projection's null space: information
along it is not directly transmitted by that projection. The artificial patch can
read it and write into a direction the projection does transmit. This is the authors'
insight. Our application makes the rival predictions and the evidential limits
explicit, then uses the same procedure to ask further questions.

## Round 1 — distinguish two explanations of the read source

**Before:** both idealized candidates reproduce the measured full-patch result by
construction. A says the effective signal comes from the output-visible component;
B says it comes from the null component. The full patch cannot distinguish them.

**Additional test:** keep the write direction fixed and read separately from each
component. A and B now make opposite predictions, anchored to the measured baseline
and full-patch endpoints of the same case.

**Result:** on 64 fresh base pairs, B predicts better in **64/64**, with exact two-sided
sign-test p = **1.0842 × 10⁻¹⁹**. Mean absolute errors are 0.2383 nat for B and 1.2225
for A. Both swap directions belong to one base pair, not two independent observations.

**After:** the declared null-read explanation is the better predictor of these added
interventions. This is a positive relative comparison. No absolute adequacy tolerance
was fixed, so B has not been shown accurate enough, exclusive, or a naturally used
semantic variable. The visible component also has an effect.

## Round 2 — ask what the selected downstream queries contribute

**Before:** knowing the preferred read-source explanation does not specify how the
written change reaches the output. Three simple accounts predict that selected
attention queries transfer the effect, act jointly with other changes, or leave the
effect essentially preserved when reset.

**Additional test:** retain the MLP patch while resetting those queries, and transfer
the changed queries without the MLP patch. The reverse transfer is needed to distinguish
the predictions that reset alone can leave together.

**Result:** on 192 fresh pairs, the transfer and joint-dependence profiles each fit
0/192; preservation fits 8/192. All three are excluded against the frozen requirement
of 80% population coverage at the declared tolerance.

**After:** none of these three simple profiles adequately describes the measured
response. The selected queries are not thereby shown irrelevant, and no fourth
mechanism is confirmed. In these data, reverse transfer adds measurements but does
not change the population-level exclusion already suggested by reset alone.

## Round 3A — separate name assignment from mention position

**Before:** a patch sensitive to a donor could be described as transferring a name
or information about its position. With only one arrangement, those descriptions can
predict the same change.

**Additional test:** keep the recipient fixed and independently cross donor answer
name and mention position, then repeat for the opposite recipient ordering. This is
an intervention on the full name assignment, not on an isolated latent person variable.

**Result:** among 512 fresh families, the position-only invariance profile fits
**301/512 (58.79%)** and identity-only fits **0/512**. Both fail the required 80%
coverage. The mean position effect is relevant, but a small signed mean name effect
does not imply name invariance: opposite effects can cancel.

**After:** a pure position-only or identity-only response is insufficient at the
declared resolution. A positional process whose strength depends on lexical context
remains possible. The result does not refute the paper's broader positional account
or establish that the vector carries a concrete person.

## Round 3B — check whether the next semantic question can be tested

**Before:** a three-role task could potentially separate a relational response rule
from simpler name and position rules. That interpretation requires the model to answer
the native role questions and produce sufficiently distinct rival predictions.

**Qualification:** test those requirements before running interpretative patches.

**Result:** across 32 development families, all 18 competence groups fail the 90%
requirement, and **0/32** families pass the geometry requirement. Numerical checks pass.
The experiment stops. **No 3B patch is run.**

**After:** this task/readout does not qualify for the planned role-transfer claim.
This is neither a comparison of patched role mechanisms nor evidence that GPT-2 has
no role representations. The useful outcome is preventing that unsupported inference.

## What the application adds

| Before asking the distinguishing question | After applying the procedure |
|---|---|
| A successful vector patch has an effect. | We know which of two declared read-source explanations predicts added interventions better. |
| Plausible downstream and semantic stories can be attached to that effect. | We have measured limits on three route profiles and two name/position invariance profiles. |
| A further role-patching result might look interpretable. | We know the proposed task does not meet its predeclared requirements, so no role claim is made. |

The application helps interpret what a vector intervention actually establishes.
It does not automatically generate hypotheses or identify a complete native mechanism.
The later rounds do not supply Q1's missing adequacy criterion, and their exclusions
or STOP do not erase Q1's supported comparison. No safety transfer is established here.

## Evidence and checks

| Round | Question and rule fixed in | Recorded outcome | Readable detail |
|---|---|---|---|
| Q1 | [Preregistration](PREREG_READ_SOURCE_Q1.md) | [Raw records](results/makelov_read_source_q1/records.jsonl), [score](results/makelov_read_source_q1/score.json) | [Confirmed case](../../docs/CONFIRMED_CASE.md) |
| 2 | [Protocol](QUERY_ROUTE_PROTOCOL.md) | [Raw records](results/query_route_confirmation/records.jsonl), [summary](results/query_route_confirmation/summary.json) | [Route result](../../docs/ROUND2_RESULT.md) |
| 3A | [Protocol](DONOR_FACTOR_PROTOCOL.md), [prospective sample-size amendment](DONOR_FACTOR_512_AMENDMENT.md) | [Raw records](results/donor_factor_confirmation_512/records.jsonl), [summary](results/donor_factor_confirmation_512/summary.json) | [Name/position result](../../docs/ROUND3A_CONFIRMATION.md) |
| 3B | [Native qualification protocol](ROLE_BASELINE_PROTOCOL.md) | [Raw records](results/role_baseline_development/records.jsonl), [summary](results/role_baseline_development/summary.json) | [Qualification STOP](../../docs/ROUND3B_STAGE_A_RESULT.md) |

[Four commands reproduce the stored decisions](VERIFICATION.md#1-check-stored-evidence-no-installation).
They check records, not fresh model executions. Q1's freeze/run order is documented
in private development history without an independent contemporaneous public timestamp;
publishing that archive does not add one. Later freezes and the 3A amendment are linked
in their result reports. The original 3A planning STOP remains recorded.

This is one model, a fixed artificial intervention and limited task families, with
different questions and statistical contracts in each round. The historical calculator
studies are separate evidence with [their own limits](../../docs/validation.md).
