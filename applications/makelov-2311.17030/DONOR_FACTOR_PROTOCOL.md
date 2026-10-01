# Round 3A: fixed-recipient donor identity/position factorial

**Execution protocol, fixed before development outcomes.** This implements only 3A
of [Plan 3](../../docs/PLAN3_PERSON_POSITION_ROLE.md). It authorizes a 32-family
development run, then confirmation only if the prespecified planning rule passes.
No 3B role task, direction training, dose selection or layer selection is included.

## Instrument and population

Use cached GPT-2 Small and the historical processed-weight conventions. At MLP8
post-GELU, final absolute prompt token, insert `(h_d-h_r) dot v_null * v_full`.
Decompose the published pinned vector as in round 1; do not renormalize components.
All prompts use the authors' held-out names, objects, places, **one template and one
prefix**. The sampled population does not cover alternate syntaxes or task wording.

Each iid family contains four native source prompts, named with fixed identities A/B:
ABB, BAB, BAA, ABA. The correct answers are A,A,B,B; their first-mention positions are
first,second,first,second. Recipients are prompt0 and prompt1, both with correct answer A.
For cells 00,01,10,11 the donor-index maps are respectively `[0,1,2,3]` and `[1,0,3,2]`.
The first factor changes the correct donor answer identity; the second changes its
mention position relative to the recipient. Name assignment changes include the repeated
giver's name: this is not a purified manipulation of a semantic answer-identity variable.

Both panels use the **A-minus-B logit margin**. Never flip it to favor the donor answer.
Each panel's self donor yields alpha=0 exactly. Its zero effect is a technical/algebraic
identity, not a scientific discovery. Baselines and all patches use the same four-row
batch; only rows0/1 receive a change, while rows2/3 are additional unchanged-output checks.

Sampling is IID with replacement from the fixed source proposal, conditional only on
declared tokenizer/source eligibility and frozen prompt exclusions. Retain repeated
accepted content with unique draw IDs. All four prompts must be checked against prior
Q1, round-2 and development prompts. No effect- or baseline-correctness filtering.

## Outcomes and restrictions

`d_ip` is the patched minus native recipient margin. In each panel:

```
I = (d10+d11-d00-d01)/2
P = (d01+d11-d00-d10)/2
J = d11-d10-d01+d00
```

Average I/P/J across the two panels inside the family, then across iid families.
Save all individual cells, simple effects, alpha values and insertion norms.

- Position-only invariance: both `d10-d00` and `d11-d01` within ±0.25 nat.
- Identity-only invariance: both `d01-d00` and `d11-d10` within ±0.25 nat.

Profile success requires both restrictions in both recipient panels and both precisions.
Require 80% population coverage, with simultaneous two-sided Clopper–Pearson intervals
for two profiles, family alpha0.025 (tail0.00625). Lower>0.8 supports adequacy;
upper<0.8 excludes it; otherwise unresolved. These are coarse invariance requirements,
not evidence of a unique or complete mechanism. All-small effects can satisfy both.

For mean I/P/J, use complete-family percentile bootstrap, family alpha0.025 across three
intervals (tail0.025/6). Use 20,000 resamples, seed320260924, and repeat with seed320260925.
Resampling indices are shared across precisions. Practical boundary is ±0.10 nat:
interval outside supports signed relevance, wholly inside equivalence, otherwise unresolved.
Require the same decision for both precisions and both seeds; an interval endpoint moving
by >0.01 nat between seeds makes that contrast's decision unresolved. Bootstrap coverage
is nominal, not a finite-sample guarantee. Any unresolved numerical family blocks mean
claims rather than being removed from the estimand.

**Cancellation limit:** averaged alpha-level P and J are algebraically zero because the
recipient ordering reverses. Output P/J can also cancel under equal affine response
slopes across recipient orders. Mean equivalence therefore concerns a signed average;
it cannot establish absence of position dependence. Per-panel distributions and the
coverage profiles remain visible. Coarse profile adequacy and finer mean relevance can
coexist and must both be reported.

## Qualification and numerical checks

Replay the same two historical Q1 examples before sampling new development cases;
require old-margin agreement within1e-4 nat. This checks runtime continuity only.
Use four deterministic CPU threads, historical library versions and explicit eager HF
loading as before. Record the TransformerLens/torch metadata override rather than hiding it.

Run all cells in float32 and then float64, using the same processed weights promoted
to double, with activations recomputed independently. Double precision is a reference,
not exact arithmetic. Local insertion budgets depend on dtype and tensor scale.
Audit one hook call, correct site/absolute position, untouched positions/batch rows,
post-cast intended delta, source mapping and recipient-self/zero identities.
Technical failure blocks interpretation and preserves the failed run.

Numerical resolution requires cross-precision discrepancy ≤0.01 nat for every cell delta,
panel simple effect, panel I/P/J and unit I/P/J. Unresolved units remain in the denominator
as profile failures; no convenient replacement. Scientific mean decisions must additionally
agree across precisions even when the discrepancy budget passes.

## Development, planning and confirmation

1. Development:32 iid families, seed24092431. All controls must pass; retain every outcome.
2. Run the planning procedure under the definitions in `donor_factor_planning.py`,
   seed320260926. Consider n128,192,256,384. Require ≥80% exact-binomial adequacy power at
   a hypothetical complete-family success probability0.90. For EACH of I/P/J, require
   ≥80% planning detection power under means±0.20 against the ±0.10 boundary, with
   development standard deviations multiplied by1.5. No contrast is dropped afterward.
3. Use10,000 Gaussian planning simulations with the simultaneous normal-reference
   interval, then at a qualifying n verify the actual bootstrap rule on1,000 simulations
   with2,000 resamples each, also using centered/rescaled empirical-family resampling.
   Require lower95% Monte Carlo bounds above0.80. Record approximation and non-normal
   assumptions. If no n≤384 qualifies, the declared outcome is **STOP: precision-limited**.
4. If authorized, freeze the selected n, seed24092432, exclusions, protocol/code/model/
   direction hashes and planning record, and push before confirmation forwards. Run once.
5. Development results remain development even when precisely estimated. No successful
   gate turns them into confirmation; do not report a planned-power calculation as data.

Release a short result page, raw paired records and a reproducible checker. A valid stop
or all-excluded result is retained. Existing rounds1/2 are not rescored or weakened by it.
