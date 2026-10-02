# Can a native margin predict which intervention families separate two accounts?

**Frozen prospective screen validation, with secondary mechanism development.**
This is one new bounded round on the previously selected Dyck head `a9g0io1r`,
layer 2/head 1 (one-based). It revises the sampling strategy after development
001; it does not rescue that failed gate or change its records. No unseen-head
generalization, saturation mechanism or general method superiority is claimed.

## Goal and why a second group is necessary

Test whether the **signed native margin <8 nat**, available before any transfer,
predicts a higher rate of separating anchor outcomes. An accepted-only sample
would estimate its yield, but would not establish enrichment over rejected cases.
The primary result compares 32 accepted with 32 rejected recipient–donor families.
The cutoff was suggested after inspecting development 001 (8/9 accepted cases
separated); that is development evidence, not a forecasted result.

This is a forecast of **one fixed, sufficient separation criterion** for two
anchor-calibrated accounts. It is not a per-case certificate that an unknown
mechanism can be identified, nor a guarantee that either account will fit.

## Population, inputs and exact sequence

1. Freeze this protocol, executable sources and prepared inputs in a local git
   commit before new native scores or transfers. Local commits/hashes establish
   an auditable local sequence, not externally witnessed preregistration.
2. Draw 1,024 candidate recipients with seed **25100202**, with replacement from
   the earlier development population: length 32, 16 opens/16 closes, some prior
   negative prefix balance, development hash partition. Exclude previously used
   full strings. Use a separate seed **25100203** for 64 donor-grid templates;
   donors are not selected by any measured outcome.
3. Extend the fixed exclusions with **all recipient and donor strings from
   development 001**. Exclude both position-20 and position-28 prefixes of every
   such string from donor pools, including across original roles and positions.
   Retain earlier public/native-run exclusions. This repairs the documented
   cross-position prefix reuse risk; original training overlap remains unknown.
4. Evaluate every candidate natively in fp32 and fp64. The screen is
   `m = logit(False) - logit(True) < 8`, **not** `abs(m)<8`. Take the first 32
   accepted and first 32 rejected in the randomized input order. Keep every
   native score, classification and selected/unselected status. If either quota
   is unavailable, stop with insufficient screening yield; do not add candidates.
5. Attach donor templates 0–31 to accepted and 32–63 to rejected cases. All templates
   have the same sampling distribution; recipient selection never inspects donor
   outcomes. Family draws are independent with replacement; repeated strings and
   prefixes are reported. Freeze the selected manifest and native screen
   predictions with a hash receipt **before any anchor transfer**.
6. Use the existing operator and controls unchanged: the highest-native-attention
   closing-bracket position in each recipient (fp64, first tie), native attention
   coefficient, single-query head contribution replacement
   `h' = h + a[j] * (v_donor[k] - v_recipient[j])`. Two balances −2/+2, positions
   20/28, minimum −4, target symbol `)`, two distinct donor prefixes per cell.
7. Measure anchors `neg_20_0` and `pos_28_0` in **both** groups and dtypes. Save
   the original H_state and H_position anchor-calibrated forecasts and hash them
   before the six remaining transfers, which are measured only for the 32
   accepted families. Rejected families need only the two anchors for the primary
   screen test; their unmeasured candidate fit is not inferred.

The 1,024 baseline candidates do not become 1,024 intervention observations.
The unit for the primary analysis is a complete recipient–donor family, n=32 per
stratum. Only the two anchors determine the primary response. The secondary
question uses all six target cells within each accepted family.

## Technical gates and missingness

Use CPU, one thread, deterministic PyTorch, the pinned upstream source/checkpoint
and the existing value-transfer fidelity/identity/off-target checks. Preserve full
node snapshots. Numerical allowance is 0.001 nat, distinct from the scientific
0.10-nat candidate tolerance. All 1,024 native margins must agree between dtypes
within 0.001 and give the same screen classification. All 64 anchor gaps must
agree within 0.001 and lie on the same side of the strict **0.202-nat** boundary
in both dtypes. The accepted six-cell prediction-error precision gate is unchanged.

Any required technical failure stops the run and blocks the result; record it,
do not remove or replace a family. No retry with another seed, head, donor,
recipient position, tolerance or screening threshold is authorized by this round.

## One primary estimand and its decision

Let `S=1[abs(anchor_D - anchor_A)>0.202]`. Estimate

`Delta = P(S=1 | m<8) - P(S=1 | m>=8)`.

Report both rates and their exact binomial Clopper–Pearson bounds. Allocate
alpha=0.0125 to each of the four one-sided bounds; the joint coverage is at least
95%. From these derive `[L_accepted-U_rejected, U_accepted-L_rejected]` for Delta.
Do not bootstrap all-zero/all-one strata into degenerate certainty.

The practical target is **25 percentage points more separating families than in
the rejected stratum**, chosen as an operational benefit (one additional
separating family per four tested families), not from residual variance.
This is not the improvement over unfiltered sampling, which also depends on how
frequently recipients pass the screen.

- Lower bound >0.25: supports the declared substantial-enrichment target.
- Lower bound >0 but <=0.25: supports positive enrichment, without establishing
  the practical target. The same interval supports this weaker directional claim.
- Interval includes zero: positive enrichment remains unestablished.
- Upper bound <0.25 excludes the practical target; upper bound <0 additionally
  supports worse separation among accepted cases. Neither proves universal
  uselessness of screening or absence of causal mechanisms.

These are readings of the **same fixed interval**, not separate hypothesis
searches. `planning.json` gives exact hypothetical power, conditional on filled
strata and valid measurements: at accepted/rejected separation rates 0.90/0.05,
the substantial target has about 99.76% power; at 0.75/0.10, about 56.44%; at
0.50/0.10, about 1.64%. These assumptions are not predictions from nine cases.

## Secondary question: unchanged feasibility gate

On accepted families only, retain the previous scientific tolerance 0.10 nat,
definite-hit limit <=0.099, possible-hit limit <=0.101 and strict anchor gap
>0.202. H_state/H_position must predict all six target margins from the same
two anchors. The original gate remains **at least 16/32 separating families and
at least 90% definite hits for one account**. Failure remains a development stop;
a primary screen success cannot override it. Passing permits planning a separate
confirmation; it does not itself confirm a semantic mechanism or start one.

Exact balance versus sign, normalized-depth aliases and correlated features of
whole donor values remain unresolved. No third balance level is introduced.

## Report and scope

Keep raw native scores, all selected/rejected records, both dtype anchor/cell
observations, forecasts and receipts, technical controls, source hashes and
complete failures. Report screening prevalence and computation separately from
intervention yield; this validation does not demonstrate net computational savings.
Publishable local claim if supported: **On this head and intervention family,
the frozen native-margin screen prospectively enriches for anchor-separable cases.**

Fresh cases on this tuned head can validate that local claim. Generalization to
unseen heads/tasks would require a separate frozen transfer evaluation. Every
outcome is reportable; only the predefined positive result supports the screen.
