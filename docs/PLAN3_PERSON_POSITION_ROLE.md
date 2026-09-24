# Plan 3: from donor dependence to a test of relational-query transfer

**Execution update, 24 September 2026:** 3A's [512-family confirmation](ROUND3A_CONFIRMATION.md)
is complete. The original [32-family development run and planning STOP](ROUND3A_RESULT.md)
remain preserved; a prospective amendment added 512 and passed the unchanged power gates.
Both declared invariance profiles were excluded on fresh data. 3B remains an unexecuted proposal.
The design below preserves the original review draft; its proposed 3A resolutions were
subsequently fixed in the [execution protocol](../applications/makelov-2311.17030/DONOR_FACTOR_PROTOCOL.md)
before development outcomes. The structural check is a check of stipulated candidate
rules, separate from the new GPT-2 measurements.

## Goal and the two decisions

**3A:** Does the fixed patch respond only to the donor answer's mention position,
or does changing the donor answer's name also change its effect at fixed position?

**3B:** On a small, separately qualified task, does the patch behave like transferring
which relation is queried and applying it to the recipient story, rather than copying
a name, selecting a mention position, or leaving the recipient unchanged?

The delivered result should be a worked example with predictions, interventions,
measurements and a bounded decision. A role-consistent result would support one
operational response model on this task. It would not identify Kaplanian character,
prove a unique semantic representation, or explain the entire native circuit.

## What is already available

| Available and executed | Reuse | What must change |
|---|---|---|
| GPT-2 Small, pinned processed weights, published vector, isolated `.venv-round2` | Same cached CPU runtime; no new checkpoint or CUDA required | Record the actual environment again |
| `patch_deltas` / `position_hook` in `makelov_read_source.py` | Fixed MLP8 post-GELU null-read/full-write operator and insertion audits | Explicit donor/recipient indices; no `h.flip(0)` assumption |
| `query_route_sampling.py` | IID proposal, source/tokenizer hashes, exclusions and unique draw IDs | Four-prompt factorial families, then the new relation-query family |
| Round-2 numerical controls and provenance | Absolute positions, fp32/fp64 checks, identity controls, public freeze | Apply to the new cells and final contrasts |
| `design.signatures` and exact-binomial interval code | Structural grouping and coverage decisions | New profiles; do not reuse the round-2 three-profile classifier |
| Records-only verification and short result-page pattern | Public reproduction without model downloads | New schema, checker and worked example |

Round 1 already found a patch effect when the correct donor name stayed fixed and its
position changed. Strict identity-only invariance therefore already has a counterexample.
**3A's new contribution is testing additional name dependence and the adequacy of a
position-only account**, not discovering positional sensitivity for the first time.
Makelov et al. already motivate the positional signal and the interpretability illusion
([sections 4.1–4.4](https://arxiv.org/html/2311.17030v2)).

## Fixed instrument and an important algebraic limit

Keep the published direction and the exact round-2 operator:

`h'_r = h_r + alpha(d,r) * v`, where `alpha(d,r) = (h_d - h_r) dot v_null`.

Do not train another direction, select a new layer, tune a dose or add a query clamp.
Record alpha, inserted-vector norm, native margins and patched margins in every cell.
For a fixed recipient, all these patches lie on one line: output is `F_r(alpha)`.
Equal alpha values produce the same intervention, whatever the donor story says.
The experiment tests how donor variables control this scalar and its output effect;
it cannot infer different internal payloads just by giving the donor conditions names.

## 3A: four donors, one recipient at a time

Example recipient: “Then, Keith and Jason were working at the cafe. Jason decided to
give a notebook to”. Correct answer Keith; first mention Keith, second mention Jason.
Use the same scaffold in all four donors:

| i,p | First two names | Repeated giver | Correct donor answer | Answer position |
|---|---|---|---|---|
| 0,0 | Keith and Jason | Jason | Keith | first |
| 0,1 | Jason and Keith | Jason | Keith | second |
| 1,0 | Jason and Keith | Keith | Jason | first |
| 1,1 | Keith and Jason | Keith | Jason | second |

`i` indicates whether the donor answer identity differs from the recipient's;
`p` indicates whether its first-mention position differs. “Position” is a mention slot,
not an arbitrary token index. Run a mirrored recipient ordering inside the same unit
and align p to that recipient. The recipient is fixed within each four-donor panel.

One donor equals its recipient exactly: its delta is structurally zero. Keep and audit
it, but do not count it as a separate scientific discovery or independent observation.
The independent unit is one sampled name-pair/scaffold family with both recipient
orders, all donors, controls and precisions. Sample names symmetrically across families.

### Readouts and operational restrictions

Let `d_ip` be the patch-induced change in the recipient-correct-name minus other-name
logit margin. Never reorient that margin to the donor's answer in individual cells.
Compute, inside each recipient panel:

```
identity effect I = (d10 + d11 - d00 - d01) / 2
position effect P = (d01 + d11 - d00 - d10) / 2
interaction J     = d11 - d10 - d01 + d00
```

Average the two aligned recipient panels within the independent unit for mean-effect
inference. Preserve all simple effects, individual cells and alpha values separately.

| Response profile | Restrictions in each panel |
|---|---|
| Position-only invariance | `d10-d00` and `d11-d01` are practically zero |
| Identity-only invariance | `d01-d00` and `d11-d10` are practically zero |

These are restrictions, not complete semantic models or automatic predictions of an
answer flip. A name-dependent scalar need not encode a transferable person identity.
Do not fit a saturated 2x2 model to its own cells and call the exact fit a discovery.

**Proposed resolutions:** 0.10 nat for practical relevance of a population mean
contrast; 0.25 nat for each per-family invariance residual. These correspond to about
11% and 28% changes in pairwise odds, respectively. They define two different claims,
not numerical noise floors. Freeze or explicitly revise them before development,
never in response to which candidate happens to pass.

For each profile, a unit succeeds only if both restrictions are within 0.25 nat in
both recipient panels and both precisions. Require 80% population coverage, assessed
by simultaneous Clopper–Pearson intervals for the two profiles. Allocate family alpha
0.025 here. Lower >0.80 means adequate, upper <0.80 excluded, otherwise unresolved.

For I/P/J, use a family bootstrap over complete units with prespecified seed and
simultaneous Bonferroni intervals, total nominal alpha 0.025. Report bootstrap Monte
Carlo stability; this component does not have a finite-sample coverage guarantee.
Practical relevance requires the interval outside ±0.10; equivalence requires it wholly
inside. A zero-covering interval is not equivalence. Neither average nor coverage claims
replace one another. Both invariant profiles may fit simply because every effect is small.
The two resolutions can also disagree: a profile can meet the coarse 0.25-nat/80%
criterion while mean identity dependence exceeds 0.10 nat. Report both; this is not an
unqualified position-only result. Require each mean decision to hold in both precisions,
using identical bootstrap draws, even when their numerical discrepancy is below 0.01.

### What 3A cannot distinguish

In two-name IOI, selecting the donor IO position and suppressing the donor subject
position have identical answer predictions. Group them. Transferring the constant
instruction “choose the recipient” predicts the recipient's original answer, like no-op.
These ambiguities are mathematical properties of this candidate menu, not missing power.

## 3B: transfer a queried relation, not an unchanging role label

3B is a new task family and first gets a baseline-only feasibility test. The transferred
candidate variable is **which relation is queried** (giver or recipient); the entities
filling those relations remain those of the recipient story.

Use the same two names in donor and recipient, so name-copy predictions do not require
an entity absent from the recipient. Example donor story:

> Bob received a book from Alice.

Query either “The person who gave the book was” or “The person who received the book was”.
Donor answers are Alice and Bob, at first-mention positions two and one.

Recipient stories cross who gives with surface ordering:

| Story | Giver | Recipient | First mention |
|---|---|---|---|
| Alice gave a book to Bob. | Alice | Bob | Alice |
| Bob received a book from Alice. | Alice | Bob | Bob |
| Bob gave a book to Alice. | Bob | Alice | Bob |
| Alice received a book from Bob. | Bob | Alice | Alice |

For each story, ask both queries and patch from both donor queries: **16 cells per
family**. Counterbalance donor surface form across sampled families. These are illustrative
strings; token-width and task-competence checks determine whether this exact instrument
is feasible. A failed task check does not license silently substituting another family.

### Target rules first, then physically compatible response models

Let `B_r(q)` return the recipient entity filling relation q, `N_r(p)` the recipient
name at mention slot p, and let `q_d`, `q_r`, `a_d`, `p_d` be the donor/recipient queried
relations, donor answer name and donor answer position.

| Candidate | Proposed answer target |
|---|---|
| Relational-query transfer | `B_r(q_d)` |
| Name copy | `a_d` |
| Mention-position transfer | `N_r(p_d)` |
| Opposite-position rule | `N_r(1-p_d)` using slots 0/1 |
| No change | `B_r(q_r)` |
| Always flip | The other recipient name than `B_r(q_r)` |

The shipped structural check finds six distinct target signatures across the 16 cells.
Without cross-query transfers, relation and no-op coincide. With only `gave_to` recipient
stories, relation coincides with opposite-position for a `received_from` donor, but
with position for a `gave_to` donor. The script checks both orientations.

**Target labels are not yet valid intervention models.** The literal opposite-position
and always-flip rules can contradict donor=recipient identity: the physical patch is
then zero, but these rules demand a change. Mark them as structurally disqualified
teaching foils, not empirical competitors defeated by new data. Opposite-position is
also not a general model of positional inhibition. The quantitative comparison below
therefore includes a delta-aware inhibitory model instead of crediting these two foils
as extra empirical wins.

One further close rival must remain grouped with the relation rule: **flip the recipient
answer exactly when donor and recipient query different relations**. With two roles and
two entities this has identical predictions, even with the surface variations. Report
that equivalence; a third queried role or a different intervention would be needed to
test beyond it. Do not add that larger task to this minimum.

This still does not enumerate all possible lexical or routing rules. In particular,
“gave/received” cue transfer can mimic a relational selector. Add one predeclared
development wording and a paraphrase family held out from patch fitting that changes the
question wording and story formulation while preserving the queried relation. Establish
baseline competence on both. Baseline qualification may inspect native outputs for the
held-out wording; never use its patch outcomes to fit gains or choose wording.
Generalization only limits the tested surface alternatives;
it never proves that all lexical accounts are impossible.

### Numerical predictions, not only answer labels

Use a fixed Alice-minus-Bob name-logit margin, with sampled names replacing the labels.
For every recipient story, its unpatched giver-query and recipient-query margins provide
two independently measured natural endpoints. Every candidate target maps to one of them.

Proposed partial-transfer response model:

`pred_H = native_margin + g_H * (target_endpoint_H - native_margin)`.

Fit one nonnegative gain `g_H in [0,1]` per non-null candidate on development families
only; freeze it for confirmation. No-op predicts the native margin and has no parameter.
All non-null candidates have the same parameter allowance. A gain of zero collapses a
candidate into no-op; nearly coincident predictions are not an identification success.
For the inhibition alternative use
`pred_inhibit = native_margin - g_inhibit * (position_endpoint - native_margin)`.
Unlike selecting the opposite position outright, this respects donor-self identity.
The five numerical comparison groups are relation/conditional-flip, name-copy,
position, inhibitory-position and no-op. Disqualify any proposed response rule that
violates exact self-identity before fitting or collecting confirmatory outcomes.
Natural-query endpoints are a testable calibration assumption, not automatically the
outcome of an internal intervention. Amplification or nonlinear effects may defeat every
candidate; retain that result rather than widening the gain after confirmation.

Evaluate both absolute prediction error and paired relative error on fresh whole
families. If making a broad adequacy claim, predeclare 0.25-nat per-cell tolerance and
80% complete-family coverage, including the held-out wording and both precisions.
Declare a separate uncertainty family for 3B before its confirmation. A relative winner
without adequacy is reported as a relative winner. If the calibrated candidates cannot
be separated at the available resolution, stop before confirmation. For the proposed
0.25-nat tolerance, require a calibrated predicted gap greater than **0.52 nat** in a
predeclared diagnostic cell for each pair of groups being claimed distinguishable:
0.50 for nonoverlapping tolerance bands plus two 0.01 numerical allowances. Inspect this
on development only to authorize confirmation. On fresh families, retain all weak-gap
cases and report their frequency; they cannot count as uniquely distinguishing successes
and may not be removed or replaced. Exact equivalences remain grouped regardless of n.

## Small execution sequence and gates

1. **Structural check now:** run `python3 examples/plan3_design_check.py`. It verifies
   the groupings, the two missing-condition failures and their interpretation using
   the shipped `design.signatures` implementation. No model or fitted parameters.
2. **Implement 3A only:** replace flip-based pairing with explicit maps, save all four
   donor activations and two recipient baselines, record scalars and margins. Include
   synthetic pure-position, pure-identity, mixed, interaction and all-small worlds;
   test the actual analysis code on these before any development outcomes.
3. **Develop on 32 new 3A families:** check identities, repeat two historical positive
   controls, qualify fp32/fp64 and measure runtime and paired variability. Exclude all
   previous Q1/round-2/development prompts, checking every new donor as well as receivers.
   Do not select the confirmatory population by observed patch effect.
4. **Choose n mechanically, then freeze:** consider n=128,192,256,384. Require at least
   80% exact-binomial power at the declared planning alternative of 90% complete-family
   profile success. Also assess mean-contrast precision/power using paired development
   variances inflated by a fixed factor 1.5 in standard deviation. The mean target is
   at least 80% planning power **for each of I, P and J separately**, under a mean of
   either +0.20 or -0.20 nat against the ±0.10 practical boundary; it is not joint power.
   Select the smallest n meeting both that target and the coverage target; cap at 384.
   Use 10,000 Gaussian unit-level planning simulations with the inflated development
   covariance and the simultaneous normal-reference interval rule (family alpha 0.025,
   three contrasts). Check the planned nominal bootstrap rule on 1,000 simulations
   with 2,000 bootstrap draws each at the selected n, and on centered/rescaled empirical
   unit resamples. Require the lower 95% Monte Carlo bound on detection probability to
   exceed 0.80; otherwise try the next n. These are explicit planning assumptions,
   not distribution-free power guarantees. Precision tables are descriptive and cannot
   independently authorize a smaller n. If none qualifies, report the
   limitation or revise the plan openly before any confirmation, not mid-run.
5. **Confirm 3A once:** freeze code, thresholds, cases, source hashes, statistics and
   the selection calculation, with a public commit before inference. Publish all cells
   and outcomes, including exclusions and unchanged rival groups.
6. **Qualify 3B separately:** baseline-only 32 development families for giver/recipient,
   both surface forms and the held-out wording. Proposed task gate: at least 90% correct
   conditional two-name choices within each query/form group, positive counterfactual
   separation, and no token-position confound. Report intervals, candidate-name mass and
   full-vocabulary argmax; the gate is feasibility, not a population competence guarantee.
   No inference about role transfer is allowed from an incompetent instrument. Stop rather
   than select only convenient items. New templates or another model require a new plan.
7. **Only if 3B qualifies:** run its development patches, freeze gains and predictions,
   repeat the resolution and precision planning for its actual losses, then publish a
   separate confirmation contract. Do not reuse 3A's sample-size justification. Keep the
   original direction; absence of a suitable effect can be a valid instrument limitation.

Numeric check for step 4 (two profiles, family alpha 0.025, coverage 0.80):

| n | Successes required for adequacy | Power if complete-family success probability is 0.90 |
|---|---:|---:|
| 128 | 114 | 0.7020 |
| 192 | 168 | 0.8960 |
| 256 | 221 | 0.9765 |
| 384 | 327 | 0.9989 |

These are exact-binomial planning calculations, **not predictions of actual performance**.
They do not establish power for I/P/J or for the later relational task.

## Technical rules, claims and budget

- Recompute caches independently in float32 and float64 with the historical processing
  flags. Higher precision is a reference, not exact truth. Validate actual post-cast
  insertions, hook count, absolute position, zero-delta and recipient-self identities.
- Proposed final-contrast discrepancy budget is 0.01 nat (one tenth of the smallest
  scientific boundary). A failing unit cannot count as profile success. Do not drop it.
  Any unresolved precision discrepancy in a mean estimand blocks its confirmatory claim;
  raw results in both precisions remain available. Technical failures block interpretation.
- Align name-token positions within each crossed family. Query words must be chosen and
  token-checked before outcomes; donor and recipient positions come from their own records.
  No silent padding, reused final-token index, or outcome-dependent prompt replacement.
- With precisely negligible responses: report limited effect under this instrument.
  With broad intervals: unresolved. With large interaction or failed profiles: these
  simple models are insufficient. None means “the model has no semantic understanding”.
- Phase 3B is not contingent on a favorite winner in 3A. It requires a competent task,
  valid nontrivial interventions and separable calibrated predictions; otherwise release
  the 3A result and the explicit feasibility limitation.

Engineering estimate: **3A about 4–8 hours**, including generator, tests, pilot, freeze
and report; **3B about another 1–2 working days if the task qualifies**. Round 2 took
207.8 seconds for 2,688 forwards; 3A has more cells and 3B longer prompts. Provisionally
budget 5–20 minutes of model computation for 3A confirmation, and tens of minutes for
3B; replace these estimates with development measurements. No paid compute is needed
for the proposed cached-model version. A failed 3B task gate ends this scope.

The final reader page should show: one story, rival predictions, one added distinguishing
condition, fresh outcomes, and exactly which explanations remain. It must distinguish
**controlled donor dependence (3A)** from **a tested relational response model (3B)** and
from **identification of a unique native semantic mechanism (still not established)**.
