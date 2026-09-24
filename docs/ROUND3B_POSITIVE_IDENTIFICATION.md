# Round 3B: a positive identification test, not just another exclusion

**Status: prospective design and executable analysis; no 3B model result.**
This replaces the two-person 3B proposal in [Plan 3](PLAN3_PERSON_POSITION_ROLE.md).
The established [3A result](ROUND3A_CONFIRMATION.md) is unchanged. Nothing here claims
that the published MLP direction already represents a role.

**Pre-run literature audit:** [the Mixing Mechanisms crosswalk](GOODFIRE_CROSSWALK.md)
records which published retrieval mechanisms are represented, undefined, or still
indistinguishable here. The [Stage A protocol](../applications/makelov-2311.17030/ROLE_BASELINE_PROTOCOL.md)
now freezes native feasibility only. The absent-target pointer test is a separate
follow-up; it is not silently added to the present three-name readout.

**Goal:** on fresh cases, positively support an accurate, nontrivial model of how a
cross-query patch responds to the recipient's role bindings, while excluding specified
name, position, unchanged-response and query-change-only alternatives. A relative winner
alone is insufficient. A supported result would identify a response class within this
candidate menu, not the unique internal implementation or the native language circuit.

## The example that makes a positive result possible

Keep this recipient fixed:

> Bob gave a book to Carol while Alice watched. The person who watched the exchange was

Its native answer should be **Alice**. Compare two donors:

| Donor story and question | Donor answer | Mention slot | Role-transfer target in the recipient | Name-copy target | Position-copy target |
|---|---|---|---|---|---|
| Alice gave a book to Bob while Carol watched. Who gave it? | Alice | First | **Bob** | Alice | Bob |
| Alice received a book from Bob while Carol watched. Who received it? | Alice | First | **Carol** | Alice | Bob |

The same donor answer and mention slot now require **different recipient answers**
under relational-query transfer. The actual prompts use the exact completions emitted
by `role_design.render`; the questions above are shortened for reading.

We also ask about **all three roles** in each donor story. For one fixed donor story
and an observer-query recipient, giver and receiver queries are both changes of question,
yet demand different answers. A rule that knows only *whether* the question changed
cannot explain that difference. A third person who is never queried would not solve
the old two-role ambiguity.

There are three names, so we measure the whole `(A-B, A-C, B-C)` logit-margin vector,
with two independent dimensions. This matters: even an arbitrary change of strength
along the Alice-to-Bob response line need not reproduce an Alice-to-Carol response.
Actual native endpoints must establish this separation; the answer labels alone do not.

## The small fixed design

- **Instrument:** GPT-2 Small, MLP8 post-GELU at each prompt's own final position;
  the existing null-read/full-write patch, unchanged. No layer search or new learned axis.
- **Two donor stories:** the matched gave/received pair above, each queried for giver,
  receiver and observer. This breaks the fixed association of donor name/position with role.
- **Six recipient contexts:** three cyclic name-to-role assignments, each in two mention
  orders. Cross every donor query with every recipient query: **108 raw patch cells**.
- **Two wording families:** `fit` and `heldout` are wording labels. On confirmation,
  both belong to fresh cases. The latter changes both story and question wording and
  contributes no patch outcomes to gain fitting or selection.
- **Primary population:** the **72 cross-query cells per wording** (`donor_query !=
  recipient_query`). The 36 same-query cells are mandatory separate diagnostics, with
  their raw effects reported; they cannot dilute primary errors or be silently omitted.
- **Unit:** one name-triple/object family containing both wordings, all contexts,
  controls and precisions. Cells and repeated forwards are not independent samples.

## Predictions that can win, and strong alternatives

Let `z_r(q)` be the three native name logits for recipient story r queried about role q.
Convert these to pairwise margins `m_r(q)`. These native measurements are calibration
inputs, obtained separately from patch outcomes. Let `y` be the patched minus native
recipient margin vector, and `q_r` its original question.

| Candidate | Absolute response prediction or allowed response set |
|---|---|
| **Relational query** | `g[q_r] * (m_r(q_d) - m_r(q_r))` |
| **Name copy** | Any signed multiple of the native endpoint difference toward the donor answer name |
| **Position copy/inhibition** | Any signed multiple of the native endpoint difference toward the recipient name in the donor answer's mention slot |
| **Question changed only** | One common response for the two different donor queries, within each fixed donor story / recipient / original-question group |
| **No effect** | Zero |

Fit exactly **three role gains**, one per original recipient query, in `[0,1]`, on
development families in the fit wording. Weight families equally; report unconstrained
estimates and saturation. Freeze the gains before confirmation. No intercept, per-case
gain fit, or expansion after looking at confirmation errors is allowed.

The name and position competitors are deliberately generous: give them the **best
possible signed gain separately in every observed cell**. Their distance to this oracle
line is a lower bound on the loss of any predictor confined to that line. Beating this
bound cannot be credited merely to choosing too weak a rival gain. It is a comparison
against a response-set restriction, not an out-of-sample fit of the oracle itself.

For question-changed-only, take the best common response to its two observations (their
mean is one minimizer of average RMS distance). Again this is an optimistic bound.

**Use absolute patch responses here.** Subtracting two patched-query outcomes would
allow a position model with independently changing gains to produce a combination of
two directions. Calling that larger set a single line would create a weak rival. Those
matched differences may be reported diagnostically, but do not drive identification.

## What counts as a positive result

The numerical choices below define the proposed scientific resolution, not a guarantee
of success. They are fixed in the executable analyzer; a change requires a documented
revision before real development patch outcomes, never a quiet adjustment to make it pass.

1. **Accurate predictions:** within each wording and precision, the average cell error
   is at most **0.25 nat**, measured as RMS error across the three pairwise margins.
   This is not a 0.25-nat bound on every individual coordinate or cell. Require the
   same error bound separately at each diagnostic cell fixed in `DIAGNOSTICS`.
2. **Genuine separation:** at those fixed diagnostics, the frozen role prediction must
   be more than **0.52 nat** from the corresponding rival response set: twice the
   adequacy tolerance plus two 0.01 numerical allowances. This is a resolution rule,
   not a general threshold theorem or power guarantee. For the switch rival the
   diagnostic is a pair of predictions and its best common response.
3. **Broad success:** require the lower exact binomial bound for the *joint event*
   in 1–2 to exceed **80% of independent families**. Weak-gap cases remain failures
   for this event; they are not removed. Allocate alpha 0.025 to this one coverage claim.
4. **Paired improvement:** require a simultaneous lower bound above **0.10 nat** for
   each rival's oracle loss minus role-model loss, in both wordings and precisions.
   Whole-family percentile bootstrap, alpha 0.025 over all 16 comparisons; this
   component has nominal, not guaranteed finite-sample coverage. Both frozen seeds
   must agree, with endpoint changes at most 0.01 nat.
5. **Valid measurement:** actual insertion/identity controls pass; final margins,
   errors and contrasts agree across fp32/fp64 within 0.01 nat. All families stay in
   the accounting. Invalid measurements block the positive claim.

Only the conjunction supports the relational response model in this declared menu.
A sole survivor, a relative winner, a tiny effect, or a synthetic success is insufficient.
The report separately exposes failure of adequacy, separation, power or instrumentation.
Same-query behavior remains outside this primary adequacy claim, even if it contradicts
a stronger account of the *entire* role-transfer operation; that contradiction must be shown.

## Execute in three bounded stages

**A. Baseline feasibility, before any 3B patches.** Sample 32 development families with
fixed names/token eligibility and a saved seed. Check all three roles in both wordings
and forms; require at least 90% conditional three-name accuracy in every declared
role × wording × form group. Report uncertainty, full-vocabulary argmax and name probability
mass. This is a feasibility threshold, not a population competence claim. Do not retain
only correctly answered individual items. Stop the task family if it fails.

Verify single-token names, actual name spans, absolute final positions and finite scores.
The three question variants must have equal token length within a story/wording, and
the matched donor collision must preserve the answer token and its position. No silent
padding or outcome-dependent replacement. At optimistic role gain 1, all fixed diagnostic
gaps must exceed 0.52 nat in both wordings and precisions for at least 29/32 families.
This is potential separation, not an estimated transfer gain or power guarantee.
A third label does not guarantee a third usable direction.

**B. Development of the fixed predictor.** If A passes, collect fit-wording patches on
these 32 families only. Use four whole-family folds to assess held-out prediction and
joint success during development, with the split seed recorded in advance. Fit the final
three gains on all development families only after this assessment. Do not inspect the
heldout-wording patch outcomes during development. Keep a historical patch positive control
to distinguish a broken hook from an instrument that simply does not transfer queries.

Plan the actual joint coverage and paired losses for candidate sizes **128, 256, 512**,
using a declared joint-success alternative of 90% and a paired-advantage alternative of
0.20 nat against the 0.10 boundary. Include complete-family covariance, numerical error,
gain-estimation uncertainty and Monte Carlo uncertainty. The confirmation size is not
chosen here: require at least 80% planning power for the conjunction, including a
prespecified sensitivity analysis for heldout-wording degradation (not measured secretly
on its patch outcomes). No qualifying size means stop or openly revise the design.

**C. Fresh confirmation.** Publish the final population definition, tokenized cases,
all native-endpoint rules, gains, diagnostics, power calculations, model/vector/code
hashes, seeds and a separate execution manifest before patch outcomes. Exclude development
families and prior exact prompts. Run both wordings and precisions once, preserve all
108 raw cells per wording plus explicit self/zero controls, and release a records-only
checker. Do not use 3A's sample size or power calculation as justification for 3B.

## What is built, and what remains

| Component | Status |
|---|---|
| Three-role prompts, full grid, rival targets and fixed diagnostics | Implemented; structural checks only |
| Gain fit and unrestricted oracle geometry | Implemented and test-covered; no neural evidence |
| Joint positive-decision rule | Implemented and tested on constructed data |
| Native task competence, token/geometry qualification | Not yet measured |
| Model runner, new power calculation, frozen execution manifest, records-only checker | Required before confirmation; not supplied by the structural demonstration |
| Empirical 3B result | None |

Run the [demonstration](../examples/role_transfer_design.py) to see the example, structural
separation and a constructed positive case. These are **synthetic checks, not model results**.

```bash
python3 -I -S examples/role_transfer_design.py  # structure; standard library only
python3 -m pip install -e '.[analysis]'         # optional, for the statistical check
python3 examples/role_transfer_design.py --exercise-analysis
```

The statistical demonstration feeds three constructed worlds into the same analyzer:
relational responses, positional responses with varying signed gain, and no effect.
Only the first supports the target response model. It uses 128 synthetic families and
2,000 bootstrap draws per seed for a short software exercise; it does not substitute
for the 20,000-draw confirmation rule, power planning, or independent method validation.

The same fixed one-dimensional patch may simply be unsuitable for this richer task.
That would limit this instrument, not prove that the method cannot identify anything.
Searching another site or direction would be a separately declared discovery stage.
Even success would leave query-sensitive lexical algorithms, other nonlinear positional
mechanisms and implementations with identical predictions unresolved. The unrestricted
physical relation `output = F_recipient(alpha)` is not a semantic rival this test can refute.

**The substantive possible gain:** a prospective, quantitatively accurate account of
cross-query intervention responses that generalizes to new role bindings and an unfit
wording and survives more than a name/position-only interpretation. This would go beyond
the current exclusions without claiming that the full identification problem is solved.
