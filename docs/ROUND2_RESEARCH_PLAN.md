# Round 2: distinguish query transfer from joint dependence

**Execution update:** see the [completed experiment](ROUND2_RESULT.md) and its
[frozen protocol](../applications/makelov-2311.17030/QUERY_ROUTE_PROTOCOL.md).
The original review draft below is retained for provenance.

**Review draft, 24 September 2026. No model experiment has run.**
This revises the proposed four-cell experiment in
[Iterative identification](ITERATIVE_IDENTIFICATION.md), not the confirmed round-1
result. Scientific thresholds below are explicit proposals, not fitted results or an
already frozen protocol. The old teaching demo still implements the four-cell design.

## Critical finding

Restoring selected attention queries can remove a patch effect without showing that
the queries carry that effect by themselves. Consider two constructed mechanisms:

| Mechanism | Equation, with natural query state `M = X` | Patch on | Patch on, query reset | Patch off, patched query inserted |
|---|---|---:|---:|---:|
| Query state sufficient | `Y = M` | 1 | 0 | 1 |
| Joint dependence | `Y = X * M` | 1 | 0 | 0 |

The old experiment cannot distinguish these examples. **One additional transfer,
without the original patch, separates them.** This is a useful next application of
causal decidability: first identify a countermodel to the proposed interpretation,
then add the cell on which it disagrees.

Neither outcome identifies a unique causal graph. For example, additional paths can
cancel. The experiment concerns the response to an artificial patch, not the native
semantic computation of the unmodified model. The selected heads are already motivated
by [Makelov et al., §5.1](https://arxiv.org/html/2311.17030v2#S5.SS1); this is not a claim
to discover a previously unknown circuit.

## One model, one intervention, one selected query group

- GPT-2 Small; reuse the published vector and round-1 processed-model conventions.
- Keep the `read_null` intervention at MLP8 post-GELU, at absolute position `P−1`:
  `delta_h = ((h_donor - h_recipient) · v_N) v`.
- Only the projected queries at heads `(9,6)`, `(9,9)`, `(10,0)` are manipulated,
  at `blocks.9.attn.hook_q` and `blocks.10.attn.hook_q`.
- Baseline queries `q0` come from the same recipient without the MLP intervention.
  Patched queries `q1` come from that recipient with the MLP intervention and no query
  clamp. Cache these at both sites; never replace them with a hybrid run's queries.
- Keys, values, other query slices and subsequent computation remain free. This tests
  query changes, not every possible route through the selected heads.
- One base pair contains both reciprocal ABB/BAB directions. Those directions retain
  the same correct IO name; they are not two independent units or a donor-answer swap.

## Five scientific cells, plus implementation identities

`Y` is the IO-minus-subject answer-logit margin.

| Cell | MLP patch | Selected queries | Purpose |
|---|---|---|---|
| A | Off | Free; cache `q0` | Baseline |
| B | On | Free; cache `q1` | Original patch effect |
| C | Off | Insert `q0` | Baseline identity control |
| D | On | Insert `q0` | Remove the patch-induced query changes |
| E | Off | Insert `q1` | Test the query changes without the original patch |

Compute `T = B-A`, `R = D-C`, and `S = E-C`. If the identity gate passes, these have
the intended common baseline. Record every cell. Also record `T-R-S` descriptively as
an interaction contrast, never as a mechanism fraction.

Add a technical self-insertion with the patch on and its own `q1` inserted; it must
reproduce B. This is a sixth forward, **not independent rescue evidence**. A zero-delta
MLP identity is another technical check.

## Three operational profiles, declared before development

| Profile | Predicted `(R,S)` | Supported interpretation if adequate |
|---|---|---|
| Transfer | `(0,T)` | Resetting these queries removes the effect; transferring them alone reproduces it. |
| Joint dependence | `(0,0)` | Neither the remaining patch nor the transferred queries reproduce the effect alone. Gating or synergy remains possible. |
| Preservation | `(T,0)` | The patch effect survives query reset; transferred queries alone do little. This does not prove the queries never participate. |

Intermediate, amplified, reversed and redundant effects can fit none. These profiles
describe controlled responses, not exhaustive mechanism families. Even the transfer
profile does not establish the only pathway or a linguistic meaning.

**Proposed scientific contract:** a profile fits a directed case when both of its
predicted contrasts are within `0.25 * |T|` of the observed contrasts. A base pair is
successful only if both directions fit in both precision evaluations. Thus the claim
is accuracy to within one quarter of the local patch effect, not a 75% mechanism share.
Set the required population coverage to 80%.

These choices express the strength of the claim. Freeze them before development and
do not relax them to obtain a result. There is no additional arbitrary minimum effect
size: report the full T distribution, and do not describe all successful effects as
behaviorally large. An unresolved or zero anchor is not a success and stays in the
denominator. Technical failure invalidates the run rather than counting against a
mechanism.

## Small execution sequence

1. **Restore the pinned environment and qualify the instruments.** GPT-2, the vector
   and historical inputs are cached. The checked Python environment lacks
   TransformerLens 2.17.0; install and pin it separately without silently substituting
   the available 2.11.0 wheel. Verify processing flags, hook shapes and execution order.
2. **Implement the five cells and two identities.** Test exact index selection, isolated
   local writes, source provenance, post-cast tensors, separate hook counts and cleanup.
   Heads 9.6/9.9 share one hook call. At layer 10, audit the query actually arriving
   after the layer-9 manipulation. Non-target tensors must be unchanged by each local
   write; later computation need not equal baseline.
3. **Extend the synthetic demo first.** Include the two equations above, preservation,
   partial effects and cancelling paths. Run the actual profile-analysis code on their
   cells. No graph label should become a stronger conclusion than its observations.
4. **Run 32 separate development base pairs.** Verify identities and numerical
   resolution, estimate runtime and profile frequencies. Exclude their prompts and
   all prior pilot/Q1 prompts from confirmation. The pilot may stop the experiment or
   motivate a separately documented redesign; it cannot modify this contract silently.
5. **Freeze, then run 192 fresh base pairs once.** Pin the independent sampling
   distribution, eligible prompt pool, duplicate policy, seeds, vector/model/code
   hashes, controls and statistics before reading confirmation outcomes. Record a
   public timestamp if this is to support a publicly verifiable preregistration.
6. **Release one short result page.** Show the five cells, T/R/S distributions, profile
   counts, simultaneous intervals, all control outcomes and one worked example. Link
   the manifest and a records-only reproduction command. Preserve round 1 regardless
   of the new outcome.

The old draft's off-target clamp and outside-route sensitivity probe remain useful for
stronger specificity claims. Their calibration is not part of this minimum: the reverse
transfer adds a discriminating observation, but does not replace those controls as a
proof against all generic disruption. Restrict the minimum's claims to the profiles above.

## Precision and decision rules

Use CPU float32 and a float64 reference for **all** confirmation cells, rebuilding
baseline and patch caches independently. Retain the same processed-model conventions;
promotion of processed weights is not a ground truth model. Tensor identity budgets
must reflect dtype and tensor scale, not a chosen tiny constant.

Proposed resolution rule: for each direction, let `t` be the smaller absolute T across
the two evaluations. Require nonzero, same-sign anchors and a maximum cross-precision
discrepancy in T, R or S of at most `0.025*t` (one tenth of the scientific tolerance).
Otherwise the pair is unresolved and is not a profile success. This is a declared
instrument-stability criterion, not a proven bound on error relative to exact arithmetic.
Report counts before and after this gate. No confirmation case is dropped.

Use simultaneous Clopper–Pearson intervals for the three base-pair success fractions:
each interval has error probability `0.05/3`, with each tail `0.05/6`.

- Lower bound above 0.80: the operational profile meets the population requirement.
- Upper bound below 0.80: the profile fails that requirement.
- Otherwise: unresolved for that profile.
- All three excluded: none of these profiles meets the requirement; this need not mean
  no causal structure exists or that one profile applies to no individual case.

The binomial calculation requires independent identically sampled base-pair units from
the frozen generator. Verify that assumption in the sampling implementation. If the
available sampling scheme instead introduces clusters, redesign the unit and power
calculation **before** freezing; never count swaps or repeated forwards as new cases.

With 192 iid units, adequacy requires **167/192 successes**. Exact binomial power is
**93.07%** if one profile's complete base-pair success probability is 90%. This includes
both directions, both precisions and the resolution rule; it is not the success rate of
one component. A 90% success probability is a planning alternative, not a prediction
from round 1. For comparison, 160 units need 141 successes and give 82.35% power under
the same alternative. The extra 32 units are cheap relative to validating the hooks.

Development estimates are uncertain. If none of the profiles appears to approach the
planning alternative, first inspect whether the candidates or intervention need a new
round; do not promise that collecting 192 cases will manufacture a positive result.

## Effort and definition of success

Engineering estimate, not a measured deadline: environment and hooks 3–5 hours; synthetic
checks and technical pilot 2–3 hours; freeze, confirmation analysis and readable release
2–3 hours. Plan roughly one working day, with a second day for technical failures.
Model computation should be a small fraction; provisionally allow 10–45 minutes for
the paired-precision run, to be replaced by the pilot benchmark. Archived Q1 metadata
reports 31.19 seconds for 360 forwards, but the new precision and hook workload differs.
No new large-model download or CUDA GPU is expected to be required.

A successful methods case is a valid distinction among declared response predictions,
with uncertainty and scope preserved. A supported profile advances identification of
this intervention's behavior. No outcome here alone establishes a unique native
mechanism, identifies semantic content, or guarantees causality beyond the tested model
and input distribution.
