# Draft round 2: does restoring Name Mover queries remove the null-read patch effect?

**Execution update:** the revised experiment is complete; see the
[192-pair result](../../docs/ROUND2_RESULT.md). This original four-cell proposal remains
below as design history.

**Status: DESIGN DRAFT — NOT FROZEN — NO MODEL RUN.**

**Review update:** see the [narrow revised plan](../../docs/ROUND2_RESEARCH_PLAN.md).
It identifies a gating countermodel to the four-cell interpretation and proposes a
reverse query transfer plus explicit adequacy rules. The original proposal below is
retained as design history; neither version is a completed experiment.

This is a second application of the method, not a new result from the stored Q1 records.
The [reader-facing walkthrough](../../docs/ITERATIVE_IDENTIFICATION.md) explains the
question. The executable teaching demo is synthetic. Tensor hooks, empirical control
thresholds, sample size and the confirmation manifest remain to be implemented and
validated before any model result can be interpreted.

This is not Q1's deferred adequacy question, historically called **Q2**. Q2 asks whether
the read-source explanation is accurate enough. **Round 2 here asks a new route question.**

## Question and precise scope

On fresh held-out IOI base pairs, does restoring selected recipient queries to their
unpatched values remove or preserve the output effect of the existing null-read,
full-write intervention at GPT-2 Small's MLP 8 post-GELU activation?

The primary intervention is exactly the historical `read_null` arm:

```text
delta_h = ((h_donor − h_recipient) · v_N) v
```

`v` remains the normalized full published direction; `v_N` is its unrenormalized
output-null component. Reuse the original rank-aware decomposition and pinned model
transforms. Do not retrain or select a new direction from route outcomes. A full-patch
reference may be reported diagnostically but must not replace this arm after results.

## Candidate route and implementation contract

Proposed heads, zero-indexed: **(9, 6), (9, 9), (10, 0)**, motivated independently by
Makelov's Name Mover analysis. They are also recorded in the existing `NAME_MOVERS`
constant in `src/resid_mid8_pilot.py`; its residual-stream experiment is a different
experiment and supplies no data for this comparison.

- Cache the original recipient queries at `P−1`, its manifest's last prompt position.
- Restore only those head slices at `blocks.9.attn.hook_q` and
  `blocks.10.attn.hook_q` while the MLP patch runs in the same forward.
- Pin and check TransformerLens hook semantics and model transforms. The targets are
  projected query vectors, after the model's normal query construction, not residual
  vectors or attention outputs. For a future different backend, revalidate the mapping.
- Always take restore values from the **unpatched recipient**. Do not recache a baseline
  after the patch or after another clamp.
- Keys, values, non-target heads, other positions and subsequent computation stay free.
  The selected heads therefore still operate; this blocks their query response only.
- Use absolute `P−1` throughout. Score the two existing answer logits at the same site
  as Q1. No generation, candidate-token position patching or cache-dependent variation.

This deliberately differs from clamping an entire head output. A query-clamp outcome
cannot be reported as testing every possible route through the selected heads.

## The four primary cells and predictions

`Ypc` is the IO-minus-subject logit margin, with patch `p` off/on and query clamp `c`
off/on, on the same directed pair.

```text
T = Y10 − Y00
R = Y11 − Y01
K = T − R
```

| Operational candidate | Prediction for remaining effect | Prediction for clamp-induced change |
|---|---|---|
| Q: removal endpoint | `R = 0` | `K = T` |
| B: preservation endpoint | `R = T` | `K = 0` |

Both are anchored to measured T. Only the separating condition evaluates their route
predictions. Their gap is `|T|`; when T is zero they coincide. Weak T means limited
resolution, not evidence that a route is absent. A sign-reversed or intermediate effect
can fail both endpoint descriptions. Do not reinterpret `K/T` as a share of mechanism.

Neither label asserts a unique graph. Compensation and cancellation can mimic preserved
or removed effects. Conclusions must retain the operational intervention wording.

## Checks required before interpretation

| Threat | Check and consequence |
|---|---|
| Wrong position/head or hook order | Assert absolute indices, separate per-site call counts, exact shapes and unchanged non-target slices; failed checks invalidate the run. |
| The clamp merely changes ordinary computation | `Y01` must reproduce `Y00` within a dtype-justified technical budget; self-restoration is also checked at the inserted tensors. |
| Wrong source or post-cast corruption | Measure the actual query tensor after restoration and actual MLP delta; compare with the cached recipient query and intended delta, including after dtype conversion. |
| Numerically empty restoration | Record each head's query immediately before and after its clamp, together with the numerical floor. At layer 10, use the actual tensor arriving after the layer-9 clamp. An unchanged query is a no-op at that site, not evidence against mediation. |
| No patch effect to explain | Report T per direction and pair, and its distribution. Plan resolution on development data; do not retrospectively select confirmation cases for large T. |
| Generic suppression of answer sensitivity | Preselect a comparison query clamp and a calibrated response-sensitivity intervention outside the targeted query route. Compare their effects under free/clamped conditions. Matching dimensions or norms alone is insufficient; the control choice and acceptable effect range must be frozen after development. Without a working control, restrict interpretation to the measured restoration effect. |
| Deterministic numerical error | Compare the final per-pair losses, R and K against a higher-precision reference on a declared subset; repeated runs at one dtype are insufficient. Propagate numerical uncertainty to the actual decision. |
| Construction forces the answer | Test the four-cell analysis on known removal, preservation, partial and cancelling worlds. Do not replace it with writing into the exact MLP output kernel or restoring the whole residual stream. Those can predetermine the outcome. |

The four-cell contrast removes **additive** clamp effects. It does not by itself establish
specific mediation or a natural causal pathway. Record all four cell values and all
control failures, not only K or a pass flag.

## Proposed inference, to be frozen after the technical pilot

**Independent unit:** one base pair with both reciprocal directions. The existing
ABB/BAB construction changes first-mention position while retaining the correct IO
name; do not describe it as a donor answer switch. All four cells and any sensitivity
conditions remain paired within that unit.

**Primary relative comparison:** for each direction compute

```text
loss_Q = |R|
loss_B = |R − T|
```

Average these losses over the two directions within each base pair. The proposed
primary test is a single exact two-sided sign test of the paired loss difference,
with ties explicitly reported. Independent pairs and the null win-probability
assumption must be stated. Candidate errors depend on the same measured T and R;
never analyze their arms as independent samples.

The current analysis API can consume per-repeat rows with conditions `remaining=R`
and `unclamped=T`, candidates `Q: remaining=0` and
`B: remaining="same_as:unclamped"`, testing only `remaining`. This tests relative
prediction quality. Freeze how loss differences within the numerical uncertainty
budget are handled before using their signs.

For example, `R = 0.4T` favors Q on absolute prediction loss while leaving a substantial
effect. A relative win is therefore not evidence that the remaining effect is negligible.

**Optional adequacy requires a separate justified contract.** Declare scientifically
meaningful tolerances for residual and preservation errors, an uncertainty procedure
with appropriate coverage, and an error family. Do not reuse Q1's missing tolerance,
the toy demo's values, or a nominal bootstrap merely because it is available.
Case-wise absolute-loss adequacy avoids cancellation; a mean signed equivalence test
would answer a different question and must be labeled accordingly.

Report relative comparison and adequacy separately. Allowed conclusions include a
better predictor without adequate fit, both excluded, unresolved, and technically
invalid. Winning more pairs cannot be translated into full mediation.

## What must be settled before a model confirmation

1. Implement and test the combined MLP/query hooks, tensor audit and controls above.
2. Run a separately labeled technical/development pilot; exclude all its prompts from
   confirmation and check overlap with the original 32-pair pilot and 64-pair Q1 set.
3. Fix the scientific target: relative comparison alone, or additional adequacy with
   justified tolerances. Freeze the control choices, inference rule, relevant effect,
   alpha/error family, tie policy and precision gate.
4. Use development variability or win rates to plan n against declared alternatives;
   retain uncertainty in that planning. No current claim that 64 pairs suffice, and the
   calculator ratio is not a power calculation.
5. Freeze code, model/vector hashes, head/query sites, runtime, generator and seeds with
   an external timestamp. Evaluate fresh cases once; retain every declared outcome.

No seed, sample size, tolerance or control threshold is approved by this draft. Its
value is the explicit next distinction and the executable design check. It must not be
presented as a completed or launch-ready confirmatory experiment.
