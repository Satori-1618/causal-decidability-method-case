# External demonstration: address transfer versus answer copying in Tracr

**Status: proposed external-repository adaptation. Source interfaces checked; no Tracr
model compiled, intervention adapter built, or experiment run for this document.**

## The precise deliverable

Produce a recorded sequence in which an initially ambiguous patch effect becomes
diagnostic after a deliberately selected input/intervention condition, followed by
predictions on untouched sequence families. Each step logs the current compatible
candidate set, the next proposed discriminator, its predictions and the measured result.

The first demonstration tests portability and correct interpretation against independently
known computation. It does **not** test superiority of adaptive selection over established
experimental design, and does not identify a native mechanism in a pretrained LLM.

## Why this repository

[Tracr](https://github.com/google-deepmind/tracr) compiles RASP programs into Transformer
weights. Its reversal example and inspection interface provide a third-party computation
with a known intermediate address. This moves beyond our own two-unit circuits. The model
is a compiled Transformer; the reversal task uses non-causal attention, not an
autoregressive language model. The repository is archived and should be pinned rather
than treated as a maintained package.

Pinned revision, checked 2026-09-29:
`9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd`.

The existing [Makelov application](CONFIRMED_CASE.md) already transfers the method to a
pretrained-model repository. Replaying it is useful evidence preservation, but is not a
new independent validation and does not supply its missing absolute adequacy test.

## A concrete example, with predictions fixed before measurement

Compile the existing `make_reverse(rasp.tokens)` program. For four content tokens,
query position 0 normally returns position 3. A donor query at position 1 instead
computes reversal address 2. Positions here are **zero-based and exclude BOS**.

Proposed intervention: copy only the donor's intermediate opposite-index coordinates
at its query position 1 into recipient query position 0, just before the final reversal
attention uses them. All recipient token content and other residual coordinates remain
unchanged. The layer and basis indices must be resolved from the compiled graph; no
guessed layer number or whole-vector patch is permitted.

Two candidate interpretations of the transferred state:

- **Address:** read the recipient token at the transferred position.
- **Answer copy:** return the donor's already determined answer at its source query.

The unchanged native output is a control. Mixed and other explanations remain possible;
an observation matching neither endpoint must not be forced into one of them.

| Stage | Recipient | Donor | Address predicts | Answer copy predicts | Purpose |
|---|---|---|---|---|---|
| Shared outcome | `[A,B,C,D]` | `[W,X,C,Z]` | `C` | `C` | A successful patch alone leaves both explanations. |
| Separate the rivals | `[A,B,C,D]` | `[W,X,Y,Z]` | `C` | `Y` | Change donor content while preserving transferred address 2. |
| Check a new prediction | `[A,B,U,D]` | `[W,X,Y,Z]` | `U` | `Y` | Change recipient content while preserving the same address. |

The recipient's native position-0 answer is `D` in all three rows. These are **symbolic
predictions, not measured results**. The third displayed row is an illustration of the
future check; actual validation must use predeclared, previously unmeasured families.

The site is chosen using known compiler structure. This is deliberately a controlled
interpretability benchmark, not discovery of an unknown address representation. The
analysis receives anonymized site IDs and candidate prediction tables, not the true
variable label or reference-output answer key. The operator who builds the adapter knows
the graph; this access split must not be described as human double blindness.

## Existing interfaces and the code still needed

| Verified upstream entry point | Use | Work remaining |
|---|---|---|
| [`compiler/lib.py`, `make_reverse`](https://github.com/google-deepmind/tracr/blob/9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd/tracr/compiler/lib.py#L40) | Defines length, opposite index, selector and aggregation | Compile with one fixed vocabulary and sufficient maximum sequence length. |
| [`compiler/assemble.py`, `AssembledTransformerModel`](https://github.com/google-deepmind/tracr/blob/9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd/tracr/compiler/assemble.py) | Parameters, model config, residual labels, compiled forward, output decoding | Resolve the target basis and layer; export opaque site IDs. |
| [`transformer/model.py`, `Transformer.__call__`](https://github.com/google-deepmind/tracr/blob/9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd/tracr/transformer/model.py) | Reference forward exposes residuals, layer outputs and attention logits | Build an instrumented forward. There is no ready-made patch callback. |
| Local `compatible_set` | Shared candidate-retention rule | Supply calibrated readout-specific radii; do not import the toy Gaussian error model. |

## Efficient execution sequence

1. **Native qualification.** Use an isolated dependency environment, pin upstream and
   record JAX/Haiku versions. Compile reversal once and compare native decoded outputs
   with the symbolic RASP reference on a small declared development set. Compare full
   output-score vectors between the upstream forward and instrumented no-op forward;
   RASP itself supplies no logits. No hypothesis classification before this passes.
2. **Intervention qualification.** Require unmodified-forward agreement, self-patch
   identity, actual post-cast coordinate replacement, no unintended coordinate/position
   changes, correct BOS offset and numerical sensitivity of the final readout. Calibrate
   dtype error empirically. Verify the copied address at the site independently of the
   final answer. This confirms the intended operator, not discovery of its semantics.
3. **Freeze the iteration contract.** Define candidate equations, allowed conditions,
   selection rule, per-case cost, maximum looks and stop states. Use independent sequence
   families as units. Choose the adequacy threshold, sample size and power/error plan
   from the development stage before looking at fresh outcomes. Statistical intervals
   must cover the whole planned sequential use, not just each isolated step.
4. **Run and log the narrowing.** Record the shared condition, choose a disagreement
   condition from the allowed menu using only retained predictions, and update the set.
   If both remain, collect only the next permitted discriminator; if neither fits, report
   misspecification; if fidelity fails, stop the mechanistic interpretation. Separate
   development-based candidate revision from the untouched confirmation.
5. **Fresh predictive check.** Evaluate the final explanation on new predeclared families.
   Report adequacy and alternatives still compatible, not just a lower relative loss.
   A later transfer test uses another compiled program, not just new reversal tokens.

Do not promise a particular sample size or runtime before compilation, hook calibration
and measured throughput. The first goal is one qualified, reproducible narrowing path;
the comparison of adaptive policies comes afterwards and needs matched budgets and a
strong fixed design. A full-patch-only comparator is insufficient for that stronger claim.

## Required data and outputs

One record per **sequence family × intervention condition**, retaining:

- upstream/model/config and adapter hashes, dependency versions, dtype and precision audit;
- donor/recipient tokens, source and target positions including the BOS mapping;
- selected coordinates, cached native tensors, inserted tensors and fidelity checks;
- complete output-score vector and decoded token, both native and intervened;
- candidate predictions, uncertainty/relevance rules, compatible set before and after;
- why the next condition was selected, cumulative forward budget and stop reason;
- separate compiler ground-truth labels for scoring after decisions are sealed.

Deliver a single per-family trajectory plot plus all trajectories and the fresh-case
summary. A persuasive result says: **the ambiguous effect was followed by a condition
where the rivals disagreed; the surviving explanation predicted independent outcomes.**

## Connection to causal abstraction

The proposed address hypothesis is a high-level computation with a proposed alignment
to an internal subspace and an intervention map. Its counterfactual predictions can be
tested against the answer-copy account. This combines the faithfulness question of
[Geiger et al.](https://arxiv.org/abs/2301.04709v4) with explicit rival discrimination;
it does not establish a stronger universal notion of causality.
