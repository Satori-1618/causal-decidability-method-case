# What is and is not identified

## 1. The actual operators

For one head, write its output as `h(A,x) = A(x)V(x)`, followed by the appropriate
slice of the output projection and the rest of the network.

- Native: `hN = A(x)V(x)`.
- Uniform: `hU = U V(x)`, where `U` is uniform over the causally allowed prefix,
  including self. The released code zeros **attention logits**, then applies
  masking and softmax; it does not zero the head's output.
- Mean: `hM = mean_x[A(x)] V(x)`, with native dataset-mean post-softmax attention
  at each position, computed over the 1,000 OOD inputs. Values still depend on the
  current input. This is a mean **attention-pattern** replacement, not mean
  replacement of the whole activation.

Both single-head procedures replace the selected head at **all query positions**.
Our proposed reference must match that scope. The all-head arm changes upstream
computations for later layers as well; its aggregate results cannot stand in for
the same single-head contrast.

Sources: [`model.py`](upstream/utils/minGPT/model.py), lines 59–86;
[`mean_ablation_dyck_single.py`](upstream/heldout/mean_ablation_dyck_single.py),
`attn_forward` and `process`. These files are vendored without modification.

## 2. Two operational rivals and their predictions

Fix one checkpoint, head, input set and readout. Let `aJ` be the measured accuracy
under intervention `J`. With the unmeasured zero-output reference `a0`, define
`R = a0 − aN` and `BJ = aJ − a0`.

| Quantity | Removal-helps rival | Replacement-rescues rival | Observed? |
|---|---|---|---|
| `R`, removal effect | `> 0` | `< 0` | No |
| `BU`, added uniform contribution | `ΔU − R` | `ΔU − R` | No |
| `BM`, added mean contribution | `ΔM − R` | `ΔM − R` | No |
| `ΔU = R + BU` | Either observed positive value is possible | Same | Yes |
| `ΔM = R + BM` | Either observed positive value is possible | Same | Yes |

Neither rival supplies a unique point forecast from its verbal label. Their
possible predictions overlap on all measured outcomes. Pure no-change and
uniform-only accounts are contradicted by the finite-set improvements on the
40 heads under both operators, but those are weaker alternatives already
addressed by the authors' controls.

For the example, the two observed equations are `0.044 = R + BU` and
`0.048 = R + BM`. They fix `BM − BU = 0.004`, **not `R`**. More accurate
estimation of these equations cannot identify the missing scalar. This is an
outcome-level identification statement; it does not claim that released model
weights prevent computing the missing counterfactual.

## 3. Constructive proof, not merely counting unknowns

[`countermodels.py`](countermodels.py) creates 1,000 illustrative negative-label
cases. Native attention alternates equally between `(0.6,0.3,0.1)` and
`(0.1,0.3,0.6)`. Its actual dataset mean is `(0.35,0.3,0.35)`; uniform attention is
`(1/3,1/3,1/3)`. For each case, an additive readout combines the head output with
a rest term and a fixed threshold. All calculations use exact rational arithmetic.

Two different choices of value vectors and rest contribution give:

| Model | Native | Uniform | Actual mean pattern | Zero output |
|---|---:|---:|---:|---:|
| Constructed removal-helps world | 779/1000 | 823/1000 | 827/1000 | **879/1000** |
| Constructed replacement-rescues world | 779/1000 | 823/1000 | 827/1000 | **679/1000** |

Both worlds match the real example's *three aggregate scores*. Their predictions
also agree **with each other** on each synthetic case in those conditions; this
does not claim a match to the real model's unavailable per-case ablation outputs.
They are not reconstructions of the trained Transformer or proofs of biological
plausibility. A concrete attention computation suffices to demonstrate the logical
gap in an inference from those observations alone.

There is a stronger invariance behind this construction. If attention rows sum
to one, replacing every value `v_j` by `v_j + c` shifts the weighted mixture by
`c` for every such attention pattern. Subtracting `c` from the rest contribution
leaves the output identical for native, uniform, mean, or any other normalized
pattern. Zeroing the weighted mixture breaks that equality. In a projected head,
the corresponding offset is `W_O c`; the output projection bias can compensate.

This explains why another normalized pattern alone cannot rule out this witness
pair. It also limits the interpretation of zero ablation: the answer depends on
the fixed parameterization/reference. An additional measurement distinguishes
these operational rivals, not all equivalent descriptions of a computation.

## 4. The smallest separating addition for this pair

Add one **zero-head-output condition** for the same frozen checkpoint and head.
Set only its `A V` slice to zero **before `c_proj`**, at all query positions;
preserve the projection bias, other heads, downstream code, tokens and readout.
Zeroing logits would just reproduce uniform replacement and fail the structural
check again. Zeroing the entire attention module would test a different object.

This adds one intervention condition, not one statistical observation. Evaluate
the same 1,000 OOD inputs and retain ID inputs to inspect collateral effects.
Recompute native, uniform and mean outputs in the same environment as controls;
store per-case binary decisions and both-class margins. The mean pattern must be
captured from the native run and frozen before replacement, as upstream does.

Before making a new empirical claim:

1. Verify no-op hook identity, untouched heads/projection bias and exact zeroing
   after dtype conversion. Confirm all-query scope and EOS readout positions.
2. Compare numerical sensitivity on the final paired estimand; native accuracy
   agreement alone is not sufficient. Persist software, checkpoint and source
   hashes. Do not assume a count-grid step is a floating-point error bound.
3. Specify a scientific relevance band and scope (this fixed set or new inputs)
   before inspecting zero outcomes. Report `R`, `BU`, `BM` and per-case changes.
   Positive `R` supports removal-helps relative to zero; negative `R` plus positive
   replacement effects supports replacement-rescues. An interval crossing the
   decision boundaries remains unresolved; an exact zero has its own outcome.
4. A generalization claim needs fresh, appropriately grouped inputs and its own
   precision plan. No sample size or power is certified by the synthetic 20-pp gap.

The deterministic exact-witness check establishes structural separation; numerical
and statistical resolution of this new neural condition is **not yet assessed**.
This proposal completes the current audit's next-intervention deliverable; it is
not a claim that the zero experiment has been performed.
