# Tracr: a qualified external demonstration

Status: development contract, before fresh confirmation. The inherited original
proposal is preserved in ORIGINAL_PLAN.md. This application implements the narrow
known-circuit validation; it does not test adaptive-design superiority.

## Scope and predictions

Use upstream Tracr revision 9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd, its reverse
program, vocabulary 0..11, length four, noncausal attention, mlp_exactness=100.
Site A is the complete compiler-identified opposite-index subspace immediately
before its sole consuming attention layer. Source query is content position 1;
target is content position 0. The model positions include one leading BOS token.
Site B, final reverse-output coordinates immediately before unembedding, is a
deliberate answer-copy positive control, evaluated in development only.

Candidate `address` predicts the recipient token at donor address 2; `donor_answer`
predicts the donor token at address 2. Both predict the complete one-hot output
vector over vocabulary 0..11. The native unchanged answer, at recipient index 3,
is a separate null control and must differ from both discriminator predictions.
The declared candidates are not exhaustive: both, neither, and invalid measurement
remain legitimate outcomes. Site selection uses known compiler structure. The
operator knows the implementation; opaque site IDs do not make this double blind.

## Sequence and measurements

Within a family, first measure a shared-outcome condition. If both candidates remain,
choose one of two allowed equal-cost donor-content changes using their predictions
only (maximal number of separated pairs, then case ID). The third condition changes
recipient content at the transferred address, preserving donor and address. It is a
new within-family prediction, not an additional independent family. In confirmation,
the entire family is freshly drawn after development and the rule is unchanged.

Record full native/patched score vectors, decoded outputs, all target-site before/
after tensors, donor values and target masks. Required controls in each measured
condition and dtype: native symbolic reversal; no-op versus upstream; self-patch;
single site execution; exact post-cast replacement; unchanged tensor complement;
no non-target output-position changes beyond technical tolerance; finite values.
Run both fp32 and fp64 arithmetic from the same compiled parameter source, explicitly
cast to each dtype, including parameter rounding. The residual stack uses the named
dtype; upstream unembedding projects into float64 in both modes. fp64 is a reference,
not exact truth. Projection scores are not nats,
probabilities, or language-model logits; they are the compiled unembedding scores.

## Tolerances, population and stopping

The scientific tolerance is fixed at 0.01 maximum absolute coordinate error from
the ideal one-hot prediction. Distinct endpoints have L-infinity separation 1.
Numerical allowance is frozen from development as max(64*eps32, 4*largest paired
fp32/fp64 score deviation), capped at 0.01; exceeding the cap blocks confirmation.
Technical native/no-op and self-patch tolerance is 64*eps(dtype)*max(1, abs(reference)).
The actual replacement and unchanged complement must be exact, not just close.
The total candidate radius is scientific tolerance plus numerical allowance and
must be strictly below 0.5. Development calibration is not a uniform mathematical
error bound. The paired precision check is repeated on every confirmation case;
a failure invalidates its dependent claim rather than becoming rival exclusion.

The generator samples without replacement from a known finite population of token
assignments (12P10 / 2, unordered discriminator options), excluding development
families by content signature. Confirmation uses 128 whole families, not conditions
or vocabulary coordinates as units. A single compound family success requires the
qualified 2 -> 1 -> 1 trajectory and the final fresh prediction. Exact finite-
population bounds, the required count and power at success rate 0.995 are frozen
before outcomes. The declared adequate population success fraction is 0.95;
one-sided alpha=0.05. No per-stage statistical testing or optional sample extension.
Adapter/site-B controls do not create additional population claims.

Preserve every failure and record it. Technical/precision failure blocks the
mechanistic interpretation; no replacement cases. A changed candidate, rule, or
site starts a new development round. Confirmation source and input hashes must
match the freeze. The report must say known-circuit validation, not native LLM
mechanism discovery or generally superior experimental design.

## Development history before this contract

One technical smoke input used recipient [0,1,2,3], donor [4,5,6,7]. Native decoding
reversed the sequence; no-op scores agreed exactly; site A returned 2 and site B
returned 6 in both dtypes. This disclosed smoke is development, not confirmation.
No production confirmation family or outcome was generated or examined.
