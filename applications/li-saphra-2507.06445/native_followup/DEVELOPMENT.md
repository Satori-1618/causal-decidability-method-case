# Development amendment: separate routing from attention mass

Status: exploratory model execution, authorized after the completed retrospective
audit. This does not change `../PROTOCOL.md` or turn inspected public data into a
holdout. No new confirmation or method-superiority claim is made by this stage.

## Concrete objective

Find a computationally specific, executable distinction on a released Dyck
checkpoint that the native/uniform/mean outcome table cannot settle. A useful
result must narrow an explanation of the *actual* model, not only distinguish
two synthetic reparameterizations. If development supports a discriminating
test, freeze its predictions, independent sampling units, practical tolerance,
controls and decision before running fresh cases.

Fixed development checkpoint: `1aez5d6p`, checkpoint 5, layer 2 head 2
(one-based). This was selected in the preceding retrospective audit as the
lexicographically first head improving by over one percentage point under both
published replacements. It is not an independently selected model. Upstream
revision remains `dedb269eca6a9cf0bc409792003e6f815f2dce34`.

## Source-derived close rival

The upstream sign-matching classification excludes BOS and EOS. Uniform
attention replacement changes both (i) routing among bracket tokens and (ii)
the mass on BOS, EOS and the bracket group. Classifying the bracket routing
does not establish which of these changes explains a replacement effect.

At the last-layer EOS query write its head output as

`h(g,r) = b*v_BOS + e*v_EOS + alpha*sum_j r_j*v_j`,

where `g=(b,e,alpha)`, `alpha=1-b-e`, and bracket routing `r` sums to one.
Cross native/uniform `g` and native/uniform `r`, with current-input values:

| Condition | Group masses | Within-bracket routing |
|---|---|---|
| Native | Native | Native |
| Routing-only | Native | Uniform |
| Gate-only | Uniform | Native |
| Full uniform | Uniform | Uniform |

Pure-routing forecast: routing-only matches full uniform and gate-only matches
native. Pure-gating forecast: gate-only matches full uniform and routing-only
matches native. These are deliberately strong, falsifiable *local sufficiency*
hypotheses, not exhaustive native algorithms. Joint dependence, opposite effects,
and neither adequate remain possible. Similar signs alone do not establish a
forecast. No numeric adequacy threshold has yet been calibrated or frozen.

## Sequence and boundaries

1. Download separately hash-locked weights, missing dependency and published
   inputs. Reproduce the published native/uniform/mean counts; inspect no fresh
   generated cases at this step.
2. Validate native identity, exact sites, causal mask, normalized replacement,
   preservation of other heads and projected bias. At this last layer, only
   the EOS query can affect its readout; confirm full-query/EOS-only equivalence.
3. Run the four development cells on the published inputs in float64, comparing
   float32 on the same cells. Report individual-input margins and accuracy;
   do not interpret the all-negative OOD accuracy as task-general improvement.
4. Assess whether a close constant or length-only rejection-margin shift can
   account for gains. A later confirmation must include both valid and invalid
   strings and separate task-specific sensitivity from an acceptance threshold.
5. If the distinction is informative, freeze a bounded fresh-input test and
   obtain adversarial review before running it. Preserve a failed candidate;
   do not adjust its threshold on confirmation outcomes.

This can establish prospective narrowing on a real checkpoint. It does not by
itself establish superiority over competent standard interpretability practice.
That stronger claim requires a fair equal-cost comparison, including rival
construction and ordinary controls, on more than one selected model–question.

Raw published inputs are cached rather than committed; the pinned fetcher and
asset hashes are required for reproduction. CPU execution is sufficient for
this small model. No model training, remote publication or author contact.
