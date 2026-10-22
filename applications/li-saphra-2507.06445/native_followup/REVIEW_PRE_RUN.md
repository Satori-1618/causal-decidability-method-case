# Pre-run review of the native factorial confirmation

Review type: separate Codex agent, not an external or human review. Read-only
inspection of the protocol, preparation, operator and runner; no confirmation
model forward was executed by the reviewer. This report records the findings
before corrections rather than silently treating them as already resolved.

## Required before execution

1. **Apply the numerical allowance to the prediction error.** The inspected
   `confirm.py` compared each arm's float32/float64 margin discrepancy with
   0.001. The protocol instead uses 0.001 as an allowance on an arm-minus-endpoint
   prediction error. These differ: discrepancies of +0.0009 and -0.0009 in the
   two margins create a 0.0018 error discrepancy. Check every actual signed
   prediction error across precisions (and any derived family error), or use
   an explicitly conservative arm-level bound of half the allowance. Preserve
   arm-level diagnostics separately.
2. **Instrument the declared operator controls.** The constructors implement
   the stated factors, but the inspected runner did not explicitly check W's
   symbol-mass preservation, T's conditional-weight preservation, stage-2
   `both` equality with R, or injected attention equality with the requested
   tensor. Add these checks on the per-task, per-dtype tensors. Unit tests of
   the constructor supplement these checks; they do not replace them.

## What the review supports

- The two stages are genuine nested interventions on a fixed network, with
  separate hybrid predictions recorded before their corresponding outcomes.
  Endpoint-conditioned forecasts are described accurately; they are not
  forecasts before all model evaluation.
- Stage 2 preserves BOS/EOS and total bracket mass. W preserves open/close
  totals while changing attention within each symbol; T does the reverse.
  Neither factor alone specifies what contextualized values mean. The text
  correctly avoids equating W with a prefix-depth algorithm.
- Gate-only multiplies all bracket attention weights by the same positive
  scalar. Therefore the upstream ordering-based sign-pattern classification
  is mathematically invariant. The runner additionally checks the actual
  dtype-level classification, where rounding-induced ties could matter.
- Restricting the experiment to a last-layer EOS query is justified: only
  positionwise operations follow. EOS-only/full-query uniform agreement is
  still checked rather than presumed numerically exact.
- Uniform string sampling followed by acceptance probability 1/orbit-size
  removes the unequal orbit-size weighting. Rejection of previously selected
  or ineligible orbits then gives sampling without replacement from the
  declared finite orbit population. The hash chooses a fixed representative
  of each condition within each orbit; it is not an additional random unit.
- Independently checked the prepared input file: 512 distinct rotation
  orbits, 1,536 cases, each orbit's members truly are rotations of one another,
  correct family hashes and condition membership, and no overlap with any
  eligible orbit in either published evaluation file. The manifest contains
  one development checkpoint and 35 transfer tasks.
- Conditional on the eligible-family count, eligible families form a uniform
  subset of the eligible finite population. A finite-population Chernoff/KL
  bound and a union bound across tasks do not require independent models.
  Shared training seeds must still prevent interpreting the cohort as iid
  trained-model replications, as the protocol already states.

## Scope that must remain visible in the result

All fresh cases have length 32 and equal open/close counts. The result therefore
does not confirm the unequal-count ID regime, where development showed a
different intervention response. Report the actual eligible fraction, both
unsuccessful and unresolved hypotheses, and the valid/invalid strata. An
adequate hypothesis is a restricted approximation on this intervention/input
population; it is not a unique native algorithm or a measured advantage over
a competent researcher applying the same factorial design.

The all-family maximum-error requirement is meaningful but demanding. The
minimum of 128 eligible families does not guarantee useful power; the protocol
already exposes that limitation. Do not relax the criterion after outcomes.

**Verdict before corrections:** the design is suitable for the bounded claim;
fix the two implementation/protocol mismatches above before execution.

## Resolution before the freeze commit

The runner now checks all eight signed arm-minus-endpoint precision differences,
as well as raw margins. Delivered W/T tensors, special-token preservation,
symbol masses and conditional ratios are checked in each run; the stage-2 joint
endpoint is executed and compared with R. A regression test catches opposite
0.0007 margin deviations producing a 0.0014 prediction-error discrepancy.
The complete followup suite passed 24 tests before confirmation execution.
The statistical rule was sharpened to binary KL/Chernoff bounds before outcomes,
with the unchanged scientific tolerance and explicit power limitations.
