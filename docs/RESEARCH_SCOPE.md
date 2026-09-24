# Research priority: causal decidability

Recorded from Felix's instruction on 21 September 2026.

**Primary goal:** explain and demonstrate a usable method for checking which causal
explanations a planned intervention experiment can distinguish. A precise intervention
effect is not sufficient to identify its explanation. The unit of reasoning is the
candidate explanation and its prediction under the executed intervention.

Current order of work:

1. **Make the method clear.** Maintain one short [method description](../README.md), a
   [worked example](WORKED_EXAMPLE.md) and an honest [evidence map](EVIDENCE_MAP.md).
   Keep structural distinguishability separate from finite-sample resolution.
2. **Apply it to Makelov and Shi.** Use the existing code, data and recent forks first.
   For each application, state the exact claim, candidates, interventions, separating
   predictions, observed compatible set and remaining limits. An implementation audit or
   power calculation alone is not a complete application of the explanation check.
3. **Only then decide what further experiment is necessary.** New gradable-adjective
   model runs and additional calculator-calibration studies are possible later work, not
   prerequisites for writing the method or the current primary deliverable.

The philosophical motivation concerns how context affects the interpretation assigned to
an internal state. It motivates rival construction; it does not establish a neural
decomposition by itself. In particular, “character”, “content”, “context” and “verdict”
need operational definitions and intervention predictions before they become experimental
candidates.

The current ratio calculator is one provisional planning tool. Its accuracy, numerical
model and future replacement do not define the structural explanation-discrimination
problem. Do not turn improving that calculator into the entire research objective.
