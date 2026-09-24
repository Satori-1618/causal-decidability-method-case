# Preregistration: which explanations of <intervention effect> remain compatible?

*The ordinary application of the method. It fixes the rivals, the conditions that separate
them, and how the data will decide, before the data exist. (To test the planning
calculator itself, use [CALCULATOR_VALIDATION_TEMPLATE.md](CALCULATOR_VALIDATION_TEMPLATE.md).)*

Freeze this document before any confirmation data are generated, and timestamp it
externally at that moment (a public push, a registry entry, a timestamping service).
Commits bind versions to each other but do not prove when they were made.

## 1. The effect and the question

The intervention, the site and the readout. The published or piloted claim about what
the intervention shows. The question: which of the explanations in §2 does this design
leave compatible?

## 2. The explanations (step 1)

For each candidate, a rule that says what the intervention transfers or changes, and
what it therefore predicts **for every condition in §3, under the intervention as
executed**. Write each prediction as a number or as "equal to the unit's measured value
in condition X". Say explicitly when a prediction is anchored to a measured endpoint; a
candidate anchored to the full patch reproduces the full patch by construction, so that
condition cannot test it.

Declare equivalence groups: candidates that are indistinguishable by construction.
Candidates not written here are neither tested nor excluded.

## 3. The design check, before any outcome (step 2)

The prediction table: candidates × conditions. From it, before any data:

- the signatures (`signatures`): which candidates no outcome of this design can separate;
- the separating conditions: for each pair of candidates, where their predictions differ,
  and by how much;
- what this design cannot decide, stated now.

## 4. The measurement (step 3)

- **Units:** what one independent case is, how units are sampled, and how many. Keep pilot
  and confirmation units disjoint.
- **Conditions and repeats:** how repeated measurements within a unit are aggregated.
- **Execution checks:** how you verify that the intervention did what §2 assumes: hook,
  position, identity controls, the inserted change after dtype conversion. A failed check
  is an invalid result, not evidence about the candidates.
- **Numerics:** dtype, and a precision reference where the numerical floor may bind.

## 5. The confirmation contract (step 4)

Fix these together; each changes what the result means.

- **Question.** Usually two parts. Which candidate predicts the separating conditions
  better on fresh units (comparative, needs no tolerance)? And which candidate reaches a
  declared prediction accuracy (adequacy)?
- **Loss.** `absolute`: the mean absolute error per unit, formed before any averaging over
  repeats or conditions, measures case-by-case prediction quality. `signed`: the mean
  signed error measures systematic deviation, and opposite errors cancel. Continuing a
  pilot, keep the pilot's loss.
- **Scope.** Per condition, or pooled over the tested conditions.
- **Tolerance.** The largest loss that still counts as adequate. Either an absolute value
  in readout units with an independent justification, or a fraction of a reference gap.
  A fractional tolerance is estimated from the same units and must be resampled together
  with the loss. A tolerance is a precision requirement, not a mechanism share: separate
  intervention effects need not add up to a joint effect after nonlinear processing. Do
  not derive it from pilot residuals so that a favoured candidate passes.
- **Uncertainty rule.** How intervals are computed, over which units, and at which family
  level. Say whether coverage is exact or nominal; a percentile bootstrap is nominal, and
  Bonferroni does not make it exact. For the comparison, the exact sign test asks how often
  a candidate wins a unit; the sign-flip test asks about the mean difference and is exact
  only if the differences are symmetric under the null, which equal expected losses do not
  imply. Declare one as primary. For small samples of discrete readouts use exact methods.
- **Statuses.** Per candidate: excluded (interval above the tolerance), adequate (interval
  below it), undecided (straddling). The compatible set is everything not excluded;
  declared groups are decided once. `evaluate` implements exactly this contract.
- **Sample size.** Planned for the primary test, with its assumptions stated. For the sign
  test: an assumed win probability, a target power and an expected tie rate. For a test of
  the mean difference: an assumed difference, the spread of the per-unit differences and a
  target power. For adequacy: the precision needed to place the better candidate's loss on
  one side of the tolerance, using the pilot's spread.

## 6. Outcomes, all of them acceptable

| outcome | reading |
|---|---|
| resolved | one group remains; the other declared candidates are excluded under this rule. Whether the remaining one is also *adequate* is reported separately |
| partially resolved | several groups remain |
| insufficient evidence | every group remains; the design did not discriminate |
| no candidate fits | every declared candidate is excluded; the explanation lies outside the declared set, or the adequacy bound is too strict for all of them |
| invalid | an execution or numerical check failed; no scientific reading |

Report, for every excluded candidate, the conditions that excluded it.

## 7. What this does not establish

That a retained candidate is the mechanism; anything about candidates not declared in §2;
anything outside the sampled units, model, site and readout.

## 8. Deviations

Every deviation is recorded in a dated section appended below, with its reason, before the
confirmation outcome is read. A deviation found after reading it makes the affected results
exploratory.
