# Glossary: how the records use a few loaded words

This page explains terms that recur across the applications. It changes no
threshold, rule or record. Each frozen protocol still governs its own case.

## What kind of check a claim offers

| Label | Meaning |
|---|---|
| **N: recompute stored evidence** | Recompute a table or decision from published records, with no model run. |
| **D: design check** | Check prediction tables and ties from public inputs, with no model at all. |
| **R: model reproduction** | Rerun the model experiment itself. No case here currently offers a complete R route. |

Upstream steps that are only logged, such as model forwards, geometry fits or intervention
mechanics, are named as such in each case. They are not presented as recomputable.

## "Adequate": one rule per case, applied to declared parameters

"Adequate" never means "true", "the mechanism" or "the only explanation". It means that a
declared prediction rule met a declared standard on a declared population. Two rule
types are used.

**Loss rule** ([guide](USING_THE_METHOD.md#from-your-own-data)):
- The unit's loss (absolute or signed) is compared with a tolerance.
- A percentile-bootstrap interval decides between adequate, excluded and undecided.
- This rule is used by the general calculator.

**Coverage rule** ([guide](USING_THE_METHOD.md#case-wise-profiles-with-a-population-coverage-requirement)):
- Each unit succeeds only if every restriction of a profile holds within tolerance.
- The share of successes is compared with a required coverage, using exact simultaneous
  bounds.
- A lower bound above the requirement means adequate; an upper bound below it means
  excluded; otherwise the result is undecided.
- A failed technical control makes the run invalid.

The applications use the coverage rule with these frozen parameters:

| Case / round | Population and unit | Per-unit success | Required coverage | Interval decision | Frozen definition |
|---|---|---|---|---|---|
| Makelov round 2 | 192 fresh pairs | both predictions within 25% of the local absolute effect T, in both reciprocal directions and both precisions | 80% | simultaneous Clopper–Pearson, joint coverage ≥ 95% | [result](ROUND2_RESULT.md), [protocol](../applications/makelov-2311.17030/QUERY_ROUTE_PROTOCOL.md) |
| Makelov round 3A | 512 fresh families | within ±0.25 nat in both recipient orders and both precisions | 80% | simultaneous Clopper–Pearson, joint coverage ≥ 97.5% | [result](ROUND3A_CONFIRMATION.md), [protocol](../applications/makelov-2311.17030/DONOR_FACTOR_PROTOCOL.md) |
| Tracr confirmation | 128 fresh token-assignment families | the qualified 2 → 1 → 1 trajectory; scientific tolerance 0.01 (projection-score units) plus a numerical allowance | 95% | one-sided 95% finite-population lower bound | [result](../applications/tracr/RESULTS.md), [freeze](../applications/tracr/CONFIRMATION_FREEZE.json) |
| Mixing Mechanisms confirmation | 300 qualifying fresh cases | concentration within κ·(T_A − T_W) = 0.115673 of the profile's anchor | 80% | simultaneous intervals; unresolved cases count against adequacy and for exclusion | [result](../applications/gur-arieh-2510.06182/CONFIRMATION_RESULT.md), [freeze](../applications/gur-arieh-2510.06182/FREEZE.json) |
| Li/Saphra native follow-up, stages 1 and 2 | sufficiently separated fresh families per stage and task | both new margins on all three family members within 0.25 nat, plus a numerical allowance | lower bound above 80% | simultaneous bounds over the frozen 288-bound family | [result](../applications/li-saphra-2507.06445/native_followup/README.md), [protocol](../applications/li-saphra-2507.06445/native_followup/CONFIRMATION.md) |
| Li/Saphra averaged anchors 004 | 151 separating families | all eight targets within 0.099 nat (definite) / 0.101 nat (possible) | 90% | four one-sided Clopper–Pearson tails at α = .01; familywise ≤ .05 | [result](../applications/li-saphra-2507.06445/value_followup/averaged_anchors_004/RESULT.md) |

Makelov Q1 is a **relative** comparison: B predicts better than A, case by case, by a sign
test. It contains no adequacy decision.

## "Confirmed" and "confirmation"

- "Confirmed" refers to a test whose question, data split, tolerance and decision rule were
  frozen before its outcomes, within the scope that test states.
- It does not mean the mechanism is identified, that the result transfers to other units,
  or that the freeze was externally timestamped.
- Where a freeze is only recorded in local commits and run manifests, the case says so.

## "Independent" in review and audit records

Historical records describe separate review passes as "independent". Unless a record names
an external person, such a pass was one of two things:

- an AI-agent audit separate from the executing agent, or
- a review text supplied by the author without a named reviewer.

These are **not external human peer review**. Records that are already archived or
hash-bound keep their original wording; this entry explains it.
