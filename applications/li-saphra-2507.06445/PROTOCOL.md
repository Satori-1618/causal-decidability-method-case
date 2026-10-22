# Li, Saphra et al.: what does an ablation improvement identify?

Status: **retrospective, exploratory audit of public artifacts**. No fresh
confirmation, author contact, publication, model training, or model execution is
part of this initial stage. Parent method repository: `53b0366`.

## Objective and scope

Audit the Dyck-1 ablation finding in §4.3 and Appendix J of
[Can Interpretation Predict Behavior on Unseen Data? v4](https://arxiv.org/html/2507.06445v4):
ablating sign-matching attention can improve OOD accuracy. The authors already
distinguish predictive association from causal support and already include
uniform, dataset-mean, and single-head interventions. Those facts are not our
contribution.

Use upstream revision `dedb269eca6a9cf0bc409792003e6f815f2dce34`, checkpoint 5,
all released 2- and 3-layer models and every actual head. Retain both/neither head
types as controls and report model and head counts separately. One-layer models
are retained in integrity checks, outside the primary paper comparison.

## Initial competing explanations

1. **Removal of a harmful native contribution:** replacement helps by removing
   a contribution of the original attention computation that favors accepting
   the OOD strings.
2. **Contribution introduced by the replacement:** replacement helps because
   the new attention-weighted value mixture favors rejecting the OOD strings;
   an improvement alone need not assign a harmful role to the native output.

These are initial verbal rivals, not yet point predictions. The code audit must
identify which parts the operators remove and add and operationalize any
comparison. Do not invent numeric forecasts from those labels or call an
under-specified comparison a successful identification. Controls and countermodels
must distinguish the native computation from the replacement's contribution.

## Analysis rules

- First check producer code, units, data joins, missing heads, source baseline
  consistency, and the meaning of the reported metric.
- Compare uniform and mean interventions on the **same head**, without choosing
  the head with the largest observed effect. Report any post-hoc illustrative
  example as such and preserve the complete cohort.
- Accuracy is a finite-set statistic over the same 1,000 OOD strings. Heads,
  interventions, seeds reused across configurations, and repeated checks do not
  create independent population samples. Primary summaries are descriptive.
- Keep the 0.8 upstream head-type cutoff. A 0.01 accuracy-change band, if used,
  is the upstream descriptive convention (10/1,000 cases), not a statistical or
  numerical confidence bound. Report continuous differences and zero-band
  sensitivity too.
- Do not infer an internal mechanism from output accuracy, make native-use or
  mediation claims from intervention success, or interpret a null contrast as
  equivalence without a valid resolution assessment.
- Public outcomes are already visible. Any further operationalization or scope
  adjustment remains retrospective and is documented; no preregistration claim.

## Deliverables and completion

A short English README with one concrete example, a candidate/intervention
matrix, reproducible source-locked analysis and machine-readable results,
appropriate tests and independent read-only verification. State exactly what the
existing interventions exclude, what remains observationally equivalent or
insufficiently measured, and a targeted extra intervention if needed. An
informative demonstrated ambiguity is allowed; a forced winner is not.
