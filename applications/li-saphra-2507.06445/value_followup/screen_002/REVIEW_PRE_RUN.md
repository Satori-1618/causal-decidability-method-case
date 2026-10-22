# Independent pre-run design review

**Verdict: ready for the declared bounded run; no scientific blocker found.**
Reviewed the current protocol, preparation, runner, analyzer, planning table and
prepared inputs before the new run. This is a separate agent's code/data review,
not independent human review or external preregistration. No model inference or
new model outcomes were used in this review.

## What was checked

- **Sampling and unit:** the candidate generator draws with replacement from a
  fixed, outcome-independent recipient population. Taking the first 32 cases in
  each native-margin stratum gives samples from the two conditional populations;
  a missing quota causes a stop. Independently generated donor templates have the
  same distribution in both strata. The unit is one recipient–donor family, not
  a cell, token, dtype or any of the 1,024 screened recipients.
- **Prospective ordering:** native scores, all classifications and the selected
  manifest are saved and hashed before anchor transfers. The accepted families'
  two-anchor forecasts are saved before their six target transfers. Neither
  eligibility nor candidate fit selects a recipient or donor template.
- **Primary inference:** the code's binomial inversion agrees with independent
  SciPy beta quantiles for every count 0…32; maximum discrepancy was
  `3.33e-16`. Four tails at 0.0125 give the stated simultaneous rectangle and
  its derived difference interval. The zero and 0.25 readings concern the same
  interval. The hypothetical power table agrees with the reviewed calculation.
- **Unchanged gates:** the primary strict anchor-gap rule is `>0.202`; both
  groups receive both anchors and precision checks. The accepted-only secondary
  analysis uses the maximum error over all six targets, definite hits `<=0.099`,
  at least 16 eligible families and at least 90% definite hits. Its status remains
  developmental. Technical failure blocks the run; cases are not replaced.
- **Prepared inputs:** regenerating both inputs with the declared seeds and
  exclusions reproduced the stored data exactly. There are 1,024 distinct
  recipient strings and 64 donor templates: 512 donor draws, 495 distinct target
  prefixes, maximum reuse two. Reuse is allowed by the sampling design and does
  not create additional independent units.
- **Freshness:** all prior development recipient and donor full strings are
  excluded, and their position-20 and position-28 prefixes are banned as donor
  targets across roles and original target positions. New recipients need not
  have every prefix unseen for the stated fresh-full-case claim; this is not a
  claim that every internal state is new. Training overlap remains unknown.

## Limits that must remain in the result

The screen and head were chosen using earlier development data. This round can
validate the frozen screen's local enrichment on fresh families, not general
method superiority, a saturation mechanism or a unique balance representation.
Separation does not guarantee either account's adequacy. Rejected families have
no measured six-cell fit. The 25-point contrast is versus the rejected stratum,
not versus unfiltered sampling; net computational savings are not established.

At true rates 0.75/0.10, the substantial-enrichment decision has only 56.44%
planning power despite high power for positive enrichment. Failure of the
stronger target must therefore retain its declared interpretation. Any future
revision requires a new protocol; it cannot repair development 001 or this run.
