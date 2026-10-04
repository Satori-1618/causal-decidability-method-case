# Amendment 002 — cell prediction and prefix variation

**2026-10-04. Analysis only; execution remains blocked for final independent
review.** Amends public freeze `b7b577c5a9dbca8674626217efc39bc6462d03e6`.
The supplied external review reports integrity **PASS**, with release withheld
pending this amendment. That is not an execution authorization or approval of
this subsequently amended code.

## What changes, and why

Add a calibration-only cell predictor and a descriptive decomposition of the
already planned margins. Averaging benefit and prefix dependence can coexist;
they are not exhaustive, mutually exclusive mechanisms. These additions measure
predictive error and variation below the existing 0.202-nat witness threshold,
without turning a small effect into an adequacy or mechanism claim.

**Unchanged:** head, model, prepared input bytes, seeds, sample size, screening,
target-start rule, interventions, tolerances, primary candidates, four adequacy
bounds, paired test, power calculation, forward budget and future planning gate.
The original analysis and original test files are unchanged. New synthetic
checks test only the add-on and its receipt integration. No neural measurements
have been made for this round; no further model runs are added.

## 1. A cell-specific forecast, without target leakage

For each dtype and family, use exactly the existing two calibration prefixes:

```
A[d,p] = (cal[d,p,0] + cal[d,p,1]) / 2
C_cell(target[d,p,r]) = A[d,p]             # both r=0 and r=1
```

Save all eight predictions per dtype in `descriptive_forecasts.jsonl`, before
any target. Bind its hash in the existing `forecast_receipt.json`. Do not average
targets into this predictor. `C_cell` is absent from the separation/start rule,
the adequacy error family and the future-confirmation planning gate.

Report every target's signed error and each family's maximum absolute error.
Use the existing 0.099/0.101 definite/possible hit conventions **descriptively**.
If the signed-error dtype discrepancy exceeds 0.001 nat, report this supplemental
classification as numerically unresolved; do not add a primary stopping rule.
Report B_avg/C_cell joint fit patterns on all 256 families and, separately, on
the frozen separating subset. No additional confidence intervals or tests.

## 2. Descriptive variation within each family

Only after targets, combine the two calibration and two target margins at each
of four `(balance, position)` cells. Thus each family supplies 16 margins,
`y[d,p,r]`, with four prefixes per cell. Calculate independently in both dtypes:

- all four cell means, sample SDs (denominator 3), and ranges;
- calibration and target cell means and their differences;
- balance contrasts at each position and the marginal balance contrast
  `Delta_B = mean(+2) - mean(-2)`;
- position contrasts at each balance and the marginal position contrast
  `Delta_P = mean(28) - mean(20)`;
- the interaction `Delta_BP = (mu[+2,28]-mu[+2,20])
  - (mu[-2,28]-mu[-2,20])`.

The interaction is necessary: marginal balance and position contrasts can both
cancel while cell differences remain. For the balanced grid, the exact
descriptive sum-of-squares decomposition is:

```
SS_balance     = 4 * Delta_B**2
SS_position    = 4 * Delta_P**2
SS_interaction = Delta_BP**2
SS_within      = sum((y[d,p,r] - mu[d,p])**2)
SS_total       = sum((y[d,p,r] - grand_mean)**2)
SS_total       = SS_balance + SS_position + SS_interaction + SS_within
pooled_within_SD = sqrt(SS_within / 12)
```

Report SDs, ranges and signed contrasts in **nats**, with magnitude/0.10 ratios.
Report sums of squares in **nat squared**, not directly against a nat tolerance.
Do not interpret these components as causal proportions or conduct an ANOVA
significance test. These are deterministic differences between selected donor
prefixes, not estimates of measurement noise. The calibration/target role hash
also separates input populations; pooled within-cell spread includes any role
mean difference. Preserve those means rather than concealing that distinction.

The cohort summary gives median, quartiles, IQR and range of family summaries,
with linear empirical quantiles at index `(n-1)*q`; families retain equal weight.
Keep all-family and separating-subset summaries distinct. Preserve both dtype
tables and report their discrepancies. Supplemental numerical uncertainty is
not evidence of an absent effect and cannot change a primary result.

## Reading the outcomes

| Pattern | Permitted reading | Not established |
|---|---|---|
| B_avg misses, C_cell fits | Retaining balance-position cells improves this prediction | Position alone is the causal explanation |
| Both miss | Neither fixed calibration rule predicts these targets adequately at this tolerance | A named history feature, or impossibility of every common cell prediction |
| Nonzero within-cell spread | The measured patched margins vary among prefixes sharing these coarse properties | That variation is large enough to defeat every common prediction |
| Existing precision-resolved target-pair gap >0.202 | No common prediction fits those two targets within 0.101 | Which prefix feature is responsible, or native use of that feature |

For example, calibration margins -0.10 and target margins +0.10 make both
calibration-based predictors miss, yet the common value 0 fits every observation
within 0.101. Thus “both fail, therefore history explains the failure” is too
strong. Conversely, failure to cross 0.202 does not establish invariance.

These outputs concern patched **output margins**, not a variance partition of
the head's internal representation. Balance, sign and distance from the fixed
minimum remain aliased, as in the original protocol.

## Historical check and review limits

The reviewer estimated approximately 62% start yield, 80% B_avg fits and 81%
cell fits from the public 32 accepted families of `screen_002`. These are old
same-head data, not measurements for round 004. The reconstructed averaged
separation rule selects 20/32 (62.5%). However, those records contain only two
prefixes per cell; they cannot directly instantiate two calibration **plus two
independent target** prefixes. Different reuse or leave-one-out choices answer
different questions. The 80%/81% figures are not adopted as validated forecasts
for this run. No threshold, sample size or start rule is tuned to them.

The [recount](review/recount.json) makes the discrepancy explicit:

| Old-data calculation (0.099 definite limit) | All 32 | 20 selected by reused averaged forecasts |
|---|---:|---:|
| Balance means fitted and scored on the same observations | 30/32 | 18/20 |
| Own-cell means fitted and scored on the same observations | 32/32 | 20/20 |
| Four single-prefix anchors predicting the other replicas | 28/32 | 16/20 |
| Same operation with replica roles reversed | 30/32 | 18/20 |
| One prefix predicting its same-cell counterpart | 26/32 | 14/20 |

Thus the observed 81.25% is reproducible for a **single-prefix predictor**, not
for the reused own-cell mean (100% in-sample). The largest observed same-cell
gap is 0.145519 nat; mean two-prefix sample SD is about 0.0353/0.0296 at balance
-2 and 0.000155/0.000105 at +2, with a maximum SD of 0.102898. These heterogeneous
old-data quantities are not a common noise parameter for the new round.
Reproduce without model loading: `python3 -B -S review/recount.py`.

The review therefore motivates richer descriptive reporting; it does not
establish that this round must fail or that its outcome is guaranteed to
identify a mechanism. Original inferential decisions remain reportable even
when the supplemental patterns are unresolved.

## Review boundary

`descriptives.py` implements the additions. The runner writes separately hashed
pre-target forecasts and post-target `descriptive_cases.jsonl` and
`descriptive_summary.json`. Existing primary outputs retain their definitions.
If the original start gate fails, there are no targets and no full-grid
decomposition; preserve the calibration forecasts and the declared stop.

Push this amendment, review its exact public commit and receipt ordering, then
obtain a separate explicit execution release. **Stop here before measurement.**
