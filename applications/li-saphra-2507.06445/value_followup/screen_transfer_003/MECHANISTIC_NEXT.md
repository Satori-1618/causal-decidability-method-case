# Later mechanism round: averaged anchors, fresh targets

**Specification for the next development round, not part of screen-transfer
003 and not authorized to run.** Its complete sampling manifest, head, seeds,
sample size, precision calculation and start rule require a separate reviewed
freeze. The calibration rule below is fixed now, before new measurements.

“Two prefix-averaged anchors per balance level” means **one anchor mean at each
position (20 and 28), each formed from two distinct donor prefixes**. Thus two
balances give **four means from eight calibration transfers per family**.
Repeating the same deterministic forward does not create a second prefix.

Keep the recipient, intervention site, native attention coefficient and operator
fixed. Retain donor balances −2/+2, minimum −4, positions 20/28, target `)`,
length 32 and 16 of each bracket. For balance d and position p, define:

`A[d,p] = (margin[d,p,calibration_1] + margin[d,p,calibration_2])/2`.

| Candidate | Prediction for a fresh target at (d,p) |
|---|---|
| Balance-class | `(A[d,20] + A[d,28])/2` |
| Absolute-position | `(A[-2,p] + A[+2,p])/2` |

Both candidates use the same eight calibration observations. Measure **two
additional, distinct target prefixes per (d,p) cell**: eight target transfers,
16 total per family per dtype. No target prefix may occur among calibration
prefixes; inspect causal prefixes at both lengths, not just full-string hashes.
Include historical exclusions across roles and positions. Draw and label all
calibration and target inputs before measuring any of them.

After calibration, save and hash both eight-entry forecast vectors **before
target transfers**. Assess each candidate by maximum absolute error over the
eight targets. Preserve the scientific tolerance **0.10 nat**, definite-hit
limit **0.099** and possible-hit limit **0.101**, with the existing dtype checks.
No averaging target errors to hide a failed cell; report each cell too.

Separability must now be computed from the new forecasts:
`max_target(abs(pred_balance - pred_position)) > 0.202`, with matching labels and
<=0.001-nat discrepancy between dtypes. This is the sufficient condition for
the two sup-norm tolerance regions to be disjoint. The old diagonal-anchor gap
is not interchangeable with it. Fix the family eligibility/start rule and its
precision before running this later round; a favorable screen is not that rule.

The question is whether averaged anchors predict held-out prefixes. Improvement
over single anchors would need a frozen comparator on the same fresh targets;
that comparison is not specified here. This does not validate history, exact
balance rather than its sign, or a unique native mechanism. A failed single-anchor predictor does not
exclude every approximate balance-only account. The earlier 19/30 result stays
failed; neither averaging nor a fitted midpoint is applied retrospectively.
