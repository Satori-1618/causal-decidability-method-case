# Screen-transfer 003: fixed-cohort prospective test

**Amended plan, awaiting final pre-run review.** See [Amendment 001](AMENDMENT_001.md):
32 becomes 64 per stratum before any new model measurements. The unit is a recipient–donor family.
This round measures anchor separability, not candidate adequacy or saturation.
Previous stops and thresholds remain part of the record.

## 1. Fixed cohort and one question

Does the model-relative screen enrich for anchor-separable families in at least
four of these six heads? Layer/head numbers below are one-based.

| Model | Layer / head | Model heads | Initialization / shuffle seed | Recipient / donor seed |
|---|---|---:|---|---|
| `50h5xlod` | 2 / 1 | 2 | 495 / 161 | 26100301 / 26100302 |
| `8a65u5l6` | 2 / 1 | 2 | 495 / 231 | 26100303 / 26100304 |
| `dsibxabs` | 2 / 4 | 4 | 92 / 161 | 26100305 / 26100306 |
| `qtda56zd` | 2 / 2 | 2 | 92 / 161 | 26100307 / 26100308 |
| `we9o801g` | 2 / 2 | 2 | 105 / 231 | 26100309 / 26100310 |
| `wfwpq5xc` | 2 / 1 | 2 | 495 / 220 | 26100311 / 26100312 |

These were selected for H_W adequacy in the earlier native-intervention study.
They are not untouched or randomly sampled heads. There are three initialization
seeds and five seed pairs. Reference head `a9g0io1r/2/1` is excluded. The claim is
transfer within this fixed cohort, not a prevalence estimate over transformers.

## 2. Screen fixed before new native scores

Let `m = logit(False) - logit(True)` at EOS. Positive means rejection. From each
checkpoint's final LayerNorm and readout, compute in float64:

```
w = W[False] - W[True]       # convert stored weights to float64 BEFORE subtraction
c = dot(w, beta)
u = w * gamma; u = u - mean(u)
R = sqrt(hidden_width) * norm(u)
z = (m - c) / R
accept if z < z_cut; reject otherwise
```

The readout has no bias in these pinned models. Validate that architecture,
finite positive R and token indices False=0/True=1. The final LayerNorm implies
an outer range `[c-R,c+R]`; this is a **model-level bound**, not a head ceiling or
evidence that a recipient is saturated. The screen is signed, not absolute.

Fix `z_cut = (8-c_reference)/R_reference = 0.8315929767566357`, using only the old
reference checkpoint. Static weight algebra gives
`c_reference=0.05487594590959401`, `R_reference=9.554101917837064`.
No reference or target forward is needed to compute these constants. Use the
float64 constants for both dtypes; record all six model cutoffs `c+z_cut*R`
before measuring their native margins. Never fit a ceiling from new outcomes.

## 3. Fixed budget, freshness and selection

- Use the existing development population and sampler: 32 symbols, 16 of each
  bracket, at least one negative prefix, the fixed development hash partition.
  Draw **1,024 recipients with replacement per head**, with the table's seeds.
  Finish that pool even if both quotas fill early. Do not replenish it.
- Draw 128 donor templates independently using the existing `design.generate`:
  balances −2/+2, positions 20/28, target `)`, prefix minimum −4, two distinct
  prefixes per grid cell. Only `neg_20_0` and `pos_28_0` will be measured.
- Preserve prior public/native exclusions and add every full donor/recipient
  string in development 001, all 1,024 screen-002 candidates, and all strings in
  its 64 prepared donor templates (including unmeasured cells). Exclude both
  length-20 and length-28 prefixes of all these full strings from new donors,
  regardless of previous role or targeted position. Original training overlap
  remains unknown. Freeze and hash the complete exclusion list and inputs.
- All six pools use the same fixed historical exclusions; do not condition one
  head's sampling on another's outcomes. Report repeats within/between pools.
  Distinct family draws, not unique strings, are the sampling units.
- Score every candidate in fp32 and fp64. Select the first 64 accepted and first
  64 rejected in input order. Attach donor templates 0–63 to accepted recipients
  and 64–127 to rejected recipients. Save all scores, assignments and hashes
  **before any transfer for that head**.
- If either stratum has fewer than 64, label the head `insufficient_yield`,
  retain its acceptance rate and cost, and perform no transfers for that head.
  No extra candidates, smaller quota, threshold adjustment or substitute head.

## 4. Measurements and technical stops

Use the pinned CPU runtime, one thread, deterministic execution, full fp32/fp64
measurements and the existing value-transfer controls. At the recipient's
highest-native-attention closing bracket (fp64, first tie), transfer only the
chosen value contribution at the EOS query:

`h' = h + a_recipient[j] * (v_donor[k] - v_recipient[j])`.

Keep the same position and native attention coefficient for both anchors.
Preserve native/self-patch, operator-fidelity and off-target node snapshots.
Check the executed tensor, including conversion to the measured dtype.
Derive the audited head width as `64 / model_heads` (16 or 32); do not reuse
the reference model's fixed width in verification.
All native margins must agree within 0.001 nat and give identical screen labels;
all **signed** anchor differences must agree within 0.001 nat and agree on
the strict test `S = 1[abs(anchor_D - anchor_A) > 0.202]`.
These thresholds are not normalized by R. Nonfinite values fail.

A technical failure makes the entire head `technical_failure`; no selective
family deletion or replacement. Preserve partial records; other prespecified
heads may proceed unchanged unless the failure is shared code/source integrity,
which stops the whole batch. Both failures and shortfalls stay in the denominator
of six; neither counts as evidence of no enrichment. No mechanism confirmation
or third-balance experiment starts automatically.

## 5. Estimand, uncertainty and decision

For each valid head, estimate `Delta = p_A - p_R`, the difference in separating
family rates for accepted versus rejected recipients. Donors follow the same
outcome-independent distribution in both groups. There are n=64 per stratum.
Use Clopper–Pearson bounds with **alpha=0.05/24 per one-sided bound** (four per
head). Derive `Delta_CI=[L_A-U_R, U_A-L_R]`. All six intervals have at least 95%
simultaneous coverage by the union bound, without independence between heads.
Do not pool heads, cells, dtypes or repeated forwards as independent families.

**Primary strong-transfer target:** at least four of six lower Delta bounds
strictly exceed **+0.25**. Report every interval and status. Also report positive
lower bounds (>0), and upper bounds below +0.25 or below zero, as weaker/negative
readings of the same intervals. No extra tests or threshold searches.
If fewer than four support the target, say **target not established**, not
"screening does not transfer"; distinguish precision limits from yield/fidelity
failures. Report how many upper bounds exclude +0.25.

`planning.json` enumerates exact hypothetical binomial power conditional on
filled, technically valid strata. At true rates 0.75/0.10, per-head power is
about 81.8%, giving about 92.2% four-of-six power under equal independent rates.
At 0.60/0.10, four-of-six power remains only about 0.4% under those assumptions.
This is a bounded test of **strong** transfer; these effect sizes are assumptions. Cohort powers under equal,
independent heads are explicitly illustrative; dependence-free bounds are also
reported. Neither calculation predicts these selected heads' effects.

## 6. Random-selection yield and costs: descriptive policy estimates

For each complete head, let `q = accepted_pool_count/1024`,
`p_A = accepted_separating/64`, `p_R = rejected_separating/64`.
Estimate unfiltered random-family yield by **`p_random=q*p_A+(1-q)*p_R`**.
Report `p_A-p_random`, not just the larger accepted-versus-rejected contrast.
This is a plug-in estimate from stratified measurements, not a measured random
arm. No formal superiority claim is attached to the cost estimates.

Count sequence-forward equivalents across both dtypes, as well as actual
batched calls and wall time. For the unchanged runtime, `b=2` forwards per
screened candidate and `f=16` additional forwards per two-anchor family:
two cells × (native, donor, self, patch) × two dtypes. This includes recomputed
recipient baselines; no unimplemented caching is credited. The random comparator
uses the same preliminary native pass and unoptimized measurement recipe.

| Quantity | Frozen calculation |
|---|---|
| Fixed-pool policy cost per separating accepted family | `(1024*b + 64*f)/(64*p_A)` |
| Random policy cost per separating family, estimated | `(b+f)/p_random` |
| Idealized streaming policy, **not this schedule** | `(b/q+f)/p_A` |
| Actual validation plan, including rejected families | `1024*b + 128*f` |
| Rejected-arm measurement overhead | `64*f` |

With this recipe the complete validation is **4,096 sequence-forwards per head,
24,576 for six**, before any separately reported retries/additional controls.
No scientific retries are permitted; aborted work is still costed. Count costs
from actual execution, reconcile deviations, and do not silently swap in an
optimized recipe. The fixed policy leaves unused accepted recipients in its
paid-for pool. The idealized streaming estimate assumes repeated sampling under
the same acceptance rate and does not remove the rejected validation overhead
from the actual bill. Include both in the report, never substitute one for the other.

Zero observed separating yield gives infinite plug-in cost; missing strata or
failed gates give **not estimable**, not zero cost. Small denominators make these
estimates unstable. A prospective equal-budget random arm would be needed to
establish savings empirically; this round does not contain one.

## 7. Public freeze and review stop

Push this plan and its source hashes before any new native score or transfer.
Record the remote commit. A later public push cannot retroactively establish
external timing for old runs. No claim of externally witnessed old freezes is made.

The supplied design review is recorded in `REVIEW.md`; its required sample-size
change is this amendment. Final pre-run review must still cover the runner and
inputs without outcome peeking. Record that review decision and publish their
freeze before execution. Changes to this plan need a dated amendment and renewed
review. The separate [anchor revision](MECHANISTIC_NEXT.md) does not run as part
of screen transfer. This handoff stops before execution; a plan-review PASS is not a final execution release.
