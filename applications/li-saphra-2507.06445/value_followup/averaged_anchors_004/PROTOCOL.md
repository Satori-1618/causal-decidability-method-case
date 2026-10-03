# Averaged-anchor mechanism round 004 — pre-run freeze

**Prospective development, one fixed head, no execution authorization.** This
fills in the sampling, precision and start rules left open in
[MECHANISTIC_NEXT.md](../screen_transfer_003/MECHANISTIC_NEXT.md). Its calibration
formulas and scientific tolerance are retained. Earlier results remain unchanged.
The question is predictive sufficiency of the transferred value on this grid,
not unique semantic content or native use of a representation.

## 1. Scope and fixed population

Use checkpoint `a9g0io1r`, layer 2/head 1, one-based. It is a **four-head model**,
hidden width 64, head width 16. Pin checkpoint/upstream hashes in `plan.json` and
inherited source locks. CPU, one thread, deterministic PyTorch; run fp32 and fp64.
The head has been selected and repeatedly studied. No untouched-head claim.

Draw exactly **2,048 recipients**, seed **26100341**, and **256 independent donor
templates**, seed **26100342**. Strings have length 32, sixteen of each bracket.
Recipients have at least one negative running balance. Sample with replacement
between families; do not delete repeated strings or count tokens as independent
observations. Report reuse. The unit is the recipient–donor family draw.

Use the existing `confirmation` hash partition for recipient full strings and
donor measured prefixes: `int(SHA256('value-prefix-v1:'+text),16)%5 != 0`.
This is a declared new input population, not permission to open earlier sealed
outcome files and not a claim that this adaptive research line is confirmed.
The old development positive-balance/position-20 pool has only **nine** unused
prefixes under the full historical bans; this input-only scarcity motivated the
choice before measurements. No model outcome is used to select the partition.

## 2. Freshness, roles and manifest

Exclude every previously prepared/opened full string, including all unmeasured
screen-transfer-003 donor cells and all six native candidate pools. The union has
18,475 full strings; form **both** their 20- and 28-symbol prefixes (16,753 and
18,447). Every new donor full string must avoid both prefix bans regardless of
its measured position. This round does not inherit the earlier shorter-prefix
exception. Original training overlap remains unknown.

Fix the global donor role by
`int(SHA256('averaged-anchor-004-role:'+first20),16)%2`:
0=calibration, 1=target. This makes calibration/target sets disjoint at both prefix
lengths across the entire round. Draw two distinct measured prefixes per role
and `(balance,position)` cell; therefore all four within-cell prefixes differ.
Across families, repeats remain allowed. Do not adapt role assignments to scores.

For each donor, retain balance −2/+2, position 20/28, running minimum −4, read
symbol `)`, length 32 and sixteen of each bracket. All strings, donor roles,
replicas, cell labels and template order are drawn and hashed before any native
score. Template i is assigned to accepted recipient i, with no donor selection
by outcomes. The prepared manifest reports 2,048 unique recipients, 4,096 donor
entries, 3,859 distinct measured prefixes (maximum reuse four), and zero
calibration/target prefix overlap. There are eleven shared 20-prefixes between
new native candidates and donors, zero shared 28-prefixes and no shared full
strings; these are reported, not used to resample. Native scores are not target
transfer observations. These overlaps do not enter the calibration forecasts.

## 3. Stage order and stops

1. Publicly freeze this protocol, code, inputs, analysis and planning tables.
   Independent review and explicit user execution authorization remain pending.
   Publish their release before any new model score.
2. Measure all 2,048 native margins `logit(False)-logit(True)` in both dtypes.
   Accept iff signed margin `<8` (not absolute margin). Labels must agree and
   score discrepancies must be <=0.001 nat. Save all scores and select the first
   256 accepted in frozen order. If missing, stop `insufficient_screening_yield`;
   no new candidates, lower quota or cutoff change.
3. Fix each recipient’s largest native fp64 EOS-attention closing-bracket
   position (first tie), native attention coefficient and intervention operator.
   Measure the eight calibration transfers for **all 256 families**.
4. Compute all forecasts and the new separation criterion in both dtypes. Save
   forecasts, scores, selected manifest, source bindings and hashes in a receipt
   **before any target transfer**. If fewer than 128 families separate, stop
   with insufficient forecast separation, preserving all calibration data.
5. If the start rule passes, measure all eight targets for **all 256 families**.
   Keep nonseparating families for the paired averaging comparison and prefix
   diagnostic. Do not delete cases for poor fit or an observed history effect.

The unchanged value-contribution replacement is
`h'=h+a_recipient[j]*(v_donor[k]-v_recipient[j])`, at the EOS query. Native/self
controls, executed-tensor fidelity, hook/position and off-target checks remain
mandatory. A required technical failure stops the entire round, preserving
partial records. No scientific retry, alternative head or replacement family.

## 4. Frozen predictions and measurements

For d in {−2,+2}, p in {20,28}, use two distinct calibration prefixes:
`A[d,p]=(cal[d,p,0]+cal[d,p,1])/2`.

| Name | Prediction for each fresh target at (d,p) |
|---|---|
| `B_avg` | `(A[d,20]+A[d,28])/2` |
| `P_avg` | `(A[-2,p]+A[+2,p])/2` |
| `B_single4` | `(cal[d,20,0]+cal[d,28,0])/2` |
| `B_legacy2` | `cal[-2,20,0]` if d=−2; `cal[+2,28,0]` if d=+2 |

`B_single4` retains the same four-cell coverage as `B_avg`; it changes only
replica averaging. The legacy comparator changes coverage too, so is descriptive.
There is no best-anchor selection, fitting to targets or rescaling tolerance.

Define `S=max_target|B_avg-P_avg|>0.202`. Require equal S labels in both dtypes
and <=0.001-nat discrepancy of every signed forecast contrast. The two sup-norm
0.101-radius prediction regions are then disjoint. Under an ideal balance-only
example, this new gap is **half** the balance contrast; old diagonal-gap yield
cannot be substituted in planning.

Candidate error is the maximum absolute error over **all eight targets**.
Definite hit: error<=0.099; possible hit: error<=0.101. Preserve every cell and
require signed prediction-error discrepancies between dtypes <=0.001. Do not
average away a missed target or treat fp64 as exact arithmetic truth.

## 5. Estimands and error family

On the m separating families (m>=128), estimate population adequacy for `B_avg`
and `P_avg`. The target remains **90%**, not 80%. For each candidate take a
one-sided Clopper–Pearson lower bound from definite hits and an upper bound from
possible hits, each at alpha=.01. Lower>.90 supports adequacy; upper<.90 excludes
it; otherwise unresolved. Four tails spend at most .04 in total. These concern
this declared generator and conditional separating population, not all prefixes.

On **all 256 families**, compare `B_avg` definite hits with `B_single4` possible
hits. A gain is averaged-definite and single-not-possible; a loss is
averaged-not-definite and single-possible. Test the paired discordants with the
one-sided exact binomial/McNemar test of gain probability<=loss probability,
alpha=.01. Numerical ambiguity therefore counts against the proposed benefit.
This fifth allocation makes the familywise error bound <=.05. Rejection supports
positive predictive improvement; it does not establish an improvement of at least
10 pp. **+10 pp is a power-planning alternative, not the tested null.** No test
is claimed for the legacy comparator or differences in individual balance groups.

Selection/start losses and technical failures remain visible. The calculation
is not a guarantee of conditional coverage given that a global start gate passes;
its error budget is for this predeclared procedure, with a failed start producing
no target claims. No extra analyses can change the frozen decision.

## 6. Direct prefix diagnostic, without additional transfers

Within each `(d,p)` compare the two fresh target margins, holding recipient,
site, attention coefficient, balance, position, symbol and minimum fixed.
A numerically supported difference **>0.202** means no common cell prediction
can fit both within0.101. It rules out a poorly chosen calibration anchor as the
**sole** explanation for that family’s discrepancy. It is evidence of additional
causal-prefix dependence under the value-transfer intervention, not a named
history feature or native circuit identification.

Report every signed gap and definite/possible witness and family count
**descriptively**. A witness-boundary straddle is numerically unresolved for this
diagnostic, not grounds to discard a family or invalidate the main comparison.
Signed contrast precision must still pass. No witness does not establish history
invariance. Averaging improvement and positive witnesses can coexist.

## 7. Sample size, power and budget

N=256 selected families; at least128 separating families before targets.
`planning.json` records exact hypothetical power and integer critical counts,
including varying screen acceptance, new separation yield, candidate fit and
paired gain/loss probabilities. The design is intended to resolve a +10-pp
averaging improvement under a declared discordance scenario; no old19/30 rate
is used as a prediction for the new averaged estimator. The 90%-adequacy question
has limited power near its boundary, especially if m is close to128. An unresolved
adequacy result remains possible even when averaging improves.

Do not multiply screening, separation and paired-test powers as independent:
the same calibration affects separation and prediction. Stage power calculations
and illustrative effect distributions are assumptions, not empirical forecasts.

Baseline: 2,048×2=4,096 sequence-forwards. Each donor transfer costs four
(native,donor,self,patch) forwards per dtype, eight across two dtypes. Calibration
costs256×8×8=16,384; targets the same. Maximum **36,864**; a calibration-stage stop
costs at most20,480. Record actual attempted/completed forwards, including partial
failure, and wall time. No caching credit or computational-saving claim.

## 8. Result and future step

Report candidate decisions, paired improvement, within-cell witnesses, start
and technical statuses separately. A future-confirmation planning gate requires
m>=128, >=90% definite hits for a candidate and the other candidate’s possible-hit
upper bound below90%. This is **feasibility**, not population adequacy: the first
candidate’s own interval may still be unresolved. Improvement over `B_single4`
is not mandatory for this gate. No confirmation is authorized automatically.

Only two balances are present, so exact balance and sign remain confounded.
Averaging success does not rule out richer prefix information. Failed calibrated
predictions alone do not falsify every balance-only representation. This single
selected head cannot establish general model prevalence or a complete mechanism.
A prospective result may narrow the next mechanistic question; no desired outcome
is guaranteed. Public freeze, independent review, then a separate release: stop here.
