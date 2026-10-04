# History 005: matched edits, fresh forecasts

**Status: prospective development freeze; review pending; no measurements.**
Date: 4 October 2026. Parent result: averaged-anchor round 004, public commit
`5d54328255f6136c71d3907a6b4788ae08f76ae4`.

## 1. Question and claim

At `a9g0io1r`, layer 2/head 1, do the effects of controlled donor-history edits
generalize to fresh prefixes as a recency pattern, a final-pattern effect, or
neither? We compare **predictions of centered transferred-margin contrasts**.
This is a new development question prompted by 004, not a repaired adequacy
test or confirmation on a previously untouched head.

004's frozen balance-only predictor remains excluded at its 90% adequacy target.
This round does not
test whether the new rules reach 90%, explain its misses completely, encode a
unique semantic variable, or describe the native use of that variable.

## 2. Fixed inputs and intervention

Checkpoint: `run_a9g0io1r_checkpoint_5.pt`, SHA256
`abec37ff9899e0349de9e19558b61dddadcdf7e15628aaa0fa55e70b435b9f4e`.
Architecture: two layers, four heads of width 16, hidden width 64. The frozen
upstream revision is in `plan.json`. Use CPU, one thread, deterministic PyTorch,
and both float32 and float64. Higher precision is a reference, not exact truth.

Seeds: recipient **26100451**, donor **26100452**. Prepare **2,048** unscored
recipient candidates and **256** donor-family templates. Each length-32 string
has 16 opens and 16 closes. Recipients have already gone below zero at some
prefix. `confirmation` in input metadata names the inherited hash partition;
it does not upgrade this round from prospective development to confirmation.

Measure all 2,048 native candidates. Require native-margin dtype discrepancy
≤0.001 nat and agreement on the strict signed screen `margin < 8`. Select the
first 256 accepted candidates; a shortfall stops without calibration or
replenishment. No outcome-based choice of donor, position, head or tolerance.

Each donor quartet is constructed as:

`18-token stem + 6-token recency core + 4-token ending + 4-token future suffix`.

The stem's current and minimum balance are both −4. The cores are
`(())((` (recency 6) and `(()()(` (recency 10); endings are `()()` and `(())`.
The read is at position 28. All cells have current balance −2, minimum −4 and
read symbol `)`. The row edit is an adjacent swap at positions 22/23; the column
edit is an adjacent swap at 26/27. All cells share the first 21 symbols and the
future suffix, containing three opens and one close. The prefix algebra and
the exact edited positions are asserted in the shipped preparer.

There are four distinct matched stems per family: two for calibration and two
for targets, producing 8 calibration and 8 target transfers. Calibration and
target roles come from a fixed first-20-prefix hash. Stems are drawn uniformly
from the eligible finite role pools; different families may reuse a stem.
This is the declared matched-stem population, not all strings satisfying the
semantic labels. Every previously prepared 20/28 donor prefix is excluded,
including unmeasured 004 inputs; full recipient strings are excluded too.
`inputs/preparation.json` reports support sizes, actual reuse and all overlap
checks. No global deduplication or resampling follows inspection of outcomes.

The intervention remains the validated value-contribution transfer:

`h' = h_recipient + a_recipient[j] × (v_donor[28] − v_recipient[j])`.

The recipient site is its native float64 EOS attention maximum over closing
brackets, with the first tie retained. Attention and other head contributions
at the intervention site stay native; downstream operations remain unpatched
and are recomputed normally. Use the existing runtime's identity,
site, delivered-tensor and finite-value controls unchanged. Score
`logit(False) − logit(True)` in nats; this is not a claim of answer-label flips.

## 3. Predictions fixed before targets

For each matched stem `k`, let `y[k,r,s]` be the four transferred margins.
Use the **fixed linear contrast**:

`z[k,r,s] = y[k,r,s] − mean_(r,s) y[k,r,s]`.

Average the two calibration `z` grids to obtain `A[r,s]`. Freeze these forecasts
for both future target stems:

| Rule | Forecast for target cell `(r,s)` |
|---|---|
| `H_recency` | row mean `mean_s A[r,s]` |
| `H_suffix` | column mean `mean_r A[r,s]` |
| `H_constant` | zero |
| `H_cell`, descriptive only | `A[r,s]` |

Here `H_suffix` means the ending of the **measured prefix**, not the four future
tokens after position 28, which are held fixed within the quartet.
Target centering is the same prespecified linear contrast. It does not estimate
a rival parameter from targets; nevertheless it changes the estimand. Shared
additive offsets are removed and cannot be explained by this test. Retain all
raw margins and report raw and centered patterns, including interaction.
The full-cell forecast is frozen prospectively but supplies no primary test
or start condition.

For each candidate, the family error `E` is the **maximum absolute forecast
error across all eight fresh centered cells**. Do not average target errors or
select a favorable target prefix. Cell repetition and dtypes never increase n.

## 4. Calibration start rule

For each of the three primary pairs, calculate the sup-norm distance between
their calibrated forecast vectors. A family supplies definite potential
separation if **any primary pair exceeds 0.022 nat in both dtypes**. Require
the frozen numerical controls; boundary straddles are reported as ambiguous,
not counted as definite separation.

Start targets only if at least **64 of 256 families** meet that condition.
This is an operational 25% minimum opportunity requirement, chosen for this
new question. It is neither the old 004 gate nor a power guarantee. The
triangle inequality gives `|E_i − E_j| ≤ ||forecast_i − forecast_j||∞`, so a
distance no greater than 0.022 cannot deliver a robust win greater than 0.022.
The converse is not guaranteed.

Report per-pair opportunity counts; do not require every pair to separate. A
recency-only signal, for example, makes `H_suffix` and `H_constant` identical.
Conversely, a pure row-by-column interaction can make all three primary
forecasts coincide. Such a case cannot pass through the descriptive `H_cell`
forecast: it remains outside this round's primary discrimination design and
may cause a start failure. A stop does not establish absence of history effects.
Write and hash **all** forecasts and the calibration records before any target.
If the gate passes, measure and analyze **all 256 families**, not merely the
separating subset. If it fails, preserve the calibration and stop.

## 5. Statistical decisions and power

The smallest relevant **comparative error advantage** is 0.02 nat. This is a
scientific resolution choice for the new history contrasts, not a dtype
tolerance, an old residual estimate, or a replacement absolute-accuracy target.

For each primary candidate and target cell, require the signed prediction error to
agree across dtypes within 0.001 nat. The difference of two candidate max-errors
then has discrepancy at most 0.002. A robust win for i over j is
`D = E_j − E_i > 0.022` in float64; a loss is `D < −0.022`; otherwise neutral.
The guard guarantees that the advantage also exceeds 0.02 in float32. Do not
require another classification agreement at the guard boundary. Technical
precision failure blocks the round's primary inference; no case dropping.

Test all three unordered primary pairs with **two-sided exact sign/binomial
tests** on robust wins and losses, using `alpha = 0.05/3` per pair. Neutrals stay
in the reported 256-family denominator but are not binomial trials. The null
is equal population probabilities of robust wins and losses; a rejected null
supports the reported direction. This does **not** test lower mean error,
absolute adequacy, equivalence or a minimum population win-rate difference.
Report pairwise directions or unresolved; allow cycles and do not force a
unique winner or claim a uniquely compatible mechanism.

`planning.py` computes exact power for declared hypothetical win/loss/neutral
rates. At n=256, rates **0.15/0.05/0.80** give approximately **0.880** power for
the specified direction; **0.20/0.05/0.75** give **0.995**. A smaller imbalance
**0.10/0.05/0.85** gives only **0.319**. These are planning alternatives, not
forecasts of this head or the probability of a whole-round success. They do
not include the chance of passing either start gate. Error control belongs to
the predeclared stopping procedure, not a conditional-on-passing guarantee.

## 6. Reporting, stops and scope

Always report screening counts, start counts by pair, controls, raw records,
all forecast vectors, all 256 errors when targets run, wins/losses/neutrals,
exact p-values, and continuous errors descriptively. Preserve signed row,
column and interaction contrasts; do not infer an interaction mechanism solely
because both one-factor candidates are poor.

Possible outcomes include screening shortfall, insufficient design yield,
technical failure, a supported comparative direction, and unresolved pairwise
evidence. A technical failure is not evidence against a candidate. A comparative
success is not a 90%-adequate explanation. No automatic confirmation follows.

The design changes two **history-edit bundles**. The row also changes local
bracket order, and the column also changes short-range nesting. Even a clear
row advantage cannot uniquely identify an abstract recency variable. Results
are limited to one selected head, these matched prefixes, the screened
recipients, this intervention and the centered-margin estimand. Adaptive
choices across preceding rounds remain part of the provenance.

## 7. Cost and release

Maximum sequence forwards: 4,096 native + 16,384 calibration + 16,384 target =
**36,864**. A calibration stop uses at most 20,480. Actual attempted/completed
forwards, batches, elapsed time and every failure must be recorded. No retry,
replacement family, threshold revision or additional model/head is authorized.

Publish code, inputs, planning, source hashes and this protocol first. Then an
independent reviewer checks the public commit. A separate explicit release
must be committed and pushed before execution; the runner checks that release
and that the computational sources/inputs still match the reviewed commit.
This preparation grants no permission to load the models or measure outcomes.
