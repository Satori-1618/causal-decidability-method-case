# Value-transfer round: current prefix state or position?

Status: new, bounded development. Earlier studies and their frozen files remain
unchanged. One released head is chosen from the prior result: `a9g0io1r`, layer
2/head 1. This is now an inspected/selected development case, not an unseen model.
No public preregistration, publication or author contact is implied.

## Question and intervention

Are donor values functionally exchangeable across positions at a fixed current
prefix balance, or across balances at a fixed position? Replace one contribution
to the recipient's final-layer EOS head output:

`h_patch = h_recipient + a_recipient[j] * (v_donor[k] - v_recipient[j])`.

All attention, all other values, other heads, other queries and recipient inputs
stay fixed. The source is a natively observed value, not a fitted direction. The
actual delivered tensor, identity and off-target controls must pass in fp32/fp64.
The implementation uses the existing validated `custom_node` path. It does not
claim that a hybrid activation is a naturally occurring state.

## Fixed development design

- 32 family blocks, seed 25100201, CPU only, no training or layer/head search.
- Recipient: a uniformly drawn equal-count length-32 string with a past prefix
  violation, excluding previously evaluated strings. Select its highest-attention
  closing bracket (first on ties) from its native fp64 EOS attention. Hold that
  recipient position and its native coefficient fixed in every arm; do not amplify.
- Donor grid: current balance -2/+2 crossed with absolute position 20/28. Every
  donor has length 32, 16 opens/16 closes, target symbol `)`, and prefix minimum
  exactly -4 through that position. Thus ever-violated status and minimum are fixed.
- Two independently sampled, distinct prefixes per cell. Within-cell differences
  directly test whether unmodelled prefix order already defeats either account.
- Prefixes are uniformly drawn conditional on these constraints. Fixed hash
  partitions allocate 20% to development and 80% to confirmation. Previously
  evaluated prefixes are excluded. A new suffix is never a fresh target value.
- Families are drawn with replacement from fixed pools, not filtered by observed
  outcomes. Repeated prefixes across families are reported, not silently removed.

## Prospective forecasts within development

Measure anchor A = (-2,20,replica0) and D = (+2,28,replica0), then save/hash the
following formulas before measuring the other six value transfers:

- H_state predicts A for negative-balance donors and D for positive-balance donors.
- H_position predicts A for position-20 donors and D for position-28 donors.

The forecasts include the second prefix at both anchor settings. No regression,
per-family refitting, or selection of the better donor replica is allowed.

Development initially uses scientific tolerance 0.10 nat (at most 0.025 in binary
conditional probability), numerical allowance 0.001 on prediction error, and
separating anchor gap >0.202. These are different quantities. The stricter
approximation tolerance is declared before this development run, not inferred
from earlier residual variance. The maximum error over all six targets defines
the family hit. Report raw margins, effects and every nonseparating family.

## Decision before any confirmation

One development run only. A confirmation is worth preparing if controls pass,
at least 16/32 families separate the predictions and at least one candidate has
definite hits (maximum error <=0.099) on 90% of eligible development families.
The score includes same-label second-prefix controls, so a selected anchor cannot
hide failed exchangeability of its other prefix. Development
does not establish these rates in a population. If the gate fails, report why;
do not sweep positions, donors, heads or tolerances until a candidate passes.

The separating-family gate aims at about 256 eligible families in a 512-family
confirmation. With four simultaneous one-sided bounds, that gives about 95%
planning power at true 90% fit (223/256 definite hits required). At only 128
eligible families the corresponding power is about 60%, not a precision guarantee.

If informative, freeze the exact source, fresh case manifest, sample size and
simultaneous inference rules before confirmation. The anticipated cap is 512 new
family blocks on this head; no additional checkpoint is automatically authorized
by a favorable result. Local freezes remain explicitly local provenance.

## Interpretation limits

With only -2/+2, exact depth and negative-current-balance status are observational
aliases. On this grid sign(d/k) follows the state profile and |d/k| follows the
position profile, because |d| is fixed at 2. Normalized balance d/k, more detailed
violation history, local patterns
and mixed dependence remain alternatives. An unrestricted function of d/k can
fit all four distinct ratios. Success means conditional exchangeability under
the tested balances and positions, not identification of a unique depth variable.
Both candidates may fail. Small anchor effects are insufficient resolution,
not proof that the transferred values or the head are causally irrelevant.

Changing whole values carries correlated prefix features. Fixed minimum and
ever-violation controls reduce specific confounds; they do not implement a pure
intervention on an isolated semantic variable. Earlier value-content diagnostics
were on a different checkpoint and provide motivation only.
