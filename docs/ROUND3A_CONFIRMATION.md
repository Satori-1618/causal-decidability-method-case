# Round 3A: position matters, but position-only invariance fails

**Update, 27 September 2026:** a [post hoc addendum](#addendum-27-september-2026-post-hoc-description)
adds a tolerance table and explains why the averaged name contrast is weak evidence of
name independence. The report below is unchanged, and its frozen decisions stand.

**Confirmed on 512 fresh GPT-2 Small families under the amended, published protocol.**
Changing the donor answer's mention position strongly changes the fixed patch's effect.
But its effect is not adequately invariant to name assignment at fixed position:
the position-only profile fits **301/512 families**, below the required 80% coverage.
The identity-only profile fits **0/512**. These are exclusions of two declared response
profiles, not identification of a transferred person or role.

## The question in plain language

Suppose a story mentions Lily and Catherine, and Catherine gives a folder to Lily.
The model should complete the sentence with Lily. We take a signal from another
story and use it in the same fixed internal patch as the earlier experiments.

Two simple accounts make different predictions. If only the donor answer's **mention
position** matters, changing which name fills that position should leave the patch
effect essentially unchanged. If only the donor answer's **name** matters, moving that
same name between first and second mention should leave the effect unchanged.

Cross both donor properties, holding the recipient fixed; then repeat for the opposite
recipient ordering. Every family includes both orders, all four donors and both numeric
precisions. They are measurements within one family, not separate independent samples.
The readout is the change in logit preference for the recipient's correct name over
the other name. It need not produce an answer flip.

This changes the full name assignment, including the giver's name. It is not a clean
intervention on a latent variable called “person identity.” The artificial patch reads
one scalar and writes along one fixed vector; its output also depends on the recipient.

## One actual example

The first family in the frozen manifest uses:

> Then, Lily and Catherine were working at the home. Catherine decided to give a folder to

Lily is the correct completion. Keep this recipient fixed and cross the donors:

| Donor's first two names | Giver | Correct donor answer; mention position | Patch effect on preference for Lily |
|---|---|---|---:|
| Lily and Catherine | Catherine | Lily; first | 0.000 |
| Catherine and Lily | Catherine | Lily; second | −0.857 |
| Catherine and Lily | Lily | Catherine; first | +0.034 |
| Lily and Catherine | Lily | Catherine; second | −0.636 |

These float64 reference measurements are changes from the native margin of **2.852**.
The self-patch is zero by construction. Here changing position causes the larger shifts;
changing the correct name at fixed position has smaller effects. Every patched margin
remains positive, so there is no answer flip in this example.

This first family meets the position-only tolerance, but others do not. The first failure
in frozen order is family 3 (Peter/Aaron): one position-matched name-assignment change
shifts the margin by **−0.279 nat**, beyond the ±0.25 tolerance. Across the full sample,
**211 families fail** at least one of the required name-invariance comparisons.

## The frozen decision rule

For each invariance account, a family succeeds only if both relevant simple effects
are within **±0.25 nat**, in both recipient orders and both precisions. Adequacy requires
80% population coverage; the lower simultaneous Clopper–Pearson bound must exceed 80%.
An upper bound below 80% excludes that coverage claim. Otherwise it is unresolved.

Mean name-assignment, position and interaction contrasts use complete-family bootstrap
intervals and a **±0.10-nat** practical boundary. A small signed mean is not the same
claim as invariance: opposite effects can cancel. The profile family receives alpha
0.025 and the mean family nominal alpha0.025, as before. The percentile-bootstrap
component has nominal, not guaranteed finite-sample coverage.

## The confirmation result

| Declared profile | Complete families meeting it | Simultaneous success-rate interval | Decision at 80% required coverage |
|---|---:|---:|---|
| Position-only invariance | **301/512 (58.79%)** | **53.20–64.22%** | Excluded |
| Identity-only invariance | **0/512** | **0–0.99%** | Excluded |

The profile intervals jointly have at least 97.5% coverage under the fixed iid-family
sampling assumptions. Mean contrasts, averaged over both recipient orders within each
family, give the following results using the other error allocation:

| Contrast | Mean, nat | Primary bootstrap interval, float64 | Decision against ±0.10 |
|---|---:|---:|---|
| Name assignment, I | −0.00776 | [−0.01186, −0.00379] | Practically equivalent |
| Position, P | **−1.28018** | **[−1.32714, −1.23298]** | Negative, relevant |
| Interaction, J | −0.00615 | [−0.03498, +0.02310] | Practically equivalent |

All decisions agree in both precisions and both frozen bootstrap seeds. I's interval
excludes exactly zero but lies well inside the practical zone: “equivalent” means small
at the declared resolution, not mathematically zero.

**Mean equivalence is not invariance.** Descriptively, the two recipient-order I effects
have opposite signs in **491/512 families**; individual name-assignment simple effects
reach **0.989 nat**. Averaging can conceal those effects. P averages −1.062 and −1.498
in the two recipient orders, so its negative mean is present in both. J's small mean
likewise does not establish that every family has negligible interaction.

## What this adds to identification

The earlier experiments already showed positional sensitivity. This confirmation
establishes a narrower additional limit: **position-only invariance is not accurate
enough across these name assignments**, even though a signed average might suggest
negligible name dependence. Both candidate profiles are excluded; no new mechanism
is declared the winner.

A primarily positional mechanism whose strength varies with lexical context remains
possible. So do changes in scalar readout magnitude and recipient response sensitivity.
The result does not refute the paper's broader positional interpretation, identify a
name-bearing payload, or show that the unmodified model uses this artificial patch's
read source. A further round would need distinct predictions from these remaining
accounts, not simply new semantic labels for the same scalar intervention.

## Why this is an amended confirmation

The [original development run](ROUND3A_RESULT.md) kept all 32 families and its initial
planning STOP. Before any confirmation outcomes, we [amended the maximum sample size](../applications/makelov-2311.17030/DONOR_FACTOR_512_AMENDMENT.md)
from 384 to 512. No effect threshold, outcome, contrast, seed, precision rule or
inferential method was relaxed.

The full amended planning passed: its lowest bootstrap power lower 95% Monte Carlo
bound was **86.79%**, above 80%. These are conditional planning results, not observed
confirmation performance. The exact 512 new families and manifest were pushed in
[`50b7a8d`](https://github.com/Satori-1618/causal-decidability-method-case/commit/50b7a8d)
before any confirmation forward. All 4,768 historical/development prompts were excluded;
the 512 sampled content families are distinct. Development observations are not pooled
into this confirmation.

## Measurement checks

- **512/512 families** passed numerical resolution; none was removed or replaced.
- All self/zero and untouched-row identities were exact. All hook and insertion checks passed.
- Largest float32/float64 discrepancy across the declared readouts: **0.00005696 nat**,
  below the frozen 0.01 budget. Float64 promotes the same processed weights; it is not exact truth.
- Largest bootstrap endpoint shift between the two fixed seeds: **0.001111 nat**,
  below 0.01; every mean decision was stable.
- There was no correctness filter. In each recipient order, **505/512** native prompts
  favored the correct name over the other name. All failures remained in the analysis;
  this is two-name scoring, not full-vocabulary correctness.
- Inference took **603.55 seconds (10.1 minutes)** on the local CPU: 6,144 forward calls,
  comprising 24,576 prompt evaluations. Statistical n remains **512 families**.
- The full records checker reproduced the analysis, including both bootstrap seeds.
  The offline suite passed **202 tests and 203 subtests**; original frozen files remain unchanged.

## Scope and reproduction

This is GPT-2 Small, MLP8 post-GELU, the final prompt position, the fixed null-read/full-write
operator, one template and one prefix. It tests the effect of the patch, not whether the
unmodified network naturally uses a person or role representation. Position selection
and giver-position inhibition remain indistinguishable in the two-name task.
**Subsequent status:** [3B's native qualification](ROUND3B_STAGE_A_RESULT.md) has since
run and stopped before patching. No empirical 3B role-transfer comparison is available.

Verify the published records without a model or additional packages:

```bash
python3 -S applications/makelov-2311.17030/scripts/check_donor_factor_512_records.py \
  --results applications/makelov-2311.17030/results/donor_factor_confirmation_512
```

Remove `-S` and add `--full-bootstrap` in an environment with NumPy to reproduce both
seeded bootstrap calculations as well. The independent standard-library checks reconstruct
raw margins, contrasts, profile counts, binomial intervals and saved planning decisions.
They check available original inputs and explicitly list missing optional ones. Stored
audits and hashes do not independently prove execution or recover activation tensors.

**Evidence:** [original protocol](../applications/makelov-2311.17030/DONOR_FACTOR_PROTOCOL.md),
[amendment](../applications/makelov-2311.17030/DONOR_FACTOR_512_AMENDMENT.md),
[planning](../applications/makelov-2311.17030/results/donor_factor_planning_512/planning.json),
[frozen cases](../applications/makelov-2311.17030/results/donor_factor_confirmation_512/cases.json),
[manifest](../applications/makelov-2311.17030/results/donor_factor_confirmation_512/manifest.json),
[raw paired records](../applications/makelov-2311.17030/results/donor_factor_confirmation_512/records.jsonl),
[complete summary](../applications/makelov-2311.17030/results/donor_factor_confirmation_512/summary.json),
[verification](../applications/makelov-2311.17030/results/donor_factor_confirmation_512/verification.json).

## Addendum, 27 September 2026: post hoc description

Computed after this report from the stored records with
[`describe_post_hoc.py`](../applications/makelov-2311.17030/scripts/describe_post_hoc.py).
It changes no frozen decision, and no row other than ±0.25 nat is a test.

**How strict was ±0.25 nat?** Complete families meeting each profile when only the
tolerance changes; all other rules are as frozen:

| Tolerance (nat) | ±0.10 | ±0.15 | ±0.20 | **±0.25 (frozen)** | ±0.30 | ±0.35 | ±0.40 | ±0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Position-only | 76 | 160 | 235 | **301** | 352 | 396 | 431 | 473 |
| Identity-only | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |

At least one name-assignment effect exceeds 0.25 nat in 41% of families, while all stay
within 0.50 nat in 92%. The exclusion at the frozen tolerance stands. A more lenient
tolerance chosen now would not make position-only a confirmed adequate profile.

**Why the averaged name contrast is weak evidence of name independence.** "Practically
equivalent" is a correct statement about the signed mean of I, but this mean is poorly
suited to showing that names do not matter:

- Both names are drawn from the same list in random order. Exchanging them reverses the
  sign of the name contrast in the scalar read by the patch, so that scalar-level contrast
  is symmetric about zero by design; it was positive in 271/512 families. The model's
  response need not preserve this symmetry.
- Measured, not guaranteed: within each recipient order the effect was close to linear
  in the read scalar (median absolute residual 0.004 nat), and the two orders had
  opposite-sign slopes in all 512 families. The same scalar change therefore moved the
  fixed A-minus-B margin in opposite directions, and averaging over the two orders largely
  cancels it; I had opposite signs in 491/512 families.
- The mean absolute name contrast was 0.089 nat and 0.128 nat in the two recipient orders.

The case-wise invariance profiles are the informative test of name independence, and
they are the basis of the exclusion above. The protocol's cancellation note addresses P
and J under equal slopes; with the opposite slopes measured here, the averaged I largely
cancels. For the averaged interaction J, the scalar-level contrast is also symmetric
about zero by design (positive in 262/512 families). In the outputs, however, the two
recipient orders add rather than cancel (same sign in 498/512 families). Measured output
J varied in sign across families (positive in 251 and 247 of 512 in the two recipient
orders), so its mean is near zero although the mean absolute J was 0.161 nat and
0.229 nat.

**Test counts.** The counts under "Measurement checks" come from the local run at that
commit. The hosted workflow for that commit failed while collecting tests because it did
not install NumPy; the workflow passes from commit `3815329`.
