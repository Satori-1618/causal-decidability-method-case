# Round 3A: position matters, but position-only invariance fails

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
