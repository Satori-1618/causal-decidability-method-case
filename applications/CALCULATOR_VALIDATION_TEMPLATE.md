# Calculator validation: does the pre-run ratio predict what a design decides?

*Use this template only to test the planning calculator itself, as the `resid_mid.8` study
did. For an ordinary application of the method, where the question is which explanations
remain compatible, use [PREREG_TEMPLATE.md](PREREG_TEMPLATE.md).*

## Title: <claim> at <site> of <model>

**Frozen before any forward pass at this site.** Two freeze points:

- **Freeze A**: this document, committed before the pilot runs.
- **Freeze B**: the numeric predictions, computed from the pilot by the pinned calculator,
  committed before the confirmation runs. The confirmation script refuses to run unless the
  Freeze B file exists with the hash recorded in its commit.

Timestamp each freeze externally at the moment it is made (a public push, a registry
entry, a timestamping service). Commits bind versions to each other but do not prove when
they were made.

State what is publicly known about this site, and how you checked that the outcome is not
already known.

## 1. What is being tested

The planning claim: before running, the procedure predicts whether a design (a readout at
a sample size) can distinguish the declared rivals. Which rival turns out to be right is
a secondary result and **is not used to judge the method**. A correct prediction that a
design cannot decide is a success.

## 2. Site, intervention, model

Model, dtype, device. Site and position. The intervention, exactly as executed. Pinned
inputs (directions, weights, data) with their sha256. The conditions run per unit.

## 3. Rivals

For each component or question, the declared rivals and what each predicts **under the
intervention actually executed**. Define every effect relative to the clean run, so that
an intervention that changes nothing has effect exactly 0. Declare equivalence groups.
Explanations not written here are neither tested nor excluded.

## 4. Readouts

Every readout, declared now. None may be dropped or added after the pilot. If a readout is
binary or ternary, note that the SD of its contrasts reveals its rate (§7).

## 5. Unit and contrasts

The statistical unit and the paired contrasts, one per rival endpoint.

## 6. Data and seeds

Distribution, builder, pilot seed and size, confirmation seed and size. Every source of
nondeterminism in data generation (for example `PYTHONHASHSEED`, set iteration order),
fixed and checked at runtime. The confirmation order, interleaved or randomised so that
every nested prefix is a draw from the declared distribution. No confirmation unit may
be generated before Freeze B.

## 7. The pilot is blinded

The pilot outputs **only** what §8 needs: the effect size to be decomposed and the SDs of
the paired contrasts. It never writes the quantities that decide between the rivals. The
pilot script is committed and hashed before it runs. Disclose any readout on which the
blinding is only nominal.

## 8. The prediction, computed at Freeze B

The exact calculator call, with every argument pinned (`alpha`, `signatures`, `z`,
`noise_factor`, dtype, `readout_scale`, `depth`), and the calculator's sha256. The
prediction model must match the readout: the normal floor assumes a known SD and adequate
n; for binary or other discrete readouts at small n, predict with exact power instead.

**The sample-size rule.** Fix the rule here, not the sizes: for example, for each cell the
sizes whose predicted ratios lie closest to {0.5, 0.75, 1, 1.5, 2, 4}, plus the largest
size affordable. The predictions then fall on both sides of ratio 1, and the test can fail
in both directions.

## 9. What "decided" means after the run

**What a decision predicts.** Any exclusion, the correct candidate set, or identification
of a relevant effect: choose one, and report the rivals that remain, with "no candidate
fits" as its own outcome.

**The realized rule,** at the same error level: which intervals or tests, and when an
endpoint counts as compatible. Its coverage must hold at every sample size used. Do not use
a normal interval from the sample SD on small discrete samples; use an exact test. State how
a zero SD is handled.

## 10. Success and failure

Statistical decisions are random, so even a correct calculator will sometimes miss.
Declare how many misses outside the band chance allows: k, the upper
(1 − error level) quantile of the number of misses expected under the declared error
rates of the realized rule.

- **Supported:** at least X % of predicted verdicts match, **and** at most k predictions
  with a ratio outside [0.5, 2] miss.
- **Contradicted:** more than k predictions outside [0.5, 2] miss.
- **Inconclusive:** otherwise.

Score the predictions against a simple baseline, at least the constant prediction, and on
separate confirmation blocks rather than nested prefixes.

Also reported, not judged: observed against predicted crossover, and the secondary answer.

## 11. What this does not test

The numerical floor, unless the dtype makes it bind. Generality. Rivals not declared here.

## 12. Prior knowledge, disclosed

What was known before this document, from the paper and from any replication.

## 13. Deviations

Every deviation is recorded in a dated section appended below, with its reason, **before**
the confirmation outcome is read. A deviation found after reading it makes the affected
predictions exploratory.
