# Applications

An application takes a causal claim about an intervention, declares the rival
explanations it has to beat, finds the conditions where their predictions differ, and
reports which explanations the data leave compatible.

## Index

The Makelov application is included in this checkout. Start with the
[plain-language case](../docs/CONFIRMED_CASE.md), then its
[records-only verification guide](makelov-2311.17030/RECORDS_ONLY.md).
No branch switch, model installation or download is needed for that path.

The next [query-restoration comparison](makelov-2311.17030/PATH_TEST_PLAN.md) is a draft,
with a [reader-facing walkthrough](../docs/ITERATIVE_IDENTIFICATION.md). It illustrates
how another round follows from the first result; it has no model-run evidence yet.

| paper | directory | what | status |
|---|---|---|---|
| Makelov, Lange & Nanda (2023), *Is this the subspace you are looking for?* arXiv:2311.17030 | `applications/makelov-2311.17030` | The four steps on the MLP8 patch: the full patch and Table 1's null patch cannot separate visible-read from null-read; a fixed write with a switched read can (read-source pilot, 32 pairs, development). Table 1 resolution per readout. Preregistered test at `resid_mid.8` | null read predicts the read patches better: pilot (32 pairs) and preregistered confirmation Q1 (64 of 64 fresh pairs, sign test p = 1.1e-19); adequacy (Q2) deferred; `resid_mid.8`: 28/28 under the frozen rule, narrow |

## How to apply the method

**Ordinary application: which explanations remain compatible?** This is the method.
Write the preregistration with [PREREG_TEMPLATE.md](PREREG_TEMPLATE.md), then:

1. **Freeze** the candidates and their predictions, the separating conditions, the units,
   the adequacy bound and the uncertainty rule, and timestamp the freeze externally.
2. **Pilot**, if needed, on units kept apart from the confirmation. It may inform the
   sample size, never the adequacy bound or the candidates.
3. **Confirm** on fresh units, once. Run `examples/from_data.py` (or `evaluate`) with the
   frozen declarations and a coverage-justified decision rule. The shipped adequacy
   intervals are nominal bootstrap intervals; calibrate them for your data, or use a
   justified relative comparison without an adequacy claim. Report the outcome and, for each excluded candidate, the
   conditions that excluded it. "No candidate fits" is a correct result.
4. **Record deviations** in the preregistration, dated, before reading the outcome.

**Retrospective use (light).** Take the per-example values behind a published contrast,
declare the rivals it is meant to separate, and compute separation and paired resolution
(`paired`). This asks which column of a table carries the claim. In the Makelov
application the same rowspace-versus-nullspace contrast sits at 85.9 paired standard
errors on logit difference and 2.65 on interchange accuracy, where it rests on 7
discordant examples against 0.

**Calculator validation (optional, separate).** To test whether the pre-run ratio predicts
what a design will decide, use
[CALCULATOR_VALIDATION_TEMPLATE.md](CALCULATOR_VALIDATION_TEMPLATE.md), as the
`resid_mid.8` study did: freeze, blinded pilot, calculator predictions frozen with the
confirmation runner, confirmation. This validates a planning tool; it is not needed to
apply the method.

## Lessons from the first applications, already in the templates

- **Declare the tolerance before the data.** With the pilot's own loss, the stored Makelov
  read-source data exclude both candidates at a tolerance of 10 % of the full effect, and
  show the null-read candidate adequate at 25 %. Both are correct answers to different
  questions, so the question has to be fixed first, on scientific grounds.
- **Choose the loss for the question.** Averaging signed errors lets +2 and −2 cancel, so
  a candidate can look perfect while missing every case. The read-source pilot compared
  absolute errors per case; its continuation keeps that loss.
- **Retained is not adequate.** Being the only candidate left says nothing about how well
  it predicts. Report excluded, adequate and undecided separately.
- **A fractional tolerance is an estimate.** If the tolerance is a fraction of an effect
  measured on the same units, resample the two together.
- **An anchored candidate reproduces its anchor by construction.** Both Makelov
  candidates are stated relative to the measured full patch, so the full patch cannot test
  them; only the read conditions can. Mark anchored predictions as such.
- **Say what a decision is supposed to predict.** At `resid_mid.8`, "decided" meant "at
  least one endpoint excluded", and in 21 of 28 checks both endpoints were excluded. Fix in
  Freeze A whether the target is any exclusion, the correct candidate set, or identification
  of a relevant effect, and report which rivals remain, not only decided / undecided.
- **Calibrate the realized rule at the sample sizes you use.** A normal interval from the
  sample SD over-excludes on small discrete samples: 3.9 % false exclusions at n = 12
  instead of 0.5 %. Use exact tests for discrete readouts, and predict with the matching
  model (exact power rather than the normal floor).
- **For calculator validation, choose sample sizes that give weak and strong designs.** At
  `resid_mid.8` every frozen prediction landed above ratio 2, so a constant prediction
  would have scored as well. Declare a *rule* that picks sizes from the pilot, score it
  against a simple baseline, and use separate confirmation blocks rather than nested
  prefixes.
- **Timestamp externally at the time.** Commits and hashes bind versions to each other;
  they do not prove when a freeze happened, and publishing a private history later does
  not add that.
- **Define every effect relative to the clean run,** so that an inert intervention has
  effect exactly 0.
- **Pin data generation** (hash seeds, symbol orders, the builder) and check it at runtime.
  Upstream IOI code uses `list(set(pattern))`, whose order depends on `PYTHONHASHSEED`.
- **Randomise or interleave the evaluation order.** Generated data often comes in blocks,
  and a prefix of one block is not a draw from the declared distribution.
- **Binary readouts leak through their SD.** For a {−1, 0, 1} contrast the SD reveals the
  rate up to p ↔ 1 − p, so a pilot is blinded only nominally on such a readout. Say so in
  advance.
- **Keep numerics out of the question** unless they are the question. In float32 the
  statistical floor binds, and the numerical floor is not tested.
- **Report effect ratios as ratios.** A single-component patch reaching 86 % of the full
  patch's effect is not an 86 % share of the mechanism: the patches are not additive and
  the network downstream is nonlinear.

## Adding an application

1. Branch from `main` as `applications/<first-author>-<arxiv-id>` (`git switch -c …`; use
   `git switch`, not `git checkout`, because the branch name matches the directory).
2. Put everything under `applications/<first-author>-<arxiv-id>/`: `README.md`,
   `PREREG_*.md`, `src/`, `scripts/`, `results/`, `tests/`, and a pinned copy of the
   calculator version you ran with.
3. Do not vendor upstream files whose licence does not allow it. Fetch them at a pinned
   revision and verify their hashes instead.
4. Add a row to the index above on `main` once the application has a result, whatever the
   result is.

## Candidate papers

Not yet checked for public per-example code or suitable rivals; a starting list only.

- Wang et al. (2022), IOI circuit, arXiv:2211.00593. Path patching, many head-level
  claims, with backup heads as natural rivals.
- Geiger et al. (2023), distributed alignment search, arXiv:2303.02536. IIA scores
  compare alignments; how far apart do rival alignments' predictions lie?
- Wu et al. (2023), Boundless DAS, arXiv:2305.08809.
- Hanna, Liu & Variengien (2023), greater-than circuit, arXiv:2305.00586.
- Conmy et al. (2023), ACDC, arXiv:2304.14997.
- Shi et al. (2024), hypothesis tests for circuits, arXiv:2410.13032. Their §5 leaves the
  Type II error of the equivalence test open.
