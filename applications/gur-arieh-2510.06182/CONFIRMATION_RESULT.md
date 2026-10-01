# Round 1: frozen confirmation result

**Status: VALID, S3 (`W_T` excluded only).** The confirmation was run once on fresh
cases after the analysis rule, cell, anchors, tolerances, sample size, seeds, model
revision, runtime and artifact hashes were frozen in [`FREEZE.json`](FREEZE.json). The
records-only checker independently reproduces the gates and decisions from the frozen
manifest and raw records. This timing is supported by local commit and run records,
not by an externally timestamped preregistration; see Verification below.

## Question

The paper fits a mixture model to distributions averaged over many patched prompts. The
same average can hide different case-level patterns. Round 1 asks a deliberately narrow
question: **does each case have roughly the same concentration as the cell average, or
is each case substantially more concentrated?** It measures concentration only. It does
not determine which mechanism operates in a case.

Two profiles were fixed on development split B:

- `W_T`: a resolved case is close to the concentration of the empirical cell average,
  `T_W = 0.533756`;
- `A_T`: a resolved case is close to the concentration of the three-way agreement
  control, `T_A = 0.996447`.

`W_T` is our narrow, operational concentration rival. It is not a general mechanism
claim made by the paper, so excluding it does not refute the paper's mixture model.

The frozen readout scores single-token, capitalised answer forms **without a leading
space**. This differs from the upstream in-context entity-token readout, which includes
a leading space. The change was selected during development, following the
[pilot readout audit](results/pilot/PILOT_REPORT.md#4-readout-validity-the-declared-readout-and-the-answer-form-readout).
The result therefore applies to this adapted readout, not an exact reproduction of the
paper's measurement.

A case matched a profile when its concentration was within
`kappa * (T_A - T_W) = 0.115673` of the corresponding anchor. A profile required at
least 80% coverage. The simultaneous intervals conservatively count unresolved cases as
non-matches for adequacy and as matches for exclusion.

## Result

The runner generated 305 fresh families to obtain the frozen quota of 300 qualifying
families. Ten were unresolved by the support rule; none failed the answer-mass rule.

| profile | resolved matches | conservative count for exclusion | simultaneous bounds relevant to the decision | status |
|---|---:|---:|---|---|
| cell-average concentration (`W_T`) | 16 / 300 | 26 / 300 | adequacy lower 0.028; exclusion upper **0.130** | **excluded** |
| agreement-like concentration (`A_T`) | 224 / 300 | 234 / 300 | adequacy lower **0.686**; exclusion upper **0.831** | **undecided** |

The aggregate pattern itself replicated: the confirmation mean over resolved cases was
`(0.5323, 0.4257, 0.0420)`, close to split B's frozen
`(0.5338, 0.4198, 0.0465)`. The maximum difference was 0.00594, below the frozen
mean-consistency limit of 0.08425.

The case-level pattern was different from that mean. Of 290 resolved cases, 282 were
more concentrated than `T_W`, and 274 exceeded it by more than the frozen profile band;
none deviated below it by that amount. Thus the average remains reproducible while its
concentration is not representative of most individual cases.

## Interpretation

This result rules out the declared `W_T` profile: in this task, cell, layer and patch
position, individual patched outputs do not usually have the concentration of the
empirical cell average. It does **not** establish the strict `A_T` profile, identify one
of the paper's mechanisms in any case, or prove that different cases use different
mechanisms. The predeclared between-case statement was therefore not earned.

For the paper, the result leaves the aggregate mixture fit intact but narrows its
interpretation. A mixed average need not mean that a typical individual patched output
is mixed to the same degree; here the typical output is substantially more concentrated.
Further interventions are needed to distinguish case-varying mechanisms from token
preferences or other explanations for that concentration.

For causal decidability, this is a positive but bounded demonstration: a condition and
decision rule fixed before fresh data eliminated one executable rival while refusing the
stronger explanation that the data did not establish.

## Verification

- Local freeze commit `e96b357` is dated 2026-09-28 17:54:57 +02:00; the recorded run
  began at 17:55:59 +02:00. [`RUN_STARTED.json`](results/confirmation/RUN_STARTED.json)
  records that commit, a clean working tree and the manifest hash below. These bind the
  recorded protocol to the run, but do not independently establish the chronology:
  the freeze was not pushed before the run and was not externally timestamped.
  A separate human review of the final freeze is not documented in this repository;
  this does not establish that the run lacked user authorization.
- frozen manifest SHA-256:
  `11a7beb825e94ca537944d329f0c74bb504a5adaaeb5e7078ddb7f6d5f8ce327`
- freeze commit: `e96b357e1e5a189656166a451ad13727d1d67375`
- all binding confirmation gates and the mean-consistency gate passed;
  identity-patch maximum answer-logit difference 0.0; MPS/CPU maximum
  `|Delta T| = 5.70e-6` against a 0.01 tolerance
- yield: 300 / 305 = 0.9836
- independent checker: `verified = true`, `run_status = VALID`, same profile statuses
- confirmation artifacts: [`results/confirmation/`](results/confirmation/), with exact
  SHA-256 coverage in `artifact_hashes.json`
