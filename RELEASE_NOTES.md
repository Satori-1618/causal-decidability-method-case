# Public method-case snapshot

## Makelov application and reproducibility update — 24 September 2026

The [current case study](applications/makelov-2311.17030/CASE_STUDY.md) explains what
each application of the existing method adds: Q1's supported relative read-source
comparison, round 2's three profile exclusions, 3A's two invariance-profile exclusions,
and 3B's native qualification STOP before any patching. It does not turn the latter
into a role-transfer result or add an absolute adequacy claim to Q1.

The [verification guide](applications/makelov-2311.17030/VERIFICATION.md) separates
standard-library records checks, NumPy analysis tests, optional PyTorch tensor tests,
and full model replay. The `test` extra now declares NumPy; `test-hooks` additionally
declares PyTorch. Tensor-dependent tests explicitly skip when that optional runtime
is absent. CI has separate stored-evidence, analysis and CPU tensor jobs. The analysis
job also regenerates the two seeded 3A bootstrap calculations from saved measurements.

This maintenance change runs no pretrained model and changes no research outcomes,
frozen contracts, raw records or historical application files. The root README and
`main` are preserved. Current entry points are the application index and case study;
older proposal text remains historical material. Local validation is recorded in
[the release check report](docs/MAKELOV_RELEASE_CHECKS.md); it is not a claim that the
updated hosted CI has already run.

## Round 2 execution update — 24 September 2026

This branch adds a new [192-pair query-route experiment](docs/ROUND2_RESULT.md), with
an earlier 32-pair development run, public pre-run freeze, full paired float32/float64
measurements, simultaneous exact-binomial profile intervals and records-only checking.
All three declared profiles fail the 80% coverage requirement; no unique mechanism is
identified. The reverse-transfer condition adds information but does not change the
population-level exclusion verdict already suggested by reset alone in these data.

The original snapshot published the general method and existing Makelov Q1 evidence;
that publication was not a new Q1 experiment or a retroactive public preregistration.
All 62 archived files remain byte-identical. The original four-cell
[teaching design](docs/ITERATIVE_IDENTIFICATION.md) is retained as history, with its
review correction and the separate executed protocol clearly linked. No claim is made
to invent model discrimination or solve mechanistic identification generally.

## Evidence and verification levels

1. **Stored evidence:** `python3 examples/confirmed_read_source.py` verifies the archived
   64-pair confirmation and checks agreement with the current analysis API. No package
   installation, downloads, models or GPU are needed.
2. **Optional input reconstruction:** the historical application scripts fetch pinned
   upstream files and rebuild directions using model weights. This requires separate
   dependencies and access to the weights; the original full verifier uses these files.
3. **Optional experiment replay:** the historical runners perform new model forwards.
   They are different from recomputing the stored result and are not run by the offline
   tests or CI. `reproduce_resid_mid8.py` also runs a model; it is not an offline checker.

Saved control fields are evidence about the recorded execution. Records-only verification
does not independently replay those tensor insertions. Four Q1 pairs have CPU32/CPU64
reference records; this is not an exhaustive numerical bound or an exact-truth reference.

## Changes to the reusable tool

- Tested self-anchors are refused: a measurement cannot count as its own prediction.
- Declared equivalent candidates must have identical tested prediction specifications;
  aliases share an adequacy decision and its multiplicity budget.
- All-tie comparisons report no evidence of a difference.
- Planning recommendations check statistical and numerical limits together. The ratio
  remains a heuristic, not power, a coverage bound or an identification guarantee.
- User-facing outputs distinguish nominal bootstrap intervals from the exact primary
  sign test; pairwise p-values require a declared error family when many are inspected.

These changes do not refit the model, change Q1's candidates or alter its frozen scorer.
The current method is cross-checked against that scorer's archived primary result.

## What has not become guaranteed

The general adequacy module uses nominal percentile-bootstrap intervals. A minimum sample
size alone does not ensure coverage, particularly for discrete or degenerate data. Use
a justified interval procedure for the intended population, or report a bounded relative
comparison rather than claiming adequate accuracy. The Q1 result uses its prespecified
single exact sign test and does not claim adequacy.

Q1 covers one model, site, direction and prompt generator. It concerns the constructed
patch operation, not a uniquely identified native semantic variable. Candidate completeness,
adequacy, other model families and safety transfer remain separate questions.

## Historical files

The Makelov application was copied from development commit
`6388102b4ee7fcc6e3025908316a9ddc816c4fe6`; the method starts from
`82e4d6ba511ab492edfa645cb8fc80dffa92a046`.
The [frozen-file inventory](release/FROZEN_FILES.json) lists the copied paths and SHA-256
digests. `python3 scripts/check_frozen_files.py` checks that none was changed.
The application's old README, runner labels, numerical field names, tests and older
experiments are retained as historical material. Start with the new records-only guide
rather than its mixed full-reproduction command block.

No private development git history, untracked drafts, model weights or fetched upstream
inputs are part of this public snapshot. The original upstream inputs are fetched at a
pinned revision rather than redistributed; the archived application documents its terms.

## Offline tests

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest -q -ra
```

The default suite covers the current method, release integrity, Q1 integration and the
records-only checker. The archived application suite is separate: several of its tests
require fetched inputs or TransformerLens. It is not silently represented as having
passed in a minimal environment.

Install `.[test-hooks]` and rerun the same command to include the optional tensor
tests. Without PyTorch, their skips are expected and visible, not counted as passes.
These tensor tests use fake models, not pretrained weights. Use the
[verification guide](applications/makelov-2311.17030/VERIFICATION.md) for fresh-environment
commands and all four result checks.
