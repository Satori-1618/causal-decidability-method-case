# Public method-case snapshot

This release contains the general method and the Makelov Q1 comparison in one checkout.
It is a new publication of existing evidence, not a new experiment or a retroactive
public preregistration. It makes no claim to invent model discrimination or to solve
mechanistic identification generally.

The added [iterative follow-up](docs/ITERATIVE_IDENTIFICATION.md) is a teaching demo and
draft design for a second application. It changes none of the archived evidence and
reports no new model experiment. Its illustrative error bounds and tolerance are not
approved thresholds for a real run.

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
python3 -m pytest
```

The default suite covers the current method, release integrity, Q1 integration and the
records-only checker. The archived application suite is separate: several of its tests
require fetched inputs or TransformerLens. It is not silently represented as having
passed in a minimal environment.
