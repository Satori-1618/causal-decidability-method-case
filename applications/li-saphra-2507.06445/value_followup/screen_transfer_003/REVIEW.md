# Independent review before execution

**Design review: PASS WITH AMENDMENT. Final pre-run review: PENDING.**
The user supplied the review and approved raising both quotas to 64, with the
1,024 budget and thresholds unchanged. See [Amendment 001](AMENDMENT_001.md).
No six-head native margin, screen assignment or intervention outcome has been
measured for this round. There is no execution approval.

Design reviewer: user-supplied review text; identity not supplied. Recorded on
3 October 2026. Its historical feasibility check used existing cohort native
margins only and did not change a cutoff. Additional agent checks are preparation
QA, not external human peer review. Final reviewer and decision: pending.

Check the public freeze, not just a narrative summary:

- [ ] The six heads and their prior selection are explicit; no untouched-cohort claim.
- [ ] The signed model-range normalization is valid for the pinned architecture.
      The old cutoff supplies the only calibration; target outcomes supply none.
- [ ] Exactly 1,024 candidate slots/head; missing either 64-family quota gives
      insufficient yield, stays among six, and cannot trigger more sampling.
- [ ] The two-anchor operator, numerical gates and 0.202-nat outcome are unchanged.
      Verification handles both 16- and 32-dimensional head values.
- [ ] The 24 simultaneous bounds, strong four-of-six target and limited power for
      moderate effects are acceptable. Directional results cannot masquerade as
      meeting the stronger target.
- [ ] The random yield is a stratified plug-in estimate. Fixed, idealized and
      validation costs are distinct; all baseline and rejected-arm costs appear.
- [ ] Historical exclusions cover full strings and both causal prefix lengths,
      including the entire previous native-screen pool and prepared donor grids.
- [ ] Averaged anchors belong to a separately frozen mechanism round; their
      calibration inputs are not also targets. No automatic run follows screen success.
- [ ] Proposed implementation and prepared-input manifest are reviewed and pushed
      before any new scores; the runner fails closed on missing review/source locks.

A review rejection or requested change produces a recorded amendment, not a
silent plan edit. The checklist above must be checked against the final runner
and prepared inputs. A design-review PASS does not approve execution; the
separate `EXECUTION_RELEASE.json` remains pending.

## Implementation handoff — preparation QA, not execution approval

The amendment was published as `1d10892` before these inputs were generated.
All six pools are now prepared: 6,144 candidate draws and 768 donor templates.
Their hashes and duplicate/prefix-reuse counts are in `inputs/preparation.json`.
No native scores or intervention outcomes have been generated for this round.

Preparation tests check deterministic regeneration, the complete historical ban
union, every donor constraint, tampering and overwrite protection. Runner tests
use fake objects only: they check the release block before model import, fixed
selection, missing quotas, dtype and signed-contrast failures, and partial cost
accounting. The separate record audit tests both head widths and corrupt records.
None of these tests is an empirical six-head runtime result.

Final review should inspect the current published commit, run the model-free
commands in the README, and check source/input bindings and failure paths.
Any eventual release must name that reviewed commit and record explicit execution
authorization. Scientific code or inputs changed after it require renewed review.

## Final pre-run review and execution release — 3 October 2026

**PASS for `23f7f7f6c6e4da8de391b131d7cb7c770ca3a5af`.** The user supplied
an independent final review and explicitly authorized the agent to publish the
release and execute all six heads exactly once. Reviewer identity is not supplied;
this is not represented as named external human peer review. The previous pending
sections above preserve the handoff history. The current release is recorded in
`EXECUTION_RELEASE.json`; the run must record its public commit.

The reviewer verified the amendment, cutoffs, unscored inputs, freshness,
fail-closed runner, 74 tests and 60 source hashes. The full 74-test suite was
rerun successfully on the still-pending reviewed state before this release.
Three `test_freeze.py` tests specifically assert that old pending state; they
remain unmodified as part of the reviewed code. Reproduce all 74 tests at the
reviewed commit. At a released checkout use the plan verifier, input validator,
runner validator and the other four test modules; do not treat intentionally
released authorization as a failure of the frozen pending-state test fixtures.

### Disclosed literal-protocol exception: shorter prefixes

§3 literally excludes both historical prefix lengths. The prepared generator
checks a donor at its measured position: all length-20 donors have fresh
20-symbol prefixes and all length-28 donors have fresh 28-symbol prefixes.
Some position-28 donors nevertheless share a historical 20-symbol beginning.
The final review disclosed and accepted this exception before execution. No
inputs are replaced and no exclusion rule is relaxed after observing outcomes.

The reviewer quoted 2–7 donors/head. A read-only check against the committed
historical union instead gives the following exact counts:

| Head | All prepared position-28 donor entries with old 20-prefix | Scheduled `pos_28_0` entries with old 20-prefix |
|---|---:|---:|
| 50h5xlod/H1 | 4 | 2 |
| 8a65u5l6/H1 | 5 | 1 |
| dsibxabs/H4 | 4 | 0 |
| qtda56zd/H2 | 3 | 1 |
| we9o801g/H2 | 11 | 1 |
| wfwpq5xc/H1 | 5 | 4 |

The scheduled column totals nine if all quotas fill. No complete new candidate
or donor string, and no donor prefix at its measurement position, overlaps the
historical exclusions. Causal attention at position 28 depends on the complete
prefix through that position, so a shared shorter beginning is not a repeated
measured activation. This does not establish training-data independence.

This release changes authorization metadata and documentation only. Scientific
code, prepared inputs, thresholds and the six-head cohort are unchanged.
