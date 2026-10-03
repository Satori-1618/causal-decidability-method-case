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
