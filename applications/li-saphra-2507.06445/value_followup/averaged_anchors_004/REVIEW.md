# Independent pre-run review — pending

## Review received and Amendment 002

On 2026-10-04 the user supplied a review of public commit `b7b577c`:
**integrity PASS; release withheld pending Amendment 002**. The supplied text
does not independently identify its reviewer; attribution is to that user-
provided review, not to a newly commissioned human or external audit here.

The [amendment](AMENDMENT_002.md) adds descriptive `C_cell` forecasts, fitted
only on calibration prefixes, and a complete four-prefix variation breakdown.
It corrects the overinterpretation of predictor failures and records why the
historical 80%/81% estimates are not direct forecasts of this design.
No original inputs, rules, inferential tests or original software tests change.

**Final review of the amended public commit is pending.** In addition to the
original checklist, check:

- [ ] C_cell forecasts use no target margins and their hash is recorded before targets.
- [ ] Four-prefix means are descriptive only and include interaction and role means.
- [ ] Supplemental precision uncertainty cannot change the primary start or outcome.
- [ ] Input bytes, original analysis, planning and original tests match `b7b577c`.
- [ ] Execution authorization remains false; no round-004 outcomes exist.

### Amendment preparation checks (not final independent approval)

The 53 original tests and 14 new descriptive/integration tests pass (67 total).
The source verifier binds 88 files including its own lock; the original planning
table and regenerated input checks pass. Byte comparison to `b7b577c` confirms
unchanged inputs, primary analysis/planning, original tests and pending release.
A separate preparation agent reviewed forecast ordering, target leakage,
interpretations and the interaction arithmetic without finding a blocker.
These are internal implementation checks, not authorization to execute.

## Original freeze checklist (retained)

**Status: no execution release; no new model measurements.** Review the public
commit containing this entire directory, including inputs and executable code.
The requested action ends at this freeze. A successful preparation check is not
an independent execution approval.

## The bounded decision to review

Can averaged calibration predict eight fresh-prefix effects on the selected
`a9g0io1r` layer2/head1, and does replica averaging improve over the same four-cell
single-prefix calibration? Target-pair differences separately test whether one
common cell prediction is already impossible. This is prospective development,
not a claim of unique representation, exact balance or natural mechanism use.

- [ ] **Architecture:** model has four attention heads, width16 per head; the
      existing reference runtime and checkpoint hash agree. Internal code review
      caught and corrected a draft2/32 mismatch before freeze, without inference.
- [ ] **Sampling:**2,048 recipients, first256 accepted by the unchanged signed
      margin<8 screen; fixed seeds26100341/26100342, no replenishment.
- [ ] **Partition change:** the old development +2/position20 pool has only nine
      fresh prefixes. The new input-only `confirmation` hash partition is explicit;
      this does not upgrade the scientific stage or open old sealed outcomes.
- [ ] **Freshness:** all historical prepared strings are included, both20/28
      prefix bans are enforced on every donor, and calibration/target roles are
      globally disjoint by a frozen first20 hash. Repeats across families are
      retained. The manifest reports eleven new native-candidate/donor first20
      overlaps; these are not calibration-target overlap or historical reuse.
- [ ] **Calibration:** four means from eight distinct-prefix transfers; the
      `B_single4` comparator retains four-cell coverage. The old two-diagonal
      account is descriptive. All four forecasts saved before any target.
- [ ] **Start rule:**>=128/256 families with new averaged forecast gap>.202,
      matching dtype labels, before any target. If passed, measure all256families.
      The old diagonal gap is not substituted. No target-based eligibility.
- [ ] **Error family:**90%candidate adequacy,4CPtails at.01 plus one paired exact
      test at.01. The latter tests improvement>0, not improvement>=10pp.
      Within-cell witnesses and the legacy comparator remain descriptive.
- [ ] **Precision/power:** numerical .001 and scientific .1 are distinct. The
      power table shows alternative discordance patterns and limited90%-adequacy
      power near the boundary. Start and final-test powers are not multiplied.
- [ ] **Witness:** same-cell target gap>.202 can exclude a common cell prediction
      locally. A missing witness does not establish invariance; a boundary straddle
      is reported as unresolved for the diagnostic, not deleted from analysis.
- [ ] **Controls/stops:** inherited operator controls, dtype checks, complete raw
      snapshots and forward costs; whole-round stop on required technical failure.
      No retries, head search, tolerance changes or automatic confirmation.
- [ ] **Implementation:** fake-only tests and model-free checks pass. Actual
      integration of this wrapper has not been exercised; review source and data
      binding rather than mistaking unit tests for a neural result.
- [ ] **Scope:** passing the future planning gate is feasibility, not population
      adequacy. Current balance, sign and distance from the fixed minimum remain
      aliases; a within-cell effect does not identify its history feature.

## Preparation QA

Separate agents handled the sampling/freshness implementation, pure analysis and
power, and the execution wrapper; the parent reconciled the protocol and release.
These checks are internal AI-assisted preparation, not the independent review
requested by the user. No weights were deserialized and no model was run.
Input generation, synthetic tests and hypothetical probability calculations do
not produce neural outcomes.

The full freeze will be pushed before review. Record reviewer, reviewed commit,
decision and explicit user authorization in `EXECUTION_RELEASE.json` only after
that review. Any subsequent scientific/source change requires an amendment and
renewed review before measurement. Do not silently modify this freeze.
