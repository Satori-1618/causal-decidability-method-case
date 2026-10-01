# Independent implementation check

**Result: all reported decisions reproduce from the saved raw margins.**
The original execution failed a technical gate; its artifacts remain unchanged.
The documented continuation accounts for all 36 frozen tasks: 35 completed and
one technically invalid. The error family remains K=288, including the invalid
task. This is a separately implemented computational check in the same research
workflow, not an independent human replication.

[`verify_confirmation.py`](verify_confirmation.py) imports neither analyzer nor
runner. It reconstructs the confidence bounds by inverting a binomial
log-likelihood level set, separately from the frozen analyzer's KL function.
The machine-readable [verification record](results/verification_report.json)
contains every task's result and verified artifact hashes.

## What was checked

- The complete cohort against the released head table, development-seed
  exclusion, unchanged multiplicity, frozen source hashes and asset locks.
- All 512 distinct rotation families, Boolean labels, initial-symbol strata,
  absence of repeated or mixed orbits, and exclusion of both public input tables.
  All 39 cached-asset hash checks passed; no required cached asset was missing.
- Every saved forecast against its preceding endpoint margins, structural
  preflight result, receipt hash and stage ordering. Local receipts establish
  local provenance; they are not external timestamps proving preregistration.
- Every eligible-family decision, maximum prediction error over both hybrids
  and all three members, definite/possible hit, confidence bound and status.
- Saved operator controls and directly recomputed per-arm and per-contrast
  float32/float64 differences. The original carried artifacts are byte-identical.
- All stratum accuracies, margin effects and descriptive interaction summaries.

Seven verifier tests pass, covering closed-form confidence endpoints, the frozen
required hit counts, strict gap eligibility, numerical guard bands, a failing
valid member, altered forecasts and incorrect orbit labels.

## The result that survives this check

For the development checkpoint on fresh inputs, stage 2 has **183 eligible
families**. The within-symbol-position hypothesis H_W scores **183/183 definite
hits**, with simultaneous lower bound **0.95378666**; the symbol-mass hypothesis
H_T scores **0/183 possible hits**, with upper bound **0.04621334**. The former is
adequate and the latter excluded under the frozen 80% rule.

This concerns the bracket-routing subexperiment. In stage 1, H_R remains
**unresolved**, with lower bound 0.78157919; H_G is excluded. Thus this is not a
completed identification of the entire full-uniform intervention mechanism.

A transfer example with both separate stages resolved is `a9g0io1r`, head 1.
It has **165/165** definite H_R hits (lower 0.94887622) against **0/165** possible
H_G hits, followed by **136/136** definite H_W hits (lower 0.93831754) against
**0/136** possible H_T hits. This example was selected for illustration **after
the results**; its task and bounds were included in the original 36-task family.
Each stage concerns its own eligible population and endpoints. These two results
do not establish a composed H_W-to-U prediction within the same 0.25-nat tolerance.

All 35 transfer-head tasks remain in the denominator:

| Stage-2 status | H_W | H_T |
|---|---:|---:|
| Adequate | 7 | 0 |
| Excluded | 21 | 33 |
| Unresolved | 5 | 0 |
| Insufficient eligible families | 1 | 1 |
| Technical invalidity | 1 | 1 |

The cohort contains 34 transfer models, all two-layer models after the specified
seed-pair exclusion. Heads and crossed training seeds are not independent model
replications. In the focus task, invalid members drive eligibility; valid
members still enter every family-level hit criterion.

## Limits and reproduction

Saved tensor-control records were checked, but the tensors were not independently
rerun. The [failure diagnostic](results/gate_failure_diagnostic.json) attributes
the invalid task to one float32 strict-order comparison becoming a tie; that
diagnostic does **not** reinstate the task or relax its frozen gate. Higher
precision remains a reference, not exact truth. No claim of native algorithm
uniqueness or superiority over the same factorial design follows.

A clean export of the final artifact tree also passed archive restoration
(213 files), the frozen-decision analyzer and this independent verifier using
standard-library Python. That export had no model cache: cached-asset and
public-input-overlap checks were explicitly unavailable there, rather than
silently treated as passed. The full local audit above performed those checks.
The final relevant test run passed 31 follow-up tests, 13 original-application
tests and 15 shared preflight tests.

From the repository root, standard-library Python:

```bash
python3 -B -S applications/li-saphra-2507.06445/native_followup/verify_confirmation.py \
  --freeze applications/li-saphra-2507.06445/native_followup/frozen/confirmation_001/freeze.json \
  --run applications/li-saphra-2507.06445/native_followup/results/confirmation_001_continued \
  --report applications/li-saphra-2507.06445/native_followup/results/confirmation_report.json
```

Cached weights and public inputs are checked when present; the output explicitly
lists missing caches and whether public-input overlap was checked. The raw-margin
and report checks need no model execution. The original failed run must remain
alongside the continued run so its hashes and carried artifacts can be verified.
