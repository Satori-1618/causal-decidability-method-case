# Execution amendment after a strict control failure

Freeze `92860b6` is retained unchanged. The first execution completed the focus
task, then stopped on `0nbysgqs`, head 2, because gate-only changed the
ordering-based sign classification. The failed run and its exception remain in
`results/confirmation_001/`. No threshold, hypothesis, input or statistical
rule is changed in response.

The protocol blocks the intended claim for a failed technical control. The
original execution loop also stopped the entire cohort, preventing accounting
for the remaining predefined tasks. A separate continuation runner now finishes
that finite cohort while preserving this failure:

- Reuse the completed focus task byte-for-byte; do not count it twice.
- Carry the failed task as **technical_invalidity**, without a scientific
  candidate decision. Do not rerun it with a looser criterion or replace it.
- Run every remaining frozen task through the original unchanged `run_task`.
  Retain any additional failure explicitly and continue accounting for tasks.
- Keep all 36 tasks in the report and the same 288-bound multiplicity family.
  Do not recalculate the correction from the successful subset.
- The reporting adapter calls the original frozen decision function only on
  technically complete tasks. Invalid tasks are not scientific nulls or
  exclusions of a hypothesis. The original complete-run analyzer remains
  unchanged and continues to reject incomplete manifests.

This amendment repairs execution and failure accounting, not the failed
intervention control. Its local commit occurs after one completed task and a
technical failure; it is not presented as wholly pre-outcome preregistration.
