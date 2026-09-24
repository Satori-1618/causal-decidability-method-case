# Notes on the Q1 record (added 2026-09-22)

The other files in this directory are the run and its score exactly as committed in
`6539ce5`, and none has changed since (see `PROVENANCE.md`). These notes came later.

- **The status label.** `summary.json` says `"status": "development_pilot_not_confirmation"`.
  The runner writes that label into every run, and the preregistration kept the runner
  unchanged since the pilot (§4), so the label came along. This run's status comes from
  the preregistration, frozen before the run, and from `score.json`.
- **The sign test's assumption.** §5 of the preregistration says the test "assumes only
  independent base pairs". In full: independent base pairs and, under the null, a
  probability of one half on average that an untied base pair favours either candidate.
  That probability need not be the same for every pair: if it varies, the two-sided test
  is conservative (Hoeffding 1956, *Ann. Math. Statist.* 27, 713–721, Theorem 5). There
  were no ties. The outcome is unchanged.
- **Rechecking.** `python3 scripts/check_read_source_q1.py` recomputes the judged result
  from `records.jsonl`, compares it with `score.json`, checks the pinned hashes and runs
  the verifier. It writes nothing.
