# Historical test failures: isolated reproduction

Audit date: 2026-09-29. This audit changes no historical tests, records, code, or tolerances.

The new benchmark's focused tests and circuit tests passed (25 tests). The full suite
reported three historical exact-JSON reproduction failures. Running only those historical
tests in a fresh Python process reproduced all three; no design-comparison test or module
was imported. A second check extracted **all input files directly from HEAD git objects**
and producer code from the commits recorded below, and executed each producer in a temporary
directory using `python -I -B`. It reproduced the same differences without importing any
working-tree application code.

Reference HEAD: `f0d0547a116fb78562bef6b754d70e9ebbc83263`.
The relevant historical tests, sources, and result directories have no working-tree
differences against that HEAD. Environment: `/opt/miniconda3/bin/python3`, Python 3.12.7,
NumPy 1.26.4, macOS 15.7.3 arm64.

| Historical result | Producer commit | Changed JSON leaves | Largest absolute difference |
|---|---|---:|---:|
| pilot | `3cdaffd` | 7 | `3.3306690738754696e-16` |
| pilot2 | `dec1f1d576ab405694c72e23e0f327f1a808a1fe` | 1 | `1.1102230246251565e-16` |
| split_A | `f822b0aaecbd11c6bf5308c3747ee349786f2dd3` | 1 | `1.1102230246251565e-16` |
| split_B | `fa83ffffa08368f2dddab668feea2a4ffd74211f` | 0 | 0 |

All changed leaves are floating-point diagnostics. Every other JSON leaf, including
gate decisions, selected conditions, sample-size decisions, counts, strings, and Boolean
flags, is unchanged. The new benchmark does not cause these failures: the archived code
and archived inputs produce them independently in this environment. The precise dependency
or platform operation responsible for the last-bit differences has **not** been isolated.

## Exact differences

JSON paths are relative to the stored `summary.json` or split decision file.

| Result / path | Stored | Recomputed |
|---|---:|---:|
| pilot / `n_rule.adequacy_power` | 0.8736186680853373 | 0.873618668085337 |
| pilot / `n_rule.exclusion_power` | 0.9706890375347821 | 0.9706890375347822 |
| pilot / `n_rule.table[0].exclusion_power` | 0.7429218768072825 | 0.7429218768072823 |
| pilot / `n_rule.table[1].adequacy_power` | 0.7671154608370332 | 0.7671154608370333 |
| pilot / `n_rule.table[2].adequacy_power` | 0.8736186680853373 | 0.873618668085337 |
| pilot / `n_rule.table[2].exclusion_power` | 0.9706890375347821 | 0.9706890375347822 |
| pilot / `n_rule.table[3].adequacy_power` | 0.9457085352518212 | 0.9457085352518213 |
| pilot2 / `audit_full_logits.max_abs_answer_mass_difference` | 1.1406891520238105e-6 | 1.1406891519127882e-6 |
| split_A / `gate_table.audit_full_logits.max_abs_answer_mass_difference` | 4.601268229764699e-7 | 4.601268228654476e-7 |

## Reproduction command

Run from the repository root; these tests themselves extract the recorded producer code
from git history and run it in subprocesses:

```bash
/opt/miniconda3/bin/python3 -m pytest -q \
  tests/test_mixing_pilot2_results.py::PilotTwoResultTests::test_summary_recomputes_with_the_recorded_code \
  tests/test_mixing_pilot_results.py::PilotOneResultTests::test_summary_recomputes_with_the_protocol_v1_code \
  tests/test_mixing_split_results.py::SplitResultTests::test_decision_recomputes_with_the_recorded_code
```

Observed: three failures; split B reproduces exactly. For the additional isolated check,
the input directories `results/pilot`, `pilot2`, `split_A`, and `split_B` were reconstructed
file by file using `git show HEAD:<path>`. The producer lists are `V1_FILES`, `SUMMARY_CODE`,
and `DECISION_CODE` in the corresponding tests. A recursive comparison then enumerated
every unequal leaf in the table above.

These remain failing exact-reproduction tests. This audit does not convert them to passes,
relax any threshold, or claim that the complete test suite is green. No model inference or
reserved structural evaluation was performed.
