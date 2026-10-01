# Release checks — 1 October 2026

These are software and stored-evidence checks for implementation commit `79eba42`.
They are not fresh scientific confirmation or a test of general methodological value.
The later documentation commit does not change the checked implementation.

## Clean-clone check

A separate full Git clone was made without shared object hardlinks. A new virtual
environment installed `.[test,verify]`; it had no PyTorch, Transformers, model weights
or application caches. Local platform: macOS arm64, Python 3.10.10, NumPy 2.2.6,
SciPy 1.15.3, pytest 9.1.1.

| Check | Observed result |
|---|---|
| `python scripts/verify_release.py` | All eight checks passed; working tree remained clean. |
| `python -m pytest -q -ra` | **445 passed, 40 skipped; 603 subtests passed.** |
| Three standard-library quick-start commands | Passed. |
| Local links in the entry documents | All targets exist. |
| Archived Makelov file inventory | All 62 files unchanged. |

The 40 skips are optional tensor/model-library tests or comparisons requiring the
separate upstream checkout. They are not counted as passes. The full Tracr application
suite is also separate; its compiled model is not rerun by the release verifier.

## Cross-version checks

- Python 3.11.15, NumPy 1.26.4, PyTorch 2.5.1, Transformers 4.57.3:
  the final preflight and three Goodfire development-replay test modules pass
  (**29 tests, 66 subtests**). The optional design-comparison module also passes
  (**17 tests**); it uses constructed tensor models, not pretrained weights.
- NumPy 2.4.6: the final Goodfire split-reproduction tests pass
  (**6 tests, 18 subtests**).
- Python 3.10.10, NumPy 2.2.6: the final three Goodfire development-replay modules
  pass (**14 tests, 18 subtests**), as does the full clean-clone suite above.

The clean installation exposed an unguarded optional PyTorch import and exact-float
comparisons in historical development tests. The optional import now skips explicitly.
Recomputed planning powers differed by at most `2.3e-13` across the checked Python
runtimes; two full-logit diagnostics differed by at most `3.6e-15` across the checked
NumPy versions. Test tolerances are limited to these named fields: `1e-12` for planning
powers, `1e-14` for the diagnostics. Counts, thresholds, gate decisions, statuses,
hashes and every other field remain exact. Regression tests reject material changes.
No archived record, scientific threshold or frozen producer was edited.

The initial Linux CI run passed all 445 analysis tests but failed Tracr's exact
population-summary comparison with automatically selected SciPy 1.17.1. The
verification extra now pins **SciPy 1.15.3**, the version in Tracr's original
frozen requirements, rather than assuming that newer tail-probability routines
reproduce every stored floating-point digit. The historical verifier is unchanged.

## What the combined verifier checks

It invokes the existing verifiers for Makelov Q1, rounds 2 and 3A, the round-3B
qualification stop, Goodfire's concentration confirmation, and Tracr's saved tensors
and family decisions. It also checks the original frozen-file inventory and recomputes
category groups from the stored Goodfire MCQA prediction table. That last step does not
re-download or independently re-derive the upstream table; the MCQA adapter's separate
command performs that source-hash-checked reconstruction.

The Tracr verifier runs against a temporary copy because its historical implementation
writes a verification report. No tracked artifact is rewritten. Full Git history is
required for checks that recover the original producer or freeze commit.

## Reproduce

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,verify]'
python scripts/verify_release.py
python -m pytest -q -ra
```

Installing dependencies uses the package index. The checks themselves need no network,
GPU or pretrained model. CI is configured to repeat records, analysis and optional
tensor checks; local results above do not claim that a hosted CI run has already passed.
