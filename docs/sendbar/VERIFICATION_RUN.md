# Verification of the local candidate (plan Teil D, step 2)

Internal working document, 2026-10-22.

**Candidate:** branch `claude/brave-thompson-9vr9bn`, commit `d070a22` (and the
documentation-only commit that adds this file). It was checked in a fresh local clone
of the candidate and a fresh virtual environment:
- Linux x86_64, Python 3.13.16
- NumPy 2.4.6, SciPy 1.15.3
- this repository installed with `pip install --no-deps -e .`
- no PyTorch, JAX or Transformers

| Command (repository root) | Result |
|---|---|
| `python scripts/verify_release.py` | **All 13 expected checks passed**: the 8 earlier checks, 4 Li/Saphra checks and the claim runner. The list of checks that ran matched the expected list exactly. |
| `python scripts/check_claims.py --network` | **48 claims in 5 files**: 27 recomputed, 7 read from summaries that a recomputing checker verified first, 1 read from a stored summary, 13 listed as logged. 0 failed, 0 skipped. |
| `python scripts/check_claims.py` (offline, as run inside the release verifier) | Same, except that 3 CausaLab download checks are reported as skipped, not passed. |
| Link and anchor check over all 120 tracked Markdown files | 0 broken relative links or heading anchors. The checker reported both errors in a deliberately broken test file. |
| `git status --porcelain` after all runs | Clean. Only ignored build metadata (`*.egg-info`) was created. |

GitHub Actions on the working branch passed for every pushed commit up to `8d1e629`,
including the full pytest suite and the tensor-test job. The run for `d070a22` was
queued when this note was written.

**Not checked here:**
- Fresh installation with Python 3.9–3.11. CI covers 3.11.
- The full pytest suite locally; CI runs it.
- Any model run.
- The original papers' wording (arXiv is unreachable from this environment).
- The website (local patch only).
- TaxiGPT and Causal Parrots, whose sources are private.
