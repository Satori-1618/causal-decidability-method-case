# Verification of the local candidate (plan Teil D, step 2)

Internal working document, 2026-10-22.

**Candidate:** branch `claude/brave-thompson-9vr9bn`, commit `b602947` (and the
documentation-only commit that updates this file). It includes the fixes from the final
repository review ([REVIEW_FINDINGS.md](REVIEW_FINDINGS.md)). It was checked in a fresh
local clone of the candidate and a virtual environment:
- Linux x86_64, Python 3.13.16
- NumPy 2.4.6, SciPy 1.15.3
- this repository installed with `pip install --no-deps -e .`
- no PyTorch, JAX or Transformers

| Command (repository root) | Result |
|---|---|
| `python scripts/verify_release.py` | **All 13 expected checks passed**: the 8 earlier checks, 4 Li/Saphra checks and the claim runner. The list of checks that ran matched the expected list exactly. |
| `python scripts/check_claims.py --network` | **60 claims in 5 files**: 37 recomputed, 7 read from summaries that a recomputing checker verified first, 1 read from a stored summary, 15 listed as source entries. 0 failed, 0 skipped. |
| `python scripts/check_claims.py` (offline, as run inside the release verifier) | Same, except that 3 CausaLab download checks are reported as skipped, not passed. |
| Same runner in a `--depth 1` clone | Makelov and Mixing stop with the message that a full Git clone is needed; Tracr, Li/Saphra and CausaLab pass. `verify_release.py` refuses to start, with the same message. A `git archive` export (no `.git`) behaves the same. |
| `python -m unittest tests.test_describe_post_hoc` | 7 tests passed. |
| Link and anchor check over all 123 tracked Markdown files | 0 broken relative links or heading anchors. |
| `git status --porcelain` after all runs | Clean. Only ignored build metadata (`*.egg-info`) was created. |

GitHub Actions on the working branch passed for every pushed commit up to `8d1e629`,
including the full pytest suite and the tensor-test job. Runs for later commits were not
checked when this note was written.

**Not checked here:**
- Fresh installation with Python 3.9–3.11. CI covers 3.11.
- The full pytest suite locally; CI runs it.
- Any model run.
- The original papers' wording (arXiv is unreachable from this environment).
- The website (local patch only).
- TaxiGPT and Causal Parrots, whose sources are private.
