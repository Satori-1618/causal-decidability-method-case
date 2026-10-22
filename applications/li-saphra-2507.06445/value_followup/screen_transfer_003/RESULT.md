# Screen-transfer 003: prospective enrichment on all six heads

**Primary target met: 6/6 heads; 4/6 required.** A rule based only on the
recipient’s native output, calibrated on an earlier head and frozen before this
run, selected fresh families whose two anchor interventions were much more likely
to separate. All six fixed heads completed; there were no technical failures,
quota shortfalls, replacements or neural retries.

This is evidence for a case-selection component of the method. It does not
identify a mechanism or establish that screening saves computation.

## Every prespecified head

Each entry is a separating family count out of 64. “Separating” means the
two patched output margins differ by more than 0.202 nat. Intervals are for
accepted minus rejected separation rates, in percentage points (pp).

| Model / head | Accepted | Rejected | Difference (pp) | Simultaneous interval (pp) | Strong target |
|---|---:|---:|---:|---:|---|
| 50h5xlod / H1 | 64/64 | 9/64 | 85.94 | [60.43, 95.54] | PASS |
| 8a65u5l6 / H1 | 62/64 | 8/64 | 84.38 | [56.43, 96.29] | PASS |
| dsibxabs / H4 | 64/64 | 6/64 | 90.62 | [66.46, 97.94] | PASS |
| qtda56zd / H2 | 59/64 | 14/64 | 70.31 | [38.20, 89.24] | PASS |
| we9o801g / H2 | 61/64 | 11/64 | 78.12 | [48.19, 93.30] | PASS |
| wfwpq5xc / H1 | 64/64 | 1/64 | 98.44 | [78.39, 100.00] | PASS |

The frozen rule required at least four lower bounds strictly above **+25 pp**.
All six exceed it; even the smallest is +38.20 pp. Bounds use 24 one-sided
Clopper–Pearson limits at alpha=0.05/24 and have at least 95% simultaneous
coverage. The recipient–donor family is the unit; heads and dtypes are not pooled.

## Yield improves; this fixed schedule does not demonstrate savings

All 1,024 native candidates per head were screened before any intervention.
The table distinguishes observed accepted-case yield from the random-selection
yield estimated by weighting accepted/rejected yields by the pool acceptance
rate. No random-selection intervention arm was measured.

| Model / head | Pool accepted / 1,024 | Accepted yield | Estimated random yield | Fixed-policy cost | Estimated random cost |
|---|---:|---:|---:|---:|---:|
| 50h5xlod / H1 | 296 | 100.00% | 38.90% | 48.00 | 46.27 |
| 8a65u5l6 / H1 | 460 | 96.88% | 50.40% | 49.55 | 35.71 |
| dsibxabs / H4 | 330 | 100.00% | 38.58% | 48.00 | 46.66 |
| qtda56zd / H2 | 539 | 92.19% | 58.89% | 52.07 | 30.57 |
| we9o801g / H2 | 541 | 95.31% | 58.46% | 50.36 | 30.79 |
| wfwpq5xc / H1 | 423 | 100.00% | 42.23% | 48.00 | 42.63 |

Costs are sequence-forwards per separating family, including both dtypes.
The fixed policy pays for the entire 1,024-candidate pool but uses only the first
64 accepted families: `(1024×2 + 64×16)/accepted_hits`. Random cost is the
descriptive estimate `18/estimated_random_yield`. **The fixed policy is more
expensive on all six heads under these estimates.** The ideal streaming
calculation in the JSON was not executed and is not a demonstrated saving.

Actual validation, including rejected families: **24,576 completed sequence-forwards**,
4,096 per head, in **13.24 seconds** on this CPU environment.
There were no unaccounted attempted forwards. This timing is specific to these
small models and excludes input preparation, review and report verification.

## Integrity and disclosed exceptions

- The reviewed implementation/input freeze was `23f7f7f`; the authorized release
  `4dd5e5c` was pushed and its remote hash checked before starting. The run
  manifest records that release commit. All six native-screen receipts precede
  every anchor intervention. Original historical runs retain their earlier
  provenance limitations.
- Every full candidate/donor string and every donor prefix at its measured
  position was fresh relative to the declared historical exclusions. Final
  review accepted a literal wording exception: **nine measured position-28
  donors share a historical 20-token beginning**. Their complete 28-token
  prefixes are fresh. Per-head counts are 2/1/0/1/1/4; all were retained.
  This is input freshness, not proof of training-data independence.
- Maximum fp32/fp64 native-margin discrepancy was **0.00003840 nat**; maximum
  signed anchor-contrast discrepancy was **0.00002027 nat**, both below 0.001.
  All screen labels and all separation decisions agreed across precisions.
- The frozen report verifier initially stopped on a metadata bug: it expected
  nested review status `approved`, while the frozen execution gate requires
  `PASS`. A [small compatibility wrapper](reporting/analyze_release_compat.py)
  checks the full release, pins the original source hashes and replaces only
  that literal in memory. Original code, release records and raw data remain
  unchanged. The report records original, corrected-source and wrapper hashes.
  This is a disclosed post-run analysis correction; no scientific rule changed
  and no model was rerun.

## What this establishes—and what remains open

The screen prospectively enriched anchor separability on fresh families in
**this fixed six-head cohort**, beyond the original calibration head. The heads
were previously selected for another intervention result; they are not a random
sample of heads or untouched models. Only three initialization seeds and five
initialization/shuffle pairs are represented, on one small Dyck-task architecture.
Sampling was with replacement; donor-prefix reuse within and between heads is
reported in the [input manifest](inputs/preparation.json). The intervals concern
family draws from this declared generator, not independently trained models.

Anchor separability is a prerequisite for the later candidate test. It is not
candidate adequacy: this run did not test whether balance, position or prefix
history explains the transferred values. It also does not establish saturation
as the reason for the screening effect. The separately specified mechanistic
round remains unexecuted.

[Canonical analysis](results/report.json) · [Verification and reproduction](VERIFICATION.md) ·
[Protocol](PROTOCOL.md) · [Final review and prefix exception](REVIEW.md) ·
[Raw archive](results/run_001.tar.gz) · [Archive/file hashes](results/raw_index.json)

**Status: one authorized run completed; primary screening target met; no further run started.**
