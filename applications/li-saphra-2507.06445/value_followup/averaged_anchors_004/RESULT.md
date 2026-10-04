# Averaged anchors 004: insufficient balance prediction, asymmetric prefix variation

**The single authorized run completed. Both frozen prediction accounts are
excluded at the declared 90% adequacy target; averaging benefit is unresolved.**
The descriptive extension shows much greater prefix variation at balance -2
than at +2. It does not identify the extra history feature or establish exact
prefix independence at +2.

Head `a9g0io1r`, layer 2 / head 1; fresh prepared inputs, one selected head.
Reviewed freeze: `80085bb`. Public execution release: `5cb6f77`, recorded in the
run manifest. No frozen input, threshold or producer-analysis/code change followed
that review. Separate reporting and audit code was added after the run.

## The complete frozen decisions

All **2,048** recipients were screened: **886** passed, and the first **256**
were selected. All 256 received eight calibration transfers. Averaged balance
and position forecasts separated in **151/256**, exceeding the frozen minimum
128; therefore **all 256** received eight fresh target transfers. There were no
technical failures, replacements, skipped target families or retries.

| Account, on the 151 separating families | Definite hits | Possible hits | Simultaneous adequacy bounds | Decision at 90% |
|---|---:|---:|---:|---|
| Averaged balance, B_avg | 123/151 (81.46%) | 125/151 (82.78%) | [72.97%, 89.31%] | Excluded |
| Averaged position, P_avg | 0/151 | 0/151 | [0%, 3.00%] | Excluded |

A family must predict **every one of its eight targets** within the fixed
tolerance. Definite/possible limits are 0.099/0.101 nat. The bounds use four
one-sided Clopper–Pearson tails at alpha .01, alongside the paired test at .01;
the prespecified familywise allocation remains at most .05. Exclusion concerns
these two calibrated predictive rules in the declared separating population,
not every possible balance or position representation.

The primary averaging comparison uses all 256 families and the matched
four-cell single-prefix comparator: **8 robust gains, 7 robust losses;
one-sided exact p = 0.50**. Averaging improvement is not established. This is not
evidence of equivalence or proof that averaging never helps.
Robust gains require B_avg definite and B_single4 not possible; losses require
B_avg not definite and B_single4 possible. Thus these counts are not obtained
by subtracting the two ordinary definite-hit totals.

The fixed two-target, same-cell witness yields **0 definite, 0 possible, 0
numerically unresolved families**. The largest target-pair difference is
**0.199846 nat**, below 0.202. This diagnostic therefore does not exclude a
common cell prediction; its silence does not establish invariance.

**No candidate meets the future-confirmation planning gate.** No confirmation
or further development run was started.

## What the descriptive amendment adds

The cell predictor C_cell uses only the two calibration prefixes at the target's
balance and position. Its forecasts were hashed before any targets:

| Population | B_avg definite / possible | C_cell definite / possible |
|---|---:|---:|
| All 256 families | 228 / 230 | 235 / 236 |
| 151 separating families | 123 / 125 | 130 / 131 |

For definite hits over all families, **220 fit both**, **8 fit only B_avg**,
**15 fit only C_cell**, and **13 fit neither**. C_cell has no preallocated
adequacy test: 235/256 is a descriptive fit rate, not a confirmed 90% result.
Retaining individual cells helps some predictions, but neither this difference
nor both predictors failing identifies position or history as the mechanism.

Each cell's sample SD uses **four prefixes: two calibration plus two targets**.
The following summaries weight each of the 256 families equally; all values are
in nats. Mean SD is an additional descriptive summary of the frozen per-cell SDs.

| Balance | Position | Mean SD | Median SD | IQR | Largest SD |
|---|---:|---:|---:|---:|---:|
| -2 | 20 | 0.028104 | 0.025125 | 0.019132 | 0.086553 |
| -2 | 28 | 0.027640 | 0.025003 | 0.021403 | 0.099550 |
| +2 | 20 | 0.000180 | 0.000153 | 0.000151 | 0.000527 |
| +2 | 28 | 0.000130 | 0.000116 | 0.000092 | 0.000379 |

The asymmetric spread persists on fresh prefixes. These are deterministic
differences in **patched output margins**, not measurement-noise estimates or
direct measurements of the head's semantic contents. Variation at +2 is small,
not exactly zero. Identifying a particular history feature remains open.

For scale, the median signed balance contrast is **-0.439918 nat**, the median
position contrast **+0.012861**, and the median balance-by-position interaction
**-0.034063**. Their full distributions and within-cell spread are preserved;
no component is interpreted as a percentage of the causal mechanism.

The all-four-prefix range reaches **0.218073 nat** in one cell. This differs from
the frozen **two-target** diagnostic and is not substituted into its decision.
An additional post-run inspection finds every family's worst B_avg error on
the -2 side; this is a descriptive observation, not an extra confirmed test.

## Integrity, limitations and records

- The release was published and its remote hash checked before the first
  measurement. All original and cell forecasts were recorded before targets.
  The supplied review and its acceptance of fail-closed add-on behavior are
  preserved in [REVIEW.md](REVIEW.md).
- All recorded operator, identity and dtype controls passed. Identity errors
  were zero; maximum primary prediction-error discrepancy was **0.000005784
  nat**, below 0.001. C_cell had no numerical-uncertainty classifications.
- Exactly **36,864 sequence-forwards**, attempted = completed, in **24.11 s**
  on the recorded CPU environment. Timing excludes preparation and review.
- Two separate agents recomputed statistics and audited provenance without
  model inference. Raw snapshots support independent reconstruction of the
  delivered node for **all 8,192 dtype-specific transfer records**. Full
  attention/off-target preservation remains a recorded producer control: the
  complete tensors needed to independently repeat those checks are not saved.
- This remains prospective development on an adaptively selected head and a
  restricted donor generator. The `confirmation` input-hash partition does not
  make it a general mechanism confirmation. Balance, sign and distance from the
  fixed minimum remain confounded. Role means are retained because calibration
  and target prefixes come from distinct hash-defined pools.

[Primary analysis](results/analysis_summary.json) ·
[Descriptive analysis](results/descriptive_summary.json) ·
[Independent statistical recount](results/independent_statistics.json) ·
[Independent provenance audit](results/independent_provenance.json) ·
[Verification and reproduction](VERIFICATION.md) ·
[Raw archive](results/run_001.tar.gz) · [File hashes](results/raw_index.json)

**Status: one authorized execution complete; neither primary account adequate;
no supported averaging gain; descriptive prefix asymmetry recorded; no further run.**
