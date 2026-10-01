# Independent audit of development_001

**Status: development result verified; the proposed superiority target is not established.**

This audit recalculated the selected-cell means and candidate compatibility directly
from `public_cases.jsonl`, `measurements.jsonl`, `private_labels.jsonl`, and
`predictions.jsonl`. It did not call the producer's summary function or classifier.
Clopper–Pearson bounds were independently checked with SciPy's beta quantiles. No
reserved graph family was executed or evaluated.

## Reproduced results

There are 512 independent cases: 268 `linear`, 244 `relu_offset`; 436 cases are in the
declared candidate set and 76 use `external_gain`. The in-set truth counts are channel A
86, channel B 75, balanced 57, joint 76, context switch 73, and inverse switch 69.

All 12,288 decision records reproduce directly from their selected measurements and
radii. Full-menu equivalence classes agree with exact prediction signatures. The
channel-A alias is preserved in every decision. Every equal-budget primary policy
consumes 56 observations; the full-menu reference consumes 128.

| Policy | Exact class / 436 | Truth excluded / 436 | Simultaneous exclusion upper bound | OOS rejected / 76 | False single class in OOS / 76 |
|---|---:|---:|---:|---:|---:|
| maximin | 398 | 1 | 1.5601% | 70 | 2 |
| mean_pairwise | 391 | 1 | 1.5601% | 70 | 2 |
| balanced_split | 332 | 2 | 1.9657% | 64 | 4 |
| random | 263 | 0 | 1.0920% | 68 | 1 |
| full_only | 0 | 0 | 1.0920% | 64 | 0 |
| full_menu | 399 | 4 | 2.6975% | 71 | 1 |

The bounds are one-sided Clopper–Pearson with alpha = 0.05/6, simultaneously covering
the six reported primary policy rates. There are zero false single-class outputs on
**in-set** cases. That does not extend to out-of-set cases: two maximin outputs retain
one false class. The original report's `False single class` column has an implicit
in-set denominator; this table makes the scope explicit without modifying the frozen
report.

The primary paired comparison is 7 maximin wins, 0 losses, and 429 ties against
`mean_pairwise`: **+1.6055 percentage points**, with the declared simultaneous conservative
95% interval **[-13.2137, +16.4247] pp**. The observed advantage is below the proposed
5 pp practical target; the interval neither establishes that target nor excludes it.
The target requiring an upper exclusion bound at most 1% is also unmet.

Against `balanced_split`, the corresponding counts are 67/1/368 and the advantage is
15.1376 pp; against random they are 136/1/299 and 30.9633 pp. These remain secondary,
developmental comparisons, not substitutes for the primary comparator.

The secondary two-extra-cell budget gives 267/436 exact classes for maximin versus
196/436 for mean-pairwise: 110 wins, 39 losses, 287 ties, or +16.2844 pp. This suggests
that the benefit may depend on budget. It is a developmental hypothesis for a separate
future contract, not permission to replace the three-extra-cell primary endpoint.

## Calibration and dependence

The full-menu reference excludes the true candidate in 4/436 cases. Each is a
`no_candidate_fits` output caused by one measured cell crossing its simultaneous
radius. Under a binomial reference with p = 0.005, the probability of at least four
exclusions in 436 trials is 0.1763; this observed count alone does not demonstrate
miscalibration. That calculation is a diagnostic at the declared probability bound,
not a new confirmation test or an estimate that the true failure probability is 0.005.
More measurements can expose additional finite-sample exclusion errors, so the
full-menu result is a higher-cost diagnostic reference, not a realized accuracy ceiling.

Policies and budgets share potential measurements within a case. Pairing is correct;
these records are not independent replications. The 512 cases sample two simple graph
compositions and known Gaussian measurement noise, not 512 distinct mechanism families.

All source hashes and the generation, prediction, and scoring seals match. Bindings:

- Manifest SHA-256: `f26ccd64abbd74d0a21065c62e618684e1f7e1d4d1c0383942d92de022aa9c86`.
- Predictions SHA-256: `99868fb591d82472721a40f72f74763ada94a113f4a7e3d57ba3270c74510e99`.

**Conclusion:** the implementation and exploratory comparison are reproducible. Maximin
has a small observed advantage over the strong primary comparator, while substantially
outperforming simpler controls on this declared distribution. Neither the practical
superiority target nor the proposed empirical exclusion-rate target is established.
Known exact candidate predictions, synthetic noise, narrow graph compositions and
misspecification errors remain material limits; the structural holdout stays unopened.
