# resid_mid.8 — confirmation result

Prospective test of `PREREG_RESID_MID8.md`. Chain of commits, each made before the next
step ran:

| step | commit | what it fixed |
|---|---|---|
| Freeze A | `6c8cdb8` | rivals, readouts, contrasts, seeds, success criterion |
| pre-run clarification | `47b6e19` | §13 entry; pilot code committed before it ran |
| Freeze B | `4d54a76` | 28 predictions from the blinded pilot; confirmation runner pinned |
| confirmation | this directory | run once at `4d54a76`, 2000 fresh pairs, none shared with the pilot |

## Primary outcome (§10)

**28 / 28 predicted verdicts match; 0 misses outside the band → supported.**

Predicted and observed crossover agree for every component × readout (n = 20, the
smallest size in the ladder).

**Reach.**

- The 28 checks are four series on seven nested prefixes of the same pairs, not 28
  independent tests.
- All 28 predictions were "decidable", at ratios 2.34–50.4, all above the [0.5, 2] band,
  so the constant prediction "decidable" would also score 28 / 28. The test can catch the
  calculator overpromising on real data. It cannot show that it correctly calls an
  undecidable design, because the frozen ladder contained none.
- The tightest cell was A / row at n = 20: the two half-widths summed to 0.552 against an
  observed E(full) of 0.650, so it could have missed.
- n = 20 already decides everywhere, so the primary grid does not locate the transition.

What the §9 rule left standing ("decided" = not both endpoints compatible):

| endpoints still compatible | checks |
|---|---:|
| neither | 21 |
| only inert (A / null, n = 20 … 1000) | 6 |
| only carries all (A / row, n = 20) | 1 |
| both | 0 |

In most checks the rule excluded **both** declared endpoints. "Not excluded" is not a
positive confirmation of that endpoint. Post hoc, the 14 A-readout checks also decide under
an exact two-sided sign test at 0.005 per endpoint.

## Secondary, added at Freeze B, reported not judged

Readout A, disjoint consecutive blocks of the interleaved confirmation order. The first
rate column uses the §9 rule; the second is a post hoc sensitivity check with an exact
two-sided sign test on the nonzero paired differences, at 0.005 per endpoint.

| n | row: ratio | row: §9 rule | row: exact | null: ratio | null: §9 rule | null: exact |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.74 | 0.321 | 0 | 0.83 | 0.515 | 0 |
| 3 | 0.91 | 0.159 | 0 | 1.01 | 0.365 | 0 |
| 4 | 1.05 | 0.406 | 0 | 1.17 | 0.654 | 0 |
| 6 | 1.28 | 0.417 | 0 | 1.43 | 0.787 | 0 |
| 8 | 1.48 | 0.408 | 0 | 1.65 | 0.828 | 0 |
| 12 | 1.81 | 0.741 | 0.120 | 2.02 | 0.970 | 0.554 |
| 16 | 2.09 | 0.968 | 0.560 | 2.34 | 1.000 | 0.944 |

**This is a diagnosis of the §9 rule at small n, not a calibration of the ratio.** The
§9 rule uses a normal interval with the sample SD. On small samples of {−1, 0, 1} values
it is not calibrated. If the values are ±1 with probability ½ each, so the true mean is 0,
it wrongly excludes 0 with probability 50 % at n = 2, 25 % at n = 3, 3.9 % at n = 12 and
2.1 % at n = 16, against a nominal 0.5 %. The exact test is conservative instead: it
cannot exclude anything with fewer than 9 discordant pairs. Neither column is a rate of
reliable decisions.

## Unblinded answer (secondary, not used to judge the method)

At n = 2000, mean ± z·sd/√n with z = 2.8070:

| readout | component | E(X) | d_inert | d_all | excludes |
|---|---|---:|---|---|---|
| L | row | 3.882 | 3.882 ± 0.089 | 0.650 ± 0.021 | inert, all |
| L | null | 0.658 | 0.658 ± 0.018 | 3.874 ± 0.087 | inert, all |
| A | row | 0.539 | 0.539 ± 0.031 | 0.179 ± 0.024 | inert, all |
| A | null | 0.006 | 0.006 ± 0.005 | 0.712 ± 0.028 | inert, all |

E(full) = 4.533 (L), 0.718 (A). The separate row-space patch achieves 86 % (L) and 75 %
(A) of the mean full-patch effect; the null-space patch achieves 15 % and 0.8 %. On A, 12
of 2000 pairs flip under the null patch and none flip back (exact two-sided p = 0.0005).

These are **effect ratios, not shares of a mechanism.** The patch along `v` is not the sum
of the patches along `v_row` and `v_null`: it has cross terms between the component it
reads and the component it writes. The network downstream is also nonlinear. On A the two
single-patch effects add up to only 76 % of the full one. Neither a mediation share nor
"both components are needed" follows. §3 of the preregistration maps "at least one
component excludes both endpoints" onto the reading "both are needed". That reading is
stronger than the endpoint rule can support; what the rule shows is that neither declared
endpoint fits.

## Corrections to this report, 2026-09-21

This report was revised after an external review, before any further run. The frozen
verdict, `result.json` and `per_pair.jsonl` are unchanged. The following changed:

- the checks are called nested rather than independent, and the table of remaining
  endpoints was added;
- the secondary table gained the exact comparison, and is no longer read as a
  calibration of the ratio;
- the effect ratios are no longer called shares carried by a component;
- an exploratory "sum of SDs" refinement was removed, because it rested on the §9 rule's
  small-n rates.

## Files

`result.json` scored cells, crossovers, block rates, unblinded intervals, provenance.
`per_pair.jsonl` every pair's logit difference and interchange indicator in all four
conditions, in evaluation order.
