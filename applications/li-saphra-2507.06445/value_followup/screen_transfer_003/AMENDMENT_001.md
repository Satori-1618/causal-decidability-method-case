# Amendment 001 — 64 families per stratum

**Date: 3 October 2026. Before any screen-transfer-003 model measurements.**
Supersedes the 32-per-stratum design published at
[`7f6ebde`](https://github.com/Satori-1618/causal-decidability-method-case/commit/7f6ebdeed0232dc8d3c08faec34be06f960335dc).
The old freeze remains available in git history. The user approved implementing
this amendment and preparing the runner/inputs, with a final review stop before
execution.

## Reason and exact change

The supplied review recommended increasing **both strata from 32 to 64**.
Under hypothetical separating rates 0.75/0.10, the probability of meeting the
strong four-of-six target rises from **1.97% to 92.17%**, conditional on complete,
technically valid groups and equal independent head decisions. These are power
scenarios, not predictions of realistic effects. At rates 0.60/0.10, the new
cohort power is still only about **0.43%**.

| Item | Initial freeze | This amendment |
|---|---:|---:|
| Native candidate budget per head | 1,024 | **1,024, unchanged** |
| Accepted / rejected families per head | 32 / 32 | **64 / 64** |
| Independent donor templates per head | 64 | **128** |
| Anchor cells per head, per dtype | 128 | **256** |
| Planned sequence-forward equivalents per head, both dtypes | 3,072 | **4,096** |
| Planned sequence-forward equivalents across six heads | 18,432 | **24,576** |

The cutoffs, six heads, seeds, sampling population, operator, two original
anchors, 0.202-nat separation boundary, 0.001-nat numerical gates, 24 confidence
bounds, +0.25 practical target and four-of-six decision are unchanged.
Donor templates 0–63 go to accepted recipients, 64–127 to rejected recipients.
Missing either quota still gives **insufficient yield**, without expanding the
1,024 pool or replacing a head. No secondary mechanism transfers are added.

The original plan explicitly acknowledged limited power. This revision buys
more precision, at one-third more scheduled computation; it does not revise a
failed new outcome or relax the scientific rule. The separate averaged-anchor
specification is unchanged.

## Review provenance and historical feasibility check

The user supplied the independent review text in this conversation. Its author's
identity and software environment were not supplied; this record does not claim
external human peer review. Additional Codex agents independently checked its
arithmetic and existing artifacts. The design-review disposition is
**PASS WITH AMENDMENT**; final runner/input review remains pending.

The reviewer used **already measured native margins** from the earlier cohort,
not new inputs or transfer outcomes, to assess quota feasibility. The following
counts were independently reproduced from the old invalid inputs only:

| Model / head | Weight-derived cutoff (nat, rounded) | Old invalid inputs accepted / 1,024 |
|---|---:|---:|
| `50h5xlod/H1` | 6.366233 | 261 |
| `8a65u5l6/H1` | 8.143526 | 456 |
| `dsibxabs/H4` | 8.340455 | 322 |
| `qtda56zd/H2` | 8.362946 | 514 |
| `we9o801g/H2` | 8.397108 | 538 |
| `wfwpq5xc/H1` | 8.337095 | 428 |

Sources: locked checkpoint tensors and
`native_followup/results/confirmation_001_continued/tasks/{model}_head{head}/cases.jsonl`,
also archived in `native_followup/artifacts/confirmation_001.tar.gz` with its
index. No cutoff, head or seed was changed using this check. These old inputs
were cyclic families with one invalid input per initial symbol; the new pool is
sampled independently from invalid strings in the development hash partition.
Consequently the counts support feasibility but do not guarantee it.

The review's numerical shorthand “about 28 versus 2” was conservative: at n=32,
26 versus 2 already passed; 26 versus 3 did not. This correction does not alter
the case for the larger sample. The revised calculator uses exact bounds.

## Release sequence

1. Publish this amendment and revised power/cost calculations.
2. Prepare deterministic inputs without model scoring; build and test the runner
   with synthetic records/fake execution only. Publish their source/input freeze.
3. Stop for final pre-run review. A separate committed execution release is
   required before any native screening or transfer. This amendment is not that release.
