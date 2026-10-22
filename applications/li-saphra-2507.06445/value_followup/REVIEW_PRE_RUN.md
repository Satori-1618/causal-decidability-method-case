# Pre-run design and statistical review

**PASS for the declared 32-family development.** The review inspected the actual
design, preparation, runner, analyzer and prepared inputs at commit `66110d8`.
No released-model inference or outcome inspection was performed for this review.
This is a separate code review within the same agent-assisted research workflow,
not an independent human replication.

## Checks performed

**Prefix sampling.** An independently written forward count over prefix depth
and minimum reproduces every support size before exclusions and phase splitting:

| Target position | Balance −2 | Balance +2 |
|---|---:|---:|
| 20 | 15,504 | 2,907 |
| 28 | 3,749,460 | 1,332,045 |

The generator chooses each branch in proportion to its number of allowed
completions. The resulting probability telescopes to the same value for every
allowed prefix. Appending the fixed closing bracket preserves minimum −4 for
the declared target balances. Rejection by fixed exclusions and the fixed hash
partition preserves uniformity within each remaining pool.

**Sampling unit.** Families are drawn with replacement from fixed pools. The
second prefix is required to differ within its cell; no cross-family uniqueness
filter is imposed. Thus the whole family is the appropriate iid sampling unit,
not individual donor values or the two conditional draws within a cell. The
prepared inputs contain 32 recipients and 256 donor draws using 252 distinct
target prefixes, with maximum cross-family reuse two.

**Current freshness.** The prepared recipients and target prefixes avoid the
stored public/previous-native-run exclusions. The phase split applies to the
actual donor prefix, so changing only a suffix cannot make that target prefix
fresh.

**Execution and decision contract.** The runner fixes the recipient's maximum-
attention closing-bracket position using its native float64 attention, then
uses that same position and recipient in every transfer. It records both anchor
outcomes and candidate forecasts before executing the six heldout transfers.
The analyzer takes the maximum forecast error across all six, including the
second prefix at each anchor setting. Strict gap `>0.202`, definite error
`<=0.099`, and the development start rule of at least 16 eligible families with
at least 90% definite hits match the document. No post-outcome donor selection,
amplification or anchor refitting is implemented.

## Required before a confirmation preparation

The current exclusion builder reads earlier public/native inputs, but does not
yet add **this value-development run's complete recipient and donor strings**.
Hash separation of the selected target prefixes alone does not cover prefixes
at other positions or in the other role. For example, a development donor whose
target is at position 28 can have a confirmation-assigned prefix at position 20.

The currently prepared development strings contain **26** eligible,
confirmation-assigned position-20 prefixes and **27** at position 28. These are
not confirmation leaks now—no confirmation inputs or outcomes have been used.
After development, all of its evaluated recipient/donor strings must contribute
their position-20 and position-28 prefixes to the fixed confirmation exclusion
file, with source hashes recorded, before confirmation inputs are drawn.

This is a requirement for the next phase, not a blocker for the bounded current
development. The current runner accepts only its 32 development families.

## Interpretation and precision limits

Six heldout transfers test conditional exchangeability, not an isolated semantic
variable. Exact depth, current-negative status and normalized-depth aliases are
explicitly unresolved on this grid; whole-value transfer carries other prefix
information. A 16/32 eligibility result would suggest about 256/512 eligible
confirmation families, but does not guarantee that count or the planned power.
The development gate is a feasibility rule, not population adequacy evidence.
