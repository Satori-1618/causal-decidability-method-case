# Round 3A amendment: extend the confirmation cap to 512

**Prospective amendment after the 32-family development run, before any confirmation
outcomes.** It does not retroactively change the original protocol or its STOP result.

The original [execution protocol](DONOR_FACTOR_PROTOCOL.md) considered n = 128, 192,
256 and 384. Its preserved [planning record](results/donor_factor_planning/planning.json)
selected no n: P power at n = 384 was 78.67% and 78.14% for the two planning signs,
below the 80% target. The original planning file has SHA256
`fa30e74a514e91a9eeac28930384addfa1ca9ae73b871d9423defd2312a40a28`.
The development evidence, its interpretation and the original producer files remain intact.

## The single scientific change

Append **512** to the candidate grid. Preserve all other commitments:

- Same 32 development families and covariance estimate, with SD multiplied by 1.5.
- Mean alternatives ±0.20 nat versus ±0.10 practical boundaries; each I/P/J contrast,
  both signs and both precisions must meet the existing planning checks.
- Same exact-binomial profile planning: at least 80% power at hypothetical complete-
  family success probability 0.90, against 80% required population coverage.
- Same 10,000 Gaussian simulations and, at a qualifying n, 1,000 Gaussian and 1,000
  empirical-family simulations with 2,000 bootstrap resamples. Every required bootstrap
  power lower 95% Monte Carlo bound must exceed 80%.
- Same seed320260926 and original per-grid-index seed offsets: the four original sizes
  keep their original schedules; 512 uses index4. No alternative seeds are tried.
- Same fixed instrument, sites, sampling population, outcome definitions, tolerances,
  error allocations, technical and numerical checks, and 20,000-resample final analysis
  with both original bootstrap seeds and the 0.01-nat endpoint-stability requirement.

The Gaussian power approximation of about 90% at n = 512 motivates checking this size;
it is not a passed planning gate or a guarantee of a decisive outcome. If 512 fails,
record another STOP. Do not relax thresholds, choose another seed, drop contrasts or
expand the grid again within this amendment.

## Authorization and separation

1. Version the new planner and runner alongside the immutable originals. Publish this
   amendment and its code before new confirmation outcomes.
2. Run the full amended planning procedure on the original development records. Bind
   the output to the amendment, original STOP, development artifacts and new planner.
3. Only if it selects 512, generate 512 fresh iid complete families with the previously
   declared confirmation seed24092432. Keep the original sampling rule; exclude every
   historical prompt and every prompt in the 32 development families. Do not pool
   development cases into confirmation, filter by correctness or replace failures.
4. Freeze and push the confirmation cases, manifest, code/model/direction hashes and
   planning binding before running any confirmation forwards. Run the complete fixed
   sample once. Technical failure remains invalid; numerical or analysis-resolution
   failures receive the unchanged unresolved decision. No interim scientific stopping.
5. Publish all results, including exclusions or unresolved findings, and verify stored
   records independently. Clearly distinguish original development, amended planning
   and fresh confirmation throughout the report.

## Claim remains bounded

The question is whether a fixed patch's effect satisfies position-only or identity-only
invariance, alongside its mean factorial contrasts. It does not identify a transferred
person, semantic role or unique native mechanism. Name assignment includes the giver;
signed mean cancellation does not establish invariance. One template and one prefix
remain the sampled population. The role task 3B is not part of this amendment.

Entry point: `scripts/run_donor_factor_512.py`. The new planner is
`src/donor_factor_planning_512.py`; the original instrument, sampler and analyzer are reused.
