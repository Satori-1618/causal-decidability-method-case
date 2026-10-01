# A fair test of intervention selection

**Development only.** This application tests whether targeted selection recovers known
causal explanation classes more often at the same measurement budget. Start with the
[research goal](../../docs/MECHANISM_DISCRIMINATION_GOAL.md) and [contract](PROTOCOL.md).

The first run is complete: [scoped results and next hypothesis](RESULTS.md),
[independent raw-data audit](DEVELOPMENT_001_AUDIT.md), and
[test audit](TEST_AUDIT.md). The intended superiority target is not yet established.

The shared full patch gives the same output for six different computations. Individual
channel interventions, crossed with context and dose, can separate them. Graphs are
executed in PyTorch; predictions use independent analytic formulas. All policies use the
repository's unchanged `compatible_set` classifier.

The principal comparison is **maximin separation versus maximum mean squared pairwise
separation**. A predetermined split, random selection, a full-only negative control and
a higher-budget full-menu reference complete the comparison. Winning against random or
full-only is insufficient to claim superiority over the main comparator.

## Run locally

Requires Python 3.10+, NumPy and CPU PyTorch. No GPU, model download or API is used. From
the repository root, choose a **new** output directory:

```bash
python3 applications/design-comparison/run.py run \
  --out applications/design-comparison/results/development_001 \
  --cases 512 --seed 2026092901
```

Or run the separate stages on the same directory:

```bash
python3 applications/design-comparison/run.py generate --out /tmp/design-dev --cases 32
python3 applications/design-comparison/run.py predict --out /tmp/design-dev
python3 applications/design-comparison/run.py score --out /tmp/design-dev
python3 applications/design-comparison/run.py verify --out /tmp/design-dev
```

Generation writes a manifest before outcomes, public candidate tables, potential
measurements and separate private labels. Prediction cannot read labels and seals its
outputs before scoring opens them. Verification checks hashes, recomputes decisions
from measurements and reproduces the summary. Sources must still match their manifest.
Existing artifacts are never overwritten. Keep each code revision's run separate.

## Read the result

- `REPORT.md`: primary paired comparison and claim limits.
- `summary.json`: every policy, exact-class recovery, exclusion bounds, false unique
  claims, none-fit detection, amplitude/noise/family strata and secondary budgets.
- `predictions.jsonl`: consumed cells, counts, costs, estimates, radii and retained sets.
- `public_cases.jsonl`: rival predictions, full-menu equivalence classes and controls.
- `generation.json`: actual generation/setup cost and numerical qualification.
- `manifest.json`, `prediction_seal.json`, `score_seal.json`: source/artifact bindings.

Budget means **consumed simulated measurements**, not measured LLM inference savings.
The harness precomputes all potential measurements to compare policies on the same
world. Graph qualification is a shared additional setup cost. Selection timings are
combined across policies; no computational-efficiency ranking follows from them.

## What remains open

The noise is deliberately simulated with known variance; the circuits are deterministic.
These are generic score units, not nats. Candidate equations and parameters are supplied.
Single-channel replacement is not Makelov's exact read/write intervention.

`squared` and `saturated_relu` downstream graph compositions remain reserved. They use
the same mechanism classes; they are not a new mechanism-discovery task. The CLI exposes
no confirmation command. A separate, powered and frozen protocol is required before
evaluating that structural holdout. A positive development comparison alone does not
establish reliability on LLMs or novelty relative to experimental-design literature.
