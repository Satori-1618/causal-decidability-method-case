# A small external application: which component supplies the patch signal?

**Completed development pilot, 2026-09-20.** GPT-2 Small, MLP8 post-GELU,
final prompt position, 32 newly sampled base pairs / 64 directed swaps. Runtime:
**20 seconds**, including model loading and a four-pair numerical reference.

The nullspace-read explanation predicts this patch's effect substantially better
than the visible-read explanation. This is a relative comparison on development
data, not a confirmation of an exclusive mechanism or an adequacy threshold.

## What was implemented

Use the [published Makelov et al. example and direction](https://github.com/amakelov/activation-patching-illusion)
at upstream commit `e0c465b74561d9c3dd1f2afa770974bf5fcaee01`.
The authors already study row/null decomposition; this application makes the
**read source** comparison explicit while holding the **write vector** fixed.

Let `v` be the normalized published direction, `v_R` its component visible to
the MLP output projection, and `v_N = v − v_R` its nullspace component. For
`Δh = h_donor − h_receiver`, run:

| Condition | Change inserted at the receiver |
|---|---|
| Baseline / identity | none / zero change through the same hook |
| Full | `v · (vᵀ Δh)` |
| Read visible | `v · (v_Rᵀ Δh)` |
| Read nullspace | `v · (v_Nᵀ Δh)` |

**No independent normalization** of `v_R` or `v_N`. The component changes sum
to the full change at the patched tensor; downstream output effects need not add.

**A predicts:** visible-read reproduces full, nullspace-read reproduces baseline.
**B predicts:** nullspace-read reproduces full, visible-read reproduces baseline.
Both are idealized endpoints. Mixed contributions or poor fit remain possible.

## Measured result

Outcome: IO-minus-subject logit margin, in nats. Negative changes weaken the
correct indirect-object answer. Both swap directions are nested within one pair.

| Receiver pattern | Full patch | Visible read | Nullspace read |
|---|---:|---:|---:|
| ABB | −1.310 | −0.216 | −1.102 |
| BAB | −1.887 | −0.318 | −1.588 |

For each direction, each candidate predicts the two component-patch margins.
We average the **absolute** prediction errors over those two conditions and
then over the two directions within a base pair. This prevents sign cancellation.

| Candidate | Mean absolute prediction error | Nominal 95% bootstrap interval |
|---|---:|---:|
| A: visible read supplies the signal | 1.338 | [1.191, 1.490] |
| B: nullspace read supplies the signal | 0.260 | [0.227, 0.294] |
| Paired error advantage of B | **1.078** | **[0.961, 1.198]** |

These are exploratory intervals over **32 base pairs**, not 64 independent
swaps. B has nonzero residual error; the visible component also has an effect.
The supported description is **nullspace reading supplies most of the effect
of this particular common-write patch**, not “only the nullspace matters.”

## Controls and limits

- All 64 directed identity controls preserve the candidate logits exactly.
- All 256 directed intervention checks pass: exactly one call at the absolute
  prompt position, other positions unchanged, inserted-tensor error within its
  dtype-derived rounding budget. Full/component decomposition also passes.
- The centered output matrix has numerical rank 767; rank-aware SVD avoids
  treating a spurious QR completion direction as visible. `‖v_N W_out‖ = 8.56e−9`.
- Numerical audit uses the first four base pairs, selected in the manifest before
  evaluation. At the final paired-error advantage, the largest directed
  discrepancy is **3.94e−5 nats** across MPS-fp32, CPU-fp32 and CPU-fp64
  comparisons. Higher precision is a reference, not exact truth; this is a small
  paired audit, not a global bound. All use the same processed float32 weights.
- The authors' centering, layer-norm folding and attention refactoring options
  are reproduced in TransformerLens 2.17.0. This is not a bitwise reproduction
  of their original software environment or published evaluation sample.
- Fresh draws use their held-out vocabulary/template split and a new seed.
  Single-token names and equal prompt lengths are the only eligibility rules;
  no filtering by behavior. Baseline IO wins in 63/64 directed cases. Original
  training/evaluation membership cannot be exhaustively certified.
- One model, one MLP site, one template. This identifies a property of the
  intervention's read/write route, not a naturally used semantic variable.

## Reuse

With `requirements-makelov.txt` installed and GPT-2 cached:

```sh
python scripts/run_makelov_read_source.py --output results/my_read_source_pilot --n 32
python scripts/verify_makelov_read_source.py results/my_read_source_pilot
```

`--device cpu` also works. Existing output directories are rejected. The runner
exports the frozen prompt/token manifest, pinned direction, every baseline and
patch margin, fidelity checks, reference values and source/model/runtime hashes.
It never loads an upstream pickle as executable code.

**Artifacts:** [summary](results/makelov_read_source_001/summary.json),
[raw directed cells](results/makelov_read_source_001/records.jsonl),
[manifest](results/makelov_read_source_001/manifest.json),
[independent verification](results/makelov_read_source_001/verification.json).

Validation: the full repository suite passed **448 tests plus 67 subtests**;
after adding two reporting regressions, the affected suite passed **42 tests**.
The frozen synthetic-world source hash remains unchanged. Historical benchmark
summaries were preserved; corrected reanalyses live in a separate audit directory.

## What this adds, and the next boundary

This is a working external application of a controlled rival comparison, with
per-case data and measured numerical sensitivity. It does **not yet validate a
before-run decidability forecast**: the separation and variability here are
calibration measurements, not predictions made before these observations.

The next confirmation can freeze this operator, candidate losses, a scientifically
chosen minimum relevant advantage and a sample-size calculation from this pilot;
then test the forecast and paired advantage on new cases. Do not reuse this pilot
as that confirmation or transfer the synthetic generator's numerical-floor model
to GPT-2 without measurement.
