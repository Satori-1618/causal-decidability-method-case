# Applications

Each case declares rivals, finds their differing predictions, and reports what its
data can decide. **They test different parts of the method.** None establishes
general reliability or identifies every mechanism compatible with a model.

## Choose a case

| Application | Read first | Evidence and limit |
|---|---|---|
| **Makelov, Lange & Nanda — GPT-2** | [Four-round overview](makelov-2311.17030/OVERVIEW.md) | Fresh comparison, later profile exclusions, then a qualification stop. Explains aspects of a fixed patch; native mechanism and Q1 adequacy remain open. |
| **DeepMind Tracr — compiled reversal** | [Result and worked example](tracr/RESULTS.md) | Known computation, two declared rivals, fresh token families. The operator chose the address site from compiler structure; this is controlled validation. |
| **Gur-Arieh, Geva & Geiger — Mixing Mechanisms** | [Confirmation result](gur-arieh-2510.06182/CONFIRMATION_RESULT.md) | One concentration profile excluded; the stronger profile undecided. No per-case mechanism identification or refutation of the aggregate model. |
| **Goodfire CausaLab — multiple choice** | [Design audit](goodfire-mcqa-preflight/README.md) | Reuses published input tables to expose ties. No new neural intervention or resolution calibration. |
| **Li, Saphra et al. — Dyck-1 attention ablation** | [Prospective causal narrowing](li-saphra-2507.06445/native_followup/README.md) · [earlier audit](li-saphra-2507.06445/README.md) | Two nested intervention tests on fresh cyclic families and a preselected transfer cohort; local adequacy, rival exclusions and substantial transfer failures all reported. |

For claim-to-artifact links and provenance, use the
[evidence map](../docs/EVIDENCE_MAP.md). Beckmann and the full historical synthetic
benchmark are not bundled; the benchmark's limitations remain in
[validation](../docs/validation.md).

## Reproduce at the appropriate level

**All included stored cases, without models or downloads during verification:**

```bash
python3 -m pip install -e '.[verify]'
python3 scripts/verify_release.py
```

This needs a full Git checkout and the small NumPy/SciPy verification extra. It also
checks Tracr's stored tensors without installing the Tracr/JAX model environment.

**Makelov, stored evidence, standard library, offline:**

```bash
python3 examples/confirmed_read_source.py
```

The [verification guide](makelov-2311.17030/VERIFICATION.md) lists the later-round
checks and separates records replay from running GPT-2 again. The application's
original README is a frozen historical artifact; use the overview as its front page.

**Mixing Mechanisms, stored evidence, standard library, offline:**

```bash
python3 applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py --results applications/gur-arieh-2510.06182/results/confirmation
```

Use a Git checkout with the included history: the checker validates the recorded
freeze commit as well as files. Its [README](gur-arieh-2510.06182/README.md) gives the
model environment. Round 2 is development only.

**Goodfire MCQA, input-table reconstruction, standard library:**

```bash
python3 applications/goodfire-mcqa-preflight/run.py
```

The first run downloads small files from a pinned upstream commit and verifies
their hashes. It executes no upstream code and loads no model.

**Tracr model replay:** follow its [setup guide](tracr/README.md#reproduce) only to
execute the compiled Transformer or its application tests. The combined check above
suffices for the stored evidence. Replaying the stored sample is not a second fresh
confirmation.

## Apply the procedure to a new study

1. Start with the [two-check guide](../docs/METHOD_PREFLIGHT.md). Use an explicit
   candidate table; preserve ties and unknown calibration.
2. Develop the design separately from confirmation. Check the implemented
   intervention and the final measurement's numerical sensitivity.
3. Freeze predictions, units, loss, tolerance if assessing adequacy, and the
   uncertainty rule before fresh outcomes. Timestamp the freeze externally at the
   time if claiming an externally verifiable preregistration.
4. Report the result and remaining ambiguity, including failed checks or no fitting
   candidate. A comparison win alone is not adequate fit.

[PREREG_TEMPLATE.md](PREREG_TEMPLATE.md) supports empirical evaluation;
[CALCULATOR_VALIDATION_TEMPLATE.md](CALCULATOR_VALIDATION_TEMPLATE.md) is for the
separate question of forecasting whether an experiment will decide. The
[technical guide](../docs/USING_THE_METHOD.md) documents the existing code paths.
