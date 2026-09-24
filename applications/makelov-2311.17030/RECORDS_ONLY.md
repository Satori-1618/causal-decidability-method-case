# Recheck the confirmed comparison without a model

From the repository root, with Python 3.9 or later:

```bash
python3 applications/makelov-2311.17030/scripts/check_read_source_q1_records.py
```

This command uses the Python standard library. It needs no installation, model download,
GPU, network access, or `directions.npz`, and writes no result files. Add `--json` for
the complete audit report. It reads the **64-pair Q1 confirmation**, not the 32-pair
development pilot used by the repository's original CSV example.

Expected result: B has smaller prediction error in **64 of 64 base pairs**, with exact
two-sided sign-test p = **1.0842 × 10⁻¹⁹**. Mean absolute prediction errors, reported
rather than judged by the preregistration, are **1.2225 nats for A** and **0.2383 for B**.
The two reciprocal swap directions are averaged within each base pair; they are not
128 independent observations.

## What the check verifies

- Thirteen frozen file hashes, including the preregistration, scorer, experiment code,
  run manifest, raw records, and numerical reference records.
- Prompt-derived case IDs, absolute positions, complete reciprocal pairs, and no shared
  case or prompt with the development pilot.
- Every recorded margin against its two answer logits; recorded identity results;
  fidelity measurements against their saved rounding budgets, not just pass flags.
- The primary sign test and mean losses, independently recomputed from the raw records
  and compared with the historical score and summary.
- The saved four-pair CPU32/CPU64 comparison at the final loss-advantage estimand. Its
  maximum directed discrepancy is approximately **2.12 × 10⁻⁵ nats**.

The 512 arm-level control records are checks on 128 directed records, not independent
experimental units. The reference comparison covers four predeclared pairs only;
float64 used the already processed float32 weights. It supplies no universal error bound.

## What the check does not verify

The command checks archived evidence. It does not rerun the model, replay hooks,
reconstruct original activation tensors, regenerate tokenization, or load upstream
vectors and model weights. It checks the *recorded hash binding* of `directions.npz`,
not that unshipped file's contents. Matching hashes do not establish when files were
created; the original freeze had no independent external timestamp.

The historical runner was reused unchanged for Q1. Some saved fields therefore still
say `development_pilot` or `MPS32`; the Q1 manifest specifies **CPU float32**. This new
checker reports the actual working runtime without rewriting frozen records. See the
historical [Q1 notes](results/makelov_read_source_q1/NOTES.md) and
[provenance](PROVENANCE.md).

For input/vector reconstruction and a fresh model execution, use the separate
reproduction path in the historical [application README](README.md#reproduce). That
path has additional dependencies and may download GPT-2 and upstream artifacts.

## What the result establishes

Q1 confirms a comparison: the declared null-read candidate predicts the two split-read
interventions better than the visible-read candidate on fresh pairs. Both candidates
use the measured full-patch and baseline endpoints. No adequacy tolerance was fixed;
the result does not establish that B is accurate enough, uniquely true, or the natural
semantic computation of GPT-2. The row/null interpretation originates with Makelov and
colleagues; this application makes the rival comparison and its separating conditions
explicit.

Run the records-only checker tests, also without third-party packages:

```bash
python3 applications/makelov-2311.17030/tests/test_check_read_source_q1_records.py
```

The tests include missing and changed artifacts, duplicate directions, altered prompts,
wrong margins, nonfinite measurements, and a purportedly passing control whose error
exceeds its budget. A clean minimal bundle is checked in an isolated Python process with
site-packages disabled, and the bundle's files are verified unchanged afterward.
