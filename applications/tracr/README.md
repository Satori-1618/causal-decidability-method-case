# An ambiguous patch, a discriminating condition, a new prediction

**Controlled external validation in DeepMind's Tracr.** We use the existing
`causal_decidability.compatible_set` implementation on a Transformer compiled from
the upstream reversal program. The circuit's computation is known independently.
This is a test of instrumentation, candidate retention and fresh prediction within
that program. The operator knows the compiler structure; this is not discovery of
an unknown mechanism, a pretrained-LLM result, or a comparison of adaptive policies.

The exact measured results and a figure are in [RESULTS.md](RESULTS.md).

## The example

The recipient normally reverses `[A,B,C,D]`, so its first answer is **D**. We copy
the donor's intermediate address from its second query to the recipient's first
query. That address points to the third content token. All other recipient
coordinates and positions stay unchanged at the intervention site.

| Condition | Recipient | Donor | Address predicts | Donor-answer predicts |
|---|---|---|---|---|
| Initial ambiguity | `[A,B,C,D]` | `[W,X,C,Z]` | C | C |
| Change donor content, hold address | `[A,B,C,D]` | `[W,X,Y,Z]` | C | Y |
| New recipient-content prediction | `[A,B,U,D]` | `[W,X,Y,Z]` | U | Y |

The letters show roles; stored cases use tokens 0..11. After the shared observation,
the same production classifier keeps both candidates. A prediction-only selector
chooses a separating condition from the declared menu; the next measurement can
retain one, both or neither. A subsequent measurement checks a new prediction.
The original candidate set is preserved when combining observations: an excluded
candidate cannot re-enter just because it fits the latest row.

The deliberate **answer-copy positive control** instead copies the final output
subspace before unembedding. The same analysis must then select the opposite
candidate. This is a separate development control, not a second mechanism found
at the main site. An unchanged-output null is also checked separately.

## Reproduce

Python 3.12 and a CPU were used; no model download or GPU is required. Installation
fetches the pinned upstream source and dependencies. Tracr is archived, so versions
are fixed and installed in this checkout's own `.venv`.

```bash
bash applications/tracr/scripts/setup.sh
```

Verify the released measurements without model execution:

```bash
.venv/bin/python applications/tracr/scripts/verify.py \
  applications/tracr/results/confirmation_001 \
  --freeze applications/tracr/CONFIRMATION_FREEZE.json
```

Re-execute the frozen model experiment into a **new** directory:

```bash
.venv/bin/python applications/tracr/scripts/run.py confirm \
  --freeze applications/tracr/CONFIRMATION_FREEZE.json \
  --out applications/tracr/results/reproduction_001
```

This reruns the published sample; it is not a second fresh confirmation. Existing
run directories cannot be overwritten. The runner requires the freeze to be
committed and rejects altered executable sources, dependencies or model parameters.
The record verifier checks both source bindings and raw tensors, not just PASS flags.

Run application tests:

```bash
JAX_ENABLE_X64=1 PYTHONPATH=applications/tracr/src \
  .venv/bin/python -m pytest applications/tracr/tests -q
```

## What is checked

- Reference/no-op agreement over complete scores and every collected half-layer;
  self-patch identity; native symbolic task correctness.
- Complete selected subspace, source/target position including BOS, one site
  execution, donor-source provenance and actual post-cast replacement.
- Exact unchanged complement at the site and unchanged other output positions
  within dtype-derived tolerance.
- Paired fp32/fp64 residual execution, including parameter rounding; upstream
  unembedding produces float64 scores in both modes. fp64 is a reference, not truth.
- One-hot candidate adequacy under a frozen scientific tolerance **0.01** plus a
  separately calibrated numerical allowance. Scores are not nats or LM logits.
- One compound success per entire fresh family; no inflated sample size from
  conditions, tokens, dtypes or repeated deterministic forwards.

## Confirmation and limits

The freeze draws **128** fresh families without replacement from the finite token-
assignment population, excluding the eight development families. With a single
final one-sided 5% finite-population test, **126** successes suffice to demonstrate
a population success fraction above **95%**. Planning power is approximately
97.31% if that fraction is 99.5%; this is a declared alternative, not a prediction.
The population is token assignments in one length-four reversal program.

The gates can block interpretation, and neither candidate may fit. A surviving
candidate is only distinguished from the declared rival under this intervention.
Because the address site was selected from known compiler structure, the outcome
is structurally expected. The useful test is whether the actual Transformer patch,
measurements, retention rule and fresh predictions deliver that expected result
without false exclusion. It does not establish general reliability or superiority
over a strong fixed experimental design.

## Files

- [Development contract](DEVELOPMENT_PROTOCOL.md) and [original proposal](ORIGINAL_PLAN.md).
- `src/tracr_demo/adapter.py`: instrumented upstream execution and source-selected sites.
- `src/tracr_demo/design.py`: candidate equations, family generation and menu selection.
- `src/tracr_demo/statistics.py`: exact finite-population planning and adequacy.
- `src/tracr_demo/experiment.py`: development, freeze and execution.
- `scripts/verify.py`: records-only tensor and decision audit.
- `results/`: raw records, full tensor snapshots, manifests and trajectories.

All stages use the original method core on `main`; it is not replaced by a Tracr-only
classifier. Geiger et al.'s [causal abstraction framework](https://arxiv.org/abs/2301.04709v4)
supplies the alignment/intervention correspondence question. This application adds
an explicit rival and a condition where their predictions separate.

Upstream: [Tracr](https://github.com/google-deepmind/tracr), pinned at
`9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd`;
[Lindner et al. (2023)](https://arxiv.org/abs/2301.05062). See [NOTICE](NOTICE).
