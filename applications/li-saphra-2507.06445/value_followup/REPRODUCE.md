# Reproduce this development round

All paths below start at the repository root. The recorded neural run used
Python 3.10.10, PyTorch 2.9.0, CPU and one thread. It completed in 1.40 seconds
after input/source checks; implementation and review time are additional.
No training or GPU is required. Verification needs only standard-library Python.

## Check saved evidence without a model

```bash
python3 -B -S applications/li-saphra-2507.06445/value_followup/verify.py \
  --run applications/li-saphra-2507.06445/value_followup/results/development_001 \
  --cases applications/li-saphra-2507.06445/value_followup/inputs/development_001/cases.jsonl \
  --report applications/li-saphra-2507.06445/value_followup/results/development_report.json
```

This also works from a clean export. Read [VERIFICATION.md](VERIFICATION.md) for
what can and cannot be checked from saved snapshots.

## Replay the model measurements

Use a full Git checkout: the runner verifies that its source and fixed inputs
match committed versions. Install the recorded PyTorch version in your chosen
Python environment, then fetch the pinned model assets using the preceding
round's downloaders. They verify hashes and do not modify the frozen results.

```bash
python applications/li-saphra-2507.06445/native_followup/fetch_assets.py
python applications/li-saphra-2507.06445/native_followup/fetch_cohort.py
python applications/li-saphra-2507.06445/value_followup/run.py \
  --cases applications/li-saphra-2507.06445/value_followup/inputs/development_001/cases.jsonl \
  --output /tmp/value-replay-001
python applications/li-saphra-2507.06445/value_followup/analyze.py \
  --run /tmp/value-replay-001 --output /tmp/value-replay-analysis.json
```

Output paths must not already exist. A replay reuses development inputs; it is
not a new confirmation. Numerical controls must pass on the replay environment;
byte-identical neural outputs across different hardware are not promised.

## Tests and figure

```bash
python -m unittest discover \
  -s applications/li-saphra-2507.06445/value_followup/tests -v
python applications/li-saphra-2507.06445/value_followup/plot_result.py
```

The tests include synthetic known profiles, sampler counts, node intervention
controls, source/forecast tampering and numerical-boundary cases. The plot needs
matplotlib and uses only the committed development report. The model code is
inherited unchanged from the source-locked preceding application.

## Scope of the preparation tool

The committed input manifest fixes the 32-family development. No confirmation
contract exists and the model runner accepts development only. A future round
must resolve the cross-role prefix exclusions in [the review](REVIEW_PRE_RUN.md)
and make a new explicit design decision; creating more samples is not automatic
authorization to bypass the failed start gate.
