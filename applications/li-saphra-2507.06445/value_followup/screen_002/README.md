# Can a baseline predict which cases will separate two explanations?

**Status: prepared; results pending.** This is a prospective test of a screen
suggested by the earlier pilot, on one previously selected Dyck head. The failed
[first development round](../README.md) remains unchanged.

## The useful distinction

A case can fail to distinguish explanations because their intervention
predictions are too close. That is different from obtaining a clear distinction
and discovering that neither explanation predicts it accurately. This round
tests the first question primarily and keeps the second as a separate result.

The screen uses a measurement available **before patching**: accept a recipient
when its native rejection margin is below 8 nat. We measure the same two anchor
transfers on 32 accepted and 32 rejected fresh families. The primary outcome is
whether their difference exceeds 0.202 nat, the already specified criterion for
separating the balance-class and position forecasts. A rejected case is not
declared causally irrelevant.

## How the test was fixed

- One head: `a9g0io1r`, layer 2/head 1. One operator and the original donor grid.
- 1,024 native candidates, then the first 32 in each screen group; all scores and
  selections are recorded before any anchor transfers.
- Same independent donor sampling in both groups, with prior full-string and
  cross-position donor-prefix exclusions.
- One primary difference in separation rates, with a simultaneous >=95% interval.
  A lower bound above zero supports enrichment; above 25 percentage points meets
  the stronger practical target. These decisions were fixed before outcomes.
- The six remaining transfers are measured for accepted families only. The
  earlier **16/32 separating plus 90% prediction matches** gate stays unchanged.

The screen cutoff was chosen after the first pilot. Fresh testing can validate
it locally, but cannot erase that development history or establish transfer to
other heads. The signed cutoff is not a general confidence rule: confidently
incorrect negative margins also pass it. It does not identify numerical balance,
its sign, a saturation mechanism or a complete circuit.

## Inspect and reproduce

From this directory, saved-record verification uses standard-library Python:

```bash
python3 -B -S verify_screen.py --run results/run_001 --inputs inputs \
  --report results/report.json
```

The verifier follows a separate implementation from the producer/analyzer. It
checks saved native scores, first-in-order selection, freshness, forecasts,
tensor arithmetic and both decisions. It is not an independent neural rerun;
whole-layer unchangedness is checked during production but full-layer tensors
are not archived. Receipts and git provenance are local, not external timestamps.

To replay model execution, use a full git checkout and the Python/PyTorch
environment and pinned asset downloaders in [the reproduction guide](../REPRODUCE.md).
Then, from this directory:

```bash
python run_screen.py --inputs inputs --output /tmp/screen-replay-002
python analyze_screen.py --run /tmp/screen-replay-002 \
  --output /tmp/screen-replay-report.json
python -m unittest discover -s tests -v
python -m unittest discover -s . -p test_analysis_screen.py -v
```

Output paths must not already exist. A replay reuses these fixed inputs and is
not a fresh confirmation. No GPU or training is needed. Numerical gates must
pass in the replay environment; byte-identical neural outputs are not promised.

[Protocol](PROTOCOL.md) · [Power assumptions](planning.json) ·
[Prepared inputs](inputs/preparation.json) · [Pre-run review](REVIEW_PRE_RUN.md)
