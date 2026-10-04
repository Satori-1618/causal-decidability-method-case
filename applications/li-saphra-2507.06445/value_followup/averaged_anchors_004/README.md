# Averaged anchors: balance, position, or additional prefix dependence?

**Prepared for independent review. No model measurements; execution blocked.**
Head `a9g0io1r`, layer 2 / head 1 (one-based); four-head model, head width 16.

**Review update:** the supplied review passed the original freeze's integrity
checks. [Amendment 002](AMENDMENT_002.md) adds a calibration-only cell predictor
and descriptive prefix variation; its final review is still pending. No primary
rule, input, sample size or tolerance changes.

The earlier single-anchor balance account matched 19/30 separating families.
This new round asks whether better calibration improves prediction of fresh
prefixes, and whether differences remain that **no common balance-and-position
value could predict**. It does not reclassify the earlier failures.

## What will happen

1. Screen **2,048 unscored, frozen recipients** using the existing signed native
   margin `<8` rule. Take the first **256 accepted**; stop if the quota is missing.
2. For each family, measure **eight calibration transfers**: two distinct
   prefixes at each of two balances and two positions. Freeze and hash every
   forecast before touching the targets.
3. Start target measurements only if the averaged balance and position forecasts
   separate in **at least 128/256 families**. The new forecast criterion is not
   the old diagonal-anchor gap. Failure is insufficient design yield, not a
   negative mechanism result.
4. If the gate passes, measure **eight fresh target transfers for every family**.
   Judge each account on its worst target error, keeping the 0.10-nat tolerance.

## Three questions, the same targets

| Question | Comparison | What a positive result permits |
|---|---|---|
| Does balance or position predict new prefixes? | Two fixed averaged-anchor accounts, tested on separating families | Support or exclusion of these specific predictive accounts |
| Does averaging help? | Averaged balance versus one prefix from each of the **same four calibration cells**, on all 256 families | A paired predictive benefit from averaging; no coverage confound |
| Is an atypical anchor the whole problem? | Two target prefixes with the **same balance and position** | A gap >0.202 nat can rule out a common cell prediction for that family |

For example, if two same-cell targets produce margins 0.00 and 0.30, no one
prediction can be within 0.101 of both. Choosing a better calibration anchor
cannot remove that discrepancy. It shows additional prefix dependence under
this intervention—not which history feature the head represents.

Averaging can help **and** residual prefix dependence can remain. Neither
outcome establishes a unique natural mechanism, exact balance rather than its
sign, or applicability to other heads. The old two-diagonal-anchor rule is
reported descriptively; it is not the averaging comparator.

The amendment also asks **how large the remaining variation is**. `C_cell` uses
the two calibration prefixes at the target's balance and position; it is tested
on separate targets. A four-prefix descriptive breakdown shows cell means,
balance, position, interaction and within-cell spread against the 0.10-nat scale.
Both predictors missing does not by itself identify a history mechanism.

## Read or check the freeze

- [Protocol](PROTOCOL.md): population, predictions, stopping and interpretation.
- [Plan](plan.json), [prepared input manifest](inputs/preparation.json), and
  [power calculation](planning.json).
- [Review checklist](REVIEW.md) and [pending execution release](EXECUTION_RELEASE.json).
- `prepare.py`, `analysis.py`, `planning.py`, `run.py`: input generation, pure
  decision arithmetic, hypothetical power and the gated execution wrapper.

The donor pool uses the existing `confirmation` **hash partition** because the
old development positive-balance pool is nearly exhausted. The name does not
make this a general mechanism confirmation. Calibration and target roles are
assigned from inputs alone and are disjoint at both prefix lengths.

From this directory, model-free checks are:

```bash
python3 -B -S verify_freeze.py
python3 -B -S prepare.py --validate-only
python3 -B -S planning.py --check
python3 -B -S run.py --validate-only
python3 -B -S -m unittest discover -s . -p 'test_*.py'
```

No head, position, cutoff, family or tolerance search is authorized. The maximum
budget is **36,864 sequence-forwards**, including both dtypes and controls.
Execution requires a later public release after independent review; this
preparation stops here. Any still-required implementation correction must be
published and reviewed before measurement.

The tests use synthetic data and fake runtimes. They are not a neural integration
run or an independent audit of future raw results; those claims remain unavailable.
