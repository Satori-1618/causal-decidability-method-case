# Round 2 development: word versus target-site invariance

**Implemented and executed; development evidence only. All declared technical gates
passed. Neither simple invariance describes most of the measured pairs.** This run does
not identify a mechanism or provide a fresh confirmation of either candidate.

## What changed, and what was predicted?

We exchanged the genre words at the positional and lexical target sites, rebuilt the
matching donor, and repeated the same block-18 patch. Both native correct answers stayed
unchanged. For each arm we measured the patch-induced change in the log odds of the
positional versus lexical answer, subtracting that arm's own no-patch log odds.

If this change follows the target site, it should retain its sign and approximate
magnitude after the exchange. If it follows the word, its sign should reverse while its
approximate magnitude stays the same. The tolerance, support requirements, native
qualification, controls, seeds and 32-family limit were committed before the run in
`0cee94e` ([protocol](ROUND2_WORD_STRUCTURE.md), [manifest](ROUND2_DEVELOPMENT.json)).

## Result

| outcome | families |
|---|---:|
| target-site invariance within the declared tolerance | 3 |
| word invariance within the declared tolerance | 3 |
| neither narrow invariance | 16 |
| outside the two-answer support requirement | 8 |
| weak patch contrast | 0 |
| **native-qualified denominator** | **30** |
| failed native qualification before conflict patching | 2 |
| **all generated families** | **32** |

All 32 planned families were generated without replacements. The two native failures
were a recipient answering Punk instead of Dub and an exchanged donor answering Funk
instead of Classical. These families received no conflict patches. Of the 30 qualifying
pairs, 22 met the declared support and signal requirements. The 8 outside-support pairs
remain in the reported denominator.

Of the 22 resolved pairs, 16 fail both simple magnitude-invariance rules. This does not
mean that words or structural relations have no influence. For example, case 005 follows
the lexical target in its generated answer in both arms, but its baseline-adjusted patch
effect changes from -10.20 to -1.87 nats; it therefore fails the stricter quantitative
target-site prediction. Sign following and magnitude invariance are different claims.

## Two illustrations from the stored cases

These examples were selected after the development run to explain the measured quantity.
They are not separate confirmatory findings.

| case | original P / L | exchanged P / L | patch-induced log-odds changes | generated answers | descriptive profile |
|---|---|---|---|---|---|
| 006 | Classical / Rap | Rap / Classical | -8.51, -6.18 | Rap, Classical | target-site-like: the change favours L in both arms |
| 017 | Classical / Rap | Rap / Classical | +6.70, -4.63 | Classical, Classical | word-like: the change favours Classical in both arms |

The same two genre words can therefore occur in different descriptive patterns in
different base contexts. This small, post hoc illustration does not determine what
causes that difference. The donor and recipient are both rebuilt by the paired exchange;
the test cannot localize the dependence to one of them or to a particular internal path.

## Checks and runtime

- Native pair yield: 30/32 = 93.75%, above the declared 90% floor.
- Agreement transfer: 57/60 = 95%; all agreement outputs passed the answer-mass check.
- Hooks, token spans, intended-token-only exchange, unchanged correct answers and
  identity patches passed.
- The first four generated families were rerun on CPU in float32, including the native
  failure. Qualification and descriptive labels agreed. Maximum difference in the
  baseline-adjusted patch contrast was 0.0000525 nat, below 0.01.
- Eight full-vocabulary audit arrays reproduce the saved selected logits exactly;
  maximum answer-mass reconstruction discrepancy was 0.00000141.
- Runtime including model loading and CPU repeats: 283.5 seconds, about 4 minutes
  44 seconds. The model was loaded from the existing cache without downloads.
- Before model inference: 461 tests and 560 subtests passed; 5 tests skipped. The 13
  new tests cover swap semantics, pure rivals, no effect, additive baseline bias,
  mixtures, insufficient support, and real hook execution on a tiny Gemma model.

## What this adds, and what comes next

Round 1 excluded a particular average-like concentration profile. Round 2 implements
the next iteration: an input transformation under which two explanations make different
paired predictions, with unchanged native answers and explicit no-patch controls.

The development result gives **no dominant simple invariant** to carry directly into a
large confirmation. It also shows why concentrated answers alone do not settle what is
transferred: the size of the patch-induced preference often changes substantially when
the words exchange sites. Preserve this as a development result and diagnose those size
changes on the saved logits before choosing one narrower, quantitative follow-up.
Do not relax the 0.25 tolerance after seeing these counts or label this run confirmation.

The test is about exact P/L answer sites, first-answer-token log odds and the declared
population of native-qualified contexts. It does not classify reflexive or neighbouring
positional outputs, all word biases, full answer sequences, or internal mechanisms.

## Artifacts

- [Raw records and controls](results/round2_word_structure_development/records.jsonl)
- [Summary and per-family contrasts](results/round2_word_structure_development/summary.json)
- [CPU repeats](results/round2_word_structure_development/cpu_reference.jsonl)
- [Artifact hashes](results/round2_word_structure_development/artifact_hashes.json)
- [Readout and record replay command](scripts/run_word_structure_development.py)

Replay without model inference:

```bash
python applications/gur-arieh-2510.06182/scripts/run_word_structure_development.py check
```

This replay shares the development analyzer; it is not the independent Round 1 checker.
The record arithmetic was additionally recomputed directly with NumPy, without importing
the new analyzer, and all 30 paired labels and contrast values agreed.
