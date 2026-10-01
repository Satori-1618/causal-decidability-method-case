# Round 2: does the patch-induced preference follow a word or a target site?

**Status: development protocol, declared before new model measurements.** This is a
narrow follow-up to Round 1, not a new confirmation and not a claim about the paper's
complete mechanisms. Round 1's freeze, implementation and results are unchanged.

## One concrete contrast

Suppose the positional candidate is Jazz and the lexical candidate is Blues. Exchange
these genres in the recipient, then rebuild the conflict donor using exactly the same
index construction. The positional candidate is now Blues; the lexical candidate is
Jazz. All musicians, instruments, queries, group positions and the token multiset stay
fixed. The donor's correct answer (at the reflexive target) and the recipient's native
correct answer also stay fixed. Token spans and unchanged tokens are checked directly.

| condition | positional candidate | lexical candidate |
|---|---|---|
| original | Jazz | Blues |
| exchanged | Blues | Jazz |

The current experiment swaps only the exact central positional target (row 3, zero
based) and the lexical target (row 5). It does not exchange the whole positional window
of Round 1. Neighbouring or reflexive answers can therefore make this new contrast
unresolved. That outcome must be reported rather than silently discarded.

## Two deliberately narrow predictions

For each arm, measure the change caused by patching in the log odds of the positional
versus lexical answer:

`D = (logit_P - logit_L)_patched - (logit_P - logit_L)_native`.

- **Target-site invariance:** `D_exchanged = D_original`.
- **Word invariance:** `D_exchanged = -D_original`.

Both compare patch-induced preference. A word bias that affects native and patched
outputs equally cancels and is not tested by this contrast. Rebinding can change the
donor vector and the recipient's response to it; this does not localize that dependence
to the transferred vector or identify a positional, lexical or reflexive circuit.
Neither candidate represents every possible structure-sensitive or word-sensitive
computation. Nonlinearities, combinations and unequal effect magnitudes can fit neither.

Use symmetric losses `|D1-D0|/(|D0|+|D1|)` and
`|D1+D0|/(|D0|+|D1|)`. The working development tolerance is 0.25. Their maximum is 1
for any nonzero pair, so the two tolerance regions cannot overlap. Weak effects are
unresolved, not evidence for both invariances. No initial winner is selected.

## Small development run

- Reuse cached Gemma 2 2B IT, music, n=7, t_entity=2, cell c4 and the input of block 18,
  last prompt token. Use float32 MPS/eager, with CPU float32 repeats.
- Exactly 32 generated families, seeds 5,000,000 through 5,000,031. No replacement and
  no outcome-based selection. Original/exchanged arms share a base family and count as
  **one unit**. Previous confirmation families are not reused.
- Both native recipients and donors must answer correctly and pass the answer-form
  readout/mass check before any conflict patch is computed. Preserve all failures.
- Primary answer-form logits and full-vocabulary answer mass; record unpatched logits,
  all prompts, generations, hook reports, matrices and model/configuration hashes.
- Identity self-patch in each arm; a three-way agreement control in each arm, with its
  target alternating between P and L by generated family parity. The control uses the
  same seeded derangement in the two arms.
- Verify both arms on CPU for the first four generated families. Maximum discrepancy
  in baseline-adjusted D must be at most 0.01 nat, with identical resolution/profile
  labels and native qualification. Retain full logits for the first two families.
- Technical error, failed provenance, native pair yield below 90%, or positive-control
  transfer/mass rates below 90% means STOP. No post-result adjustment.

Development resolution requires at least 0.5 full-vocabulary mass on answer-form tokens
and at least 0.1 of that conditional mass on the two target answers, in **each** patched
arm. Mean `|D|` across arms must reach 1 nat (an odds multiplier of e); this avoids
classifying negligible effects. These are declared working thresholds, not established
universal relevance boundaries. Sensitivity can be reported separately later.

## What this run can deliver

The summary reports structure-like, word-like, neither, weak, and outside-support
counts, preserving the qualifying-family denominator. These are development
descriptions, without confirmatory adequate/excluded labels or a mechanism winner.
The raw-margin and native-margin records make it possible to see what the baseline
subtraction changed. All criteria and seeds are committed before running.

If the development run yields enough measurable, separated cases, the next action is
to freeze a narrow prediction, sample-size calculation and paired decision rule for new
families. If it does not, record the reason. Changing the task, site, tolerances or
candidate definitions requires a separate development version; this run is never
retroactively promoted to confirmation.

## Reproduction

From the repository root, in the existing pinned model environment:

```bash
python applications/gur-arieh-2510.06182/scripts/run_word_structure_development.py run \
  --upstream /path/to/mixing-mechs
python applications/gur-arieh-2510.06182/scripts/run_word_structure_development.py check
```

The runner requires a clean committed tree, the byte-identical committed manifest,
frozen hashes and runtime versions, a locally cached model, and a new output directory.
The check command replays the stored analysis without model inference. This replay
shares the development analyzer, unlike the separate Round 1 independent checker; it
also checks the baseline-subtraction identity with a second expression. The new
countermodel and tiny-model tests cover the swap semantics, ambiguity, baseline bias,
identity and qualification-before-patching.
