# Better after ablation — better because of what?

## For readers of Li, Saphra et al. (2025)

**The replacement effect, recomputed from the released data.** In 180 released multilayer
Dyck-1 models, replacing one sign-matching head's attention with uniform or dataset-mean
weights improves out-of-distribution accuracy for 40 of 42 heads by more than 1 pp under
both replacements. The mean change is +7.97 pp (uniform) and +8.00 pp (mean). The same
single-head replacements lower accuracy for the 64 violation-detecting heads, by −7.32
and −6.85 pp; 47 and 48 of them lose more than 1 pp. For the remaining 1,244 heads the
changes are −0.23 and −0.17 pp. All means are head-weighted and recomputed from the
released result tables ([§4.3 and Appendix J](https://arxiv.org/html/2507.06445v4#S4.SS3);
[summary](results/summary.json)).

**What the prospective follow-up adds.** Each test was fixed in local commits before it
ran, on fresh inputs. The native follow-up used the same 512 new length-32 cyclic families
for all 36 tasks; each family has one valid and two invalid strings. The value rounds used
their own recipient–donor families (round 004: 256).

1. **Which part of the uniform replacement does the work** ([native follow-up](native_followup/README.md)).
   Two stages were fixed before measurement. A family counts as a match when both new
   output margins of all three members lie within 0.25 nat of the prediction. Each stage
   uses only the families that separate its predictions, so the stages have different
   populations and their composition is not itself tested.
   - Stage 1 tests whether redistributing attention between bracket positions alone
     reproduces the full uniform effect, while changing only group masses (BOS/EOS and
     bracket-group masses) leaves native behaviour (H_R). The reverse assignment is the
     rival (H_G).
   - Stage 2 tests whether redistributing between occurrences of the *same* bracket
     reproduces the routing effect, while changing only symbol masses leaves native
     behaviour (H_W). The reverse assignment is the rival (H_T).

   Results:
   - **Development checkpoint** (`1aez5d6p` layer 2/head 2, fresh inputs):
     - stage 2: H_W adequate on 183/183 separating families (bound 95.38–100%); H_T 0/183;
     - stage 1: H_R undecided (172/192); H_G excluded.
   - **Illustrated transfer head** (`a9g0io1r` layer 2/head 1):
     - stage 1: H_R adequate on 165/165 separating families (94.89–100%); H_G 0/165;
     - stage 2: H_W adequate on 136/136 (93.83–100%); H_T 0/136.
     - It was the only task of 36 that was adequate at both stages, and it was picked for
       illustration **after** the run.
     - Its published single-head effect is small: 83.0% native, 85.1% uniform and 84.7%
       mean accuracy on the 1,000 OOD strings.
     - For this head only invalid strings crossed the separation threshold. The member
       counts are `invalid_open` 119 and `invalid_close` 68 at stage 1, and 97 and 57 at
       stage 2; a family can contribute several members. A family still counted as a
       match only if all three members, including the valid string, were predicted. In 20
       transfer tasks at stage 1 and 15 at stage 2, valid strings also crossed the threshold.
2. **Across the transfer cohort** (35 tasks in 34 two-layer models):
   - Stage 2, H_W: 7 adequate, 21 excluded, 5 undecided, 1 with too few separating
     families, 1 technically invalid. H_T was excluded in 33.
   - Stage 1, H_R: 1 adequate, 33 excluded, 1 technically invalid. H_G was excluded in 34.
   - "7 of 35" is the stage-2 H_W status across the cohort; the other 28 are not all
     exclusions.
   - The stages use different conditional populations, so their composition is not a
     tested account of the full uniform effect.
   - The decomposition holds locally and does not describe sign-matching heads in general.
3. **What the relevant states carry** ([value rounds](value_followup/README.md)).
   - Rounds 001, 002 and 004 use the same head `a9g0io1r`/1; screen 003 tests six other
     heads.
   - In the latest completed round, 004 (prospective development on one head), an
     averaged balance-only predictor matched 123 of
     151 separating families (simultaneous bounds 72.97–89.31%). A position-only
     predictor matched 0. Both are excluded at the 90% target.
   - Averaging was not shown to help: against the four-cell single-prefix comparator there
     were 8 robust gains and 7 robust losses, p = 0.50.
   - A further matched-history round, 005, is prepared but **not run**.

**What is not claimed:** which semantic feature the states carry, a unique native
algorithm, necessity, or complete mediation. No claim is made against the paper's result.

**A further question, not a missing control.** The uniform and mean replacements answer
whether replacing the head's attention helps. A different question is whether the native
head output helps or hurts relative to a zero-output reference. The retrospective stage
below shows, with two constructed countermodels, that the replacement results do not
decide that question. We have **not** run a zero-output condition, and zero is not a
neutral reference. We would welcome the authors' view on whether a zero-output reference is informative for
their aims.

**Provenance.**
- The native follow-up's freeze was committed locally 7.7 s before execution, with no
  external timestamp.
- The first execution stopped on a failed technical control. A documented amendment,
  committed after one completed task and that failure, then completed the fixed cohort
  without changing predictions, cases or thresholds. It is not pre-outcome preregistration
  ([continuation](native_followup/CONTINUATION.md)).
- Pre-run reviews of the native follow-up and the value rounds were either separate
  AI-agent reviews or review texts supplied without a named reviewer; none is documented as
  external peer review. See the
  [glossary](../../docs/GLOSSARY.md#independent-in-review-and-audit-records).
- Full attention and off-target tensors are not archived, so they cannot be recomputed.

**Check it without a model:**
`python3 scripts/check_claims.py --case li-saphra-2507.06445`, run from the repository
root, recomputes the numbers above from the stored records and archives. Provenance,
scope and status statements are listed as source entries with their evidence.

## Earlier retrospective stage: what the published observations leave open

**The published improvement is robust. Its decomposition is not identified.**
In Li, Saphra and colleagues' Dyck-1 models, replacing a sign-matching head's
attention improves OOD accuracy for 40 of 42 heads by more than one percentage
point, under **both** uniform and dataset-mean replacement. These observations
do not determine whether zeroing the native head output would help or hurt.

This is a **retrospective, local application** of the causal-explanation
preflight to [§4.3 and Appendix J of the paper](https://arxiv.org/html/2507.06445v4#S4.SS3).
The authors already report predictive/causal dissociation and both replacement
controls. We reproduce their evidence and make one remaining inference boundary
executable; we do not refute that finding or claim to identify a native mechanism.

![Measured replacement effects and the remaining causal ambiguity](figures/ablation-and-rivals.png)

## A real example, followed by two explicit explanations

Released model `1aez5d6p`, checkpoint 5, layer 2/head 2; 1,000 OOD strings:

| Intervention on that head | Published accuracy | What the code does |
|---|---:|---|
| Native attention | 77.9% | Uses the input's attention-weighted values |
| Uniform attention | 82.3% | Replaces the weights; continues to transmit values |
| Dataset-mean attention | 82.7% | Replaces the weights by their mean; continues to transmit values |
| Zero head output | **Not reported** | Would remove this head's value mixture at all query positions |

The example is the lexicographically first head with both improvements above 1 pp,
selected **after** reading the data for illustration. It supports no confirmatory
claim. The analysis keeps every head, including weak and zero effects.

Let `a0` be accuracy with the head output set to zero, `aN` native accuracy,
and `aU` accuracy with uniform replacement. The following identity is exact:

**`aU − aN = (a0 − aN) + (aU − a0)`**

The first term is removal relative to a specified zero reference; the second is
the contribution of adding the replacement relative to that reference. No linear
network assumption is needed for this decomposition of three outcome values.
The same identity holds for mean replacement.

| Rival | Native, uniform and mean observations | Additional zero-output condition |
|---|---|---|
| **Removal helps:** the native contribution hurts relative to zero | Can match all three published accuracies | `a0 > aN` |
| **Replacement rescues:** removal hurts, but the replacement more than compensates | Can also match all three accuracies | `a0 < aN` |

These are operational causal claims about a **fixed zero reference**, not complete
semantic explanations. A zero effect, mixtures, and insufficient resolution remain
possible; the pair is not an exhaustive mechanism inventory.

## What the method adds

1. **Reproduce the relevant controls.** All 1,620 heads in 270 released models
   pass the source/join checks; the paper's multilayer comparison has 1,350 heads
   in 180 models. Among its 42 sign-matching heads in 41 models, 41 improve under
   each replacement, one is unchanged, and the same 40 exceed 1 pp under both.
   Average gains are **+7.97 pp uniform / +8.00 pp mean** with equal head weights,
   or **+8.02 / +8.06 pp** when each model receives equal weight. The difference
   does not require selecting each model's strongest ablation.
2. **Exhibit the ambiguity.** Two small attention models reproduce the example's
   three accuracies exactly, with identical per-case predictions across the two
   models in those conditions. In one, zeroing gives **87.9%**; in the other,
   **67.9%**. These two zero scores are **constructed witnesses, not forecasts**
   for the released neural model. They show that the measured improvements alone
   do not determine the sign of the removal effect.
3. **Specify the missing intervention class.** The witnesses remain identical
   under **any row-normalized attention-weight replacement**: a constant shift
   in every value can be offset downstream. Another normalized replacement, more
   repeated forwards, or a more precise estimate of the same measurements cannot
   distinguish this pair. A zero-output condition does distinguish them.

The existing method calculator is used unchanged: the two exact witnesses form
one prediction group under the three existing conditions and two groups when
the zero-reference condition is added. See the generated
[countermodel results](results/countermodels.json) and the
[derivation and proposed next intervention](IDENTIFICATION.md).

**What survives:** these replacement patterns outperform the native patterns on
the measured OOD set. **What remains open:** whether the original head output
helps or hurts relative to zero, and which semantic computation explains that
effect. The paper's main predictive result is not challenged.

## Reproduce locally

From the repository root, standard-library Python 3.9+:

```bash
python3 -B -S applications/li-saphra-2507.06445/analyze.py --check
python3 -B -S applications/li-saphra-2507.06445/verify_independently.py
python3 -B -S -m unittest discover -s applications/li-saphra-2507.06445/tests -v
```

The first command reconstructs the stored results and runs the existing preflight;
the second independently reads the raw CSVs and checks the central numbers and
countermodels without importing the producer. No model, installation, or network
is needed. If upstream files are absent, `python3 .../fetch.py` retrieves only the
12 files in [SOURCE_LOCK.json](SOURCE_LOCK.json), from the fixed commit, and checks
their hashes. Upstream code is included for inspection and is **not executed**.

For new output construction, `python3 .../analyze.py` writes missing results and
refuses to overwrite different existing results. To redraw the figure, use
`python3 .../plot.py` with Matplotlib installed.

## Scope and evidence

- Source: [public repository at `dedb269e`](https://github.com/vli31/id-predict-ood/tree/dedb269eca6a9cf0bc409792003e6f815f2dce34).
  [Protocol](PROTOCOL.md), [machine-readable summary](results/summary.json),
  [all head records](results/heads.csv), [provenance](results/manifest.json).
- This is **stored-data reconstruction**, not a new model replication, fresh
  confirmation, or a measured zero-ablation result. Cross-producer native
  accuracies agree in all 270 models; that does not certify tensor fidelity or
  cross-precision stability.
- These are descriptive finite-set results. Heads share models and test inputs;
  the model sweep reuses 15 initialization/shuffle seed pairs. We report no
  heads-as-independent confidence interval, population exclusion, or power claim.
  The 1-pp band comes from upstream's descriptive convention, not a significance
  or numerical threshold; the count grid is 0.1 pp, not a numerical error bound.
- The actual primary pair is **structurally unresolved** in the outcome tables,
  even before statistical resolution. Synthetic point forecasts demonstrate
  ambiguity; they do not supply missing forecasts or measurement calibration for
  the neural checkpoint.
- Zeroing is a conventional reference, not a uniquely neutral absence. Changing
  value offsets and compensating elsewhere can preserve behavior under every
  normalized attention replacement but change a zero-output result. Even the
  added experiment would establish a local effect in the fixed checkpoint,
  not a representation-invariant notion of what the head "really does."
- All OOD test strings have the negative Nested label. Accuracy gains are more
  correct rejections on that set, not by themselves evidence of an improved
  general-purpose parentheses-balancing algorithm. Same-head uniform ablation
  reduces ID accuracy by 1.00 pp on average; both distributions should be retained
  in a follow-up.

**Status of this retrospective stage: reconstruction complete; two operational causal
explanations remain compatible; the separating zero-output measurement has not
been run.**
