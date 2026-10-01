# Can this experiment distinguish the explanations?

**Before you run a mechanism experiment, check whether its result could distinguish
your explanation from a rival.** This small toolkit compares the predictions you
supply. If you can estimate measurement uncertainty, it also compares the predicted
differences with that uncertainty under your stated assumptions.

In activation patching, you copy an internal activation from a source run (**donor**)
into a target run (**recipient**) and observe how the answer changes. Several
explanations can fit that change.

**You supply:** rival rules and predicted answers for your proposed test cases.
**You get:** which rules predict the same answers, which supplied cases separate
them, and what measurement information is still missing. You propose the rules and
cases; the calculator checks them.

## A 30-second example

Does a patch transfer **a value** or **a completed choice**? Both rules predict
“Object” in the first case. Raising the recipient's alternative from 40 to 80
points makes their predictions disagree:

![Hypothetical predictions: with the same donor choosing Object at 60 versus 10 points, both value transfer and choice transfer predict Object for a recipient at 20 versus 40. At 20 versus 80, value transfer predicts Alternative while choice transfer still predicts Object.](docs/assets/preflight-example.svg)

This separates two proposed rules **in the design**; an experiment must still test
them. The [full example](docs/METHOD_PREFLIGHT.md#start-here-if-you-are-searching-for-a-mechanism)
also includes “always Object” and “recipient answer preserved.”

## Try it in a minute

Start with choice predictions. **No GPU, model run or statistical calibration is
needed for this first check.** Use Python 3.9+, from the repository root:

```bash
python3 examples/causal_preflight.py --config examples/data/causal_preflight_choice_example.json --structure-only
```

**Expected result:** all four proposed rules have different answer patterns across
the three example cases. This checks your prediction table; it does not yet test
which rule matches the model. Measurement resolution remains unassessed.
The output calls answer preservation `no_effect`; this label concerns only the answer.

For your own study, [copy the template and replace the rules](docs/METHOD_PREFLIGHT.md#the-reusable-files).
If you only have a tentative hypothesis, start with the
[short prompt](docs/prompts/CAUSAL_PREFLIGHT_PROMPT.md#quick-start).
You do not need to know the true mechanism first.

## Two checks before the experiment

| Check | What you supply | What you learn |
|---|---|---|
| **Do the predictions differ?** | Your rival rules and their predicted answers for each test case. | Which rules remain tied, and which supplied cases separate them. |
| **Can the difference be measured?** | Numerical predictions, number of independent examples, variation across examples and numerical imprecision. | Whether the gap clears the declared planning screen, or which calibration is missing. |

The first check is useful even when calibration is unknown. More samples cannot
distinguish identical predictions for the quantities you chose. The second check is
conditional on its measurement assumptions, **not a general power calculation or a
validated guarantee**. Leave missing inputs unknown; the teaching numbers are not
research defaults.

**Then test the explanations:** verify the intervention, freeze the evaluation rule,
and compare predictions on fresh cases. Report what is excluded, retained or still
uncertain. A better prediction is not automatically an adequate explanation; a unique
survivor is not automatically the true mechanism. Failed technical checks block the
mechanistic interpretation rather than count against a candidate.

<details>
<summary>Two more runnable examples: measurement planning and stored model evidence</summary>

```bash
# Hypothetical numerical example: a tie, then a conditional resolution check.
python3 examples/causal_preflight.py

# Recompute the stored GPT-2 read-source comparison; no model or download.
python3 examples/confirmed_read_source.py
```

The final command verifies the frozen records and reproduces **B better in 64/64
fresh pairs**. It checks stored evidence; it does not rerun GPT-2.

</details>

## What has been demonstrated?

| Case | Result | What remains open |
|---|---|---|
| [Tracr: known reversal circuit](applications/tracr/RESULTS.md) | Fresh token families reproduce a two-candidate tie, a separating intervention, and a new prediction. | The site was chosen from known compiler structure. This validates a controlled application, not discovery of an unknown mechanism. |
| [Makelov: GPT-2 patch](applications/makelov-2311.17030/OVERVIEW.md) | Null-read B predicts better than visible-read A on 64 fresh pairs. Later rounds exclude simple route and name/position profiles; a role-task check stops before patching. | B's absolute adequacy and the native model's mechanism are not established. |
| [Mixing Mechanisms](applications/gur-arieh-2510.06182/CONFIRMATION_RESULT.md) | On 300 qualifying fresh families, the declared cell-average concentration profile is excluded; the agreement-like profile is undecided. | Concentration does not identify a mechanism. The paper's aggregate mixture account is not refuted. The excluded profile is, by a pre-declared caveat, biased toward exclusion. |
| [Goodfire CausaLab: MCQA](applications/goodfire-mcqa-preflight/README.md) | Published input tables expose tied predictions and suggest an extra condition for one fixed patch. | A design audit only: no new model run or measurement-resolution result. |

These are different kinds of evidence, not interchangeable successes. The
[evidence map](docs/EVIDENCE_MAP.md) connects each claim to its protocol, artifacts,
verification and limits. The [application index](applications/README.md) separates
records-only checks from experiments that require model dependencies.

## Limits of this release

- **Only declared rivals are tested.** An omitted explanation is neither excluded
  nor made implausible by a good result.
- **The forecasting target was missed.** The historical synthetic planning test
  achieved **88.47% and 88.20%**, below its **90%** target. The current numerical
  preflight is a transparent conditional screen, not a newly validated replacement.
  See [the full validation report](docs/validation.md).
- **Numerical assumptions need evidence.** A precision discrepancy is a sensitivity
  estimate, not automatically an error bound. Repeating one dtype exactly does not
  establish numerical accuracy; higher precision is a reference, not exact truth.
- **AI-assisted rival generation has not been evaluated.** The starter prompt is
  an aid, not evidence that an AI finds the right alternatives more reliably.
- **Freeze provenance varies.** Hashes bind protocols and records. In particular,
  Mixing Mechanisms has local commit/run chronology, not an independently timestamped
  preregistration. Publishing that history later does not change this distinction.
- **Beckmann is a separate, unbundled case.** It is not part of this release's
  reproducible evidence and is not required to use the method.

## Connection to causal abstraction

[Geiger et al. (2025)](https://www.jmlr.org/papers/v26/23-0058.html) define when an
aligned high-level causal model reproduces low-level outcomes under corresponding
interventions, exactly or approximately. This procedure asks a complementary design
question: **can these interventions distinguish that explanation from its rivals?**
Agreement with one explanation need not exclude another. The toolkit applies
experimental model-comparison principles; it does not replace causal abstraction.

## Where to go next

| Need | Start here |
|---|---|
| Plan your own experiment | [Practical guide](docs/METHOD_PREFLIGHT.md) and [copy/paste prompt](docs/prompts/CAUSAL_PREFLIGHT_PROMPT.md) |
| Understand a real result | [Makelov's four-round overview](applications/makelov-2311.17030/OVERVIEW.md) |
| Evaluate recorded outcomes | [Technical guide and data contract](docs/USING_THE_METHOD.md) |
| Inspect evidence and limitations | [Evidence map](docs/EVIDENCE_MAP.md), [validation](docs/validation.md), [application index](applications/README.md) |
| Read the implementation | [`src/causal_decidability/`](src/causal_decidability/) and [`examples/`](examples/) |

The core has no runtime dependencies. For the default tests:

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest
```

To verify the principal bundled results together, without model execution or downloads
during verification, use Python 3.10+, a full Git checkout and the verification extra:

```bash
python3 -m pip install -e '.[verify]'
python3 scripts/verify_release.py
```

Application-specific model reruns and optional tests have their own environments;
see their READMEs. Historical protocols, amendments and results are retained for
audit. They are supporting records, not extra steps required to start using the tool.

By Felix Borck. [MIT licence](LICENSE); upstream assets retain their own terms.
For attribution, see [CITATION.cff](CITATION.cff).
