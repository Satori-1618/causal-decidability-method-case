# Before you run: can your experiment distinguish the explanations?

**Goal:** identify which causal distinctions an experiment can support **before**
spending the main measurement budget. The output is a small prediction table,
a resolution assessment, and the next justified design change.

**You do not need to know the true mechanism.** Start with a question about a
candidate component and two hypothetical rules for what it might do. The check
asks what would follow **if each rule were true**. It helps choose the next
experiment; it does not automatically discover components or generate correct rivals.
Use it during discovery to improve a design, then freeze it before confirmation.

This complements causal abstraction: a high-level explanation may predict an
intervention faithfully while a rival predicts it equally well. Here we compare
their predictions on the **chosen interventions and readouts**. We do not establish
which mechanism is true by inspecting a plan.

## The two checks

| Check | Researcher's question | Input | Useful action |
|---|---|---|---|
| **1. Prediction separation** | Would these explanations predict anything different here? | Explicit rival rules and their predictions for every planned condition. | Group identical predictions; add a condition or readout where an important pair differs. |
| **2. Measurement resolution** | Is that difference large enough for this measurement plan? | Prediction gap, independent-unit count, estimator variability, numerical sensitivity. | Identify missing calibration or consider more units, a more discriminating condition, or better numerical accuracy. |

**Run check 1 first.** More data cannot distinguish identical predictions for the
same quantities. If only means are specified, this statement concerns those means:
the rivals might still differ in their variances or full distributions.
Choice predictions are enough to start; statistical and numerical calibration can
remain unknown while you check whether the proposed rules disagree at all.

## Start here if you are searching for a mechanism

Write: **“I suspect component ___ does ___. An alternative is ___.
My planned intervention is ___; I will measure ___.”** If you cannot yet fill a
slot, use the [short starter prompt](prompts/CAUSAL_PREFLIGHT_PROMPT.md#quick-start)
to identify the missing specification. A missing mechanism is normal; a missing
prediction is a reason to develop a hypothesis, not to invent a number.

**Example: does a patch transfer a value or a completed choice?** Activation
patching copies an internal activation from a source run (the **donor**) into a
target run (the **recipient**). Suppose discovery has suggested a particular layer
and token position to patch. You read the answer “Object” or “Alternative”; the
task asks which option gives more points.

- **Value transfer:** replace the recipient's Object value with the donor's value,
  then compare it with the recipient's Alternative value.
- **Choice transfer:** output the donor's observed choice, ignoring the recipient's offer.
- **Forced Object:** output Object regardless of the donor's choice or either value.
- **Recipient answer preserved:** output the recipient's observed unpatched answer.
  Include this baseline: the same answer does not imply an internally inactive patch.

These are proposed rules, not descriptions of an already discovered circuit.
Keep the patch operator/site and answer scoring fixed while changing these cases:

| Case: Object / Alternative points | Value transfer | Choice transfer | Forced Object | Recipient answer preserved |
|---|---|---|---|---|
| Donor 60/10 → recipient 20/40 | Object (60 > 40) | Object | Object | Alternative |
| Same donor → recipient 20/80 | Alternative (60 < 80) | Object | Object | Alternative |
| Donor 60/90 → recipient 20/40 | Object (60 > 40) | Alternative | Object | Alternative |

The first row groups the first three rules and separates them from the baseline.
The second leaves value transfer tied with the baseline (Alternative); the third
leaves choice transfer tied with it. **Together, rows 2 and 3 separate all four
rules.** Without the baseline column, these ties stay invisible.
No hidden-activation values or guessed logit margins were needed to derive this.
The table specifies **choices**, not their confidence or probability.

The toy table stipulates unpatched donor choices Object (60/10) and Alternative
(60/90), and recipient choices Alternative (20/40 and 20/80). In a real study,
measure **both donor and recipient baselines**; points alone establish neither.
The value candidate explicitly hypothesizes the comparison rule—it is not assumed
to be what the LLM does. Verify that the concrete hook implements the proposed
patch. If the site is still unknown, locating a candidate site remains discovery.
Other rules can survive these three cases; [the full example](WORKED_EXAMPLE.md)
shows additional rivals and remaining ambiguity.

**A useful preflight report already exists at this point:** “The original case
separates the first three rules from recipient-answer preservation. Each extra case
alone leaves one rule tied with that baseline; together they give all four rules
different choice patterns. Measurement resolution is still unknown: no quantitative
margin gap or calibrated error model has been supplied. Next: develop and check
these cases on separate pilot data.”
Do not feed Object/Alternative encoded as 1/0 into the Gaussian calculator and
treat their gap as one nat. The structural check accepts them as categories:

```bash
python3 examples/causal_preflight.py --config examples/data/causal_preflight_choice_example.json --structure-only
python3 examples/causal_preflight.py --config examples/data/causal_preflight_choice_example.json --structure-only --cells d60_10_to_r20_80
```

The second command shows the tie that remains if only one new case is run.

## Use it on your own study

1. **Write the rivals as rules.** State what the intervention changes, what stays
   fixed, and how each rival predicts the readout. Include the closest alternative
   and preservation of the observed unpatched readout. This is not a claim that
   internal computation stays unchanged.
   Record the source or derivation of every prediction. Missing predictions block
   the corresponding comparison; labels such as “semantic” are insufficient.
   Use logical consequences, explicit equations, or separate development data.
   Do not fit predictions to the confirmation outcomes they are meant to explain.
2. **Make one complete prediction table.** Compare each relevant pair. For rivals
   with identical prediction patterns, propose a new condition from their rules
   and recompute the table.
   Scope equality to this table; numerical agreement alone does not prove an
   analytic identity. If nothing separates a pair, report it together.
3. **Calibrate only the promising distinctions.** Define the independent sampling
   unit and the exact quantity being averaged. For a paired contrast, calculate
   that contrast within each unit before estimating its SD. Measure precision
   sensitivity on the same final quantity, using matched cases and implementations.
4. **Record the decision before the main run.** Freeze the rival table, population,
   readout, uncertainty assumptions, and subsequent empirical decision rule. Check
   the intervention's fidelity separately, including after dtype conversion.

| Before-run output | Meaning | Next step |
|---|---|---|
| **Predictions missing** | Separation itself cannot yet be assessed. | Specify the rule or gather development evidence; record UNKNOWN. |
| **Identical declared predictions** | This quantity gives no prediction-based distinction. | Change condition/readout, or retain the group. |
| **Different, resolution unknown** | A gap exists; calibration is missing. | Obtain the missing pilot or numerical check. |
| **Below the planning threshold** | This declared screen is not cleared. | Revise the design/calibration; do not call separation impossible. |
| **Numerical allowance blocks this screen** | Increasing sample size alone cannot clear the assumed numerical allowance. | Improve/check precision or increase prediction separation. |
| **Provisionally separated** | At least one condition clears the screen for the pair. | Validate fidelity and plan the actual empirical test. |

Report pairs separately: separating A from B says nothing about A versus C.
None of these outputs excludes an explanation using empirical data.

## The reusable files

**Use your own case:** copy the choice template, edit its rules and predictions,
then run the structural check on your file:

```bash
cp examples/data/causal_preflight_choice_example.json my_preflight.json
# Edit my_preflight.json: replace the cases, predictions and prediction_source.
python3 examples/causal_preflight.py --config my_preflight.json --structure-only
```

Each rule needs one predicted answer per case, in the same order as `cells`.
If a prediction is unknown, develop that rule first; do not substitute a guess.
Use the numerical template below when you have justified numerical predictions
and calibration for the second check.

- [Runnable code](../examples/causal_preflight.py): uses the repository's existing
  prediction-grouping code; requires Python 3.9+, no packages, GPU, or model.
  Use it when you have fixed numerical mean predictions, or choice categories for
  the structural check only. Use the prompt/table for sign-only, fitted, or missing
  predictions; the code does not handle them.
- [Editable JSON template](../examples/data/causal_preflight_example.json): replace
  the teaching inputs with your rules and calibration sources; use `null` for unknown
  `n`, `sd_per_unit`, or `numerical_allowance`. Use `pilot_estimate` for estimated SD.
- [Choice template](../examples/data/causal_preflight_choice_example.json): the
  quick-start table above as categories, for the structural check only.
- [Copy/paste research prompt](prompts/CAUSAL_PREFLIGHT_PROMPT.md): helps build the
  table and assess missing evidence. A language model must not invent calibration.

Whether this prompt improves AI-assisted rival generation is untested. A separate
pilot protocol exists; that evaluation is outside this release.

From the repository root:

```bash
python3 examples/causal_preflight.py
python3 examples/causal_preflight.py --structure-only --cells full_patch
python3 examples/causal_preflight.py --config examples/data/causal_preflight_example.json --n 256 --json
```

`--structure-only` requires only `cells` (with an `id` each), `predictions` (each row
a JSON list in cell order), and `prediction_source`. It needs no `n`, SD, `alpha`, or numerical
allowance. With `"prediction_kind": "category"`, predictions may be labels such as
`"Object"`; they are compared for equality only. Use this for outcomes that are
themselves categories, not for signs or directions of a graded quantity: equal signs
are not equal predictions. `UNKNOWN` is refused, because two unknown predictions
are not a shared prediction. Categories cannot enter the second check.
The choice template retains the key `no_effect` for compatibility; it means only
“recipient answer preserved,” as defined above.
To add the second check, use the full JSON template and replace its
teaching calibration; do not inherit its values as research defaults.

The separate numerical demo uses hypothetical rules `full_patch = a+b`,
`path_a_only = a`: A has `(a,b)=(0.2,1.8)`, B has `(0.3,1.7)`.
With its **assumed** Gaussian SD 0.3 and numerical allowance 0.005, the full patch
ties; the 0.1 gap in the extra condition fails the planning screen at `n=64`
and clears it at `n=256`. This illustrates arithmetic, not a measured mechanism,
a power estimate, or a recommendation to collect 256 cases.

**What the example computes.** For `m` declared measurement cells, including the
optional cells in the template, `z = NormalDist().inv_cdf(1 − alpha/(2m))`.
For each cell, `r_stat = z × SD_per_unit / sqrt(n)` and
`r_total = r_stat + numerical_allowance`. A rival pair clears this conservative
screen if **any** selected cell has `prediction_gap > 2 × r_total`.
The addition treats the numerical allowance as an absolute error budget, not
random noise that shrinks with `n`.

With independent Gaussian units and known SD, the statistical intervals have
simultaneous coverage at least `1 − alpha` by Bonferroni. Pilot SDs make this an
approximation whose estimation uncertainty is not included. An observed precision
discrepancy is a sensitivity estimate, not automatically a bound on error; using it
as an allowance makes the conclusion conditional. Compare plausible allowances.
Fitted rival predictions, ratios, probabilities, and other estimators may need
different uncertainty calculations. This small example treats predictions as fixed.

## A focused implementation and validation plan

| Step | Deliverable | Completion criterion |
|---|---|---|
| **1. Package the method.** | This guide, the prompt, JSON template, and executable example. | A researcher can obtain both checks without editing the package or running a model. |
| **2. Apply it to one independently planned study.** | A sourced rival table and existing calibration inputs. | Every assessed gap has a derivation; missing inputs stay unknown; the author can understand the recommended change. |
| **3. Freeze and test the forecast.** | A separate protocol using fresh independent units and the intended empirical rule. | Assess both predicted resolvable and predicted weak designs, without changing the rule after outcomes. |
| **4. Measure practical value.** | Forecast errors, unresolved cases, cost, and decisions versus the original plan. | Show an avoided unsupported conclusion or a useful design change, while tracking mistaken warnings and useful distinctions lost. |

Steps 1–2 produce a usable planning aid. Step 3 is needed to validate its resolution
forecasts, and step 4 to demonstrate its practical benefit. A teaching example or
passing software tests does not supply that evidence. No new model run is part of
this package delivery.

The structural check is scoped to a finite, declared candidate set. The numerical
screen here is a transparent example, not a newly validated predictor; the older
calculator's missed forecasting target remains documented in [validation](validation.md).
Use the [technical guide](USING_THE_METHOD.md) for subsequent empirical evaluation
and the [research goal](../applications/tracr/METHOD_VALUE_GOAL.md) for the value claim.
