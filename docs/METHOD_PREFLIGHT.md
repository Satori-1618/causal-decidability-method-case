# Before you run: can your experiment distinguish the explanations?

**Goal:** identify which causal distinctions an experiment can support **before**
spending the main measurement budget. The output is a small prediction table,
a resolution assessment, and the next justified design change.

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

## A small example

Two hypothetical circuits produce a score through two paths: `score = a + b`.
Explanation A specifies `a = 0.2, b = 1.8`; B specifies `a = 0.3, b = 1.7`.
A full intervention exposes their combined contribution; a selective intervention
exposes only path A. These are teaching rules, **not measured LLM mechanisms**.

| Planned measurement | Explanation A | Explanation B | What this adds |
|---|---:|---:|---|
| Full intervention: `a + b` | 2.0 | 2.0 | No distinction in the predicted score. |
| Selective intervention: `a` | 0.2 | 0.3 | A difference of 0.1 score units. |

The extra condition creates a distinction, but its size still matters. Assume,
for illustration, independent Gaussian family scores with known SD 0.3, a numerical
error allowance of 0.005, and simultaneous 95% intervals over these two measurements.

| Design | Independent families | Gap | Twice the planning radius | Assessment |
|---|---:|---:|---:|---|
| Full intervention only | 64 | 0.0 | 0.178 | Identical declared predictions. |
| Add selective intervention | 64 | 0.1 | 0.178 | Planning neighborhoods still overlap. |
| Same two measurements | 256 | 0.1 | 0.094 | Planning neighborhoods are separated. |

The method tells the researcher **what to change and why**. The last row is a
conditional planning result, not an experiment, a power estimate, or proof of A or B.

## Use it on your own study

1. **Write the rivals as rules.** State what the intervention changes, what stays
   fixed, and how each rival predicts the readout. Include the closest alternative.
   Record the source or derivation of every prediction. Missing predictions block
   the corresponding comparison; labels such as “semantic” are insufficient.
2. **Make one complete prediction table.** Compare each relevant pair. For tied
   rows, propose a new condition from the rival rules and recompute the table.
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
| **Identical declared predictions** | This quantity gives no prediction-based distinction. | Change condition/readout, or retain the group. |
| **Different, resolution unknown** | A gap exists; calibration is missing. | Obtain the missing pilot or numerical check. |
| **Below the planning threshold** | This declared screen is not cleared. | Revise the design/calibration; do not call separation impossible. |
| **Numerical allowance blocks this screen** | Increasing sample size alone cannot clear the assumed numerical allowance. | Improve/check precision or increase prediction separation. |
| **Provisionally separated** | At least one condition clears the screen for the pair. | Validate fidelity and plan the actual empirical test. |

Report pairs separately: separating A from B says nothing about A versus C.
None of these outputs excludes an explanation using empirical data.

## The reusable files

- [Runnable code](../examples/causal_preflight.py): uses the repository's existing
  prediction-grouping code; requires Python 3.9+, no packages, GPU, or model.
- [Editable JSON template](../examples/data/causal_preflight_example.json): replace
  the teaching inputs with your rules and calibration sources; use `null` for unknown
  `n`, `sd_per_unit`, or `numerical_allowance`. Use `pilot_estimate` for estimated SD.
- [Copy/paste research prompt](prompts/CAUSAL_PREFLIGHT_PROMPT.md): helps build the
  table and assess missing evidence. A language model must not invent calibration.

From the repository root:

```bash
python3 examples/causal_preflight.py
python3 examples/causal_preflight.py --config examples/data/causal_preflight_example.json --n 256 --json
```

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
