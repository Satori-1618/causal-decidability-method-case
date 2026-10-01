# Reusable causal preflight prompt

Use with the [two-part method guide](../METHOD_PREFLIGHT.md). Replace the input
slots below, then copy the fenced prompt into a research assistant conversation.
This is a design audit; it does not run an experiment or decide which rival is true.
Whether this prompt improves AI-assisted rival generation is untested.

## Quick start

Use this when you have a research idea but no finished rival table. The longer
prompt below is for a design you can already specify. Neither requires you to
know which mechanism is true.

```text
Help me plan a mechanism test before a confirmatory run.
My question: [what I want to explain]
Candidate component/site and planned patch: [what I know, or UNKNOWN]
My current explanation: [hypothesis, not a fact]
An alternative explanation: [hypothesis, or UNKNOWN]
What I would measure: [choice, logit margin, etc.]
Existing evidence/rules: [paste or give paths; UNKNOWN is allowed]

First help me specify at most three competing rules, plus preservation of the
observed unpatched readout. This does not mean the patch is internally inactive.
Use observed donor and recipient baselines, or mark them UNKNOWN; do not substitute
task-correct answers. Label any rule you suggest
as a proposal, and identify assumptions that need my input or development data.
Show one concrete donor/recipient example and derive each rule's prediction.
Keep predictions as choices, signs, or UNKNOWN when that is all the rules imply;
do not invent scores, probabilities, variances, or numerical error allowances.
If the current case does not discriminate, propose one justified change that
makes the rules disagree. If no such change follows, explain what is missing.
Shared signs alone do not prove equal quantitative predictions. Unknown is not
the same as indistinguishable. Give one next action and say whether it is
hypothesis development, structural checking, or measurement calibration.
Do not run models or present any proposed mechanism as already identified.
```

## Full design audit

```text
Act as a careful, useful methods reviewer. Audit whether my planned experiment
can distinguish the declared explanations and provisionally measure their
predicted difference. Use only supplied plans, derivations, and existing artifacts.
Do not launch models, collect new data, or invent missing numerical inputs.

INPUTS (write UNKNOWN where unavailable)
- Scientific question, population, model, and intended claim: [fill in]
- Finite declared rival set, each rival's causal rule and assumptions: [fill in]
- Planned conditions, including donor/recipient/context/control: [fill in]
- Observed unpatched donor and recipient readouts: [fill in]
- Exact intervention operator, site/token, and what is held fixed: [fill in]
- Readout, estimand, units, aggregation, and any normalization: [fill in]
- Each rival's prediction for every condition/readout, with its derivation or
  artifact path/field; say whether it predicts a mean or a distribution: [fill in]
- Independent sampling unit, pairing/clustering, target population, and planned
  number of independent units n: [fill in]
- Existing pilot: unit-level measurements or variance of the relevant estimator's
  unit-level quantity, pilot unit count, provenance, and relation to target data:
  [fill in]
- Existing numerical calibration: actual dtypes/backends compared on the SAME
  final estimand and cases, observed discrepancies, aggregation, and provenance:
  [fill in]
- Intervention fidelity checks: intended versus implemented operation, including
  after dtype conversion, identity/sham and positive-reference evidence: [fill in]
- Declared planning rule, meaningful separation/precision target, uncertainty
  assumptions, and any alpha/power/multiplicity choices with rationale: [fill in]
- Existing discovery/calibration/confirmation split and freeze plan: [fill in]

PART 1 — DO THE DECLARED PREDICTIONS DIFFER HERE?
1. Restate the exact operator, estimand, and scope. Distinguish an intervention's
   effect from the unmodified model's native computation.
2. Construct the condition-by-rival prediction table, citing every value's source.
   Derive a value only when the supplied rule and inputs determine it; show the
   derivation. Otherwise mark it missing. Do not infer quantitative predictions
   from labels such as "semantic", "routing", or "direct effect".
   If only choices/signs are justified, keep those and route the analysis through
   a qualitative table. Opposite predictions can motivate a test; shared signs
   alone are overlapping predictions, not proof of structural equivalence. The
   numerical example requires fixed mean predictions, not arbitrary category codes.
3. For each relevant rival pair, compare the complete declared prediction pattern.
   Report "identical on declared quantities", "different", or "unassessable".
   Exact equality needs an exact derivation or an explicitly finite/exact contract.
   Floating-point agreement or closeness is not an equivalence proof. Preserve
   approximate comparisons as diagnostics with the supplied tolerance and source.
4. Equal predicted means do not imply equal predicted distributions. Scope a tie
   to the measured quantities. More samples of those same quantities cannot fix
   identical declared predictions; changing the readout might introduce a new test.
5. If an important pair ties, name the smallest condition/readout/operator change
   justified by the rival rules that could separate it. If none follows, say so.
   Do not silently add a rival or claim that unmentioned rivals have been excluded.

PART 2 — IS A DECLARED GAP PLAUSIBLY RESOLVABLE?
6. For a differing pair, specify the relevant gap and estimator before calculating
   uncertainty. Use the pilot variance of that estimator's unit-level quantity;
   for a paired contrast use within-unit differences, not marginal SDs. Repeated
   tokens, conditions, directions, or paraphrases are not automatically new units.
7. For an independent-unit sample mean, s/sqrt(n) is a provisional standard-error
   estimate, not a power guarantee. Use the declared planning rule only if its
   assumptions fit; ratios, clustering, small pilots, or other estimators may need
   different calibration. Report pilot uncertainty and population mismatch.
8. Compare numerical discrepancies on the final estimand to the relevant gap and
   uncertainty budget. Dtype names or machine epsilon alone cannot supply that
   budget. Observed sensitivity is not a proven worst-case bound or an independent
   random noise term. Do not assume it shrinks with n or combine it with variance
   without a justified rule.
9. Check the fidelity evidence separately. Missing or failed fidelity checks block
   a causal interpretation even if the planned predictions differ and gaps are big.
10. Report resolution as "provisionally sufficient under stated assumptions",
    "below the declared planning threshold", or "unassessable". Never invent an
    SD, n, tolerance, numerical bound, or decision rule. Below threshold is not
    impossibility; absent inputs are not a pass. Recommend the minimum calibration
    or design change justified by the limiting evidence, without executing it.

OUTPUT (brief and concrete)
- A short verdict keeping structural separation and measurement resolution apart.
- One small before-run table with columns: rival pair; prediction status/source;
  gap; independent-unit uncertainty; measured numeric sensitivity; fidelity;
  provisional resolution; missing evidence/minimum next change.
- The single highest-priority justified next design or calibration change, and
  what must be specified and frozen before fresh confirmation.
- A claim boundary: preflight establishes only a conditional design assessment.
  Empirical retain/exclude/adequacy decisions belong to a later valid experiment
  with its declared decision rule. No unique-mechanism or adequate-fit claim here.
```
