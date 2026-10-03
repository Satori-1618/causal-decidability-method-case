# Does the screen transfer to six other heads?

**Status: reviewed design amended to 64 per stratum; awaiting final pre-run review.
No screening or intervention run.**

**Goal:** test whether one fixed rule, using only a recipient's native output,
enriches for cases where the planned interventions can separate two explanations.
The rule is transferred to six previously selected heads without tuning it to
their value-transfer outcomes. This tests a useful selection step, not whether
either explanation describes the mechanism.

The earlier [screen test](../screen_002/README.md) found 30/32 separating accepted
families versus 0/32 rejected families on one head. Its mechanism gate still
failed (19/30 balance matches). Those results remain unchanged.

## The bounded experiment

1. Score exactly **1,024 fresh candidate recipients per head** with the fixed,
   model-normalized version of the old signed-margin cutoff. No transfers inform
   this selection.
2. Take the first **64 accepted and 64 rejected families**. If either quota is
   missing, report **insufficient yield**; do not extend the pool or replace the
   head. Measure only the original two anchor transfers in each family.
3. Report each head's enrichment with simultaneous bounds. The strong target is
   **at least four of the six lower bounds above +25 percentage points**. Smaller
   positive effects are reported separately; missing this target does not show
   that screening is useless.
4. Use the observed acceptance rate to estimate random-selection yield and
   cost per separating family. Include the entire fixed screening cost and the
   rejected validation arm. These are policy estimates, not an observed random
   control or a demonstrated saving.

Maximum scope: **6,144 native candidates, 768 families, 1,536 anchor cells**, each
measured in fp32 and fp64. No target-cell mechanism test is part of this run.

## Review materials

- [Protocol](PROTOCOL.md): cohort, cutoff, budget, stopping rules and claims.
- [Machine-readable plan](plan.json) and [hypothetical precision calculation](planning.json).
- [Later mechanistic round](MECHANISTIC_NEXT.md): two prefix-averaged anchors per
  balance level, with calibration and target prefixes kept separate.
- [Review record and final checklist](REVIEW.md): design review accepted with the
  [sample-size amendment](AMENDMENT_001.md); final execution review remains pending.
- [Source lock](SOURCE_LOCK.json): binds this plan, calculators and inherited sources.

From this directory, these commands use only files and hypothetical numbers:

```bash
python3 verify_plan.py
python3 -m unittest discover -s . -p 'test_*.py'
python3 plan_analysis.py --check
```

There is deliberately **no model-run command** here. After independent review,
the runner and prepared inputs must be checked against this public plan, frozen
and pushed before any new native score is measured. Any design change requires
an explicit amendment, not an edit hidden inside execution.
