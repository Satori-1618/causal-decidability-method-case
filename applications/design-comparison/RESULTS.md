# Development 001: a working comparison, not yet the target result

**Status: development.** [Raw run and sealed report](results/development_001/REPORT.md).
The source/artifact verifier passes. Reserved downstream graph compositions have not
been executed. The [research goal](../../docs/MECHANISM_DISCRIMINATION_GOAL.md) is still open.

512 independently generated instances were evaluated: 436 from six declared explanation
classes and 76 from an outside-candidate graph. Policies shared the prediction matrix,
measurements and production classifier. The primary budget was 56 measurements per case.

| Selection policy | Correct full-menu class, in-set | In-set truth excluded | In-set false single class | Out-of-set false single class | Out-of-set correctly rejected |
|---|---:|---:|---:|---:|---:|
| Maximin | 398/436 (91.28%) | 1/436 | 0/436 | 2/76 | 70/76 |
| Mean pairwise separation — main comparator | 391/436 (89.68%) | 1/436 | 0/436 | 2/76 | 70/76 |
| Fixed channel split | 332/436 (76.15%) | 2/436 | 0/436 | 4/76 | 64/76 |
| Random selection | 263/436 (60.32%) | 0/436 | 0/436 | 1/76 | 68/76 |
| Full-only negative control | 0/436 | 0/436 | 0/436 | 0/76 | 64/76 |
| Full menu, **128-measurement reference** | 399/436 (91.51%) | 4/436 | 0/436 | 1/76 | 71/76 |

**Reading correction for the sealed report:** its column “False single class” refers to
**in-set cases only**. There are false single-class outputs on out-of-set cases, as shown
above and stored in the original summary. “No false identifications” without this scope
would be incorrect. The sealed artifacts are retained unchanged.

## What the main comparison says

Maximin minus mean-pairwise recovery is **+1.61 percentage points**, with the declared
simultaneous conservative 95% interval **[−13.21, +16.42] pp**. There are seven paired
wins, no losses and 429 ties. The proposed practical superiority threshold is +5 pp;
this run does not establish it. A different, post-hoc interval must not turn development
into confirmation, and narrower uncertainty alone cannot enlarge the point difference.

The observed exclusion rate for maximin is 1/436. Its simultaneous one-sided 95% upper
bound is **1.56%**, so the proposed 1% empirical upper-bound target is also not established.
The known-Gaussian construction separately gives a model-conditional exclusion bound of
0.5% per case/policy; that analytical guarantee and the finite-sample empirical bound
answer different questions.

Against fixed split, the paired difference is +15.14 pp [0.32, 29.96]; against random,
+30.96 pp [16.14, 45.78]. These are useful development comparisons. They do not establish
superiority over the main discrimination-design comparator or over full published
interpretability workflows.

## A possible next hypothesis, explicitly secondary

With **two** added cells (48 measurements total), maximin resolves 267/436 cases and
mean-pairwise resolves 196/436: a descriptive difference of **16.28 pp**. With four added
cells the counts are 398 versus 401. The benefit therefore appears budget-dependent in
these development data; a universal advantage would be an overstatement.

This does **not** replace the three-cell primary result. A reasonable next study would
predeclare the narrower claim about constrained budgets, retain the strong comparator,
and test it on reserved downstream compositions under a new powered protocol. It should
also compare to a hand-designed cross-context split, to avoid mistaking a weak fixed
schedule for a substantive methodological advantage. No such confirmation was run here.

## Scope and cost

Actual graph generation/qualification took about five seconds in this environment,
including 155,648 common qualification graph forwards and 8,192 truth-graph cell forwards.
This is a cheap CPU testbed. Selection, artifact writing and verification are additional
work; policy-level GPU or LLM runtime savings have not been measured.

The graphs are small, fixed neural computations. Their parameters and candidate
predictions are known; measurement noise is added under an explicit known-variance
model. No unknown mechanism was discovered inside a transformer. The contribution of
this implementation is a reusable, fair measurement of selection performance, including
its failures, rather than a positive result guaranteed by an easy comparator.
