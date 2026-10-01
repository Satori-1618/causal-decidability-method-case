# Frozen conditional predictions on real checkpoints

This is a prospective test of **restricted explanations of an intervention**,
following exploratory measurements on checkpoint `1aez5d6p`. It is not a test
of unique native algorithm identification, nor of superiority to a competent
researcher using the same factorial design. No neural network is trained here.

## Question and successive narrowing

The published uniform replacement changes several computational quantities at
once. We separate them in two stages; all names refer to changes in attention,
not mutually exclusive meanings of the contextualized value vectors.

1. **Routing versus group mass.** Native N and full uniform U do not separate
   an explanation based on routing between bracket positions from one based on
   BOS/EOS/bracket-group masses. R changes only bracket routing; G changes only
   group masses. Forecast H_R: `(R,G)=(U,N)`; H_G: `(R,G)=(N,U)`.
2. **Which part of bracket routing?** Native N and bracket-uniform R do not
   separate redistribution among occurrences of the same symbol from changed
   total mass on open/close symbols. W flattens weights within each symbol while
   preserving symbol masses; T changes symbol masses to count proportions while
   preserving the within-symbol distributions. Forecast H_W: `(W,T)=(R,N)`;
   H_T: `(W,T)=(N,R)`.

Values, other heads, BOS/EOS masses in stage 2, projection bias and downstream
computation are unchanged. Only the final-layer EOS query is modified. All
attention distributions remain causal, nonnegative and normalized. Interactions
are allowed, so both simple hypotheses can fail. H_W does not identify prefix
depth: positional or other contextual differences remain possible.

## Population, sampling and development separation

- Focus checkpoint: the already inspected `1aez5d6p`; only its inputs and hybrid
  outcomes are fresh. It is never called an unseen model.
- Transfer cohort: every released last-layer sign-matching head in 2/3-layer
  models, excluding the development initialization/shuffle pair `(365,220)`.
  Selection uses the existing source table, not hybrid effects. The prepared
  manifest enumerates the tasks before execution.
- Report the released model cohort as a **finite cohort**, not iid models.
  The training seeds form a crossed 5-by-3 grid. Heads, models with shared seeds,
  labels and arms are not independent training replications.
- Sample 512 distinct cyclic-rotation families of length 32 and 16 opens/16
  closes. Correct the string-to-orbit size bias by accepting a uniformly shuffled
  string with probability `1 / number_of_distinct_rotations`, then reject public
  or previously sampled orbits. Require an invalid rotation of each initial
  symbol. This is uniform sampling without replacement from that finite family
  population. Each family has one valid rotation and two invalid rotations, one
  starting with each symbol; within-orbit selection is deterministic by hash.
- Exclude every cyclic orbit represented in either published evaluation CSV.
  Unknown training overlap cannot be excluded: these are newly evaluated cases,
  not a proof of absence from the original training stream.

## Predictions before outcomes

Save/hash N/U-based stage-1 forecast rows before running R/G; save/hash N/R-based
stage-2 rows before running W/T. Invoke the repository's actual structural
preflight on these predictions. Old endpoints alone must leave the two rivals
equivalent; either hybrid can separate them when their endpoint gap is nonzero.
This is a conditional prediction using measured endpoints, not a forecast made
before any model evaluation. The symbolic rules are fixed before all new inputs.

## Frozen outcome rules

Primary substantive test: stage 2. Stage 1 is reported fully, including failures.
The same decision rule covers all four hypotheses, without switching outcomes.

- Margin: `logit(False) − logit(True)`, i.e. the log odds within the two task
  classes. Scientific tolerance **0.25 nat**: at most 0.0625 change in binary
  conditional probability by the logistic derivative bound. This tolerance is
  a declared approximation standard, not inferred from model variability.
- Numerical allowance **0.001 nat on prediction error**, comfortably above the
  observed development dtype discrepancy. Recompute every arm in float32 and
  float64; stop claims for any failed numerical or operator control. Higher
  precision is a reference, not mathematical ground truth.
- A family is separating for a stage if at least one of its three members has
  endpoint gap **greater than 0.502 nat**. Determine this from endpoints before
  either hybrid. Report eligibility separately for valid/invalid members; this
  conditioning may be driven mostly by invalid cases.
- For each hypothesis and family, error is the **maximum absolute prediction
  error across both hybrids and all three family members**. A definite hit has
  error <=0.249; a possible hit has error <=0.251. No averaging can hide a failed
  member or hybrid. The target is at least **80%** hits among eligible families.
- Below **128 eligible families**, label the inferential result insufficient;
  still report every measurement. A small endpoint effect is not equivalence.
- Simultaneous binary KL/Chernoff bounds valid for sampling without replacement:
  invert `n_eligible * KL(observed_hit_rate || bound) = log(K/0.05)`, where
  `K = tasks * 4 hypotheses * 2 one-sided bounds`. Lower bound uses definite
  hits; upper uses possible hits. Sampling without replacement has a moment-
  generating-function bound no larger than sampling with replacement. No
  Gaussian/noise assumption or independent-model assumption is used. Conditional
  on n, eligible families form a uniform subset of the eligible population.
  The without-replacement reduction is Hoeffding's convex-order result, stated
  in [Bardenet & Maillard (2015), Lemma 1.1](https://arxiv.org/abs/1309.4029).
- Adequate if lower >=0.80; excluded if upper <0.80; otherwise unresolved.
  Report each retained set, including both inadequate or multiple candidates.
  An adequate restricted hypothesis is not unique among undeclared mechanisms.
- Report per-case margins, exact finite accuracies, losses, label and starting-
  symbol strata, technical gates and the interaction. A better loss alone is
  not adequacy. No ratio is interpreted as a mediated fraction.

Precision planning, before new outcomes: with 36 tasks, K=288. The adequate
hit counts are 120/128, 230/256 and 446/512 for definite hits. Under an iid
binomial *planning reference* with true hit rate 0.90, approximate powers are
0.097, 0.585 and 0.986; the actual design samples without replacement. Thus 128
is a minimum reporting requirement, not a promise of adequate precision. This
binary bound replaces a proposed Hoeffding radius **before confirmation**; the
change addresses worst-case-variance conservatism, not observed test outcomes.

## Controls and stopping

Per task/dtype, native identity and last-layer EOS-only/full-query uniform must
agree within 1e-10 (float64) / 1e-5 (float32). All current values and off-target
preprojection slices must stay exactly unchanged. Verify stage-2 endpoint equals
R, symbol-mass preservation in W, within-symbol ratios in T, and unchanged
special tokens. Validate source/model hashes, all expected cells, finite values
and hook calls. Any failure is technical invalidity, not a scientific null.

After this bounded test, report the outcome. Do not search another model, change
the threshold, or add an undeclared followup to force adequacy. This application
can demonstrate successive prospective restrictions of an intervention account;
it cannot guarantee a generally superior method or certainty about all causal
explanations.

The freeze is a **local commit and hashed manifest**, not an externally timestamped
preregistration or a user-reviewed freeze. The user authorized continuing this
research; no remote publication is performed.
