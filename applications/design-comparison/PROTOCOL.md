# Equal-budget intervention selection: development contract v1

Status: **development only**. The structural holdout is not authorized by this runner.
No model download or LLM inference is needed. This is a benchmark of experiment selection,
not a claim that maximin experimental design is new.

## Question and executed computations

Can prediction-guided selection recover the correct full-menu explanation class more
often than strong alternatives, given identical measurements and inference?

The world module executes interventions on intermediate activations of small PyTorch
CPU graphs. Their native and ordinary full-transfer endpoints match across candidates.
Their internal computations differ: channel A, channel B, additive combination, joint
requirement, and context-dependent switches. An algebraic alias must never be split;
an out-of-set graph tests `no_candidate_fits`. This is a controlled computation benchmark,
not a simulation of the distribution of mechanisms inside real transformers.

Candidate predictions are independent analytic expressions, checked against executed
graphs. They are supplied to **every** policy. Knowing these predictions and parameters
is an explicit advantage of this testbed, not a result established for LLMs.
The interventions replace individual hidden-channel coordinates. They are inspired by
split controls but do **not** reproduce Makelov's read-source/full-write operator.

## Shared budget and policies

Four compulsory cells measure native/full transfer in both contexts. The primary budget
adds **three** distinct cells, with **eight** measurement samples per cell: 56 observations.
Secondary budgets add one, two or four cells, and are not independent replications.
Each candidate cell has unit measurement cost. Graph qualification and prediction-table
construction are common setup costs and reported separately; selection time is recorded.

| Policy | Rule | Role |
|---|---|---|
| maximin | Exhaustively maximize the smallest distinguishable-pair gap divided by twice the simultaneous radius; mean capped separation breaks ties | Proposed selection rule |
| mean_pairwise | Exhaustively maximize mean squared pairwise gap in units of known standard error | Main substantive comparator, inspired by standard discrimination design; no claim to reproduce a particular paper's entire workflow |
| balanced_split | Replace A then B in context 0, then A then B in context 1, at unit dose, then the other doses | Predetermined channel-split control |
| random | Uniform sample without replacement from the same extra cells | Equal-budget selection baseline |
| full_only | Spend the whole budget on the four compulsory native/full cells | Nondiscriminating negative control, never the headline competitor |
| full_menu | Measure every cell | Higher-cost diagnostic reference; never treated as a budget-matched competitor |

The exhaustive selectors see only public predictions, menu, uncertainties, equivalence
groups and budget. They cannot read graph identity or realized measurements. They do not
adapt to outcomes. The same potential measurements are shared across policies, making
the comparison paired. Baseline order and tie rules are frozen in code before each run.

## Noise, numerical fidelity, and inference

The graphs are deterministic. Artificial measurement error is explicitly **independent
Gaussian noise with known standard deviation**, not asserted to model LLM rounding or
empirical noise. Noise levels are 0.03, 0.10 and 0.30 score units, with public per-cell
heteroscedastic factors. The graph output is a generic score, **not nats**.

For M cells in the full permitted menu, the simultaneous radius in cell j is
`z(1-alpha/(2M)) * sigma_j / sqrt(n_j) + numerical_bound`, with `alpha=0.005`.
Bonferroni controls exclusion of a correctly specified true candidate over all cells
under the stated known-noise model. It also covers any selected subset; no candidate-count
penalty is needed for retention of the true candidate. It is not a universal error bound.
Different budgets/policies use the same first eight noise draws except that full-only
spends additional draws on anchors, as explicitly recorded.

The numerical term covers reference/float32 discrepancies at the actual intervened
graph outputs on this finite declared graph/menu, with a documented allowance. It is
separate from statistical uncertainty. Analytic-reference, self-patch and shared-endpoint
checks must pass. A failed technical check is `invalid`, not a candidate exclusion.
Scientific approximation tolerance is zero: these are fully specified known graphs.

All policies call unchanged `causal_decidability.compatible_set`. Equivalence classes
are fixed over the **full menu**, never recomputed from a policy's subset. Correct
resolution means exact equality between the returned set and the true full-menu class.

## Population, splitting, and evidence boundary

Development uses `linear` and `relu_offset`; `squared` and `saturated_relu` are reserved
structural transformations. The runner refuses to execute any reserved family.
This is separation by downstream graph composition, not merely fresh seeds; the same
six mechanism classes remain. It is not a holdout of wholly new mechanism classes and
covers only a small hand-designed set. Code for a holdout structure being visible is not
operator blindness. Candidate discovery and parameter estimation are outside this test.

Each independent case draws family, amplitude band, amplitude, noise level and truth
from the declared distribution. Amplitude bands are [0.1,0.3], [0.3,1] and [1,3]. Six
distinct in-set classes and the out-of-set truth are sampled uniformly; the equivalent
alias is not double-weighted. Outputs are stratified by family, truth, amplitude band,
noise and pre-outcome separation/uncertainty band. No overall rate is field prevalence.
Policies, cells and budgets within a case are paired, never additional independent n.

512 independent development cases is the default. A smaller smoke run checks plumbing,
not scientific success. Case IDs are opaque. Public case inputs, private truth labels,
measurements, predictions and scores are separate artifacts. Hashes bind each stage;
scoring reads labels only after predictions are written. This single-operator workflow
provides an access boundary and tamper checks, not proof that the author was blind.

## Outcomes and proposed later confirmation target

Primary: paired difference in exact correct-class recovery on in-set cases at three extra
cells, maximin minus mean_pairwise. Report the same difference against balanced_split and
random. All denominators and costs are explicit. Also report true-class exclusion,
false single-class identification, partial/no resolution and out-of-set rejection.

Development differences receive simultaneous conservative Hoeffding intervals over the
three primary comparisons. Per-policy exclusion rates receive exact one-sided binomial
upper bounds with a declared multiplicity correction. These quantify this sampled
development distribution; development selection still prevents confirmation claims.

Proposed confirmation target: lower bound on the primary improvement **above 0.05**, and
upper bound on false exclusion **at most 0.01**. This is a proposal to power and freeze,
not a target to optimize the current generator against. A tie with the strong comparator
is not success on superiority. No post-hoc replacement of comparator, amplitude mixture,
budget or threshold is permitted to turn a developmental loss into success.

Before confirmation: independent code audit, sample-size/power plan against declared
alternatives, immutable versioned protocol/code/generator, and explicit authorization of
the reserved structural evaluation. The current CLI cannot run confirmation.

## Commands and provenance

See README for generation, prediction, scoring and verification. Each new directory is
create-only. A manifest records seed, configuration, source hashes, Git revision/dirty
state, Python, NumPy, Torch, CPU platform, timings and all artifact hashes. Previously
frozen application files and their claims are unchanged.
