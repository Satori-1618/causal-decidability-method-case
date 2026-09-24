# Round 2 execution protocol

Status: specified before development outcomes. The scientific contract implements
[the reviewed plan](../../docs/ROUND2_RESEARCH_PLAN.md). A separate confirmation
manifest will be committed before confirmation forwards.

## Target and scope

GPT-2 Small, fixed published MLP8 `read_null` / full-write intervention, queries of
heads 9.6, 9.9 and 10.0 at the last prompt token. Five scientific conditions A–E and
two implementation identities F/G follow `query_route.CELL_MEANINGS`.

Three operational predictions for `(R,S)` are frozen: transfer `(0,T)`, joint dependence
`(0,0)`, preservation `(T,0)`. They are not exhaustive causal graphs. In particular,
the joint-dependence profile is compatible with gating, not proof of a particular gate.

Each profile must predict both contrasts within 25% of the absolute local T in both
reciprocal directions and both float32/float64 evaluations. The numerical gate requires
nonzero same-sign anchors and every cross-precision T/R/S discrepancy at most 2.5% of
the smaller absolute T. Unresolved cases remain in the denominator as non-successes.

The population requirement is 80%. Three simultaneous two-sided exact-binomial
Clopper–Pearson intervals use alpha/6 = 0.05/6 in each tail. Lower > 0.80 means adequate;
upper < 0.80 means excluded; otherwise unresolved. This uncertainty statement is about
sampling for the declared instrument, not exact arithmetic or semantic identification.

## Independent sampling

The source vocabulary/template split is the same as round 1. Its Cartesian proposal
contains over 31 million tuples before token eligibility, so it is defined factorwise
rather than exhaustively materialized. This is a computational implementation of the
frozen distribution, not an outcome-driven restriction of its support.

`sample_iid_cases` draws proposal tuples independently with replacement, rejects only
token/text ineligibility and frozen historical exclusions, and retains duplicate
accepted content if it occurs. Each draw has a distinct ID. Both reciprocal directions
belong to one unit. The manifest hashes the full population **definition**, including
source files, tokenizer, eligibility, exclusion set and draw policy; it does not pretend
to hash an enumerated population. No case is filtered by baseline correctness or effect.

- Development: 32 draws, seed 24092401.
- Confirmation: 192 draws, seed 24092402; condition on exclusion of all development
  prompts as well as the historical artifacts listed in the manifest.
- This is prompt-level freshness within the same templates/model, not new mechanism
  families or a claim about original training-data membership.

## Technical qualification and development decision

Restore the historical runtime in an isolated environment. TransformerLens 2.17.0's
metadata asks for torch >= 2.6, while historical Q1 used torch 2.5.1. Preserve and record
that override, verify offline loading, and replay the first two archived Q1 pairs before
new forwards. Require maximum saved-margin difference <= 1e-4 nat; report the actual
difference. This check does not validate the new hooks.

For each new forward, audit absolute position, head slices, execution order, one call per
declared site, finite scores, unchanged non-target slices locally, and post-cast insertion
fidelity. Require baseline query self-insertion, patched query self-insertion and the
zero-delta MLP control to reproduce their references within dtype/scale-derived budgets.
Technical failures block interpretation; repair must precede a newly frozen run, with
any already exposed cases retained as development data.

Proceed to confirmation if all development technical controls pass and at least 26/32
base pairs pass the declared numerical-resolution gate. This is an instrument-feasibility
gate, not evidence for a candidate. Inspect profile frequencies for planning, but do not
change the profiles, thresholds, sites or n. A pilot that suggests all profiles fail can
still justify confirmation of that restricted negative conclusion. If the instrument
gate fails, release the development result and its limitation rather than score fresh
confirmation data with a defective instrument.

## Freeze and execution

Commit code, tests and this protocol before development forwards. Before confirmation,
commit its exact cases and manifest (code, model, vector, source, runtime, sites,
thresholds and input hashes); push that branch before execution to provide an external
timestamp. No author contact is authorized or needed. Run each frozen directory once.
The runner verifies input hashes, writes a start ledger, preserves measurements in both
precisions and refuses overwriting an existing run.

The fp64 reference promotes the same processed float32 weights; caches and patch deltas
are recomputed independently. Agreement is instrument stability, not exact ground truth.
No interim outcome stopping, unplanned sample extension, or silent replacement of cases.

## Reporting

Publish all seven cells, final T/R/S and interaction contrasts, profile counts and
simultaneous intervals, weak-anchor/precision failures, tensor controls, total effects,
sampling provenance and a records-only checker. All-excluded means no declared profile
meets 80% population coverage; it does not mean no case fits or no causal effect exists.
Retain round 1. Claim neither a unique native mechanism nor identification of meaning.
