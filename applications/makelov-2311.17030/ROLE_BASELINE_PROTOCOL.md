# Round 3B Stage A: native feasibility protocol

**Frozen before any 3B model output. No intervention and no confirmation claim.**
The question is whether GPT-2 Small can handle the three-role prompts and whether
its native endpoints could distinguish the proposed response classes. This is not a
search for a layer, direction, wording or gain that produces a desired result.

Read the [positive-identification design](../../docs/ROUND3B_POSITIVE_IDENTIFICATION.md)
and [Goodfire crosswalk](../../docs/GOODFIRE_CROSSWALK.md) first. The latter separates
our mention-position rival from group-index retrieval and records unresolved lexical
and reflexive alternatives. No absent-target pointer assay is included here.

## Fixed population and measurements

- **32 IID family draws**, seed **24092441**, with replacement from the declared
  tokenizer-eligible population. Each family is a distinct ordered name triple plus
  one object; duplicate content is retained as separate draws, not removed for diversity.
- Names come from the held-out half of the authors' source list and must be distinct
  single leading-space GPT-2 tokens. Objects come from the held-out source half,
  intersected with the count-noun whitelist recorded in `role_baseline_sampling.py`.
  The whitelist is fixed for grammatical `a {item}` prompts before model outcomes.
- Exactly **48 native rows per family per precision**: two wordings, eight labeled
  contexts (two donors plus six recipients), and three role questions. Duplicate
  context text is retained under its declared IDs. Rows are repeated conditions, not n.
- Exact earlier prompts are excluded before sampling. Source lists, exclusions, token
  IDs, absolute positions, name spans, proposal/rejection counts and tokenizer revision
  are saved. Eligibility never uses model answers. A bounded unsuccessful sampler stops.
- Use `role_design.render` unchanged, including `Then, `. Each story's three query
  variants must have equal token length. The matched d0/giver and d1/receiver donors
  must preserve the answer token, all name positions and total length. No padding.
- GPT-2 Small, cached weights only, CPU, four threads, fixed processed-weight transforms
  matching earlier rounds. Native next-token logits at each prompt's own final position;
  **no MLP patch, attention intervention or vector loading**.
- Run fp32 and promoted-fp64 separately with the same processed weights. Save all
  three name logits, full-vocabulary argmax and total probability mass of those names.
  Higher precision is a reference, not exact truth.

## Gates fixed before outcomes

**Technical validity.** Check input hashes, tokenizer replay, finite outputs and complete
row coverage. One native batch-versus-single replay per family/precision must meet
`64 * dtype epsilon * max(1, absolute selected logit)`; report actual errors and bounds.
Cross-precision pairwise name-margin discrepancies must all be at most **0.01 nat**;
the same bound applies to derived role predictions, oracle projections and diagnostic
gaps. Discrete correctness changes are reported and each precision must pass separately.
A failed technical check is invalid measurement, not scientific evidence against roles.

**Competence.** Require at least **90% conditional three-name correctness** in every
`query × wording × form` group, independently in each precision. The forms are `gave`,
`received`, `watched`; queries are giver, receiver, observer. Tied maxima fail. Average
within each family first, then across the 32 families. Report full-vocabulary accuracy,
name mass and nominal whole-family bootstrap intervals; they do not override the
prespecified point threshold. This is a feasibility rule, not a confidence-qualified
claim of 90% population competence.

**Potential geometric separation.** Use native recipient endpoints to construct the
fixed primary directions and `role_design.DIAGNOSTICS`. At the **maximum permitted
role gain of 1**, require every marked diagnostic gap from the name-line, position-line,
question-change-only and no-op rival sets to exceed **0.52 nat RMS**, in both wordings
and precisions, for at least **29 of 32 families** (at least 90%). Keep weak families
in the denominator. This optimistic screen can fail even when the model knows the
answers; three labels do not guarantee three distinct quantitative directions.

The 0.52 resolution comes from the existing 0.25-nat adequacy rule and 0.01 numerical
allowance. Gain 1 is neither estimated nor evidence of transfer. Passing means only that
these native directions could be separating at that gain. It does not establish
adequacy, power or separation at the gains that a later patch experiment might fit.

**Descriptive uncertainty.** 20,000 whole-family bootstrap draws, seed **320260940**,
nominal 95% intervals. No multiplicity-adjusted population claim is made from these
development intervals. The reported unit is 32 families, never 1,536 prompts or 3,072
precision-specific rows. All groups and all failure reasons must be reported.

## Decisions and scope

| Outcome | Required action |
|---|---|
| Technical or numerical failure | Preserve artifacts; block scientific interpretation |
| Competence failure | Stop this task formulation; no 3B patches or outcome-selected prompt repair |
| Geometry failure | Stop this quantitative design; no inference that role information is absent |
| All gates pass | Eligible for a separately frozen development run; no automatic confirmation |

If several gates fail, report each. A failed native task gate concerns GPT-2 on these
prompts; it cannot localize a problem to MLP8, which was not intervened on. A new task
or model would require an openly different development design and fresh cases.
Heldout wording here means a wording intended to be withheld from **patch gain fitting**;
its native outputs are deliberately inspected in Stage A. Neither wording is confirmation.

## Freeze and execution

Prepare uses the tokenizer and file hashes only, without model forwards. Commit the
protocol, code, cases and manifest before execution. The runner requires the exact
manifest hash and a clean committed tree; it refuses changed inputs or an existing
execution marker. Preserve partial and failed runs. It never selects replacements or
calls Stage B. Freeze source and runtime hashes, including the historical dependency
override (TransformerLens 2.17.0 declares torch >=2.6, while the qualified earlier
runtime uses torch 2.5.1).

```bash
.venv-round2/bin/python applications/makelov-2311.17030/scripts/run_role_baseline.py prepare \
  --output applications/makelov-2311.17030/results/role_baseline_development
# Commit the prepared inputs; use the printed manifest digest below.
.venv-round2/bin/python applications/makelov-2311.17030/scripts/run_role_baseline.py run \
  --output applications/makelov-2311.17030/results/role_baseline_development \
  --manifest-sha256 DIGEST_FROM_PREPARATION
```

The output retains every row and precision, the full gate analysis, timings and hashes.
An independent records-only recheck is required before the result is summarized publicly.
