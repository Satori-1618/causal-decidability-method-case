# Round 3B stopped before patching: the native task did not qualify

**Result:** GPT-2 Small did not reliably answer the proposed three-role questions,
and its native responses did not provide the required separation between rivals.
Both prespecified feasibility gates failed; the numerical checks passed. **No 3B
patch was run.** This is a development result about this task formulation, not a
negative result about role representations or a completed mechanism comparison.

The [Goodfire crosswalk](GOODFIRE_CROSSWALK.md), native protocol, cases and code were
committed and pushed as [`24cc60e`](https://github.com/Satori-1618/causal-decidability-method-case/commit/24cc60e)
**before measurement**. The run used 32 fixed families, without replacing failures.

## A concrete example

The first sampled family starts:

> Then, Dean gave a scarf to Savannah while Paige watched.

The correct names are Dean for giver, Savannah for receiver, and Paige for observer.
Among those three names, the model ranked **Dean highest for all three questions**
in this context. Its unrestricted next-token choice was Dean for the giver question
and ` a` for the other two. This is the first draw, not a selected worst case; all
prompts and scores are retained.

The intended next experiment would ask whether a patch transfers *which role to
retrieve*. If the unpatched questions do not reliably elicit their intended answers,
a later change in the names' scores cannot straightforwardly establish that claim.
The geometry check asks a second question: even under the most favorable permitted
gain, are the declared predictions far enough apart to test? Here that also failed.

## What was measured

32 independently sampled name-triple/object families each contained 48 native rows:
two wordings, eight labeled contexts and three queried roles. Every row was evaluated
in fp32 and promoted-fp64. These are repeated conditions within **32 units**, not
3,072 independent examples. Native batch-versus-single replays added 64 evaluations.

The model run and analysis took **57.14 seconds on CPU**: 320 forward calls,
3,136 prompt evaluations, zero patch forwards. Implementation, source audit and
verification time are additional; this is measured computation time only.

| Prespecified requirement | Observed | Decision |
|---|---|---|
| At least 90% conditional name accuracy in every role × wording × form group | All 18 groups failed; group accuracies ranged from 6.25% to 75.00% | Stop competence |
| All marked rival gaps >0.52 nat at optimistic gain 1, in both wordings/precisions, for at least 29/32 families | 0/32 families passed | Stop geometry |
| Native and derived precision discrepancies at most 0.01 nat | Maximum 0.0001681 nat; all 32 families passed | Pass numerics |
| Batch-versus-single native replay within recorded dtype tolerance | All 64 passed; maximum selected-logit discrepancy 0 | Pass replay |

The two scientific failure reasons are preserved together. A replay with zero discrepancy
does not establish numerical accuracy; that is why the separate precision comparison
was retained. Neither check establishes a semantic mechanism.

### All competence groups

The fp32 and fp64 accuracies are identical. Each entry first averages the contexts
inside a family and then the families, as frozen. `fit` and `heldout` are wording
labels; both are development measurements here, not confirmation data.

| Wording | Story form | Giver | Receiver | Observer |
|---|---|---:|---:|---:|
| fit | gave | 46.88% | 50.78% | 14.84% |
| fit | received | 28.13% | 75.00% | 6.25% |
| fit | watched | 45.83% | 45.83% | 18.75% |
| heldout | gave | 29.69% | 51.56% | 17.97% |
| heldout | received | 50.00% | 50.00% | 18.75% |
| heldout | watched | 60.42% | 31.25% | 22.92% |

Descriptively, accuracy across all logical rows is **36.46%** when restricted to the
three names and **1.95%** for unrestricted next-token argmax. The names receive **7.69%**
of total next-token probability on average in fp64. These extra diagnostics were
reported, not substituted for the frozen conditional-accuracy gate. Nominal family
bootstrap intervals for all groups are in the summary; none rescues a failed point gate.

### Geometry has its own failure

At gain 1, no family passes all fixed diagnostics even in either wording separately.
For the fit wording, zero families pass the mention-position diagnostic and zero pass
the question-change-only diagnostic. This is a failure of the available native
separation at the declared resolution, **not a measurement of patch behavior**.
The largest family-minimum gap is 0.2293 nat for fit wording and 0.4458 nat for heldout
wording, both below 0.52. A smaller allowed gain cannot fix these diagnostic gaps.

## What the literature audit changes

The paper's group-position address is not our name's mention slot, and its lexical
retrieval uses an entity-valued query rather than our role question. A reflexive pointer
and answer copying also share target labels when every donor name exists in the
recipient. The [crosswalk](GOODFIRE_CROSSWALK.md) records these distinctions and the
requirements for a future absent-target assay.

Thus, even a passing 3B experiment would not automatically separate all mechanisms in
*Mixing Mechanisms*. The gain would be a quantitatively supported response class,
with equivalent implementations explicitly left open.

## What follows

**Do not run Stage B or confirmation on this frozen task.** No model size, wording,
threshold, cases or layer was changed after the result. A new discovery stage may
qualify a different task/model combination and then freeze a fresh test. Directly
adapting the published multi-group entity-binding task is another separate option;
it must first qualify its own model, intervention and readout.

This run does not show that GPT-2 lacks roles, that MLP8 cannot carry them, or that the
earlier Makelov-derived results were wrong. MLP8 was not intervened on. What it shows
is that **this native task and calibration do not support the proposed next
identification step at the prespecified resolution**. The method caught that before
spending on, or interpreting, an unqualified patch experiment. It adds no positive
mechanism identification on its own.

## Check the evidence without running a model

```bash
python3 -I -S applications/makelov-2311.17030/scripts/check_role_baseline_records.py \
  applications/makelov-2311.17030/results/role_baseline_development --check-only
```

The independent implementation uses only the standard library and checks the
execution-commit input blobs, hashes, raw row scoring, replay controls, geometry and
both stop reasons. It does not independently rerun the model or verify bootstrap/CP
intervals. The post-run `--check-only` convenience option leaves the original
`verification.json` untouched and changes no scoring rule. Freeze-bound model
execution requires the original freeze revision.

- [Frozen protocol](../applications/makelov-2311.17030/ROLE_BASELINE_PROTOCOL.md)
- [Manifest and input hashes](../applications/makelov-2311.17030/results/role_baseline_development/manifest.json)
- [Cases and exact prompts](../applications/makelov-2311.17030/results/role_baseline_development/cases.json)
- [Raw measurements](../applications/makelov-2311.17030/results/role_baseline_development/records.jsonl)
- [Analysis with every group, interval and diagnostic](../applications/makelov-2311.17030/results/role_baseline_development/summary.json)
- [Independent verification](../applications/makelov-2311.17030/results/role_baseline_development/verification.json)

Manifest SHA-256: `47bdebd54c1b9c89c17f73c960d28ec06e120ffce7b32414ad3ccd180c63e784`.
Raw-record SHA-256: `fa5f7fae1b1f306a3813d44b394c45b9ddfcf42dd9484458153623e402051b72`.
