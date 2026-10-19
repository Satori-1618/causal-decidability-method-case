# Evidence map

**Claim of this release:** an explicit rival table can expose an uninformative
experiment, guide a separating condition, and keep empirical conclusions within the
comparisons actually tested. The planning screen, rival generation and empirical
mechanism identification are separate claims. The cases below do not jointly prove
that the whole procedure is generally reliable.

## Included evidence

| Case | What was tested and found | Canonical record | What it does not establish |
|---|---|---|---|
| **Makelov Q1: read source** | Holding the write direction fixed, compare visible-only and null-only reads. B has lower prediction error in 64/64 fresh base pairs. | [Result](CONFIRMED_CASE.md), [protocol](../applications/makelov-2311.17030/PREREG_READ_SOURCE_Q1.md), [records-only check](../applications/makelov-2311.17030/RECORDS_ONLY.md). | Absolute adequacy, exclusive truth, or native use of the null component. Both rivals reproduce their measured full-patch anchor by construction. |
| **Makelov round 2: query route** | Reset selected queries and transfer those queries alone. All three predefined response profiles are excluded on 192 fresh pairs. | [Result](ROUND2_RESULT.md), [protocol](../applications/makelov-2311.17030/QUERY_ROUTE_PROTOCOL.md). | A complete route or a binary conclusion that the queries participate/do not participate. The reverse condition did not change the exclusion already implied by the reset. |
| **Makelov round 3A: donor factors** | Cross donor name and position. Position-only and name-only invariance profiles fail the declared coverage requirement on 512 fresh families. | [Result](ROUND3A_CONFIRMATION.md), [protocol](../applications/makelov-2311.17030/DONOR_FACTOR_PROTOCOL.md), [prospective size amendment](../applications/makelov-2311.17030/DONOR_FACTOR_512_AMENDMENT.md). | A transferred person or semantic role. The original planning STOP and amendment remain part of the record. |
| **Makelov round 3B: role-task qualification** | Native competence and separation fail on 32 frozen families; numerical checks pass. No patches are run. | [Result](ROUND3B_STAGE_A_RESULT.md), [protocol](../applications/makelov-2311.17030/ROLE_BASELINE_PROTOCOL.md). | Absence of role representations. This task cannot support the proposed comparison. |
| **Tracr: known reversal program** | 128/128 fresh token-assignment families reproduce the declared two-candidate tie, separation, and new prediction under the frozen test. | [Result](../applications/tracr/RESULTS.md), [freeze](../applications/tracr/CONFIRMATION_FREEZE.json), [raw records and summary](../applications/tracr/results/confirmation_001/). | Blind discovery: the address site was selected from compiler structure. No generalization to other algorithms, pretrained models, or an advantage over a strong fixed design is shown. |
| **Mixing Mechanisms: concentration** | On 300 qualifying fresh families, the cell-average concentration profile W_T is excluded; the agreement-like profile A_T remains undecided. | [Result and scope](../applications/gur-arieh-2510.06182/CONFIRMATION_RESULT.md), [freeze](../applications/gur-arieh-2510.06182/FREEZE.json), [raw records](../applications/gur-arieh-2510.06182/results/confirmation/). | A per-case causal mechanism, exclusive lexical/positional switching, or a refutation of the paper's aggregate mixture account. W_T is this application’s restricted profile; its pre-declared caveat says it is biased toward exclusion when case-level argmaxes vary. |
| **Goodfire CausaLab: MCQA input designs** | Each published input family leaves two of three declared rules tied; their combined input menu separates those three. Correct-color transfer remains tied with task-answer preservation. | [Audit](../applications/goodfire-mcqa-preflight/README.md), [source hashes](../applications/goodfire-mcqa-preflight/artifacts/upstream/source_manifest.json), [predictions](../applications/goodfire-mcqa-preflight/results/). | A measured neural mechanism or resolved measurement plan. Testing the combined menu requires one fixed patch; the upstream designs fit separate patches. |

These counts have different units. A family can contain several conditions, tokens,
or precision runs; they do not become additional independent cases.

## Planning validation: an explicit limitation

The [historical validation report](validation.md) retains the corrected results:
**D1 forecast agreement is 88.47% / 88.20%, below the frozen 90% target.** The earlier
91.38% / 91.48% figures used an incorrect bfloat16 mantissa constant and are superseded.
The full synthetic benchmark is not bundled here, so its reported rates are not
recomputed by this release's tests.

The separate Makelov resid_mid.8 planning study met its 28/28 rule, but all forecasts
were “decidable,” sample sizes were nested, and 21 outcomes excluded both endpoint
candidates. A constant positive forecast would also succeed. This does not validate
predictions of weak designs, small-sample calibration or general mechanism recovery.

The [current preflight](METHOD_PREFLIGHT.md) is deliberately narrower: a structural
comparison plus a conditional numerical screen. Known Gaussian variability, pilot
estimates and numerical allowances have different evidential status. A sensitivity
estimate is not automatically a bound, and passing this screen is not a power claim.

## What “frozen before the run” means here

Hashes bind records to specific protocol and code files; they do not establish an
independent date. A records-only replay validates stored computations, not historical
human review or a new model run.

- **Makelov Q1:** the result manifest records development commit `829228c`; raw records
  were committed in `6539ce5`. Earlier records report a pre-run push to the private
  source repository; this release does not independently establish that timestamp. The
  [provenance record](../applications/makelov-2311.17030/PROVENANCE.md) preserves original
  file hashes; [verification](../applications/makelov-2311.17030/VERIFICATION.md) distinguishes
  the later rounds and their source records.
- **Tracr:** the freeze, seeds, source hashes and run records are included. Freshness
  means new token assignments within one known program and design. Consult the
  [development and confirmation history](../applications/tracr/RESULTS.md#development-history-and-limits)
  rather than treating all development reruns as confirmations.
- **Mixing Mechanisms:** freeze commit `e96b357e1e5a189656166a451ad13727d1d67375` and the
  run manifest support local before-run chronology. That freeze was not pushed before
  confirmation; no independent timestamp or separate human sign-off is established
  by the repository. The original contract and artifacts remain unchanged.
- **Goodfire MCQA:** a retrospective audit of inputs pinned at upstream commit
  `8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e`, not a fresh-data confirmation.

Publishing local history later does not turn it into a contemporaneous public freeze.
For a new study, record the freeze externally at the time and distinguish approval,
protocol binding, software replay and fresh empirical testing.

## Scope of the remaining material

- [Worked examples](WORKED_EXAMPLE.md) demonstrate logical possibilities by construction.
  They do not measure their frequency in research or a benefit over expert planning.
- [Mixing Mechanisms round 2](../applications/gur-arieh-2510.06182/ROUND2_DEVELOPMENT_RESULT.md)
  is development only. It is not an additional confirmed mechanism.
- The [rival-generation prompt](prompts/CAUSAL_PREFLIGHT_PROMPT.md) is supplied for use,
  but improved AI-assisted rival generation has **not been evaluated**. A two-case
  pilot was proposed and remains future work.
- Beckmann is a separate application in another repository, outside this release's
  reproducible evidence. No conclusion here depends on its unavailable data.
- The earlier Shi resolution audit is also unbundled. It addressed statistical test
  power and an implementation discrepancy, not a second mechanistic comparison.

The useful output is a **scoped distinction or a documented limit**: one or several
rivals retained, none fitting under the declared criterion, missing resolution, or
an invalid intervention. None of these outputs licenses a stronger claim about an
undeclared rival or the model's complete native mechanism.
