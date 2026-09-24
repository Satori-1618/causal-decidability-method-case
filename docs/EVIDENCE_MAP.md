# What already supports the method?

**The central claim is about experimental design:** an intervention effect may be real
while several explanations predict it. The method specifies those explanations, finds
conditions that separate them, and reports the remaining ambiguity. Calculator accuracy
is a separate empirical claim.

**Current snapshot:** the Makelov application and its raw Q1 records are now included
in this checkout. Run `python3 examples/confirmed_read_source.py`; see the
[case explanation](CONFIRMED_CASE.md) and [verification scope](../RELEASE_NOTES.md).
Historical development branch names below are provenance, not branches readers need
to fetch. Shi and the full synthetic benchmark are not bundled with this release.

**Next step, not a new finding:** [round 2](ITERATIVE_IDENTIFICATION.md) applies the same
logic to removal versus preservation of the patch effect under a selected-query clamp.
Its four-cell demo is constructed, and the real-model contract is a draft. There is no
empirical route result, no approved adequacy tolerance and no implication that Q1's
candidate has become a uniquely identified mechanism.

## 1. The structural claim: demonstrated by construction

The [worked example](WORKED_EXAMPLE.md) has executable candidate rules. A single patch
effect fits value transfer, choice transfer, choice reversal and a fixed output choice.
Expanding the design separates some of these explanations. Nine candidates produce five
prediction patterns in six cells and seven patterns in twelve cells; three point-shift
rules remain indistinguishable.

This proves the stated possibilities for this toy. It does not establish how frequently
they occur in LLM research. The distinction exists before any statistical calculation.

## 2. Makelov: an existing empirical rival comparison

The strongest direct evidence is the **preregistered confirmation (Q1)** of a common-write,
split-read comparison that a development pilot introduced. Both use the direction published
by [Makelov, Lange and Nanda](https://arxiv.org/abs/2311.17030). Both are distinct from the
`resid_mid.8` planning experiment below.

**Question:** which component supplies the signal read by this patch? At GPT-2 Small's
MLP8 post-GELU activation, decompose the normalized direction `v` into an output-visible
component `v_R` and an output-null component `v_N`. For donor–recipient difference `Δh`,
hold the write direction fixed and change only the read source:

| Intervention | Inserted change |
|---|---|
| Full | `v (vᵀ Δh)` |
| Visible read | `v (v_Rᵀ Δh)` |
| Null read | `v (v_Nᵀ Δh)` |

| Idealized candidate | Visible-read prediction | Null-read prediction |
|---|---|---|
| A: visible component supplies the signal | Full-patch output | Baseline output |
| B: null component supplies the signal | Baseline output | Full-patch output |

A full patch alone does not decide between these endpoint descriptions. The two added
read-source conditions yield opposing predictions whenever full and baseline differ.
These are conditional predictions using measured full/baseline endpoints, not an
independent forecast of those endpoints.

**Pilot (development data).** On **32 base pairs**, with both swap directions grouped
within each pair, mean absolute prediction error was **1.338 nats for A** and **0.260 for
B**. B's paired advantage was **1.078 nats, nominal 95% bootstrap interval [0.961,
1.198]**. The saved verification reports passing identity, intervention-fidelity and small
paired numerical-reference checks.

**Supported by the pilot:** B predicts these interventions better than A. Q1 then tested
this on fresh pairs.

**Confirmed prospectively (Q1, 21 September 2026).** Frozen and pushed before the run
(`PREREG_READ_SOURCE_Q1.md`, development repository `829228c`). On 64 fresh base pairs,
with the pilot's case-wise loss, B predicted the two read interventions better than A in
**64 of 64** pairs (exact sign test p = 1.1e-19). Reported, not judged: mean loss 0.238
vs 1.223 nats. This confirms the comparison, not an adequacy claim.

**Not established:** that B is accurate enough (Q2, deferred until a tolerance can be
justified), exact, exclusively true, or a naturally used semantic variable.
No adequacy threshold was used to convert this loss comparison into a unique compatible
mechanism. The authors already examine row/null decomposition; this is an application
of the explanation check, not discovery of that decomposition.

**Historical packaging (22 September 2026):** both runs were packaged on development branch
`applications/makelov-2311.17030`. Its files are now included under that directory;
the new [records-only guide](../applications/makelov-2311.17030/RECORDS_ONLY.md) is the
entry point for verification.
Its `PROVENANCE.md` lists every copied file with its hash; each is byte-identical to the
private development repository. The upstream commit is
`e0c465b74561d9c3dd1f2afa770974bf5fcaee01`.

- **Pilot:** development repository `7401b78`; raw `records.jsonl` SHA-256
  `5e63b3b95cfd50324530ee62b703d37fcfda629460641926c07c803a39a93829`. A fresh CPU re-run
  reproduces the prompts and directions exactly and every summary statistic within 1e-6.
  The pilot was committed only on 21 September, a day after it ran, so its own manifest is
  the record of when it ran. It remains a development pilot without an adequacy threshold.
- **Q1:** preregistration frozen and pushed in `829228c` (SHA-256
  `f5bb62caaea253e7789f919b20a9bb78c68fc4e14e6219e260110f0f415f8a72`) before the run; run
  and score committed in `6539ce5`; raw `records.jsonl` SHA-256
  `eb4428e174b67dd65d3a9948a550273dc6fefe6adbfb4c014baadf82542f0ef1`. The run's manifest
  records `829228c` as its `git_head`. The added `scripts/check_read_source_q1_records.py`
  recomputes the judged result without downloads; the historical `check_read_source_q1.py`
  additionally depends on reconstructed tensor inputs. The development
  repository is private, so the order of freeze and run can be checked there, not here.
  There is no external timestamp.

## 3. Shi: a supporting resolution audit, not yet a second mechanistic comparison

For the sign-count test specified by [Shi et al.](https://arxiv.org/abs/2410.13032),
the existing module computes the probability that a declared alternative escapes
rejection. At `n = 40`, `ε = 0.1`, `α = 0.05`, an alternative with `θ = 0.80` still
passes about **16.1%** of the time. A passed test is therefore not a bound on `θ`.

This helps answer **whether an existing test can resolve a specified statistical
departure**. It does not supply two mechanistic explanations or a separating internal
intervention. The inspected upstream implementation also uses raw score differences
where the paper specifies a sign indicator. The power calculation applies to the
specified sign-count test; its applicability to published results requires resolving
the implementation/version discrepancy.

**Not included in this repository yet.** The power module (`src/equivalence_power.py`,
corrected in commit `1a2da38`), its tables (`docs/equivalence_power.md`), the audit
(`EQUIVALENCE_POWER_AUDIT_20260921.md`) and `docs/circuitry_issue_draft.md` are in the
private development repository. The inspected upstream version is
`ec9850b445b7dd0ed6fd93e9eaa44089b3a0a61e`.

## 4. Calculator validation: useful, with a narrower role

The [validation report](validation.md) preserves both successes and failures:

- The synthetic benchmark tests known constructed worlds. Its corrected pre-run
  agreement is 88.47% / 88.20%, below the frozen 90% target; reported false identifications
  are zero on that grid. Neither number is universal reliability.
- The prospective `resid_mid.8` application met its frozen 28/28 criterion, but all
  predictions were positive, the checks use nested data, and 21 outcomes excluded both
  endpoint candidates. This supports a limited planning claim, not mechanism recovery.

## Next work, in order

1. Use the method and worked example as the main explanation of the research goal.
2. ~~Package the existing Makelov read-source pilot~~ and ~~confirm the comparison
   (Q1)~~ done: see the included application. Q2, whether the better explanation is
   accurate enough, needs a tolerance with an independent scientific justification first.
3. Present Shi as the resolution audit it currently is. To make it a full second
   mechanistic application, first specify rival mechanisms and their intervention
   predictions; a new power table alone cannot do that.

New adjective experiments and further calculator tuning are not prerequisites for this
sequence. See the recorded [research priority](RESEARCH_SCOPE.md).
