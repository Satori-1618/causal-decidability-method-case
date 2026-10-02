# Independent artifact verification

**PASS for the recorded development run; the confirmation-start gate remains
false.** A separate agent in the same AI-assisted workflow implemented
[verify.py](verify.py) using only Python's standard library. It imports none of
the producer, runtime, design or analysis modules. This is an independent code
path over saved artifacts, not an external replication or a new model run.

## What was checked

- All 17 manifest-bound source files and five output files match their recorded
  hashes. The prepared inputs, run records and forecast receipt agree.
- Every donor satisfies the declared length, bracket counts, target symbol,
  position, current prefix balance and prefix-minimum constraints. Prefix hashes,
  phase partition, exclusions and within-cell distinctness were recomputed.
- Recipient positions are the first highest-attention closing brackets in their
  saved fp64 EOS attention rows. Their attention coefficients and native margins
  match the transfer records.
- Anchor forecasts have the specified six-target formulas and match the final
  anchor records. Local timestamps put the forecast receipt before target
  measurement. These records are not an external timestamp guarantee.
- For all **512 snapshots**, the verifier reconstructed the intended head vector
  `h_r + a_r * (v_d - v_r)`, including fp32 rounding after each arithmetic step.
  The recorded delivered vector equals the intended vector exactly. Construction
  roundoff, margin differences, prediction errors and dtype discrepancies were
  recomputed.
- The maximum error over all six targets, same-label prefix controls, strict
  eligibility rule, definite/possible hit rules and start gate reproduce the
  [analysis report](results/development_report.json).

## Recomputed result

| Quantity | Verified value |
|---|---:|
| Development families | 32 |
| Eligible families: anchor gap strictly above 0.202 nat | 8 |
| Balance-class candidate definite hits | 7/8 |
| Position candidate definite hits | 0/8 |
| Maximum same-label prefix difference | 0.0725724 nat |
| Maximum fp32/fp64 prediction-error difference | 0.00000307823 nat |
| Confirmation-start gate | **Not met** |

The gate requires at least 16 eligible families and at least 90% definite hits
for one candidate. These are development results, not confirmed exclusions or
identification of a semantic variable.

## Reproduce without model inference

From the repository root, using Python 3.9 or later:

```bash
python3 -B -S -m unittest discover \
  -s applications/li-saphra-2507.06445/value_followup/tests \
  -p test_verify.py -v

python3 -B -S applications/li-saphra-2507.06445/value_followup/verify.py \
  --run applications/li-saphra-2507.06445/value_followup/results/development_001 \
  --cases applications/li-saphra-2507.06445/value_followup/inputs/development_001/cases.jsonl \
  --report applications/li-saphra-2507.06445/value_followup/results/development_report.json
```

Both commands were executed: **11 tests passed**, and the artifact audit passed.
The saved [verification report](results/verification_report.json) includes the
recomputed family results and artifact hashes. Optional `--output PATH` writes a
new report and refuses to overwrite an existing file.

The combined follow-up suite passed **30 tests**. A clean export of the final
artifact tree also passed this verifier with standard-library Python and no
model cache; running the analyzer there reproduced the committed report exactly.

## Verification boundary

The audit verifies records and arithmetic; it does not rerun the neural model.
Full attention and value arrays were not saved, so unchangedness outside the
intervention is checked as a producer-reported control, not independently
reconstructed from complete tensors. Recipient selection is verified against
the saved attention row, not a fresh forward. Local hashes and timestamps bind
the recorded workflow but do not establish external preregistration. A future
confirmation would additionally need the cross-role prefix-freshness audit
described in the [README](README.md); none was started here.
