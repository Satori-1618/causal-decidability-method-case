# Preparation checks — 4 October 2026

**Implementation tested; scientific decision not evaluated.** No checkpoint was
loaded and no new model measurement was made. These internal checks do not
constitute the independent pre-run review or an execution release.

| Check | Result |
|---|---|
| New model-free suite | **50 tests passed** |
| Unchanged averaged-anchor 004 suite | **67 tests passed** |
| Full input regeneration | Byte-identical prepared files |
| Finite matched-stem construction | 13,260 possible stems; 5,158 eligible; 2,538 calibration / 2,620 target |
| Prepared input count | 2,048 candidates; 256 families; 1,024 matched quartets; 4,096 donors |
| Historical donor-prefix and cross-role overlap | Zero at both 20 and 28 symbols |
| Reuse, reported without resampling | 937 distinct stems; maximum reuse three; one new recipient–donor first-20 overlap, zero full-string or first-28 overlap |
| Candidate arithmetic | Recency, ending, shared offset, pure interaction, equivalence and dtype-boundary synthetic cases exercised |
| Independent sign-test arithmetic | All 33,153 win/loss counts with at most 256 non-neutral cases matched SciPy; maximum p discrepancy 1.77×10⁻¹³ |
| Independent power calculation | Ten hypothetical scenarios agreed within 10⁻¹¹ |
| Fake execution path | 63 separating families stop; 64 start all 256 targets; forecasts precede targets; failed forwards are recorded without retry |
| Pending release | Refused before Torch import or output creation |

The independent arithmetic check used SciPy 1.15.3. The shipped calculations
and normal tests need only Python's standard library; planning-table comparison
allows tiny rounding differences only in calculated floating-point outputs,
never in scientific thresholds or decision constants.

Internal review corrected two overbroad phrases before freezing: downstream
computation is **unpatched and recomputed**, and the previous exclusion concerns
its **specific frozen balance-only predictor**, not every balance representation.
The prefix-ending rule is explicitly distinguished from the future suffix.

Reproduce from this directory:

```bash
python3 -B -S -m unittest discover -p 'test_*.py'
python3 -B -S planning.py --check
python3 -B -S verify_freeze.py
python3 -B -S run.py --validate-only
```

Source hashes include the unchanged inherited dependency closure. New input
generation reads prior prepared strings for exclusions, not previous outcomes.
All scientific settings must survive independent review unchanged or receive a
public, dated amendment before a separate release. The outcome of this round
is still unknown.
