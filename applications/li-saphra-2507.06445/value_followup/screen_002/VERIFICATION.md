# Independent verification of screen 002

**PASS.** The frozen [verifier](verify_screen.py) was executed against the
completed raw run and [analysis report](results/report.json). Its independent
recount confirms a positive **screen-enrichment** result and a failed **secondary
mechanism-development gate**. No model inference was repeated.

| Quantity | Independently recomputed |
|---|---:|
| Separating families, native margin <8 | **30/32** |
| Separating families, native margin ≥8 | **0/32** |
| Difference in separation rates | **93.75 percentage points** |
| Joint-coverage ≥95% interval for that difference | **[64.13, 99.47] percentage points** |
| Declared 25-percentage-point enrichment target | **Supported** |
| Secondary balance-class candidate, definite matches | **19/30** |
| Secondary position candidate, definite matches | **0/30** |
| Secondary confirmation-start gate | **Not met** |

The interval uses four one-sided Clopper–Pearson bounds at alpha=0.0125, then
subtracts the stratum bounds. In particular, 0/32 is not treated as certainty:
the rejected stratum's upper bound is **0.127976**. The secondary results remain
developmental and do not establish a unique balance variable or a confirmed
semantic explanation.

## Checks performed

The verifier uses the standard library and the earlier independent verifier's
arithmetic helpers; it imports no producer, analyzer, design or model runtime.
It checked **30 source hashes, nine output hashes, all 1,024 native candidate
records and 640 saved transfer snapshots**, including:

- Both dtype scores and classifications; the signed `<8` rule; the first 32
  cases in each stratum; and the unchanged, outcome-blind donor-template order.
- Donor constraints and both-position prefix exclusions, extending the bound
  historical ban list with all previous recipient and donor strings.
- Receipt hashes and local ordering: selection before anchors, and the fixed
  anchor-calibrated forecasts before the six target measurements.
- First-maximum closing-bracket recipient selection from saved attention rows;
  exact post-dtype reconstruction of the intended and delivered node vectors;
  fixed recipient state within each family; and measured precision gates.
- All primary counts and bounds, six-target prediction errors and the original
  secondary stopping rule, compared with the saved analysis report.

The maximum reconstructed fp32 node-construction roundoff was
**1.42680 × 10⁻⁸**; the corresponding fp64 arithmetic check was exact on the
stored numbers. This is a construction check, not a claim of exact model truth.
The verifier's **16 tests passed**, including selection tampering, threshold
straddles, receipt ordering, nondegenerate binomial endpoints and an export with
no ignored public-data cache.

The complete staged repository was also exported without Git metadata or ignored
caches, and the standard-library audit passed against the real saved run there:
30 sources, nine outputs, 1,024 native records and 640 snapshots. In total,
**28 new screen tests and the 30 existing value-transfer tests passed**. This is
the relevant test surface, not a claim that every unrelated repository suite was
rerun.

## Reproduce without model inference

Run from the repository root:

```bash
python3 -B -S applications/li-saphra-2507.06445/value_followup/screen_002/verify_screen.py \
  --run applications/li-saphra-2507.06445/value_followup/screen_002/results/run_001 \
  --inputs applications/li-saphra-2507.06445/value_followup/screen_002/inputs \
  --report applications/li-saphra-2507.06445/value_followup/screen_002/results/report.json
```

The checked run manifest hash is
`61fdb44c19d0dc9ed43d2f9cb78938da5b0a1d729b4393a73cd467ee4bd68a45`.
Its local source freeze is commit `7e808be`.

## Scope

This is a separate agent's independent code path within the same AI-assisted
workflow, not external replication. Local receipts do not establish externally
witnessed preregistration. Historical-list continuity is checked; unavailable
original public files are not reparsed. Full-layer unchangedness remains a
producer-reported control because complete layer tensors were not archived.

The result validates a **post-pilot frozen screen prospectively on new cases
at this selected head and intervention family**. It does not show that native
margin causally determines separability, that rejected cases lack mechanisms,
that the procedure saves computation overall, or that the screen transfers to
other heads. The primary success does not override the secondary failed gate.
