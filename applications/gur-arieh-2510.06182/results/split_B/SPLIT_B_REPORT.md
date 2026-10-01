# Round 1 split B (protocol v2): freeze-value development report

**Development data; not a confirmation result.** Split B re-estimates every value that
would enter the freeze after split A selected cell **c4**. The run followed
[SPLIT_A_B_PROTOCOL.md](../../SPLIT_A_B_PROTOCOL.md), which was committed before split A
or B (`f822b0a`). No value from split A other than the selected cell is used below, and
the confirmation seed block has not been touched.

**Outcome: PROCEED.** All declared gates passed. The 200 qualifying families contained
five unresolved conflict cases, giving a resolution rate of 0.975. Under the declared
sample-size rule this raises the proposed confirmation size from 200 to **300** families.
At N = 300, planned adequacy power is 0.911 and planned exclusion power is 0.925.

## Run

| item | value |
|---|---|
| selected cell | c4: i_P = 3, i_L = 5, i_R = 1, i_N = 6 |
| model | `google/gemma-2-2b-it` at `299a8560…`, local cache, offline flags on, no download |
| execution | float32 on MPS, eager attention; CPU float32 for gate 7 |
| families | 200 generated and 200 qualifying; seeds 3,000,000 + draw index |
| gate-7 cases | draw indices 0–31, CPU float32 |

## Gate table

| # | gate | value | status |
|---|---|---|---|
| 1 | model and tokenizer hashes | all 9 locked files match | pass |
| 2 | native competence and yield | recipient 200/200; conflict donor 200/200; first token and answer-form readout agree with generation 400/400; yield 200/200 | pass |
| 3 | tokens and pools | pools equal the lock; `trance` excluded by the declared one-token rule; no alignment failures | pass |
| 4 | hooks and shapes | 1,400 checks, exactly one declared write per hook; no design-index or technical failure | pass |
| 5 | identity self-patch | max answer-logit and full-vocabulary difference 0.0; identical argmax and generation 200/200 | pass |
| 6a | agreement transfer | 0.975 across 600 runs | pass (≥ 0.90) |
| 6b | agreement resolution | 598/600 = 0.9967; all three target-specific anchors defined | pass (≥ 0.90) |
| 7 | MPS fp32 vs CPU fp32 | all 32 declared cases; max \|\Delta T\| = 2.16 × 10⁻⁵; identical resolution and labels 32/32 | pass (≤ 0.01) |
| 8 | support | 195/200 resolved; five unresolved by support, none by answer-token mass; s_min = 0.10 from split-B no-patch runs | pass (≥ 0.90) |
| 9 | anchor separation | T_W = 0.533756, T_A = 0.996447, d = 0.462691 | pass (d ≥ 0.20) |
| 10 | sample-size rule | N = 300; adequacy power 0.911, exclusion power 0.925 | pass |
| — | mean gate | \delta = 0.084247 from 10,000 declared resample pairs, seed 251006182 | recorded |

The answer-form mass remained well above its 0.5 floor: the minimum in the primary
conflict arm was 0.813 and the median was 0.985. The primary answer-form argmax matched
the generated entity in 200/200 conflict cases. The upstream context-token readout is
descriptive only; its argmax matched generation in 141/200 conflict cases and its median
mass was 1.22 × 10⁻⁵.

## Values proposed for the freeze

| value | split-B estimate |
|---|---|
| s_min | 0.10 |
| T_W | 0.533756 |
| T_A | 0.996447 |
| d | 0.462691 |
| mean q (P/L/R), resolved cases | 0.533756 / 0.419777 / 0.046466 (m = 195) |
| unresolved rate | 0.025 |
| final N | 300 |
| adequacy / exclusion power | 0.911 / 0.925 |
| \delta | 0.084247 |

These are development estimates for review. They do not identify either profile and do
not answer the Round 1 question. That answer requires one fresh confirmation run after a
separate final freeze.

## Descriptive diagnostics

Among the 195 resolved conflict cases at block 18, the largest target mass was positional
in 54.4%, lexical in 43.1%, and reflexive in 2.6%. These shares are not a confirmation
result and do not enter the decision rule. At the block-19 diagnostic, the mean mass moved
toward the reflexive target; this diagnostic never determines a gate or frozen value.

## Runtime and files

The MPS run took 1,292 seconds for 200 families (mean 6.46 seconds per family); the
32-case CPU reference took 86 seconds. The result directory contains the manifest,
start record, raw records, CPU reference, full-logit audit, timings, decision, this report,
and a generated artifact index. Reproduce the decision with:

```bash
python applications/gur-arieh-2510.06182/scripts/run_mixing_split.py \
  decide --split B \
  --output applications/gur-arieh-2510.06182/results/split_B
```
