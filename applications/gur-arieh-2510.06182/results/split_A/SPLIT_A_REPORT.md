# Round 1 split A (protocol v2): cell selection report

**Development data.** Split A selects the cell and nothing else: no value from split A is
frozen, and every value that would be frozen comes from split B. The run followed
[SPLIT_A_B_PROTOCOL.md](../../SPLIT_A_B_PROTOCOL.md), committed before it (`f822b0a`,
clean tree). The confirmation seed block was not touched.

**Outcome.** Every gate passed and every quota was met. By the declared rule (largest d,
ties c1, c2, c3, c4, d ≥ 0.20) split A selects **c4** (i_P 3, i_L 5, i_R 1, i_N 6) with
d = 0.522, **0.053** above the second-largest d (c1, 0.469). Split B runs in c4 only.

## Run

| item | value |
|---|---|
| model | `google/gemma-2-2b-it` at `299a8560…`, local cache, offline flags on, no download |
| environment | `/Users/felixb/causal-decidability-method-case/.venv-round2/bin/python` (read-only), Python 3.11.15, torch 2.5.1, transformers 4.57.3; every package version in `manifest.json` |
| execution | float32 on MPS, eager attention; CPU float32 for gate 7 |
| families | 204 generated, 200 qualifying: c1 50/50, c2 50/51, c3 50/50, c4 50/53; seeds 2,000,000 + i with i = 4j + k (cell k) |
| gate-7 cases | draw indices 0–31 (the first 8 generated families of each cell) |

## Gate table (development data)

| # | gate | value | status |
|---|---|---|---|
| 1 | model and tokenizer hashes | 9 files match | pass |
| 2 | native competence | recipient 203/204 correct (i_N = 0: 101/101; i_N = 6: 102/103), conflict donor 201/204; first token is the answer form: recipient 203/204, donor 201/204; answer-form argmax agrees with generation 408/408 | reported |
| 2 | yield | 200/204 = 0.980 (c1 1.00, c2 0.98, c3 1.00, c4 0.94) | pass (≥ 0.50) |
| 3 | tokens and pools | pools equal the lock; `<bos>` prefix; 0 alignment failures | pass |
| 4 | hooks and shapes | 1,428 checks (7 per family), one write at position 93 on [1, 94, 2304]; design indices correct; no technical failure | pass |
| 5 | identity self-patch | max \|Δ logit\| 0.0; argmax and generation identical 204/204 | pass |
| 7 | MPS float32 vs CPU float32 | 32/32 declared cases (8 per cell): max \|ΔT\| 1.2 × 10⁻⁵; resolution and labels identical 32/32 | pass (≤ 0.01) |
| 6a | agreement transfer | 0.958 over 600 runs (c1 0.940, c2 0.953, c3 0.973, **c4 0.967**) | selected cell passes (binding on B) |
| 6b | agreement resolution | 1.000 in every cell | selected cell passes (binding on B) |
| 8 | support | unresolved 2/200 (c1 0/50, c2 1/50, c3 1/50, **c4 0/50**), both by support, none by answer mass; s_min = 0.10 (the floor) in every cell | selected cell passes (binding on B) |
| — | answer-token mass | median 0.98–0.99 in every run type; minimum 0.68 (layer-19 diagnostic); none below 0.5 | — |
| — | audit of full logits | consistent | pass |

## Selection (the declared rule)

| cell | T_W | T_A | d | mean q (P/L/R) |
|---|---|---|---|---|
| c1 | 0.523 | 0.992 | 0.469 | 0.523/0.007/0.470 |
| c2 | 0.537 | 0.998 | 0.461 | 0.537/0.412/0.052 |
| c3 | 0.527 | 0.994 | 0.466 | 0.473/0.000/0.527 |
| **c4** | 0.471 | 0.993 | **0.522** | 0.465/0.471/0.064 |

Selected: **c4**, margin 0.053 over c1. All four cells exceed d_min = 0.20. These split-A
anchors served the selection only and are not carried forward; split B re-estimates
everything in c4.

## Runtime

5.91 s per family on average (204 families, 1,206 s), model load 5 s, gate-7 CPU
reference 71 s for 32 cases, MPS driver memory 11.4 GiB.

## Files

`manifest.json`, `RUN_STARTED.json`, `records.jsonl`, `gate7_cpu_reference.jsonl`,
`audit_full_logits.npz`, `timings.json`, `split_A_decision.json` (the gate table, the
selection and the checks above), this report and the generated `artifact_hashes.json`.
Reproduce the decision with `python scripts/run_mixing_split.py decide --split A
--output results/split_A`.
