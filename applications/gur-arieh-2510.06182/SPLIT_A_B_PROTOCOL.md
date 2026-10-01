# Round 1 development splits A and B: protocol (declared before either run)

**Status: authorized by the user on 28 September 2026. Committed before split A runs.**
Splits A and B are development data. Nothing here freezes a value; the freeze and the
confirmation are not authorized. The confirmation seed block (4,000,000 + i) is not used,
not even by smoke runs. The machine-readable form of this protocol is `SPLIT_A` and
`SPLIT_B` in [scripts/run_mixing_split.py](scripts/run_mixing_split.py); the decisions are
in [src/mixing_splits.py](src/mixing_splits.py) and
[src/mixing_round1_analysis.py](src/mixing_round1_analysis.py).

Everything follows protocol v2 ([PREFLIGHT_PLAN.md](PREFLIGHT_PLAN.md)): float32 on MPS;
the answer-form readout; a case is resolved iff S ≥ s_min and the answer-token mass is
≥ 0.5; the paper's in-context readout is descriptive only; the v2 pools recorded in
`SOURCE_LOCK.json`; native correctness with the first-token criterion; eager attention;
layer 18, last token; the layer-19 patch and greedy generation under every patch as
diagnostics; environment `.venv-round2` read-only, with the interpreter and every package
version recorded in each manifest.

## Approvals recorded here (28 September 2026)

1. **Pool-rule scope.** The answer-form condition applies to genres only, because only
   genres are ever answers: the readout scores the n genres, native correctness compares
   the first generated token with a genre's answer form, and musicians and instruments
   only appear in the context and in the question. Their context form must still be one
   token.
2. **Gate-7 cases** are fixed below for both splits.
3. **N = 200 from pilot 2 is only a planning value.** The final N and the "adequacy
   powered" label come from split B's unresolved rate in the selected cell, by the
   recorded rule.
4. **Environment:** `.venv-round2`, read-only.

## Split A: cell selection

| item | value |
|---|---|
| cells | the four candidates c1 (i_P 3, i_L 1, i_R 5, i_N 0), c2 (3, 5, 1, 0), c3 (3, 1, 5, 6), c4 (3, 5, 1, 6) |
| size | 50 qualifying families per cell (200 in all) |
| order and seeds | family i = 4j + k uses cell k (c1, c2, c3, c4 for k = 0…3) and seed 2,000,000 + i; a cell stops at its 50th qualifying family |
| cap | at most 100 generated families per cell; STOP if a cell does not reach 50 |
| gate-7 cases | draw indices 0–31 = the first 8 generated families of each cell (j = 0…7), rerun in float32 on CPU |
| audit sample | full-vocabulary logits of the conflict patch of draw indices 0–3 |
| s_min | per cell, the declared rule on all no-patch runs of the cell's qualifying families (used for the selection only) |
| selection | per cell T_W, T_A (P, L and R weighted equally) and d; the cell with the largest d, ties broken c1, c2, c3, c4, subject to d ≥ d_min = 0.20; otherwise `NOT_DECIDABLE_WITH_CURRENT_INTERVENTIONS` (S1). A cell whose anchors are undefined counts as d = −1 |
| STOP | any of gates 1–5 or 7 fails; a quota is not met; no cell reaches d_min; the selected cell fails gate 6a, 6b or 8 on split A (these gates are binding on split B; failing them already on A stops the run before B, a conservative rule declared here) |
| output | `results/split_A/`: manifest, records, gate-7 CPU reference, audit logits, timings, `split_A_decision.json`, report, generated index |

If split A stops, split B is not run.

## Split B: the values that would be frozen

| item | value |
|---|---|
| cell | the cell selected on split A only (read from the committed `split_A_decision.json`, status PROCEED) |
| size | 200 qualifying families |
| order and seeds | family i uses the selected cell and seed 3,000,000 + i; generation stops at the 200th qualifying family |
| cap | at most 400 generated families; STOP if 200 are not reached |
| gate-7 cases | draw indices 0–31 = the first 32 generated families, rerun in float32 on CPU |
| audit sample | draw indices 0–3 |

Split B computes, from split B only (never from the pilots or split A):

1. **s_min** by the declared rule (upper order statistic at 0.99, floor 0.10) on all
   no-patch runs of split B's qualifying families.
2. **Resolution rate** under the combined gate; STOP below 0.90. The unresolved rate is
   1 − resolution rate.
3. **T_W** = T(mean of resolved q), **T_A** = mean over P, L and R of the mean T over
   resolved agreement runs with that target, **d** = T_A − T_W; STOP if d < d_min.
4. **Agreement gates:** resolution ≥ 0.90 and transfer ≥ 0.90; STOP otherwise.
5. **Final N and the adequacy label** by the recorded N rule at split B's unresolved
   rate; STOP if the rule stops (exclusion power below 0.80 even at N = 500).
6. **δ** by the declared two-sample resampling over split B's resolved q-vectors (sizes
   m = resolved split-B cases and the final N; 10,000 resample pairs; seed 251006182;
   false-INVALID rate 0.05; the upper order statistic at 0.95 of the sup-norm difference).
7. **Gates 1–5 and 7**, as on split A.

Output: `results/split_B/` with `split_B_decision.json`, a report and the generated index.

**Recorded conflict.** The preflight plan and `PROPOSED_VALUES.json` said the frozen
s_min would come from split A. The user's instruction of 28 September 2026 says every
value that would be frozen, s_min included, comes from split B. This protocol follows the
later instruction; split A's s_min for the selected cell is reported next to it. In both
pilots s_min was the floor 0.10 in every cell.

## After split B

A draft freeze manifest (`FREEZE_DRAFT.md`, application directory) lists every value that
would be frozen, the confirmation seed block (4,000,000 + i) and its gate-7 cases, and
the code and model hashes, for the user's review. It is not a freeze. No freeze commit,
no confirmation run and no push happen without a further authorization.

## Smoke runs

Only from the smoke block (1,900,000 + i), with outputs outside `results/`. Never from
the split A, split B or confirmation blocks.
