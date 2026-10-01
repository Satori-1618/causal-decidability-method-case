# Round 1 development pilot: report

**Every number in this report is pilot/development, descriptive.** The pilot was
authorized by the user on 28 September 2026 for the pilot only. Its data are development
data: they are not used to estimate the frozen anchors or δ, and nothing was tuned on
them. The pilot's T values, anchors and label shares are feasibility information, not a
result about the model. Split A, split B, the freeze and the confirmation were not run.

**Outcome in one paragraph.** The pipeline works offline on the cached model: file
hashes, tokens, hooks, identity self-patch, native competence, yield, agreement control
and support all pass their declared gates. **Gate 7 (bfloat16 against float32) fails.**
A readout-validity diagnostic shows a second problem that no declared gate covers: the
declared, paper-faithful readout scores the in-context token form (`▁country`), which
carries about 2 × 10⁻⁵ of the model's next-token mass; the model answers with the
capitalised form (`Country`). Under conflict patches the declared readout's argmax names
the entity the model actually generates in only 58 of 79 cases. Both findings need the
user's protocol decisions before split A (Section 6). The N rule gives **N = 400** at a
pooled unresolved rate of 3.8% (3 of 79) under the declared readout.

## 1. What was run

| item | value |
|---|---|
| model | `google/gemma-2-2b-it` at `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, local Hugging Face cache, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; no download |
| code | commit `a8e4ea1` (clean tree), `scripts/run_mixing_pilot.py` |
| upstream | clone at `c53372c`, checked against `SOURCE_LOCK.json` (11 files, 58 cited lines, 10 tasks) |
| environment | existing environment reused read-only: `.venv-round2` of the original working tree (Python 3.11.15, torch 2.5.1, transformers 4.57.3, tokenizers 0.22.2, safetensors 0.8.0, numpy 1.26.4), `PYTHONDONTWRITEBYTECODE=1`; pinned in `requirements-model.txt` |
| device | MPS (Apple M1 Max, 64 GB) |
| execution | bfloat16, eager attention, logits read as float32; float32 reference on families 0–31 |
| task, n, layer | music performance, t_entity = 2, n = 7, residual stream entering block 18, last token; block 19 as a diagnostic |
| families | 80, seeds 1,000,000 + i (i = 0…79), cell c(i mod 4 + 1), 20 per cell |
| runs per family | recipient (no-patch) and conflict donor with greedy generation, conflict patch at 18 and at 19, three agreement donors and patches, identity self-patch; greedy generation under every patch (diagnostic) |
| audit | full-vocabulary logits of the conflict patch of families 0–3 (`audit_full_logits.npz`) |

Before the pilot, two smoke runs of the same script (4 and 8 families, seeds
1,000,000–1,000,007) checked the pipeline; their outputs were kept in a scratch directory
and are not part of these results. The first smoke run's audit logits showed the readout
mismatch of Section 4, so the answer-form logits and the greedy generation under every
patch were added to the runner as recorded diagnostics (commit `a8e4ea1`) before the
pilot proper. No declared value was changed.

## 2. Gate table (pilot/development, descriptive)

| # | gate | pilot value | criterion | status |
|---|---|---|---|---|
| 1 | model and tokenizer hashes | 9 files hashed from the local snapshot; each matches its cache content address (sha256 for LFS files, git blob id otherwise) and the locked size | all match | **pass** |
| 2 | native competence per position group | recipient 40/40 at i_N = 0, 40/40 at i_N = 6; conflict donor 80/80 at position 3; agreement donors 237/240 (i_P = 3: 80/80; i_L = 1: 40/40; i_L = 5: 38/40; i_R = 1: 40/40; i_R = 5: 39/40, recorded, not filtered); readout agrees with generation: recipient 79/80, donor 80/80 | reported | reported |
| 2 | yield | 79/80 = 0.988 (c1 20/20, c2 20/20, c3 19/20, c4 20/20); the one miss (pilot-0078): recipient generated "Country", the readout argmax named "gospel" | ≥ 0.50 | **pass** |
| 3 | tokens | all 70 entities single tokens with a leading space (23 musicians, 24 genres, 23 instruments; none dropped); chat template renders `<bos><start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`, the dropped prefix is `<bos>`, one BOS token in each recipient prompt; the 7 genre tokens found in design order in every prompt (0 alignment failures); every recipient prompt has 94 tokens | pools ≥ n + 2; alignment exact | **pass** |
| 4 | hooks and shapes | 560 hook checks (7 per family: conflict at 18 and 19, three agreement patches, identity forward, identity generation): each patched forward 1 call and 1 write at position 93 on a [1, 94, 2304] input with a [2304] vector, and 1 write per patched generation; design indices: conflict = cell and P = L = R = j for all 240 agreement cases | exactly one write, last position | **pass** |
| 5 | identity self-patch | max \|Δ logit\| = 0.0 (entity and full vocabulary), same argmax 80/80, same greedy generation 80/80 | ≤ 0.001, same argmax and generation | **pass** |
| 6a | agreement transfer | 220/237 = 0.928 argmax on the common target (qualifying families) | ≥ 0.90 (binding on split B) | pass (margin 0.028) |
| 6b | agreement resolution | 232/237 = 0.979 | ≥ 0.90 (binding on split B) | pass |
| 7 | bfloat16 vs float32 | 32 families: max \|ΔT\| = **0.062**, median 5 × 10⁻⁵, 7/32 above 0.01; resolution and labels identical 32/32; max entity-logit difference 0.42. With the unembedding in float32 the gap is the same (max 0.063), so it arises in the bfloat16 forward pass, not in the final logits | \|ΔT\| ≤ 0.01 and identical resolution and labels | **FAIL** |
| 8 | support | s_min = 0.10 in every cell (the floor; no-patch S at most 0.013); unresolved 3/79 = 3.8% (c1 2/20, c2 1/20, c3 0/19, c4 0/20); resolution rate 0.962 | ≥ 0.90 (binding on split B) | pass |
| 9 | separation (descriptive) | d = 0.412 (c1), 0.470 (c2), 0.424 (c3), 0.439 (c4) | some cell ≥ 0.20 (binding on split A) | feasible |
| 10 | N rule | unresolved 3.8% > 2% → smallest N in {300, 400, 500} with A_T-adequacy power ≥ 0.80: **N = 400** (adequacy power 0.874; exclusion power 0.971; N = 300 gives 0.767) | the rule | N = 400, awaiting approval |
| — | readout audit | logsumexp recomputed from the stored full logits of 4 runs: max difference 8 × 10⁻⁷ | — | consistent |
| — | 18 vs 19 (diagnostic) | Section 5 | never a STOP | reported |

Technical failures: none. Pilot s_min: the declared rule applied per cell to the
qualifying cases' no-patch runs (19–20 per cell, so the upper order statistic at 0.99 is
their maximum); here the floor 0.10 binds.

## 3. Dtype (gate 7) and float32 execution

Gate 7 fails for the declared readout and also for the answer-form readout of Section 4
(conflict patch: max \|ΔT\| 0.121 on 22 families, 5 above 0.01, one label differs).

What float32 execution would look like, from a rerun of families 0–31 in float32 on MPS
(`diagnostic_fp32_execution.json`, descriptive):

| quantity | value |
|---|---|
| runtime per family, float32 on MPS | mean 5.97 s, median 5.44 s, max 8.22 s (bfloat16 on the same families: mean 5.85 s) |
| memory | 9.74 GiB allocated after loading, at most 11.44 GiB MPS driver memory after a family, process peak resident size 12.1 GiB, machine 64 GiB: fits |
| reproducibility | the float32 family rerun reproduces the stored float32 reference exactly (32/32 identical conflict logits) |
| float32 MPS against float32 CPU, conflict patch, 8 families | max \|ΔT\| 2.2 × 10⁻⁵ (declared readout), 9.4 × 10⁻⁶ (answer form); max logit difference 8.4 × 10⁻⁵ |

With float32 execution, a gate comparing the MPS run with a float32 CPU reference would
see residuals about three orders of magnitude below the 0.01 tolerance on these 8
families. MPS bfloat16 is not faster than float32 here.

## 4. Readout validity: the declared readout and the answer-form readout

**Declared readout (paper-faithful, as `tasks/dist.py:358`).** The logits of the n
in-context entity tokens, which carry a leading space (`▁country`).

**Answer-form readout (diagnostic only).** The token form the model emits first at the
answer position. It was determined from the pilot's 400 unpatched native generations
(recipient, conflict donor and agreement donors): the first generated token is the
single-token capitalised form without a leading space of the correct entity in 378
cases, the first piece `T` of the two-token `Trance` in 19, and another entity's
capitalised form in 3 (wrong answers). The rule is therefore "first letter capitalised,
no leading space" (`Country`). 23 of 24 genres are one token in that form; **`trance` is
not** (`T` + `rance`). The answer-form readout is evaluated only on the 54 qualifying
families whose seven genres all have a single-token answer form. The runner stored these
tokens' logits for every run; no model rerun was needed.

Absolute full-vocabulary mass on the n scored tokens, and how often each readout's
argmax names the entity greedy generation produces (54 families; for patched runs the
generation is under the same patch):

| run type | runs | mass, declared (median; min) | mass, answer form (median; min) | argmax = generated entity, declared | same, answer form |
|---|---|---|---|---|---|
| no-patch recipient | 54 | 2.6 × 10⁻⁵; 3.7 × 10⁻⁷ | 0.98; 0.88 | 54/54 | 54/54 |
| native conflict donor | 54 | 1.9 × 10⁻⁵; 1.4 × 10⁻⁷ | 0.99; 0.88 | 54/54 | 54/54 |
| native agreement donors | 162 | 2.1 × 10⁻⁵; 7 × 10⁻⁸ | 0.99; 0.89 | 162/162 | 162/162 |
| conflict patch, block 18 | 54 | 1.5 × 10⁻⁵; 5.8 × 10⁻⁸ | 0.99; 0.86 | **38/54** | 54/54 |
| conflict patch, block 19 (diagnostic) | 54 | 6.6 × 10⁻⁶; 4.3 × 10⁻⁸ | 0.99; 0.88 | **36/54** | 54/54 |
| agreement patches, block 18 | 162 | 1.3 × 10⁻⁵; 7.1 × 10⁻⁸ | 0.99; 0.90 | 157/162 | 162/162 |

On all 79 qualifying families the declared readout names the generated entity in 58/79
conflict runs at block 18, 54/79 at block 19 and 230/237 agreement runs; its median mass
is 2.0 × 10⁻⁵ (conflict) and 3.3 × 10⁻⁵ (no-patch). The answer-form agreement with
generation is close to guaranteed by construction (greedy decoding emits the top token,
which is an answer-form token here); it shows that the form is the one the model uses,
not that its distribution is right. Every generation under a patch named one of the
seven in-context genres.

Per cell, both readouts side by side (T_W, T_A with P, L and R weighted equally, d,
mean q = (P-window, L, R), unresolved cases, label shares among resolved cases, agreement
transfer and resolution; s_min by the declared rule on each readout's own no-patch runs,
0.10 in every case). **Descriptive, pilot/development; not a result.**

| cell | readout | families | unresolved | T_W | T_A | d | mean q (P/L/R) | labels P/L/R | transfer | agreement resolved |
|---|---|---|---|---|---|---|---|---|---|---|
| c1 | declared, all | 20 | 2/20 | 0.576 | 0.988 | 0.412 | 0.424/0.000/0.576 | 0.39/0.00/0.61 | 0.933 | 0.983 |
| c1 | declared, subset | 16 | 2/16 | 0.545 | 0.985 | 0.440 | 0.455/0.000/0.545 | 0.43/0.00/0.57 | 0.917 | 0.979 |
| c1 | answer form, subset | 16 | 0/16 | 0.529 | 0.993 | 0.464 | 0.470/0.000/0.529 | 0.50/0.00/0.50 | 0.938 | 1.000 |
| c2 | declared, all | 20 | 1/20 | 0.529 | 0.999 | 0.470 | 0.529/0.316/0.154 | 0.53/0.32/0.16 | 0.967 | 0.983 |
| c2 | declared, subset | 13 | 1/13 | 0.587 | 0.999 | 0.412 | 0.587/0.241/0.172 | 0.58/0.25/0.17 | 0.974 | 1.000 |
| c2 | answer form, subset | 13 | 0/13 | 0.646 | 0.992 | 0.346 | 0.646/0.241/0.114 | 0.54/0.31/0.15 | 0.974 | 1.000 |
| c3 | declared, all | 19 | 0/19 | 0.567 | 0.991 | 0.424 | 0.431/0.003/0.567 | 0.42/0.00/0.58 | 0.912 | 0.982 |
| c3 | declared, subset | 10 | 0/10 | 0.617 | 0.982 | 0.365 | 0.383/0.000/0.617 | 0.40/0.00/0.60 | 0.900 | 0.967 |
| c3 | answer form, subset | 10 | 0/10 | 0.791 | 0.997 | 0.206 | 0.208/0.001/0.791 | 0.20/0.00/0.80 | 1.000 | 1.000 |
| c4 | declared, all | 20 | 0/20 | 0.554 | 0.993 | 0.439 | 0.554/0.367/0.080 | 0.55/0.35/0.10 | 0.900 | 0.967 |
| c4 | declared, subset | 15 | 0/15 | 0.540 | 0.991 | 0.451 | 0.540/0.356/0.104 | 0.53/0.33/0.13 | 0.933 | 0.978 |
| c4 | answer form, subset | 15 | 0/15 | 0.499 | 0.993 | 0.495 | 0.496/0.499/0.005 | 0.47/0.53/0.00 | 0.956 | 1.000 |

With 10–20 families per cell these values move by several hundredths between the two
readouts and between the full set and the subset. Which cell split A would select is not
settled by the pilot: under the declared readout on all families c2 has the largest d
(0.470, 0.031 above c4); on the answer-form subset c4 has the largest d (0.495) and c3
comes close to d_min. The N rule's input also depends on the readout: 3/79 unresolved
under the declared readout (N = 400), 0/54 under the answer form on the subset (the rule
would give N = 200, but that subset excludes families with `trance`).

## 5. Layer 18 against 19 (diagnostic, decides nothing)

In this conflict design the completed-answer copy and the reflexive pointer predict the
same token, so this comparison cannot separate them. Pooled over the 79 qualifying
families, the entity generated under the conflict patch sits in the P-window in 48%, at
i_L in 19%, at i_R in 32% and at i_N in 1% of cases at block 18; at block 19 the shares
are 6%, 5%, 89% and 0%. Mean q (declared readout) per cell at 18 → 19: c1 (0.42, 0.00,
0.58) → (0.14, 0.00, 0.86); c2 (0.53, 0.32, 0.15) → (0.37, 0.16, 0.47); c3 (0.43, 0.00,
0.57) → (0.11, 0.00, 0.89); c4 (0.55, 0.37, 0.08) → (0.25, 0.27, 0.48).

## 6. Runtime

bfloat16 on MPS: 6.28 s per family on average (median 6.24, range 5.35–9.44), 503 s for
80 families; model load 3 s; float32 reference for 32 families 23 s; hashing the 5.2 GB
snapshot was not timed separately. Pilot families include the identity self-patch, the layer-19 patch
and generation under every patch, so the projections below are upper bounds. Generated
families = qualifying / 0.988.

| stage | qualifying | generated (expected) | hours (upper bound, bfloat16; float32 is about the same) |
|---|---|---|---|
| split A (50 per cell) | 200 | 203 | 0.35 |
| split B (selected cell) | 200 | 203 | 0.35 |
| confirmation | N = 400 | 406 | 0.71 |

## 7. STOP conditions active after the pilot

- **Gate 7 (dtype) failed** under the declared protocol (bfloat16). Under the plan this
  is a STOP until the execution dtype is decided.
- **Readout validity (no declared gate).** Not a declared STOP, but the declared statistic
  rests on tokens carrying about 2 × 10⁻⁵ of the next-token mass, whose argmax disagrees
  with the patched model's generated answer in 21 of 79 conflict cases. Proceeding
  without a decision would test a quantity whose relation to the model's answer is
  unclear.
- Armed for split A/B and confirmation, not triggered in the pilot: gates 6a and 6b
  (margins 0.028 and 0.079), 8 (margin 0.062), 9 and the mean-consistency gate.
- The N rule's edge band (1.8–2% unresolved) was not reached.

## 8. Decisions the user must make before split A

**(i) Execution dtype.**
- (a) Keep bfloat16: gate 7 stays failed (STOP, S1), unless the tolerance is changed,
  which would be tuning on pilot data.
- (b) float32 on MPS: fits in memory (≤ 11.4 GiB), costs about the same time, reproduces
  exactly, and differs from float32 on CPU by ≤ 2.2 × 10⁻⁵ in T on the 8 families
  checked. Gate 7 would then compare MPS float32 with a float32 CPU reference on 32
  development cases, with the same 0.01 tolerance.
- *Recommendation (conservative):* (b), with gate 7 re-declared as MPS float32 against
  CPU float32 and the tolerance unchanged.

**(ii) Primary readout, and an absolute-mass gate.**
- (a) Declared, paper-faithful in-context form (`▁country`): faithful to upstream and to
  the brief's p_k, but its mass is about 2 × 10⁻⁵ and its argmax disagrees with the
  generated answer under conflict patches in 27% of cases, so T may not describe what
  the model answers.
- (b) Answer form (`Country`): about 0.98 of the next-token mass, agrees with
  generation; departs from upstream's readout, so the statistic describes the model's
  answer distribution rather than upstream's scored tokens. It needs single-token answer
  forms, so `trance` would leave the genre pool (23 genres remain, ≥ n + 2), which
  changes the family sampler from the pilot's.
- (c) Both, with one declared primary and the other reported descriptively (no second
  test, so no α split).
- Absolute-mass gate: only meaningful with (b) or (c)-with-(b)-primary; a declared floor
  on the n answer-form tokens' full-vocabulary mass per run (the pilot's minimum was
  0.86), with runs below it counted as unresolved or as technical failures, declared
  before new data. With (a) any sensible mass floor would fail almost every run.
- *Recommendation (conservative):* (c) with the answer form as primary, a pre-declared
  mass floor of 0.5 on the answer-form tokens counted as unresolved, `trance` removed
  from the pool, and the in-context readout kept as a descriptive, paper-faithful
  secondary. This needs the user's approval because it changes the brief's statistic
  definition (the p_k of Section 5.4).

**(iii) Repeat the pilot under the changed protocol.**
- If (i) or (ii) changes, the pilot's gates, the unresolved rate that feeds the N rule
  and the runtime were measured under a different protocol.
- *Recommendation (conservative):* yes. Rerun the 80-family pilot (about 9 minutes) with
  the chosen dtype, readout, pool and mass gate on a fresh pilot seed block disjoint from
  these pilot seeds and from A, B and confirmation (for example 1,100,000 + i), recompute
  N by the recorded rule from that pilot, and only then approve the final table and draw
  split A.

**(iv) Other points for the final approval.**
- N: 400 by the rule under the declared protocol; to be recomputed if (iii) is done.
- Agreement transfer (0.928 against a 0.90 floor under the declared readout) has little
  margin; the answer form gives 0.938–1.000 per cell on the subset. No change proposed.
- The environment used is `.venv-round2` of the original working tree, read-only; the
  user may prefer a dedicated environment outside both trees before split A.

## 9. Files

| file | content |
|---|---|
| `manifest.json`, `RUN_STARTED.json` | pilot specification, code and model hashes, pools, gate-3 checks, environment; the start record holds the manifest hash |
| `records.jsonl` | 80 families, bfloat16, one per line, in the records-only checker's schema plus the pilot measurements |
| `fp32_reference.jsonl` | float32 conflict and no-patch entity logits for families 0–31 |
| `audit_full_logits.npz` | full-vocabulary logits of the conflict patch, families 0–3 |
| `timings.json`, `summary.json` | timings; the gate table computed by `src/mixing_pilot_summary.py` |
| `diagnostic_answer_form.json` | Section 4, from `scripts/diagnose_pilot_readout.py` (stored records and the pinned tokenizer only) |
| `diagnostic_fp32_execution.json`, `diagnostic_fp32_records.jsonl` | Section 3, from `scripts/diagnose_fp32_execution.py` (float32 rerun of families 0–31, CPU check of 0–7) |
| `artifact_hashes.json` | sha256 of every data file above |

Reproduce the summary and the answer-form diagnostic from the stored files:

```bash
python scripts/run_mixing_pilot.py summarize --output results/pilot
python scripts/diagnose_pilot_readout.py --results results/pilot
```
