# Round 1 development pilot 2 (protocol v2): report

**Every number in this report is pilot/development, descriptive.** Pilot 2 was authorized
by the user on 28 September 2026, together with protocol v2, with no go for split A or
split B. Its data are development data: they are not used to estimate the frozen anchors
or δ, and nothing was tuned on them. The pilot's T values, anchors and label shares are
feasibility information, not a result about the model. Split A, split B, the freeze and
the confirmation were not run. `results/pilot/` (pilot 1) is unchanged.

**Outcome in one paragraph.** Under protocol v2 every declared gate passes: file hashes,
tokens and the locked pools, hooks, identity self-patch, native competence (with the new
first-token criterion) and yield (80/80), agreement transfer (0.967) and resolution
(1.000), the re-declared gate 7 (MPS float32 against CPU float32: max |ΔT| 8.8 × 10⁻⁶ on
the 32 declared cases) and support (1 of 80 unresolved). The answer-token mass has a
median of about 0.98–0.99 in every run type and never falls below 0.5. The pooled
unresolved rate of 1.25% gives a **provisional planning N = 200**; the final N is set at
the freeze from split B. Descriptively, d ranges from 0.27 (c1) to 0.52 (c2); c2 leads c4
by only 0.014.

## 1. What was run

| item | value |
|---|---|
| protocol | v2 (commit `dec1f1d`, clean tree): float32 on MPS, answer-form readout with the combined gate S ≥ s_min and answer-token mass ≥ 0.5, the paper's readout descriptive only, v2 pools, gate 7 against CPU float32 |
| model | `google/gemma-2-2b-it` at `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, local Hugging Face cache, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; no download |
| environment | `/Users/felixb/causal-decidability-method-case/.venv-round2/bin/python` (read-only, `PYTHONDONTWRITEBYTECODE=1`), Python 3.11.15, torch 2.5.1, transformers 4.57.3, tokenizers 0.22.2, safetensors 0.8.0, numpy 1.26.4; all 78 package versions recorded in `manifest.json` |
| device | MPS (Apple M1 Max, 64 GB) for the run; CPU for the gate-7 reference |
| families | 80, fresh seeds 1,100,000 + i (i = 0…79), cell c(i mod 4 + 1), 20 per cell |
| runs per family | recipient (no-patch) and conflict donor with greedy generation; conflict patch at block 18 and at 19 (diagnostic); three agreement donors and patches; identity self-patch; greedy generation under every patch (diagnostic) |
| gate 7 | families 0–31 (8 per cell), conflict patch rerun in float32 on CPU |
| audit | full-vocabulary logits of the conflict patch of families 0–3 |

**Smoke run.** Before the protocol commit, one smoke run of 4 families used the declared
smoke seed block (1,900,000–1,900,003) and wrote to a scratch directory outside
`results/`; its outputs are not kept. It showed one problem: the audit of the stored
full logits compared the answer-token mass recomputed in float64 with the run's float32
value at a tolerance of 10⁻⁶ and failed by 1.6 × 10⁻⁶. The audit tolerance for the mass
was set to 10⁻⁵ (float32 precision) before the commit. The audit is a consistency check,
not a declared gate. No pilot-2 seed was used before the pilot.

**Pools.** Musician 23, Genre 23, Instrument 23. `trance` is dropped: its answer form
`Trance` is two tokens (`T`, `rance`). All kept genres are one token in both forms, with
unique ids (recorded with the ids in `SOURCE_LOCK.json`, `round1_entity_pools`; the run
checked that its pools equal the lock). The answer-form condition was applied to the
answered category (genres) only; under an all-category reading the instruments `flute`,
`trumpet` and `whistle` would also have been dropped (see Section 7).

## 2. Gate table (pilot/development, descriptive)

| # | gate | pilot value | criterion | status |
|---|---|---|---|---|
| 1 | model and tokenizer hashes | 9 files; every sha256 matches its cache content address and the locked size | all match | **pass** |
| 2 | native competence per position group | recipient 40/40 at i_N = 0 and 40/40 at i_N = 6; conflict donor 80/80 at position 3; first generated token is the answer-form token: 80/80 recipients, 80/80 donors; answer-form argmax agrees with generation 160/160; agreement donors 238/240 (recorded, not filtered) | reported | reported |
| 2 | yield | 80/80 = 1.000 (20/20 in every cell) | ≥ 0.50 | **pass** |
| 3 | tokens and pools | pools equal the lock (`trance` dropped); `<bos>` is the dropped prefix; 0 alignment failures; every recipient prompt 94 tokens | pools as locked; ≥ n + 2 per category | **pass** |
| 4 | hooks and shapes | 560 checks (7 per family), all one call and one write at position 93 on [1, 94, 2304]; design indices correct for every conflict and agreement case; no technical failure | exactly one write, last position | **pass** |
| 5 | identity self-patch (MPS float32) | max \|Δ logit\| = 0.0 (answer form and full vocabulary); same argmax 80/80; same generation 80/80 | ≤ 0.001, same argmax and generation | **pass** |
| 6a | agreement transfer | 232/240 = 0.967 (answer-form argmax on the common target) | ≥ 0.90 (binding on split B) | pass |
| 6b | agreement resolution | 240/240 = 1.000 (combined gate) | ≥ 0.90 (binding on split B) | pass |
| 7 | MPS float32 vs CPU float32 | 32/32 declared cases (8 per cell): max \|ΔT\| 8.8 × 10⁻⁶, median 7 × 10⁻⁸; resolution 32/32 and labels 32/32 identical; max answer-logit difference 7.6 × 10⁻⁵ | \|ΔT\| ≤ 0.01, identical resolution and labels | **pass** |
| 8 | support | s_min = 0.10 in every cell (the floor; no-patch S at most 2 × 10⁻⁴); unresolved 1/80 = 1.25% (c2: 1/20 by support; 0 by answer mass) | resolution ≥ 0.90 (binding on split B) | pass |
| 9 | separation (descriptive) | d = 0.273 (c1), 0.519 (c2), 0.460 (c3), 0.505 (c4) | some cell ≥ 0.20 (binding on split A) | feasible |
| 10 | planning N | 1.25% ≤ 2% → **N = 200** (adequacy power 0.879, exclusion power 0.813) | the rule, provisional | N = 200 (planning only) |
| — | audit of full logits | logsumexp within 7.8 × 10⁻⁷, answer mass within 1.1 × 10⁻⁶, answer logits identical | consistency | pass |
| — | 18 vs 19 (diagnostic) | Section 5 | never a STOP | reported |

The planning N is provisional. With 1 unresolved case in 80, the rate is imprecise (a
one-sided 97.5% upper bound is about 6.8%). The final N, and whether adequacy counts as
powered, come from split B's unresolved rate in the selected cell at the freeze.

## 3. Answer-token mass per run type

Full-vocabulary probability on the n answer-form tokens (all 80 families qualify):

| run type | runs | min | 10% | median | below 0.5 |
|---|---|---|---|---|---|
| no-patch recipient | 80 | 0.843 | 0.935 | 0.979 | 0 |
| native conflict donor | 80 | 0.754 | 0.956 | 0.988 | 0 |
| native agreement donors | 240 | 0.816 | 0.946 | 0.986 | 0 |
| conflict patch, block 18 | 80 | 0.580 | 0.957 | 0.990 | 0 |
| conflict patch, block 19 (diagnostic) | 80 | 0.757 | 0.973 | 0.991 | 0 |
| agreement patches, block 18 | 240 | 0.868 | 0.960 | 0.989 | 0 |

The paper's in-context tokens carry a median of 1.0 × 10⁻⁵ (conflict) and 2.8 × 10⁻⁵
(no-patch) of the mass (descriptive only).

## 4. Per cell: primary readout and the paper readout (descriptive)

T_W, T_A (P, L and R weighted equally), d, mean q = (P-window, L, R), unresolved cases,
label shares among resolved cases, agreement transfer and resolution. The primary rows
use the combined gate with s_min = 0.10; the paper rows resolve by S alone with their own
s_min (0.10) and decide nothing. **Pilot/development, descriptive; not a result.**

| cell | readout | families | unresolved | T_W | T_A | d | mean q (P/L/R) | labels P/L/R | transfer | agreement resolved |
|---|---|---|---|---|---|---|---|---|---|---|
| c1 | primary (answer form) | 20 | 0/20 | 0.724 | 0.996 | 0.273 | 0.271/0.005/0.724 | 0.30/0.00/0.70 | 0.983 | 1.000 |
| c1 | paper, descriptive | 20 | 2/20 | 0.713 | 0.999 | 0.285 | 0.238/0.049/0.713 | 0.22/0.06/0.72 | 0.933 | 1.000 |
| c2 | primary (answer form) | 20 | 1/20 | 0.470 | 0.989 | 0.519 | 0.470/0.455/0.075 | 0.47/0.47/0.05 | 0.967 | 1.000 |
| c2 | paper, descriptive | 20 | 1/20 | 0.539 | 0.991 | 0.452 | 0.404/0.539/0.057 | 0.37/0.58/0.05 | 0.933 | 1.000 |
| c3 | primary (answer form) | 20 | 0/20 | 0.540 | 1.000 | 0.460 | 0.460/0.000/0.540 | 0.45/0.00/0.55 | 0.950 | 1.000 |
| c3 | paper, descriptive | 20 | 0/20 | 0.506 | 0.988 | 0.483 | 0.494/0.000/0.506 | 0.50/0.00/0.50 | 0.933 | 1.000 |
| c4 | primary (answer form) | 20 | 0/20 | 0.491 | 0.997 | 0.505 | 0.491/0.464/0.045 | 0.50/0.45/0.05 | 0.967 | 1.000 |
| c4 | paper, descriptive | 20 | 0/20 | 0.541 | 0.996 | 0.455 | 0.373/0.541/0.086 | 0.35/0.55/0.10 | 0.967 | 0.967 |

Likely selection if split A looked like this pilot: **c2** (d = 0.519), with a margin of
**0.014** over c4 (0.505). That margin is well within what 20 families per cell can
resolve; the selection happens on split A. c1 (0.273) stays above d_min = 0.20.

**Readout validity.** Under the conflict patch the primary argmax names the generated
entity in 79/79 runs whose generation is an in-context genre (one generation, in
pilot2-0040, was "Bass", an instrument); the paper readout's argmax does so in 67/79
(descriptive). At block 19: 80/80 against 58/80; agreement patches: 240/240 against
228/240; native runs 160/160 for both. As in pilot 1, the primary agreement is close to
guaranteed by construction (greedy decoding emits the top token, here an answer-form
token); it confirms the form, not the distribution.

## 5. Layer 18 against 19 (diagnostic, decides nothing)

Answer copy and the reflexive pointer predict the same token in this design, so this
comparison separates nothing. Mean q (primary) and the position of the generated entity
under the conflict patch, block 18 → block 19:

| cell | mean q at 18 | mean q at 19 | generated at 18 | generated at 19 |
|---|---|---|---|---|
| c1 | 0.27/0.01/0.72 | 0.03/0.00/0.97 | P 25%, R 70%, not a genre 5% ("Bass") | P 5%, R 95% |
| c2 | 0.47/0.46/0.08 | 0.07/0.24/0.70 | P 45%, L 45%, R 5%, other 5% | L 20%, R 80% |
| c3 | 0.46/0.00/0.54 | 0.09/0.00/0.91 | P 45%, R 50%, N 5% | P 5%, R 95% |
| c4 | 0.49/0.46/0.05 | 0.11/0.20/0.69 | P 50%, L 45%, N 5% | P 10%, L 20%, R 70% |

## 6. Runtime

float32 on MPS: 5.54 s per family on average (median 5.46, range 4.68–7.90), 443 s for 80
families; model load 4 s; the gate-7 reference on CPU 75 s for 32 families (about 2.3 s
each) after a 7 s load; MPS driver memory 11.4 GiB. Pilot families include the identity
self-patch, the layer-19 patch and generation under every patch, so the projections are
upper bounds. Yield 1.000.

| stage | qualifying | generated (expected) | hours (upper bound) |
|---|---|---|---|
| split A (50 per cell) | 200 | 200 | 0.31, plus about 1.3 min for 32 gate-7 CPU cases |
| split B (selected cell) | 200 | 200 | 0.31 |
| confirmation at the planning N | 200 | 200 | 0.31 (the final N is set at the freeze) |

## 7. STOP conditions and discrepancies

**Active STOP conditions: none triggered.** Armed for split A/B and the confirmation:
gate 6a (margin 0.067), 6b, 7, 8 (margin 0.088 on 80 cases), 9, the N rule at the
freeze, and the mean-consistency gate.

**Discrepancies and choices to confirm.**
1. **Pool rule scope.** The answer-form condition was applied to genres only (the only
   answers). Under an all-category reading, `flute`, `trumpet` and `whistle` would also
   leave the instrument pool; no measurement depends on their answer form.
2. **Audit tolerance.** Set to 10⁻⁵ for the answer mass after the smoke run (float32 vs
   float64 recomputation); not a declared gate.
3. **Gate-7 cases for split A** are not fixed yet; they must be fixed in split A's
   protocol commit (proposal: the first 8 generated families of each cell).
4. **Pilot-1 code.** The protocol-v1 producers are superseded in the tree; pilot 1's
   summary is reproduced in a test from the code at `3cdaffd`, its answer-form diagnostic
   still reproduces byte for byte, and its fp32 diagnostic script now refuses to run
   against the v2 runner (check out `3cdaffd` to rerun it).
5. **Environment.** `.venv-round2` of the original working tree was reused read-only, as
   permitted; the interpreter path, version and all package versions are recorded in
   `manifest.json`.
6. One conflict generation named an instrument ("Bass", pilot2-0040); it is recorded, and
   the case stays resolved (answer mass and S above the gates).

## 8. What the user must approve before split A

1. Final approval of the full value table under protocol v2 (`PROPOSED_VALUES.json`),
   including the pool rule's scope (item 1 above).
2. The 32 gate-7 cases of split A (8 per cell), fixed in split A's protocol commit before
   that run.
3. Acceptance that the planning N = 200 is provisional and that the final N and the
   adequacy label follow from split B at the freeze, by the recorded rule.
4. Authorization to draw split A (and then split B), with the environment used here or
   a dedicated one.
5. For public text: the statistic now uses the answer-form readout, not upstream's
   in-context tokens; the technical footnote should say so.

## 9. Files

| file | content |
|---|---|
| `manifest.json`, `RUN_STARTED.json` | protocol, pilot specification, producer and model hashes, pools and token ids, gate-3 checks, full environment; the start record holds the manifest hash |
| `records.jsonl` | 80 families, float32 on MPS, in the records-only checker's v2 schema plus the pilot measurements |
| `gate7_cpu_reference.jsonl` | the CPU float32 conflict and no-patch readouts of families 0–31 |
| `audit_full_logits.npz` | full-vocabulary logits of the conflict patch, families 0–3 |
| `timings.json`, `summary.json` | timings; the gate table computed by `src/mixing_pilot_summary.py` |
| `artifact_hashes.json` | generated index: sha256 of every file here and of every producer script |

Reproduce the summary and the index from the stored files:

```bash
python scripts/run_mixing_pilot.py summarize --output results/pilot2
python scripts/run_mixing_pilot.py index --output results/pilot2
```
