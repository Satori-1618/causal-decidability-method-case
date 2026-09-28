# Round 1 preflight and development plan (PROPOSED)

**Status: proposed, awaiting approval.** Nothing in this plan is frozen, no model weights
were downloaded and no model was run. Every value the brief leaves open is listed once,
in the table below, with its justification. The same values are in
[PROPOSED_VALUES.json](PROPOSED_VALUES.json). Model runs start only after the user
approves these values, accepts the Gemma licence and authorizes the runs.

## What Round 1 asks, in plain words

The paper fits its mixture model to average distributions: 150 patched cases per
combination of target positions, averaged after a softmax. The same average can come
about in two ways. Every case may spread its probability like the average (W), or cases
may differ (H, an explanation nobody had written down). Round 1 checks, on fresh cases
and under a rule fixed before the data, whether single cases look like the average in
one respect: **how strongly a case favours one of the three candidate positions**. It
compares concentration only, never which position is favoured, and it does not identify
a mechanism.

## Setting

- **Model:** `google/gemma-2-2b-it` at revision `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
  (from metadata; not declared upstream). Model and tokenizer hashes are recorded at
  gate 1.
- **Task:** music performance, target entity 2 of 3 (the genre); the question names the
  musician and the instrument ("What music did {Musician} play on the {Instrument}?").
- **Groups:** n = 7; middle positional index (0-based 3).
- **Patch:** the residual stream entering decoder block ℓ, last token only (`[-1]`), from
  donor to recipient; ℓ proposed as 18.
- **Readout:** logits at the last position for the n in-context genre tokens; p is their
  softmax. With the window w = 1 around i_P:
  S = window mass + p[i_L] + p[i_R]; q = (window mass, p[i_L], p[i_R]) / S; T = max(q).
- **Admissible cells:** i_P, i_L, i_R and i_N distinct, and i_L, i_R, i_N each more than
  one position from i_P, checked on design indices.

## How cases are built

The runner, written after authorization, renders prompts from the upstream schema
templates through the adapter; it does not import `tasks/dist.py`, which cannot be
imported at `c53372c` (see [SOURCES.md](SOURCES.md)). The index logic below is already
implemented and tested in `src/mixing_round1_design.py`, without any model.

- **Conflict case (one per fresh base context).** Draw a recipient binding matrix G with
  seven distinct musicians, genres and instruments. The recipient asks about group i_N.
  The donor G′ swaps the genres of groups i_P and i_R and the musician–instrument pairs
  of groups i_P and i_L, and asks about the pair now at i_P. Then the positional
  mechanism points to i_P, the lexical one to i_L and the reflexive one to i_R. This
  mirrors upstream's main template, except that the cell is fixed and i_L ≠ i_R is
  enforced (upstream draws collide in 20% of cases at n = 7, and 77% of its draws would
  be inadmissible under the window).
- **Agreement control (three per case).** For j in (i_P, i_L, i_R): keep group j in
  place, move every other group to a different position with its bindings intact (a
  derangement), and let the donor ask about group j. The design-index checker confirms
  that positional, lexical and reflexive indices all equal j. This construction is this
  application's own; the paper's App. D.1 aligns only two mechanisms. The blog shows
  averages for cells where all three coincide, so no novelty is claimed.
- **No-patch run (one per case):** the recipient without a patch, for anchoring s_min.
- **Design-index check:** every case's indices are recomputed from the two matrices and
  both questions; a mismatch is a technical failure, not a lost case.

## Population and yield

A case enters only if the unpatched recipient and the donor are both natively correct
under greedy generation, and the argmax over the seven genre logits names the same genre
as the generation. Generation stops at the N-th qualifying case, with a cap of 2N
generated contexts. The yield is qualifying over generated, reported next to every
status.

## Development

1. **Optional early pilot (advice, needs its own authorization).** About 20 base
   contexts per candidate cell on disjoint seeds, run at ℓ and ℓ+1. It settles the three
   likely STOP causes early: whether cell averages are mixed at n = 7, what S looks like,
   and whether T agrees between bfloat16 and float32. Pilot cases are never reused. A
   pilot can stop the plan or send a question back to the user; it does not tune κ, the
   coverage requirement or the candidates.
2. **Split A** (50 qualifying cases per candidate cell). Fix s_min per cell from its
   no-patch runs; estimate d = T_A − T_W for each cell; select the cell with the largest
   d, ties by declared order. If no cell reaches d_min: `NOT_DECIDABLE_WITH_CURRENT_INTERVENTIONS`, S1.
3. **Split B** (200 qualifying cases, selected cell only). Resolution rate under the
   frozen s_min (STOP below 0.90); T_W = T(mean of q) with equal weights; T_A = mean of
   T over cases and the three agreement targets; d must reach d_min; δ is resampled from
   B's resolved q-vectors.
4. **Freeze** the selected cell, T_W, T_A, d, w, s_min, δ, κ, the coverage and
   unresolved rules, the gates, seeds, N, model and code hashes, the runner and the
   checker. The checker (`scripts/check_mixing_round1_records.py`) is already committed.
5. **Confirmation:** N fresh qualifying cases on disjoint seeds, analysed once.

## Gates, before any interpretive patch

| # | gate | proposed criterion | on failure |
|---|---|---|---|
| 1 | model pinned | revision above; sha256 of weight and tokenizer files recorded after licence acceptance | STOP |
| 2 | native competence and yield | yield ≥ 0.50 per split; native accuracy per position group reported | STOP if N is unreachable |
| 3 | token alignment | every entity used is one token with a leading space; entity positions located by design, not by string search | STOP if a category keeps fewer than n + 2 entities |
| 4 | hooks | exactly one write per patched forward, at block ℓ, last position; tensor shapes as expected | technical failure (INVALID) |
| 5 | identity self-patch | entity logits within 0.001, same argmax, same greedy generation | STOP |
| 6 | agreement control | on split B, ≥ 90% of agreement runs put the entity argmax on the common target | STOP |
| 7 | precision | on 32 development cases, \|T(bf16) − T(fp32)\| ≤ 0.01 and identical resolution and labels | STOP |
| 8 | support | resolution rate on split B ≥ 0.90 (fixed by the brief) | STOP (population) |
| 9 | separation | d ≥ d_min on split B; some candidate reaches d_min on split A | STOP / NOT_DECIDABLE (S1) |
| 10 | power | exclusion power ≥ 0.80 at the split-B unresolved rate for N in (200, 250, 300) | STOP |
| — | layer check (pilot) | at ℓ the donor's answer copy at or below chance, at ℓ+1 the majority | STOP, return to the user |
| — | mean consistency (confirmation) | ‖q̄_B − q̄_conf‖∞ ≤ δ | INVALID ("stale anchor") |

Every STOP is a valid result (S1).

## The one table of proposed values

Items marked *brief* are fixed by the brief and restated for completeness. All others are
proposals awaiting approval.

| value | proposal | justification |
|---|---|---|
| w | **1** (*brief*) | Fixed before any data; the paper reports positional predictions spread near i_P. |
| s_min rule | **max(Q, 0.10)**, Q = upper order statistic at 0.99 of S over the cell's no-patch runs on split A | Anchored to no-patch runs: at most 1% of unpatched runs would count as resolved. The floor makes a resolved case put at least 10% of entity mass on the three candidates, so q is not a ratio of tiny masses. |
| d_min | **0.20** | Band half-width κ·d ≥ 0.05 on the T scale (1/3 to 1): five times the dtype tolerance and above the ≈ 0.02 sampling error of T_W from 200 cases. |
| δ procedure | source **split B** (resolved q-vectors, selected cell); resample sizes m = resolved B cases and N; **10,000** resample pairs; seed **251006182**; δ = upper order statistic at 0.95 of the sup-norm difference of means | As in the brief. A passed gate also bounds \|T(q̄_conf) − T_W\| by δ, because max is 1-Lipschitz in the sup-norm. |
| false-INVALID rate | **0.05** | One drift-free confirmation in twenty would be declared stale. If fewer than N cases resolve, the realized rate is higher (conservative toward INVALID). |
| candidate cells (≤ 5) | **c1** (i_P 3, i_L 1, i_R 5, i_N 0), **c2** (3, 5, 1, 0), **c3** (3, 1, 5, 6), **c4** (3, 5, 1, 6); tie-break c1 → c4 | Middle i_P; lexical and reflexive at the nearest admissible distance on opposite sides, outside the window and not adjacent to each other; native answer at an end group for yield; mirror images of one layout. At n = 7 two of i_L, i_R, i_N are always adjacent. |
| sizes of A and B | **A: 50 per candidate cell (200); B: 200** qualifying cases; cap 2× per split | A only ranks four cells; B fixes the anchors (SE of T_W ≈ 0.02) and supplies δ at m = N = 200. |
| N and target power | **N = 200**; declared power **0.80** for exclusion when 70% of resolved cases conform, at the split-B unresolved rate; else the smallest of 250, 300 that reaches it; else STOP | The brief's orientation reproduces exactly (0.831 adequacy at N = 150; 0.842 exclusion at N = 200). Adequacy power is reported, not gated, because unresolved cases lower it steeply (table below). |
| per-label tail | **0.0125** (*brief*) | α/4, as for the profile intervals. |
| agreement-control transfer | **≥ 0.90** of split-B agreement runs with argmax on the common target | T_A must describe a response in which the three signals agree; the paper reports consistent answers under two-way alignment (App. D.1). |
| yield floor | **≥ 0.50** | With a cap of 2N generated contexts, N stays reachable. |
| dtype tolerance | **\|ΔT\| ≤ 0.01** on **32** development cases, plus identical resolution and labels | One fifth of the smallest band half-width. |
| task | **music performance, t_entity = 2** (genre; query names musician and instrument) | Figure 5 (right) is the music task at t_entity = 2; App. A.1 reports lexical and reflexive balanced there. On boxes, t_entity = 2 is the last entity and the lexical mechanism dominates. |
| n | **7** | The brief's minimum. If averages are not mixed, the result is NOT_DECIDABLE (S1); a larger n is then the user's decision, before split A. |
| layer ℓ | **18** (input of block 18, `hidden_states[18]`), checked in the pilot by the paper's D.2 criterion | Not printed in the paper; Figure 2 and the notebook use 18, the blog places retrieval at 19–25; the command-line default is 17. |
| precision, attention | **bfloat16**, **eager** attention, logits read in float32; float32 reference for gate 7 | As in the upstream notebook; eager because Gemma 2 soft-caps attention logits. |
| identity self-patch | **\|Δ logit\| ≤ 0.001**, same argmax and same greedy generation | Gate 5; allows only bfloat16 kernel noise. |
| native correctness | greedy, 3 new tokens; **first word equals the target**; readout argmax agrees; agreement donors recorded, not filtered | Stricter than upstream's substring checker. |
| entity pools | **single-token entities only**, checked on the pinned tokenizer before sampling; STOP if a category keeps fewer than n + 2 | Upstream asserts rather than filters; the gated tokenizer could not be checked in this pass. |
| seeds | disjoint blocks: pilot 1,000,000 + i; A 2,000,000 + i; B 3,000,000 + i; confirmation 4,000,000 + i | Fresh base contexts from disjoint seeds for confirmation. |

## Power, and why the unresolved rate matters

Exact power with the existing `clopper_pearson` helper (α/4 per tail), where "coverage"
is the share of *resolved* cases that conform and unresolved cases occur independently
(adequacy counts them as non-matches, exclusion as matches):

| N | unresolved | adequacy power at 0.90 | exclusion power at 0.70 |
|---|---|---|---|
| 150 | 0% | 0.831 | 0.731 |
| 200 | 0% | 0.957 | 0.842 |
| 200 | 1% | 0.90 | 0.82 |
| 200 | 2% | 0.81 | 0.79 |
| 200 | 5% | 0.39 | 0.71 |
| 200 | 10% | 0.03 | 0.53 |
| 250 | 5% | 0.45 | 0.79 |
| 300 | 5% | 0.57 | 0.88 |

The brief's orientation figures are the 0% rows and reproduce exactly. The resolution
gate allows up to 10% unresolved cases; at that rate adequacy is practically out of
reach and exclusion power drops to about 0.5. **Decision for the user:** accept that
adequacy is reported but not powered when unresolved cases are more than about 2%, or
tighten the resolution requirement. The plan does not change the brief's rule on its own.

## Orientation from the blog (not used for any value)

The blog's interactive data for n = 7, target 2 (music, construction and patch positions
undocumented, changed two days before retrieval) hold one mean vector per cell. For the
index triples of c1/c3 and c2/c4, the window of i_P holds about 0.55 and 0.40 of the
mean mass, and the far-side candidate about 0.39 and 0.57, while the candidate at 0-based
index 1 receives about 0.01. The q-map of these means gives T ≈ 0.58 for both. This
suggests mixed averages at n = 7, but it is a mean of p, weighted by support, and not
the q-mean that T_W uses. The candidate at index 1 may carry little mass; the
development data decide.

## Active STOP conditions

- Gate 1: model and tokenizer hashes are not recorded (licence not accepted). Blocking
  for every model run.
- Gate 3: single-token status of the entities is unverified (gated tokenizer).
- Layer check: ℓ = 18 is inferred, not printed in the paper.
- Gates 2 and 5–10 and the mean-consistency gate: armed, untested until data exist.
- Power: armed, and sensitive to the unresolved rate (above).
- Openness check: no STOP (no case-level test found).

## Authorizations still needed

1. Approval of, or changes to, the proposed values above.
2. The user accepts the Gemma licence on Hugging Face (only the user can do this) and
   makes access available in their own environment; no token passes through this work.
3. Download of the model and tokenizer at the pinned revision (about 5.2 GB).
4. Installation of a pinned model-run environment outside the repository (proposed:
   `torch==2.5.1`, `transformers==4.57.3`, as in the Makelov application; to be
   confirmed, with Gemma 2 support checked).
5. Model runs, in order: the optional pilot, then splits A and B, then the freeze, then
   confirmation.
6. Whether to timestamp the freeze externally, as the application guide recommends.
7. Any push of this branch.

## What Round 1 will leave open

Which mechanism each case follows; whether token preferences drive differences between
cases; other cells, layers and the patch positions `[-1, -4, -6, -8]`; and, unless the
between-case sentence is earned, whether cases differ at all. Round 1 does not run that
next round.
