# Round 1 preflight and development plan

**Status: approved for the pilot only (28 September 2026); final approval pending.**
The user approved three protocol corrections, recorded the N rule, and approved the
value table below *for the pilot only* (task music performance, t_entity = 2, n = 7, the
four candidate cells, layer 18, d_min = 0.20, bfloat16 with eager attention and a
float32 reference). Final approval of the full table happens after the pilot report.
Nothing in this plan is frozen. Split A, split B, the freeze and the confirmation are not
authorized. Every value the brief leaves open is listed once, in the table below, with
its justification; the same values are in [PROPOSED_VALUES.json](PROPOSED_VALUES.json).

### Corrections of 28 September 2026

1. **Layer.** The study site is fixed by declaration: the residual stream entering
   decoder block 18 (Hugging Face `hidden_states[18]`), last token only, as upstream
   `tasks/dist.py:221`. The earlier 18 → 19 "answer copy" check is removed as a gate. In
   this conflict design the completed-answer copy (A) and the reflexive pointer (R)
   predict the same token (Round 0's own result), so the check cannot separate anything.
   A comparison of layers 18 and 19 is kept only as a labelled diagnostic and is never a
   STOP.
2. **s_min wording.** With 50 no-patch runs the upper order statistic at 0.99 is simply
   their maximum. The earlier sentence "at most 1% of unpatched runs would count as
   resolved" was wrong. The correct property: for a new no-patch run that is
   exchangeable with the m runs (and without ties), the chance that its S reaches the
   maximum of the m runs is 1/(m + 1), about 2% at m = 50; since s_min is at least that
   maximum, the chance that a new unpatched run counts as resolved is at most 1/(m + 1).
3. **Agreement anchor.** P, L and R weigh equally:
   T_A,c = mean over j ∈ {P, L, R} of (mean over resolved agreement runs with common
   target j of T(q)). New gate: at least 90% of the agreement-control runs must be
   resolved, else STOP.
4. **N rule** (applied after the pilot, recorded now; see below).

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
  (from metadata; not declared upstream), loaded offline from the local Hugging Face
  cache. Model and tokenizer hashes are recorded at gate 1.
- **Task:** music performance, target entity 2 of 3 (the genre); the question names the
  musician and the instrument ("What music did {Musician} play on the {Instrument}?").
- **Groups:** n = 7; middle positional index (0-based 3).
- **Patch:** the residual stream entering decoder block 18 (`hidden_states[18]`), last
  token only (`[-1]`), from donor to recipient. Fixed by declaration.
- **Readout:** logits at the last position for the n in-context genre tokens; p is their
  softmax. With the window w = 1 around i_P:
  S = window mass + p[i_L] + p[i_R]; q = (window mass, p[i_L], p[i_R]) / S; T = max(q).
- **Admissible cells:** i_P, i_L, i_R and i_N distinct, and i_L, i_R, i_N each more than
  one position from i_P, checked on design indices.

## How cases are built

The model runner (added in the next commit) renders prompts from the upstream schema
through the adapter, which reads `grammar/schemas.py` from the pinned clone; it does not
import `tasks/dist.py`, which cannot be imported at `c53372c` (see
[SOURCES.md](SOURCES.md)). The index logic is in `src/mixing_round1_design.py`.

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

1. **Pilot (authorized 28 September 2026).** 80 base contexts, 20 per candidate cell,
   seeds 1,000,000 + i, on the cached model only. It reports the gates, the unresolved
   rate for the N rule and descriptive anchors per cell, and runs the 18 vs 19
   diagnostic. Pilot data are development data: they are never used to estimate the
   frozen anchors or δ, and nothing is tuned on them except what this plan declares (N by
   the rule below; confirming that declared values are feasible). The full pilot
   specification is the `pilot` entry of [PROPOSED_VALUES.json](PROPOSED_VALUES.json).
2. **Split A** (50 qualifying cases per candidate cell; not yet authorized). Fix s_min
   per cell from its no-patch runs; estimate d = T_A − T_W for each cell; select the cell
   with the largest d, ties by declared order. If no cell reaches d_min:
   `NOT_DECIDABLE_WITH_CURRENT_INTERVENTIONS`, S1.
3. **Split B** (200 qualifying cases, selected cell only; not yet authorized). Resolution
   rate under the frozen s_min (STOP below 0.90); T_W = T(mean of q) with equal weights;
   T_A with P, L and R weighted equally; at least 90% of agreement runs resolved (else
   STOP); d must reach d_min; δ is resampled from B's resolved q-vectors.
4. **Freeze** the selected cell, T_W, T_A, d, w, s_min, δ, κ, the coverage and
   unresolved rules, the gates, seeds, N, model and code hashes, the runner and the
   checker. The checker (`scripts/check_mixing_round1_records.py`) is already committed.
5. **Confirmation:** N fresh qualifying cases on disjoint seeds, analysed once.

## Gates, before any interpretive patch

| # | gate | criterion | on failure |
|---|---|---|---|
| 1 | model pinned | revision above; sha256 of weight and tokenizer files recorded from the local snapshot and matched to the cache's content addresses | STOP |
| 2 | native competence and yield | yield ≥ 0.50 per split; native accuracy per position group reported | STOP if N is unreachable |
| 3 | token alignment | every entity used is one token with a leading space; entity positions located by design, not by string search | STOP if a category keeps fewer than n + 2 entities |
| 4 | hooks | exactly one write per patched forward, at block 18, last position; tensor shapes as expected | technical failure (INVALID) |
| 5 | identity self-patch | entity logits within 0.001, same argmax, same greedy generation | STOP |
| 6a | agreement transfer | on split B, ≥ 90% of agreement runs put the entity argmax on the common target | STOP |
| 6b | agreement resolution | on split B, ≥ 90% of agreement runs are resolved (S ≥ s_min) | STOP |
| 7 | precision | on 32 development cases, \|T(bf16) − T(fp32)\| ≤ 0.01 and identical resolution and labels | STOP |
| 8 | support | resolution rate on split B ≥ 0.90 (fixed by the brief) | STOP (population) |
| 9 | separation | d ≥ d_min on split B; some candidate reaches d_min on split A | STOP / NOT_DECIDABLE (S1) |
| 10 | power | the N rule below | STOP only as the rule states |
| — | 18 vs 19 (diagnostic) | argmax shares and mean q of the conflict patch at blocks 18 and 19, reported | never a STOP |
| — | mean consistency (confirmation) | ‖q̄_B − q̄_conf‖∞ ≤ δ | INVALID ("stale anchor") |

Every STOP is a valid result (S1).

## The N rule (recorded before the pilot)

N = 200 if the pilot's unresolved rate among conflict cases is at most 2%; otherwise the
smallest N in {300, 400, 500} that gives A_T-adequacy power ≥ 0.80 at the observed
unresolved rate (at 90% conformity among resolved cases); if even 500 does not, keep
N = 500 and label adequacy as not powered (exclusion must still reach power ≥ 0.80, else
STOP). The rate is taken over the pilot's qualifying conflict cases, pooled over the four
cells, each resolved under its cell's pilot s_min. The rule is implemented as
`n_rule` in `src/mixing_round1_design.py`. The resulting N awaits the user's final
approval.

**Open edge, flagged, not acted on.** At N = 200 exclusion power (70% of resolved cases
conform) drops below 0.80 for unresolved rates from about 1.8% to 2% (0.794 at 2%). The
rule text makes exclusion power a STOP only in the N = 500 branch. If the pilot's rate
falls in that band, the report says so and the user decides.

## The one table of values

Items marked *brief* are fixed by the brief, items marked *user* by the user's
corrections of 28 September 2026. All others are approved for the pilot only.

| value | value | justification |
|---|---|---|
| w | **1** (*brief*) | Fixed before any data; the paper reports positional predictions spread near i_P. |
| s_min rule | **max(Q, 0.10)**, Q = upper order statistic at 0.99 of S over the cell's no-patch runs (qualifying cases) on split A | With m ≤ 100 runs Q is their maximum. For an exchangeable new no-patch run, P(S reaches the maximum of m runs) = 1/(m + 1), about 2% at m = 50, so at most that share of new unpatched runs would count as resolved (*user* wording correction). The floor makes a resolved case put at least 10% of entity mass on the three candidates. |
| d_min | **0.20** | Band half-width κ·d ≥ 0.05 on the T scale (1/3 to 1): five times the dtype tolerance and above the ≈ 0.02 sampling error of T_W from 200 cases. |
| δ procedure | source **split B** (resolved q-vectors, selected cell); resample sizes m = resolved B cases and N; **10,000** resample pairs; seed **251006182**; δ = upper order statistic at 0.95 of the sup-norm difference of means | As in the brief. A passed gate also bounds \|T(q̄_conf) − T_W\| by δ, because max is 1-Lipschitz in the sup-norm. |
| false-INVALID rate | **0.05** | One drift-free confirmation in twenty would be declared stale. If fewer than N cases resolve, the realized rate is higher (conservative toward INVALID). |
| candidate cells (≤ 5) | **c1** (i_P 3, i_L 1, i_R 5, i_N 0), **c2** (3, 5, 1, 0), **c3** (3, 1, 5, 6), **c4** (3, 5, 1, 6); tie-break c1 → c4 | Middle i_P; lexical and reflexive at the nearest admissible distance on opposite sides, outside the window and not adjacent to each other; native answer at an end group for yield; mirror images of one layout. At n = 7 two of i_L, i_R, i_N are always adjacent. |
| sizes of A and B | **A: 50 per candidate cell (200); B: 200** qualifying cases; cap 2× per split | A only ranks four cells; B fixes the anchors (SE of T_W ≈ 0.02) and supplies δ. |
| N | **the N rule above** (*user*); target power 0.80 | See above; the brief's orientation reproduces exactly (0.831 adequacy at N = 150; 0.842 exclusion at N = 200). |
| T_A | **mean over j ∈ {P, L, R} of the mean T over resolved agreement runs with target j** (*user*) | Equal weights keep a target with fewer resolved runs from weighing less. |
| agreement resolution | **≥ 0.90** of agreement runs resolved (*user*) | T_A must not rest on a selected minority of agreement runs. |
| per-label tail | **0.0125** (*brief*) | α/4, as for the profile intervals. |
| agreement-control transfer | **≥ 0.90** of split-B agreement runs with argmax on the common target | T_A must describe a response in which the three signals agree; the paper reports consistent answers under two-way alignment (App. D.1). |
| yield floor | **≥ 0.50** | With a cap of 2N generated contexts, N stays reachable. |
| dtype tolerance | **\|ΔT\| ≤ 0.01** on **32** development cases, plus identical resolution and labels | One fifth of the smallest band half-width. |
| task | **music performance, t_entity = 2** (genre; query names musician and instrument) | Figure 5 (right) is the music task at t_entity = 2; App. A.1 reports lexical and reflexive balanced there. On boxes, t_entity = 2 is the last entity and the lexical mechanism dominates. |
| n | **7** | The brief's minimum. If averages are not mixed, the result is NOT_DECIDABLE (S1); a larger n is then the user's decision, before split A. |
| layer ℓ | **18** (input of block 18, `hidden_states[18]`), fixed by declaration (*user*); 19 only as a diagnostic | As upstream's notebook and Figure 2. The paper does not print ℓ for gemma-2-2b-it; the command-line default is 17. |
| precision, attention | **bfloat16**, **eager** attention, logits read in float32; float32 reference for gate 7 | As in the upstream notebook; eager because Gemma 2 soft-caps attention logits. |
| identity self-patch | **\|Δ logit\| ≤ 0.001**, same argmax and same greedy generation | Gate 5; allows only bfloat16 kernel noise. |
| native correctness | greedy, 3 new tokens; **first word equals the target**; readout argmax agrees; agreement donors recorded, not filtered | Stricter than upstream's substring checker. |
| entity pools | **single-token entities only**, checked on the pinned tokenizer before sampling; STOP if a category keeps fewer than n + 2 | Upstream asserts rather than filters. |
| seeds | disjoint blocks: pilot 1,000,000 + i; A 2,000,000 + i; B 3,000,000 + i; confirmation 4,000,000 + i | Fresh base contexts from disjoint seeds for confirmation. |

## Power, and why the unresolved rate matters

Exact power with the existing `clopper_pearson` helper (α/4 per tail), where "coverage"
is the share of *resolved* cases that conform and unresolved cases occur independently
(adequacy counts them as non-matches, exclusion as matches):

| N | unresolved | adequacy power at 0.90 | exclusion power at 0.70 |
|---|---|---|---|
| 150 | 0% | 0.831 | 0.731 |
| 200 | 0% | 0.957 | 0.842 |
| 200 | 1% | 0.899 | 0.819 |
| 200 | 2% | 0.806 | 0.794 |
| 200 | 5% | 0.390 | 0.705 |
| 300 | 2.5% | 0.911 | 0.925 |
| 400 | 4% | 0.850 | 0.969 |
| 500 | 5% | 0.814 | 0.986 |
| 500 | 6% | 0.627 | 0.980 |
| 500 | 10% | 0.036 | 0.929 |

The brief's orientation figures are the 0% rows and reproduce exactly. The resolution
gate allows up to 10% unresolved cases; the N rule recovers adequacy power up to about
5% unresolved, and above that adequacy is reported as not powered.

## Orientation from the blog (not used for any value)

The blog's interactive data for n = 7, target 2 (music, construction and patch positions
undocumented, changed two days before retrieval) hold one mean vector per cell. For the
index triples of c1/c3 and c2/c4, the window of i_P holds about 0.55 and 0.40 of the
mean mass, and the far-side candidate about 0.39 and 0.57, while the candidate at 0-based
index 1 receives about 0.01. The q-map of these means gives T ≈ 0.58 for both. This
suggests mixed averages at n = 7, but it is a mean of p, weighted by support, and not
the q-mean that T_W uses. The candidate at index 1 may carry little mass; the
development data decide.

## Active STOP conditions before the pilot

- Gates 1–3: checked by the pilot (model hashes from the local snapshot, the rendered
  chat template, single-token pools).
- Gates 4–8: checked for feasibility by the pilot; the binding checks are on split B.
- Gate 9 and the N rule: the pilot gives descriptive anchors and the unresolved rate;
  the binding decisions are on splits A and B.
- Openness check: no STOP (no case-level test found).

The pilot's results and the STOP conditions still active after it will be in
`results/pilot/PILOT_REPORT.md`.

## Authorizations still needed after the pilot

1. Final approval of, or changes to, the value table, including the N that the rule
   gives.
2. Drawing split A and split B (model runs).
3. The freeze, then the confirmation run.
4. Whether to timestamp the freeze externally, as the application guide recommends.
5. Any push of this branch.

## What Round 1 will leave open

Which mechanism each case follows; whether token preferences drive differences between
cases; other cells, layers and the patch positions `[-1, -4, -6, -8]`; and, unless the
between-case sentence is earned, whether cases differ at all. Round 1 does not run that
next round.
