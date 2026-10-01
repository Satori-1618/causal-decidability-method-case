# Round 0: which designs separate which explanations (RETROSPECTIVE)

**RETROSPECTIVE.** No model was run and no new data were collected. This round
reconstructs, from the paper's text and the released code, which of the authors' designs
can tell which candidate explanations apart. Readings of the paper's reported outcomes are
not frozen decisions. The levels S1 to S3 do not apply to this round.

## The candidates

| | candidate | what it predicts after the patch |
|---|---|---|
| P | positional | the entity in the group at the donor's query position, i_P |
| L | lexical | the entity bound to the donor's query entity, at i_L |
| R | reflexive pointer | the donor's target entity, found through a pointer, at i_R if it occurs in the recipient |
| A | completed-answer copy | the donor's already retrieved answer, copied as such |
| N | no effect | the recipient's own answer, at i_N |

Each prediction has one of four types: a **point** (one entity), a **set** (several
entities, any of which fits; used for the positional window i_P ± w), **undefined** (the
mechanism produces no output), or **outside scope** (the readout cannot represent the
prediction, or the condition is not one the candidate speaks to). An undefined prediction
is not evaluable: it gives no support and no separation, and the candidate stays
compatible, untested. It is never a number.

## Predictions per design

Indices are the paper's Figure 1 example after patching (n = 4, i_P = 2, i_L = 1,
i_R = 3, recipient asking about group 4).

| design | P | L | R | A | N | groups that cannot be told apart |
|---|---|---|---|---|---|---|
| §3.2 (donor answer present), layer ℓ | point 2 | point 1 | point 3 | point 3 | point 4 | {R, A} |
| same, but i_L = i_R (upstream allows this) | point 2 | point 3 | point 3 | point 3 | point 4 | {L, R, A} |
| same, P as window ±1 | set {1, 2, 3} | point 1 | point 3 | point 3 | point 4 | P overlaps L, R and A |
| §3.4 (donor answer absent), layer ℓ, readout sees the absent token | point 2 | point 1 | undefined | point ABSENT | point 4 | none among P, L, A, N; R not evaluable |
| §3.4, layer ℓ, in-context entity readout only | point 2 | point 1 | undefined | outside scope | point 4 | R and A not evaluable |
| §3.4 repeated at layer ℓ+1 | outside scope | outside scope | outside scope | outside scope | outside scope | a different intervention |

With admissible indices (n = 7, i_P = 4, i_L = 2, i_R = 6, i_N = 1), the window ±1 is
disjoint from every other prediction, and only {R, A} remain together.

## Round 0 in the table form

| Step | What happened |
|---|---|
| **Open** | The paper's designs are meant to tell positional, lexical and reflexive retrieval apart. Which designs separate which explanations, once two more are written down: a plain copy of the donor's finished answer (A), and no effect (N)? |
| **Added test** | None; this round is retrospective. Every candidate's prediction was written down per design as a point, a set, undefined, or outside scope. |
| **Result** | §3.2 separates P, L and N; R and A predict the same entity and form one group. If the lexical and reflexive indices coincide, which the released sampler allows, L joins that group. In §3.4, with the donor's answer absent from the recipient, A predicts the absent token and R is not evaluable. Layers ℓ and ℓ+1 are different interventions. |
| **Narrowed** | The authors' §3.4 design can exclude a strong answer copy, provided the readout sees the absent token. The paper reports that at layer ℓ the model did not answer with the absent entity, and at ℓ+1 it did, which shows the paper's readout could represent it; the released default readout, which scores only in-context entity tokens, could not. A small answer-copy contribution below that resolution remains possible. R stays compatible and is not positively identified. |
| **Still open** | The paper's mixture model describes average distributions per cell. Whether single cases look like that average was not checked. This is the question of Round 1. |

The §3.4 design is the authors' own. This round only states what it separates.

## Files

- `src/mixing_round0.py`: the typed prediction table and the separation analysis, using
  the repository's `causal_decidability.signatures` for exact equivalence.
- `scripts/round0_reconstruction.py --check`: recomputes
  `results/round0_retrospective/round0.json` and compares.
- `tests/test_mixing_round0.py`: the structural tests.
