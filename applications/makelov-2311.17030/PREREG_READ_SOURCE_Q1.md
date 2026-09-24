# Preregistration: which read source predicts the Makelov MLP8 patch better? (Q1)

**Frozen in the commit that adds this file, before any fresh pair is run.** That commit is
pushed at once to the private development repository. It has no public timestamp:
nothing was registered outside that repository.

Follows the ordinary application template (`applications/PREREG_TEMPLATE.md` in
`~/causal-decidability`). Pilot: `MAKELOV_READ_SOURCE_PILOT.md`,
`results/makelov_read_source_001/` (32 base pairs, run 2026-09-20).

## 1. The effect and the question

Patching the published DAS direction of Makelov, Lange & Nanda (arXiv:2311.17030) at the
MLP 8 post-GELU activation of GPT-2 Small shifts the IO-minus-subject logit margin by a
large amount. The pilot held the write vector `v` fixed and switched only the read source,
between the output-visible component `v_R` and the output-null component `v_N`.

**Q1, the only question tested here:** on fresh base pairs, does explanation B predict the
two read interventions better than explanation A, case by case?

**Q2, "is the better explanation accurate enough?", is deferred.** No tolerance is
declared, and no adequacy claim will be made from this run.

## 2. The explanations

Both are anchored to measured endpoints of the same directed pair:

| candidate | predicted `read_row` margin | predicted `read_null` margin |
|---|---|---|
| A, visible read supplies the signal | the full-patch margin | the baseline margin |
| B, null read supplies the signal | the baseline margin | the full-patch margin |

Both reproduce the full patch by construction, so the full patch cannot test them. No
equivalence groups. Candidates not written here are neither tested nor excluded.

## 3. The design check, before any outcome

Conditions per directed pair: `baseline`, `full` (read `v`, write `v`), `read_row` (read
`v_R`, write `v`), `read_null` (read `v_N`, write `v`). On `full` the candidates do not
differ. On each read condition their predictions differ by the pair's own full-effect
gap, `|full − baseline|`, and in opposite directions. The design separates A from B
through the two read conditions only.

## 4. The measurement

- **Code:** `src/makelov_read_source.py` and `scripts/run_makelov_read_source.py`,
  unchanged since the pilot (commit `7401b78`). Operator, site, rank-aware SVD split and
  model transforms are as in the pilot.
- **Model and numerics:** GPT-2 Small, float32, CPU (`--device cpu`). The runner's
  four-pair CPU-float64 reference is recorded and not judged.
- **Units:** 64 fresh base pairs from the authors' held-out vocabulary and template split,
  seed **20260922** (unused before), both swap directions per pair. Before this freeze the
  64 pairs were generated with the tokenizer only, without any model forward pass: they
  share no case id and no single prompt with the pilot's 32. The unit is the base pair.
- **Command:**
  `python scripts/run_makelov_read_source.py --output results/makelov_read_source_q1 --n 64 --seed 20260922 --device cpu`
- **Execution checks:** the runner's identity and fidelity controls. If either fails, the
  run is **invalid** and carries no scientific reading.

## 5. The confirmation contract

- **Loss:** the pilot's. For each directed pair and candidate, the mean over the two read
  conditions of |observed margin − predicted margin|. Then the mean over the pair's two
  directions gives one loss per base pair and candidate.
- **Primary test (the only one judged):** the exact two-sided sign test on the per-pair
  loss differences `loss_A − loss_B`, ties dropped, α = 0.05. It asks how often each
  explanation predicts a fresh case better, and assumes only independent base pairs. With
  n = 64 it rejects when one candidate wins at least 41 pairs (actual size 0.033). Power:
  0.88 if B wins 70 % of cases, 0.98 at 75 %, 0.62 at 65 %.
- **Reported, not judged:**
  - the mean per-pair loss difference with a nominal 95 % percentile-bootstrap interval
    (10,000 resamples, seed 0);
  - the sign-flip permutation p-value of that mean, which assumes the differences are
    symmetric about zero under the null;
  - each candidate's mean loss.
- **Scoring:** `scripts/score_read_source_q1.py`. It checks the controls, the pair count
  and the disjointness from the pilot, and writes
  `results/makelov_read_source_q1/score.json`.

## 6. Outcomes, all of them acceptable

| outcome | condition |
|---|---|
| B predicts better | p ≤ 0.05 and B wins more pairs |
| A predicts better | p ≤ 0.05 and A wins more pairs (this would contradict the pilot) |
| no difference shown | p > 0.05 |
| invalid | a control failed, the pair count differs from 64, or a pair overlaps the pilot |

## 7. What this does not establish

- That B is accurate enough (Q2), or that its reading is the mechanism.
- Anything about the mean loss difference as a judged claim: the primary test concerns
  how often a candidate wins a case, not by how much.
- Anything about other sites, models, prompt families, or explanations not declared here.

## 8. Deviations

Every deviation is recorded in a dated section appended below, with its reason, before the
outcome is read.
