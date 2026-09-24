# Application: Makelov, Lange & Nanda (2023), arXiv:2311.17030

*Is this the subspace you are looking for? An interpretability illusion for subspace
activation patching.* Code and published directions:
[amakelov/activation-patching-illusion](https://github.com/amakelov/activation-patching-illusion)
at revision `e0c465b7`.

Patching the published DAS direction in MLP 8 of GPT-2 Small moves the model's answer
by a large, precisely measured amount. The paper shows that this success can be an
illusion. This application asks the method's question of the same patch: **which
explanations of that shift can a given design tell apart, and at what resolution?**

## The four steps on this paper

### 1. Specify the explanations

Split the direction `v` into `v_R`, the part the MLP's output matrix can transmit, and
`v_N = v − v_R`, the part it maps to zero. The patch reads the donor–recipient difference
along a direction and writes it back along a direction. Two endpoint explanations of
where the effective signal comes from:

- **A, visible read:** the effect comes from reading along `v_R`.
- **B, null read:** the effect comes from reading along `v_N`, the component the output
  matrix cannot transmit. This is the illusion the paper describes.

### 2. Check the design before any outcome

| condition | A predicts | B predicts | separates A from B? |
|---|---|---|---|
| full patch: read `v`, write `v` | the full effect (anchored) | the full effect (anchored) | **no**, by construction |
| Table 1 null patch: read and write `v_N` | no change | no change | **no**: the write lies in the kernel of the output matrix |
| read `v_R`, write `v` | the full effect | no change | yes |
| read `v_N`, write `v` | no change | the full effect | yes |

The full patch has a large effect, 1.543 nats on the logit difference (paired SE 0.016,
n = 2000). Both explanations are stated relative to the measured full-patch and baseline
endpoints, so both reproduce the full patch by construction: that condition cannot test
them, and matching it is no evidence for either. The null patch of Table 1 cannot separate
them either: measured, it changes no example by more than 2.1e-5 nats. The informative
test is the pair of read conditions, where holding the write vector fixed and switching
the read source gives opposite predictions. There the predicted gap is the size of the
full effect in each pair.

### 3. Check that the difference is measurable

- **Read-source pilot** (32 base pairs, both swap directions): the predicted gap between
  A and B is 1.598 nats per pair on average. The measured paired advantage of B over A in
  prediction error is 1.078 nats, with a nominal 95 % bootstrap interval of [0.961, 1.198].
  The difference is measurable by a wide margin.
- **Table 1, retrospectively:** the published rowspace-versus-nullspace contrast is 85.9
  paired standard errors from zero on logit difference, but only 2.65 on interchange
  accuracy, where it rests on 7 discordant examples against 0 (exact McNemar p = 0.016).
  The same question has a different resolution depending on the readout. The paper's
  Appendix A.2 leans on this fine contrast; the coarse contrasts sit at 9 to 100.
- **`resid_mid.8`, prospectively:** a preregistered test of the calculator's planning
  claim, on a published direction for which no evaluation was found in the paper, the
  saved metrics or the notebook outputs.

### 4. Return the explanations still compatible

- **Read-source pilot:**

  | receiver pattern | full patch | read `v_R` | read `v_N` |
  |---|---:|---:|---:|
  | ABB | −1.310 | −0.216 | −1.102 |
  | BAB | −1.887 | −0.318 | −1.588 |

  B predicts the component patches far better: mean absolute prediction error 0.260
  against 1.338 nats; all 32 paired advantages favour B. No adequacy threshold was
  declared in advance, so this is a relative fit, not a compatible-set decision. Reading
  along the null component alone gives 84 % of the full patch's mean effect, reading along
  the visible component alone 17 %. The sum of the separately measured mean effects is
  about 101 % of the full-patch effect, on the fresh pairs of Q1 as well. In single
  directed pairs the sum departs from the full effect by up to 15 % (pilot) and 25 % (Q1).
  These are effect ratios, reported and not judged. They do not show that only the null
  component matters: the visible read has an effect of its own.

  **Confirmed on fresh pairs (Q1).** A preregistration froze the question "does B predict
  the two read interventions better than A, case by case?", the pilot's loss, and an exact
  sign test (`PREREG_READ_SOURCE_Q1.md`). It was committed and pushed before the run. On
  64 fresh base pairs B predicted better in **64 of 64** (p = 1.1e-19); mean loss 0.238 against
  1.223 nats (reported, not judged). This confirms the comparison. Whether B is accurate
  *enough* (Q2) was deferred: it needs a tolerance with an independent justification.

  Applying a confirmation contract to the same data shows what a declared tolerance would
  decide (`examples/from_data.py` on `main`, with the pilot's own loss: absolute error,
  pooled over the two read conditions). At a tolerance of 10 % of the full effect, both
  candidates are excluded. At 25 %, B is shown adequate and A is excluded. The tolerance
  is resampled together with the full effect it is a fraction of. No tolerance was fixed
  before these data were seen, so this is an illustration, not a result. A confirmation
  fixes loss, tolerance and uncertainty rule together, first, and runs on fresh pairs.
  A tolerance there is a precision requirement, not a share of the mechanism: the two
  read effects need not add up to the full effect.
- **`resid_mid.8`:** 28 of 28 predicted verdicts matched, which the frozen criterion calls
  supported. The result is narrow: every prediction was "decidable", the checks share
  nested data, and in 21 of 28 the rule excluded both endpoint explanations. Neither
  endpoint fits there. The separate row-space patch reaches 86 % (logit difference) and
  75 % (interchange accuracy) of the full patch's mean effect. These are effect ratios,
  not mechanism shares.

**What this adds to the paper.** The illusion is the authors' insight, found by
reasoning about the output kernel; the method would not have found it. What the method
adds is explicit: which conditions cannot separate the two explanations (the full patch
and the null patch of Table 1), which can (a fixed write with a switched read), and at
what resolution the comparison is decided.

## The four experiments

| experiment | status | files |
|---|---|---|
| Table 1 MLP8 rows, reproduced | exact on every published digit of the patched accuracies | `src/makelov_table1.py`, `scripts/run_makelov_table1.py`, `scripts/analyze_makelov_table1.py`, `scripts/decide_makelov_table1.py`, `results/makelov_replication_001/` |
| read-source pilot, MLP8 | development pilot, 2026-09-20; no threshold, not a confirmation | `MAKELOV_READ_SOURCE_PILOT.md`, `src/makelov_read_source.py`, `scripts/run_makelov_read_source.py`, `scripts/verify_makelov_read_source.py`, `results/makelov_read_source_001/` |
| read-source confirmation Q1, MLP8 | preregistered, frozen before the run; 64 fresh pairs; B better in 64 of 64 | `PREREG_READ_SOURCE_Q1.md`, `scripts/score_read_source_q1.py`, `scripts/check_read_source_q1.py`, `results/makelov_read_source_q1/` (with `NOTES.md`) |
| `resid_mid.8`, preregistered | Freeze A → blinded pilot → Freeze B → confirmation | `PREREG_RESID_MID8.md`, `src/resid_mid8_*.py`, `scripts/*resid_mid8*.py`, `results/resid_mid8_*/`, [result](results/resid_mid8_confirmation/README.md) |

The two MLP8 experiments split `v` slightly differently. The Table 1 replication follows
the authors' QR basis, which counts one numerically dead direction as visible. The pilot
uses a rank-aware SVD (rank 767).

## Reproduce

```bash
pip install -r requirements.txt
python3 scripts/fetch_upstream.py                    # upstream files, hash-verified
python3 scripts/rebuild_read_source_directions.py    # the pilot's directions, hash-verified
python3 -m pytest tests -q
python3 scripts/verify_makelov_read_source.py results/makelov_read_source_001
python3 scripts/check_read_source_q1.py
PYTHONHASHSEED=0 python3 scripts/reproduce_resid_mid8.py
python3 scripts/run_makelov_read_source.py --output /tmp/read_source_rerun --n 32 --device cpu
```

- The rebuild step also downloads GPT-2 into the local cache, which the pilot's runner
  requires.
- A fresh CPU re-run of the pilot gives identical prompts and directions, every directed
  margin within 4.2e-5 nats and every summary statistic within 1e-6 of the original run
  on Apple MPS. The comparison is archived in
  `results/makelov_read_source_001_cpu_rerun/comparison.json`, written by
  `scripts/compare_read_source_runs.py`.
- The pilot's own note refers to `requirements-makelov.txt` in the development
  repository; here the same pins are in `requirements.txt`.
- `reproduce_resid_mid8.py` compares all 2,000 confirmation pairs with the committed
  files and writes nothing.
- `check_read_source_q1.py` rechecks the Q1 confirmation in one command and writes
  nothing. With the frozen scorer's own functions it recomputes wins, ties, the sign-test
  p and the outcome from the raw records and compares them with `score.json`. It checks
  the pinned hashes and the disjointness from the pilot, and runs the verifier (controls,
  directed pairs, ids, summary means, input hashes). The frozen scorer itself refuses to
  score twice.
- Fetched and rebuilt files are untracked and can go missing, for example after a
  rebase. Run the first two commands again when they are.

The original resid_mid.8 runners are byte-identical to the versions that ran. They refuse
to run again, because their outputs exist and their inputs are not committed here.
[PROVENANCE.md](PROVENANCE.md) lists every copied file with its hash.

Upstream files are fetched, not redistributed: the upstream repository has no licence.
For the same reason the pilot's `directions.npz`, which contains the authors' vector, is
rebuilt locally rather than shipped.
