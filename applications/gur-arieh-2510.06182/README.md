# Application: Gur-Arieh, Geva & Geiger, "Mixing Mechanisms" (arXiv:2510.06182)

Yoav Gur-Arieh, Mor Geva and Atticus Geiger, *Mixing Mechanisms: How Language Models
Retrieve Bound Entities In-Context*, arXiv:2510.06182v2 (ICLR 2026).
Code: [yoavgur/mixing-mechs](https://github.com/yoavgur/mixing-mechs) at commit
`c53372c`. Blog post: <https://yoav.ml/blog/2025/mixing-mechs/>.

## What this application does

The paper shows that a language model retrieves an entity from a list of bound entities
by mixing three signals: the entity's position (positional), the words used in the
question (lexical), and a direct pointer to the entity itself (reflexive). The authors
patch activations from one prompt into another, collect many such patches per
combination of target positions, average the resulting distributions, and fit a model
that mixes the three signals.

This application uses that study to show, simply, that the method's four-step loop runs
on a second, independent published case:

1. Write the rivals down as executable predictions.
2. Find a condition where their predictions differ.
3. Check that the difference is measurable; fix the rule before the data.
4. Decide on fresh cases.

What stays open becomes the next question. An explanation nobody formulated is neither
tested nor excluded. This is a demonstration of the procedure, not an identification of
the model's mechanism.

## Scope

- **Round 0** reconstructs, without a model, which of the paper's designs separate which
  candidates. It is labelled RETROSPECTIVE.
- **Round 1** asks one question the paper leaves open: the paper's mixture model
  describes *average* distributions per cell. Does a single case look like that average,
  or do cases differ? It uses one task, one cell, one layer and last-token patching.
- The levels S1 to S3 apply to Round 1 only: S1, a declared gate stopped the run;
  S2, the loop ran completely on fresh cases under a frozen rule; S3, S2 and at least
  one declared profile was excluded (always named).

**First pass:** sources and scope, Round 0, the Round 1 analyzer and checker with
synthetic tests, and a preflight plan. No model was run in the first pass.

**Status (28 September 2026):** the user approved three protocol corrections (layer 18
fixed by declaration, the s_min wording, an equal-weight agreement anchor with a 90%
resolution gate), recorded the N rule, and approved the value table for a development
pilot only, on the locally cached model. The pilot ran on 80 families
([results/pilot/PILOT_REPORT.md](results/pilot/PILOT_REPORT.md)): the bfloat16 dtype gate
failed and the declared readout's validity is in question. The user then approved
**protocol v2** (float32 execution on MPS with a CPU float32 reference for gate 7; the
answer form 'Country' as the primary readout with a 0.5 answer-token-mass gate; the
paper's in-context readout kept as descriptive only; pools restricted so `trance` drops;
the final N set at the freeze from split B) and a second pilot on fresh seeds. There is
no go for split A or split B; the freeze and the confirmation are not authorized.

## Files

| file | what |
|---|---|
| [SOURCE_LOCK.json](SOURCE_LOCK.json) | paper version, upstream commit and licence, file hashes, task registry, chat-template format, model revision from metadata, verified upstream facts with file:line, openness check |
| [SOURCES.md](SOURCES.md) | the same facts in plain text, with the discrepancies found |
| [requirements-analysis.txt](requirements-analysis.txt) | the pinned environment used for this pass (no model) |
| [scripts/lock_sources.py](scripts/lock_sources.py) | small adapter: checks a separate upstream clone against the lock; nothing is vendored |
| [ROUND0_RETROSPECTIVE.md](ROUND0_RETROSPECTIVE.md) | Round 0 (RETROSPECTIVE): which designs separate which candidates; hands one open question to Round 1 |
| [src/mixing_round0.py](src/mixing_round0.py), [results/round0_retrospective/round0.json](results/round0_retrospective/round0.json) | the typed prediction table and its stored reconstruction (`scripts/round0_reconstruction.py --check`) |
| [src/mixing_round1_analysis.py](src/mixing_round1_analysis.py) | Round 1 analyzer: q-map, the declared unresolved rule, mean-consistency gate, between-case check, development gates; reuses the existing `clopper_pearson` helper |
| [scripts/check_mixing_round1_records.py](scripts/check_mixing_round1_records.py) | records-only checker: recomputes every status, gate, the level and the between-case decision from the frozen manifest and raw records; imports no runner |
| [src/mixing_synthetic_worlds.py](src/mixing_synthetic_worlds.py), [results/synthetic_worlds/worlds.json](results/synthetic_worlds/worlds.json) | the six synthetic worlds (analyzer verification only, not evidence; `scripts/run_synthetic_worlds.py --check`) |
| [PREFLIGHT_PLAN.md](PREFLIGHT_PLAN.md), [PROPOSED_VALUES.json](PROPOSED_VALUES.json) | Round 1 preflight and development plan with the one table of values (**approved for the pilot only; final approval pending**), the corrections of 28 September 2026, the N rule, gates, power and the authorizations still needed |
| [src/mixing_round1_design.py](src/mixing_round1_design.py) | design-index checker, the conflict case of a fixed cell, the three-way agreement control, exact power and the N rule; no model |
| [src/mixing_prompts.py](src/mixing_prompts.py) | prompt construction exactly as upstream (row definition, `raw_input`, chat template with the first five characters dropped), from the task spec the adapter reads from the clone (`scripts/lock_sources.py`, `schema_spec`) |
| [src/mixing_runner.py](src/mixing_runner.py) | model runner (torch, protocol v2): offline snapshot and file hashes, the v2 pool rule, token alignment, last-token forward-pre hook at block ℓ with call and write counts, the primary answer-form readout (logits, answer-token mass, logsumexp over the vocabulary and its complement) and the descriptive paper readout, greedy generation, identity self-patch, and one record per family in the checker's schema |
| [scripts/run_mixing_pilot.py](scripts/run_mixing_pilot.py), [src/mixing_pilot_summary.py](src/mixing_pilot_summary.py) | the development pilot under protocol v2 (offline, cached model), its gate table from the stored files, and the generated artifact index; pilot 1 used the versions at commit `3cdaffd` |
| [requirements-model.txt](requirements-model.txt) | the environment used for model runs (torch 2.5.1, transformers 4.57.3) |
| [results/pilot/](results/pilot/PILOT_REPORT.md) | the development pilot: records, gate table, readout and dtype diagnostics (`scripts/diagnose_pilot_readout.py`, `scripts/diagnose_fp32_execution.py`) and the report |

Tests live with the repository's other tests: `tests/test_mixing_*.py` (the runner tests
use a tiny, randomly initialised Gemma 2 and a word-level tokenizer built locally in
`tests/mixing_tiny_model.py`; they need torch and are skipped without it) and
`tests/test_check_mixing_round1_records.py`.

Check a clone (standard library only):

```bash
python applications/gur-arieh-2510.06182/scripts/lock_sources.py \
    --upstream /path/to/mixing-mechs --check
```
