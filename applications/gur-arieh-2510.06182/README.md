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

**First pass (this branch so far):** sources and scope, Round 0, the Round 1 analyzer
and checker with synthetic tests, and a preflight plan whose numerical values are
proposals awaiting approval. No model weights were downloaded and no model was run.
Model runs need the user's approval of the proposed values, acceptance of the Gemma
licence by the user, and explicit authorization.

## Files

| file | what |
|---|---|
| [SOURCE_LOCK.json](SOURCE_LOCK.json) | paper version, upstream commit and licence, file hashes, task registry, chat-template format, model revision from metadata, verified upstream facts with file:line, openness check |
| [SOURCES.md](SOURCES.md) | the same facts in plain text, with the discrepancies found |
| [requirements-analysis.txt](requirements-analysis.txt) | the pinned environment used for this pass (no model) |
| [scripts/lock_sources.py](scripts/lock_sources.py) | small adapter: checks a separate upstream clone against the lock; nothing is vendored |
| [ROUND0_RETROSPECTIVE.md](ROUND0_RETROSPECTIVE.md) | Round 0 (RETROSPECTIVE): which designs separate which candidates; hands one open question to Round 1 |
| [src/mixing_round0.py](src/mixing_round0.py), [results/round0_retrospective/round0.json](results/round0_retrospective/round0.json) | the typed prediction table and its stored reconstruction (`scripts/round0_reconstruction.py --check`) |

Check a clone (standard library only):

```bash
python applications/gur-arieh-2510.06182/scripts/lock_sources.py \
    --upstream /path/to/mixing-mechs --check
```
