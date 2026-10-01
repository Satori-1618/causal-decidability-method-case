# What Round 3B can learn from *Mixing Mechanisms*

**Pre-run audit, 24 September 2026.** This was written before any Round 3B model
outputs. Its purpose is to limit and improve [our proposed test](ROUND3B_POSITIVE_IDENTIFICATION.md),
not to claim a reproduction. The next authorized measurement is native-only Stage A.

Gur-Arieh, Geva and Geiger already construct interventions on which rival retrieval
mechanisms disagree, and connect the analysis to causal abstraction. Their framework
also allows mixtures. Comparing explicit rivals is therefore established practice;
our contribution here must be the checked application and reusable evaluation procedure,
not a claim to have invented that experimental logic.

## The distinctions are not interchangeable

In [the paper, §§3.1–3.4](https://arxiv.org/html/2510.06182v2), entities are bound in
groups. A question supplies an entity and requests its counterpart. Positional retrieval
carries a group index, lexical retrieval the query entity, and reflexive retrieval the
answer entity as a pointer. The paper intervenes on the residual stream.

Our task has one three-person event and asks for a role. Our prospective intervention
is a fixed scalar MLP8 perturbation. The following mapping is **our analysis of the
two designs**, not terminology imported unchanged from the paper.

| Candidate in the audit | Applicability to the current 3B grid | What a positive 3B result could resolve |
|---|---|---|
| Paper positional mechanism: group address | Not directly instantiated: there is one event, and the requested role changes | Does not exclude group addressing combined with transferred role selection |
| Paper lexical retrieval: entity-valued query | Undefined without an adapter: our query supplies a role, not a person | No direct conclusion about this published mechanism |
| Adapted lexical role lookup | A stipulated role→person lookup predicts the same targets as relational-query transfer | These implementations remain in the same target-prediction class |
| Paper reflexive pointer: donor answer as address | Every donor answer name also occurs in the recipient | Same target labels as answer-name copying; the present grid cannot separate them |
| Final-answer/name copy | Operationalized here by a generous signed response line toward a native name endpoint | Can reject that quantitative response restriction, not every possible output-copy implementation |
| Mention-position copy/inhibition | Operationalized by a signed line toward the recipient name at the donor answer's mention slot | Can reject this restriction; it is not the paper's group-index mechanism |
| Relational-query response class | Three development-fitted gains predict the recipient's native response to the donor query | Could support accurate cross-query responses within the declared quantitative menu |
| Question-change-only; no-op | Already have executable response sets | Remain explicit competitors |

Equal target labels alone do not make full logit distributions equal. Numerical
equivalence additionally requires the same declared output calibration. Conversely,
mechanisms with no defined prediction must not be relabeled as excluded or equivalent.

### Pairwise implications for the register

- **Relational vs adapted lexical-role lookup:** identical targets on every current
  cell; identical quantitative predictions if they share our native-endpoint calibration.
- **Reflexive target pointer vs final-answer copy:** identical targets here. Their
  algorithms differ, and our name-line envelope does not exhaust either algorithm.
- **Relational vs name/mention-position:** some targets differ. Only measured native
  endpoint geometry can establish sufficient separation from the numerical rival sets.
- **Paper positional vs paper lexical vs either 3B class:** no direct cross-study
  verdict. An entity-binding adapter must first define comparable interventions and outputs.
- **Mixed or nonlinear implementations vs the narrow response classes:** may remain
  unresolved even if one pure response class wins. Failure of the whole menu can reflect
  these missing alternatives; an invalid instrument is a separate outcome.

This covers the potentially misleading pairs rather than assigning numerical distances
to undefined models. The executable 3B grid continues to compare only its explicitly
defined response sets.

## The absent-target test is a separate next round

The paper's §3.4 removes the donor answer from the recipient context: answer copying
can still produce it, whereas the reflexive component cannot dereference it. A later-layer
control shows that the absent answer can be transmitted. An unresolved pointer does not
by itself specify a no-op or a replacement answer.

For our instrument, a future test therefore needs all of the following:

1. Donor-only answer names, with names and positions counterbalanced.
2. An output space that explicitly includes the absent donor answer. The current
   three-recipient-name readout would make answer copying impossible by construction.
3. A declared response prediction for unsuccessful dereferencing, or a deliberately
   weaker set-valued claim. No invented fallback answer after seeing results.
4. A positive control showing that an absent token can be transmitted through the
   tested intervention. A changed layer/operator is a separately declared instrument.
5. Competence, numerical fidelity, rival separation and fresh confirmation for that
   new design. Failure to produce an absent name alone does not identify a pointer.

No such patches are authorized by this Stage A freeze. This extension is explicitly
deferred; running it now would silently change both the question and the readout.

## Released-code audit and reproduction hazards

Reviewed revision: [`c53372c606e7cadf2494d2ac7b08e466042052df`](https://github.com/yoavgur/mixing-mechs/tree/c53372c606e7cadf2494d2ac7b08e466042052df),
dated 9 August 2026. It postdates paper v2 (28 May 2026); it is not a verified archive
of the exact paper run. An independent read-only audit agreed with these distinctions.

- [`training.py:1203–1354`](https://github.com/yoavgur/mixing-mechs/blob/c53372c606e7cadf2494d2ac7b08e466042052df/training.py#L1203)
  constructs the retrieval targets and absent-answer cases (`--messiness 4`). Verify
  separation per case: randomly sampled swaps can leave two predicted targets equal.
- [`tasks/dist.py:358`](https://github.com/yoavgur/mixing-mechs/blob/c53372c606e7cadf2494d2ac7b08e466042052df/tasks/dist.py#L358)
  normally restricts answer selection to recipient-present tokens. A pointer/copy
  reproduction needs unrestricted generation or explicit absent-token scoring. This is
  a configuration hazard, not evidence that the paper used the wrong setting.
- The main `get_dist` caller specifies the last token; the helper default and example
  notebook use several positions. Pin the executed configuration rather than treating
  every released entry point as one intervention.

| Reviewed file | SHA-256 |
|---|---|
| `training.py` | `9544d9d1ba52f2a3fbc052a3648b9c32bc387e1788c1bfee29713f369d664d1a` |
| `tasks/dist.py` | `846cf94bcc9c0c4806b21219b6448757a297ab2aa6004e3e0e8baae06177257e` |
| `example.ipynb` | `8a35150eb87210f13f03c5429239e2f928df0641b7afdac32893723d1025187b` |

## Decision before measurement

Proceed only with [Stage A](../applications/makelov-2311.17030/ROLE_BASELINE_PROTOCOL.md):
32 families, native answers and endpoint geometry, no patches. A failure would disqualify
these GPT-2 prompts as the present test of role transfer; it would not disprove role
representations, independently implicate MLP8, or retract the earlier rounds.

Even a later successful 3B confirmation would identify a bounded response class,
including observationally equivalent lexical implementations. A direct application to
the independent entity-binding study remains a separate, potentially useful replication.
