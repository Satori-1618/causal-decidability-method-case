# Why context motivates competing explanations

A word can contribute a stable rule while its use depends on the context. That suggests
a useful mechanistic question: when a patch changes an answer, did it transfer a rule,
a contextual input, their combination, or the already computed answer?

This is motivation for constructing rivals, not evidence that a network stores those
parts separately. In [Kaplan's account](https://fitelson.org/proseminar/kaplan.pdf),
*character* is a rule from contexts to contents, not a constant subvector. For an
operational example, [Kennedy's treatment of gradable adjectives](https://semantics.uchicago.edu/kennedy/docs/vaguenessandgrammar-final.pdf)
helps separate a height scale from a context-dependent standard for “tall”.

Consider the toy rule `tall = 1[height > standard]`. One candidate says the patch
transfers the donor's standard; another says it transfers the donor's completed answer.
Let the donor have height 190 and standard 180, hence answer Yes:

| Recipient | Standard-transfer prediction | Donor-answer prediction |
|---|---|---|
| Height 190, standard 200 | Yes: 190 > 180 | Yes |
| Height 165, standard 150 | No: 165 ≤ 180 | Yes |

The first patch changes an answer under both explanations and cannot separate them.
The second recipient produces different predictions. Those predictions follow from the
explicit rules and numbers; the words “standard” and “answer” alone would not determine
them. This is a teaching example, not a completed LLM experiment.

Three boundaries matter when extending it:

- A changed standard need not change the answer; the measured height and readout matter.
- A model's answer to a recombined natural prompt is not automatically the counterfactual
  outcome of an internal patch. That correspondence must be justified and tested.
- An invertible relation between two representations is not sufficient on its own for
  intervention equivalence: the mechanisms and intervention mappings must correspond too.

The philosophical contribution is to help name plausible competing explanations before
testing them. The methodological claim is broader: once specified, their predictions can
be compared even in entirely formal systems with no linguistic ambiguity. Neither a
language theory nor this check guarantees that the declared candidate set is exhaustive.

Return to the [method](../README.md) or the larger [worked example](WORKED_EXAMPLE.md).
