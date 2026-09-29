# How interventions narrow causal explanations

**Retrospective replay of one measured synthetic development circuit.** No new model inference and no prospective adaptive selection claim.

The first linear `context_switch` case in the archived generation order was selected after the run for illustration. This is not a representative sample or a new confirmation.

| Cumulative evidence | Displayed score ± comparison radius | Compatible classes |
|---|---:|---|
| Native/full anchors in both contexts | 0.203736 ± 0.046763 | channel_a / channel_a_alias; channel_b; balanced; joint; context_switch; context_switch_inverse |
| Add A-only in context 0 | 0.200112 ± 0.047731 | channel_a / channel_a_alias; context_switch |
| Add A-only in context 1 | -0.002327 ± 0.040024 | context_switch |

Seven implementation labels form six full-menu classes: `channel_a_alias` remains equivalent to `channel_a` under every allowed intervention. The final class agrees with the known executed graph. This does not assert unique implementation or natural LLM usage.

The classifier consumes all previous cells at each stage; candidate equivalence is never redefined using only the selected cells. Both native/full context anchors are included, although the figure displays the context-0 full patch.

## Measurement and uncertainty

Each score is the mean of the first eight stored artificial noisy measurements of a deterministic float32 circuit. The known Gaussian model gives Bonferroni radii over the complete 16-cell menu at alpha=0.005; the calibrated numerical allowance is added. These are generic scores, not nats and not posterior probabilities of mechanism truth. The retrospective selection of an illustrative case is not a fresh population test.

The displayed sequence was constructed after the run. The implemented selector chooses a batch of interventions before outcomes, not the next cell after each retained set. Other B-only cells are already in the archive: they cannot be described as unseen validation.

## Real-model evidence is separate

[Q1](CONFIRMED_CASE.md): null-read predicts better in 64/64 fresh GPT-2 pairs. No adequacy tolerance was declared. [Round 2](ROUND2_RESULT.md) and [Round 3A](ROUND3A_CONFIRMATION.md) exclude different narrow profiles; they are not successive subsets of this six-class space. [Round 3B](ROUND3B_STAGE_A_RESULT.md) stopped before patching.

## Reproduce and inspect

`python3 docs/make_iterative_narrowing_onepager.py` requires reportlab and pypdf. It reads and hash-checks existing data; it loads no model.

[PDF](../output/pdf/ITERATIVE_CAUSAL_NARROWING.pdf) | [Exact values, cells and input hashes](ITERATIVE_CAUSAL_NARROWING.json)

**Connection to Geiger:** causal abstraction relates a high-level explanation to an underlying computation through corresponding interventions. This demonstration concerns which measured interventions distinguish competing candidate computations; the present six-class example is not itself a validated semantic abstraction of an LLM. [Geiger et al.](https://arxiv.org/abs/2301.04709v4)
