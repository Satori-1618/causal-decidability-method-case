# Round 2: the selected queries affect the patch, but none of three simple stories fits

**Executed on 192 fresh GPT-2 Small base pairs, after a separate 32-pair pilot.**
All pairs passed the technical and cross-precision checks. All three predefined
response profiles failed the required population coverage. This is a confirmed
exclusion result under the declared instrument, not a uniquely identified mechanism.

## The question, in plain language

[Round 1](CONFIRMED_CASE.md) compared where an artificial patch reads its signal.
Round 2 kept that `read_null` patch fixed and asked what selected downstream attention
queries contribute to its output effect.

Simply resetting the queries is not enough: a change might carry the effect itself,
or merely enable another change to act. We therefore added a **reverse transfer**:
insert the patched queries while leaving the original MLP patch switched off.

The targets were fixed in advance: heads 9.6, 9.9 and 10.0, at the last prompt position.
Other queries, keys, values and later computation remained free. Five scientific
conditions and two technical identities were run in both float32 and float64.

## Three predictions, one unchanged rule

Let **T** be the original patch effect, **R** the effect remaining after query reset,
and **S** the effect of transferring the changed queries alone.

| Predefined response profile | Prediction | Successful base pairs | Simultaneous interval for success rate |
|---|---|---:|---:|
| Transfer | Reset removes the effect; queries alone reproduce it: `(R,S)=(0,T)` | **0/192** | **0–2.46%** |
| Joint dependence | Neither intervention part reproduces it alone: `(R,S)=(0,0)` | **0/192** | **0–2.46%** |
| Preservation | Reset preserves the effect; queries alone do little: `(R,S)=(T,0)` | **8/192** | **1.48–9.00%** |

A success required both predictions to be within **25% of the local absolute T**, in
both reciprocal directions and both precisions. The required population coverage was
**80%**. The intervals jointly have at least 95% coverage under the frozen iid sampling
model. Every upper bound is below 80%, so all three profiles are excluded as sufficiently
broad descriptions. This does not mean each profile fails every individual case.

## One actual example

This is the first pair in the frozen confirmation manifest, not an example selected
for a favorable result:

> Then, Keith and Jason were working at the cafe. Jason decided to give a notebook to

The other prompt reverses the first two names; **Keith remains the correct recipient**.
For the first direction, the measured preference for Keith over Jason was:

| Condition | Answer-logit margin | Change from baseline |
|---|---:|---:|
| Baseline | 2.882 | — |
| Original MLP patch | 1.731 | **−1.150** |
| Original patch, queries reset | 1.953 | **−0.929** |
| Changed queries alone | 2.586 | **−0.295** |

Resetting the queries leaves much of the original shift. Transferring them alone causes
a smaller shift. Thus neither “the selected queries reproduce everything” nor “they do
nothing” describes this example. All four margins remain positive: the measurement is
a change in preference, **not a demonstrated switch of the selected answer**.

Across all 384 paired directions, the descriptive mean effects were T = −1.322,
R = −0.962 and S = −0.366 nats. These means suggest a partly shared, approximately
additive response worth testing next; they are not a newly confirmed fourth profile,
a mechanism percentage, or a substitute for the case-wise primary decision.

## Why this is a valid result rather than a failed measurement

- **192/192 pairs** met the frozen numerical-resolution rule; none was dropped.
- All three identity comparisons were **bit-exact** in both precisions on every pair.
- The largest float32/float64 discrepancy in T, R or S was **0.0000504 nat**.
- Query source, head, position, local write fidelity and hook order were audited.
- Cases were sampled iid with replacement from the fixed token-eligible proposal;
  all actual draws were distinct. Historical and development prompts were excluded.
- Confirmation cases, code hashes and rules were pushed in commit
  [`e07084a`](https://github.com/Satori-1618/causal-decidability-method-case/commit/e07084a)
  before confirmation forwards. The contract was unchanged after the pilot.

The pilot took **40.75 seconds**, confirmation **207.79 seconds** on the local CPU.
These are measured model-run times; implementation and verification took additional
work. The runtime reproduces historical Q1, but retains its documented dependency
override: TransformerLens 2.17.0 with torch 2.5.1 despite the package's newer requirement.

## What this adds to identification

The additional condition can distinguish two explanations that survive query reset
alone. On the real model, the measurements instead reject the three simple endpoints.
**The method has narrowed what can be claimed without forcing a winner.** It has not
identified a unique native path, a semantic variable, or a complete circuit; the selected
queries are only one intervention target, and cancelling or redundant paths remain possible.
Round 1's confirmed comparison remains intact.

**Limit on the incremental empirical gain:** a retrospective check using reset alone
gives 0/192 cases meeting the removal criterion and 17/192 meeting preservation. Both
already fall far short of the 80% requirement. The fifth condition reduces case-wise
preservation fits from 17 to 8 and measures what the queries do alone, but **does not
change the overall exclusion verdict in this dataset**. Its ability to distinguish
pure transfer from joint dependence is demonstrated by the synthetic countermodels;
neither explanation fits the real data here. This is not an empirical identification
victory between those two mechanisms.

The next candidate would need to predict the partial response quantitatively, then face
new separating conditions and fresh data. It must not be declared confirmed by fitting
these already inspected records.

## Reproduce without a model

From this branch's checkout, using only Python's standard library:

```bash
python3 -S applications/makelov-2311.17030/scripts/check_query_route_records.py \
  --results applications/makelov-2311.17030/results/query_route_confirmation

python3 -S examples/iterative_query_transfer.py

python3 -S applications/makelov-2311.17030/scripts/summarize_query_route.py \
  --results applications/makelov-2311.17030/results/query_route_confirmation
```

The checker validates the stored evidence; it does not rerun GPT-2. Unavailable optional
original model/source/vector files in a clean checkout are explicitly listed. The teaching
demo is synthetic and uses the same profile-analysis code as the actual experiment.
This path also [passed from a clean git export](../applications/makelov-2311.17030/results/query_route_verification/clean_checkout.json)
with the original model, source and vector files deliberately unavailable.

Evidence: [protocol](../applications/makelov-2311.17030/QUERY_ROUTE_PROTOCOL.md),
[frozen manifest](../applications/makelov-2311.17030/results/query_route_confirmation/manifest.json),
[primary summary](../applications/makelov-2311.17030/results/query_route_confirmation/summary.json),
[independent verification](../applications/makelov-2311.17030/results/query_route_verification/confirmation.json),
[descriptive calculations](../applications/makelov-2311.17030/results/query_route_verification/descriptive_summary.json).
