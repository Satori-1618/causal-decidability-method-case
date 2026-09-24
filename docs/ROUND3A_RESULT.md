# Round 3A: position matters, but position alone does not describe the patch

**Development result on 32 GPT-2 Small families. No confirmation run was authorized.**
The fixed patch responds strongly when the donor answer changes mention position.
However, changing the name assignment at the same answer position also matters in
many cases. Neither of the two simple invariance profiles meets the declared coverage
requirement in this development sample.

## What changed, and what stayed fixed?

[Round 1](CONFIRMED_CASE.md) compared the patch's read source; [round 2](ROUND2_RESULT.md)
tested selected downstream queries. Here the patch itself stayed fixed: read a scalar
from the donor/recipient activation difference along the published vector's null
component, then write along the full vector, at MLP8's last prompt position.

Each family crosses two donor properties: **which name is the correct answer** and
**whether that name was mentioned first or second**. The recipient stays fixed within
each four-donor comparison. Repeat with the recipient names in reversed order, as part
of the same family. The measurement is the change in preference for the recipient's
correct name over the other name, in logit-margin units (nats).

These are interventions on the source of an artificial patch. The name manipulation
changes the whole name assignment, including the repeated giver. It does not isolate a
latent variable called “person identity.”

## One actual family

The first family in the frozen development manifest uses this exact recipient prompt:

> Then, Ray and Bailey were working at the market. Bailey decided to give a coins to

Ray is the correct completion. The second recipient reverses the first two names,
while Bailey remains the giver and Ray the correct answer. The source wording,
including “a coins,” was retained unchanged.

| Donor's first two names | Giver | Correct donor answer; position | Patch effect: Ray-first recipient | Patch effect: Ray-second recipient |
|---|---|---|---:|---:|
| Ray and Bailey | Bailey | Ray; first | 0.000 | −1.625 |
| Bailey and Ray | Bailey | Ray; second | −1.075 | 0.000 |
| Bailey and Ray | Ray | Bailey; first | +0.039 | −1.687 |
| Ray and Bailey | Ray | Bailey; second | −1.082 | +0.009 |

Numbers are float64 reference measurements. The zero cells are self-patches, zero by
construction. For this family, moving the donor answer to the other mention position
reduces preference for Ray, whether the donor answer is Ray or Bailey. All margins
remain positive: this example shows a preference change, not an answer flip.

**That simple position account does not generalize sufficiently.** In the very next
frozen family (Mia/Aaron), changing name assignment at fixed position changes the margin
by +0.105/+0.194 in one recipient order, but −0.760/−0.325 in the other. These are the
two position-matched comparisons in each panel. The declared tolerance is ±0.25.

## What the full development sample says

A family meets an invariance profile only if both required comparisons are within
±0.25 nat in **both** recipient orders and **both** precisions. The required population
coverage was 80%, fixed before the run.

| Profile | Meaning | Families meeting it | Simultaneous coverage interval | Development decision |
|---|---|---:|---:|---|
| Position-only | Name assignment can change without materially changing the effect, at fixed answer position | 14/32 | 22.56–66.73% | Excluded at the declared 80% coverage |
| Identity-only | Answer position can change without materially changing the effect, at fixed correct name | 0/32 | 0–14.67% | Excluded at the declared 80% coverage |

These Clopper–Pearson intervals jointly have at least 97.5% coverage under the fixed
iid-family sampling assumptions. The following mean intervals use the other 2.5%
error allocation: a whole-family percentile bootstrap over three contrasts, with
nominal rather than guaranteed coverage. The practical mean boundary is ±0.10 nat.

| Mean contrast | Estimate | Primary interval, float64 | Development decision |
|---|---:|---:|---|
| Name-assignment change, I | −0.0206 | [−0.0508, +0.0031] | Signed mean practically equivalent to zero |
| Position change, P | −1.2546 | [−1.4431, −1.0881] | Negative, practically relevant |
| Interaction, J | +0.0386 | [−0.0729, +0.1484] | Unresolved |

Every decision agrees in float32, float64 and both bootstrap seeds. **Small mean I does
not mean name independence:** descriptively, the two recipient-order I effects have
opposite signs in all 32 families. Individual name-assignment effects reach 0.781 nat.
Position effects average −1.018 and −1.491 in the two orders; they are present in both.
All cells, simple effects and panel contrasts are preserved in the records.

## Why there is no confirmation yet

The frozen rule required at least 80% planning power for **each** mean contrast at a
true ±0.20-nat effect against the ±0.10 boundary, with development standard deviations
inflated by 1.5. It considered n = 128, 192, 256 and 384, plus a separate profile-power
requirement. Even at n = 384, the initial normal-reference screen gave only 78.67% and
78.14% detection for the two signs of P; neither Monte Carlo interval reached 80%.
No candidate reached the subsequent bootstrap planning checks.

The recorded outcome is therefore **STOP: precision-limited under this planning rule**.
It concerns sensitivity to a hypothetical small effect, not failure to measure the much
larger observed position effect. It is not a universal sample-size limit. Increasing n,
changing the target or dropping a contrast would require a new protocol; none was done.
The development findings above have not been replicated on a fresh confirmation sample.

## What this adds—and what remains open

Earlier rounds already established positional sensitivity. The new contribution is
checking whether position alone is a sufficiently accurate description: the development
data reject that coarse invariance requirement, even though the signed average name
effect is small. This illustrates why comparing means alone can miss dependence that
changes sign across contexts.

It does **not** show that a person, a role or Kaplanian character is transferred.
For each fixed recipient, every intervention is still one scalar times the same vector;
different donor effects can arise from scalar magnitude and the recipient's response.
Selecting the answer position and inhibiting the giver's position remain indistinguishable
in this two-name task. Round 3B has not run.

The population is narrow: one model, one patch site, one template and one prefix, with
sampled names, objects and places. No baseline-correctness filtering was used. Descriptively,
all recipient prompts favor the correct name over the other name; two donor prompts do
not, and remain included. This is two-name scoring, not an audit of the full vocabulary.

## Checks, provenance and reproduction

- 32/32 complete families passed numerical resolution; no cases were removed.
- Self/zero and untouched-row identities were exact. All insertion and hook checks passed.
- Maximum fp32/fp64 discrepancy over the declared readouts: **0.0000606 nat**, below 0.01.
- Float64 promotes the same processed weights; it is a reference, not exact arithmetic.
- All four prompts per family were checked against 4,640 historical excluded prompts.
- Code and cases were pushed in [`17ee572`](https://github.com/Satori-1618/causal-decidability-method-case/commit/17ee572)
  before development inference. The model run took **34.73 seconds on CPU**; planning
  took 6.45 seconds. Implementation and verification took additional time.
- The complete offline suite passed: **186 tests and 160 subtests**. The full records
  verification reproduced both bootstrap seeds; all 62 archived files stayed unchanged.

From this branch's checkout, verify the saved evidence with the standard library:

```bash
python3 -S applications/makelov-2311.17030/scripts/check_donor_factor_records.py \
  --results applications/makelov-2311.17030/results/donor_factor_development
```

For full reproduction of both seeded bootstrap calculations, run the same command
without `-S` and add `--full-bootstrap` in an environment with NumPy. Neither command
loads a model. The independent checker reconstructs margins, profile counts, exact
binomial intervals and mean contrasts; full-bootstrap mode additionally reruns the frozen
analyzer. Missing optional original inputs are listed, not silently verified. Stored
hashes and audits do not recover activation tensors or independently prove model execution.

**Evidence:** [execution protocol](../applications/makelov-2311.17030/DONOR_FACTOR_PROTOCOL.md),
[frozen manifest](../applications/makelov-2311.17030/results/donor_factor_development/manifest.json),
[raw paired records](../applications/makelov-2311.17030/results/donor_factor_development/records.jsonl),
[complete summary](../applications/makelov-2311.17030/results/donor_factor_development/summary.json),
[verification report](../applications/makelov-2311.17030/results/donor_factor_development/verification.json),
[planning result](../applications/makelov-2311.17030/results/donor_factor_planning/planning.json).
