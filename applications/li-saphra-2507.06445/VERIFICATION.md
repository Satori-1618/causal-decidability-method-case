# Completion and claim audit

Status: **retrospective analysis completed locally**. No new neural run, remote
publication or author contact. No new mechanism-identification claim.

| Required outcome | Current evidence |
|---|---|
| One concrete published ablation finding | Paper §4.3 / Appendix J, Dyck-1 checkpoint 5; source revision in `SOURCE_LOCK.json` |
| Two operational causal explanations | `IDENTIFICATION.md`: sign of removal relative to a fixed zero-output reference |
| Predictions for the same existing interventions | Both admit the recorded native/uniform/mean observations; generated exact attention witnesses demonstrate this rather than merely asserting it |
| Full public-data comparison | `results/heads.csv`: all 1,620 heads; primary 1,350 heads/180 multilayer models, including 42 sign-matching heads |
| Actual method reused | `countermodels.py` invokes unchanged `examples/causal_preflight.py`; one group on existing conditions, two with zero |
| Measurability and resolution distinguished | Existing accuracy count grid is 0.1 pp; numerical sensitivity/population uncertainty not certified; absent zero condition gives structural nonidentification before any power calculation |
| Claim narrowed with measured evidence | Same 40/42 heads improve >1 pp under both replacements; no empirical winner between the two zero-reference rivals |
| Concrete next intervention | `IDENTIFICATION.md` §4: zero only selected head's `AV` slice before output projection, preserve bias/other heads, all query positions; store ID/OOD per-case outcomes and qualify fidelity/numerics |
| Accessible account and visual | `README.md`, real example and figure; synthetic zero values visibly distinguished from measurements |
| Reproducible result | Offline standard-library reconstruction, separately implemented read-only verification, focused tests, clean-export verification |

## Verification levels

`analyze.py --check` validates 12 source hashes, complete head joins, redundant
accuracy/delta fields, baseline consistency, head classification, witness
calculations, the existing preflight, and byte-identical result reconstruction.

`verify_independently.py` does not import either producer. It independently
recomputes primary head counts and changes from upstream CSVs, all serialized
witness predictions and margins, the actual mean attention pattern, and
normalized-mixture invariance over a rational simplex grid. Here "independent"
means **a separate implementation**, not a blinded analyst or external reviewer.

The 13 application tests cover invalid accuracy counts, strict reporting-band
boundaries, head-type classification, complete cohorts, model weighting, exact
countermodels across different effect patterns, normalized-mixture invariance,
and tie preservation/separation in the delivered method code. They do not certify
the original model implementation or the scientific adequacy of every rival.

## Adversarial interpretation check

- **The authors already use mean and single-head controls.** Included; the
  descriptive robustness result supports their finding rather than overturning it.
- **Their claim can be about a pattern relative to a replacement.** That claim
  survives. Our additional question concerns the whole head output relative to
  zero and must not be attributed to the authors as an unqualified assertion.
- **Zero is not a uniquely neutral reference.** Correct. The value-offset
  construction exposes this dependence; a future zero result would be local to
  the fixed checkpoint and declared operator.
- **The witnesses were chosen to match the example.** Yes, deliberately and
  retrospectively. They prove nonidentification from these observations, not
  successful prediction of held-out neural behavior or general method reliability.
- **Known weights could determine the missing counterfactual.** Yes, by computing
  it. Our nonidentification claim concerns the released outcome tables and
  normalized-attention observations, not impossibility given full executable code.
- **Robustness is not an interval on a population effect.** Correct. Counts and
  averages are descriptive, and heads and repeated seeds are not treated as IID.
- **Does this identify the mechanism?** No. It identifies the specific missing
  comparison needed for the declared operational question.

The scope has not been enlarged to question formation, new checkpoints, or a
new model run. The result is a completed application of structural preflight and
public-evidence reconstruction, with measurement resolution for the proposed
additional neural condition explicitly open.
