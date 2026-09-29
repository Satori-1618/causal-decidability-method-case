# Intervention selection: development results

**Development only. No reserved structural family was evaluated.**

Independent circuit instances: 512. Primary: four anchors + three additional cells; 8 samples per cell.

| Policy | Correct class (in set) | Truth excluded | False single class | OOS rejected | Cost |
|---|---:|---:|---:|---:|---:|
| maximin | 398/436 | 1/436 | 0 | 70/76 | 56.0 |
| mean_pairwise | 391/436 | 1/436 | 0 | 70/76 | 56.0 |
| balanced_split | 332/436 | 2/436 | 0 | 64/76 | 56.0 |
| random | 263/436 | 0/436 | 0 | 68/76 | 56.0 |
| full_only | 0/436 | 0/436 | 0 | 64/76 | 56.0 |
| full_menu | 399/436 | 4/436 | 0 | 71/76 | 128.0 |

Full-menu is a higher-cost reference. Full-only is a negative control.

| Maximin minus | Paired advantage | Simultaneous conservative 95% interval | Wins / losses / ties |
|---|---:|---:|---:|
| mean_pairwise | 1.61 pp | [-13.21, 16.42] pp | 7 / 0 / 429 |
| balanced_split | 15.14 pp | [0.32, 29.96] pp | 67 / 1 / 368 |
| random | 30.96 pp | [16.14, 45.78] pp | 136 / 1 / 299 |

## Interpretation

The primary comparator is mean_pairwise, not full-only or random. An advantage over the latter does not establish an advantage over established discrimination design. All policies use the same production compatible-set classifier.

Truth-exclusion upper bounds (simultaneous over six reported policies), per-family, amplitude/noise/ratio strata, and secondary budgets are in summary.json. Graph outputs are deterministic; Gaussian measurement noise is added deliberately and its variance is known. These are generic scores, not LLM nats. Parameterized two-unit circuits and their known prediction tables are a restricted testbed.

The structural holdout remains unopened. A fresh evaluation requires a reviewed, powered confirmation contract. This development run cannot be promoted by relabeling fresh seeds.

Development comparison of explicit selection objectives on declared known circuits. No confirmation, novelty, native LLM mechanism, or general superiority established.
