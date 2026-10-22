# Model-free verification commands (U05)

Internal working document. Each command below was checked by reading its call chain.
None of them initializes a model or tokenizer, downloads weights, or runs a forward pass.
Where a command can write files, the table says so.

**Environment:** an isolated virtual environment with `numpy>=1.24` and `scipy==1.15.3`
(the `verify` extra), and this repository installed with `pip install --no-deps -e .`.
PyTorch, JAX and Transformers are deliberately **absent**. A passing run is therefore
also evidence that no code path imported them.

**Working directory:** repository root of a fresh clone or export, unless stated otherwise.

## Release verifier: eight existing checks

`python scripts/verify_release.py` runs each check with `-I -B`, `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.

| # | Check | Command | Writes? | Audit note |
|---|---|---|---|---|
| 1 | archived Makelov files unchanged | `scripts/check_frozen_files.py` | no | hashes 62 files listed in `release/FROZEN_FILES.json` |
| 2 | Makelov Q1 | `examples/confirmed_read_source.py` | no | `causal_decidability.evaluate` only |
| 3 | Makelov round 2 | `applications/makelov-2311.17030/scripts/check_query_route_records.py --results …/query_route_confirmation` | no | imports `query_route_analysis` (NumPy only) |
| 4 | Makelov round 3A | `…/check_donor_factor_512_records.py --results …/donor_factor_confirmation_512` | only with `--output` (not used) | `--model-snapshot`, `--use-hf-cache` and `--require-original-inputs` are **not** passed |
| 5 | Makelov round 3B stop | `…/check_role_baseline_records.py …/role_baseline_development --check-only` | not with `--check-only` | without the flag it would create `verification.json` |
| 6 | Mixing Mechanisms confirmation | `applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py --results …/confirmation` | no | reads the stored zip and records |
| 7 | CausaLab MCQA prediction groups | `examples/causal_preflight.py --config applications/goodfire-mcqa-preflight/results/combined_predictions.json --structure-only` | no | design calculation only |
| 8 | Tracr records | `scripts/check_tracr_records_portable.py <temp copy> --freeze …/CONFIRMATION_FREEZE.json` | writes `verification.json` **into the temporary copy only** | the frozen verifier imports `tracr_demo.experiment`; that module imports `tracr`/`jax` only inside model-building functions, which are not called |

Labels 6 and 7 are renamed in this revision (plan 1.1): the old label "Goodfire" for
check 6 referred to the Mixing Mechanisms case.

## Commands not on the positive list

- `python -m pytest` over the whole repository: some tests construct small tensor models
  or need optional libraries. It is not run as part of this revision.
- `applications/goodfire-mcqa-preflight/run.py`: it **overwrites** versioned files in
  `results/`. Run it only on a copy, with outputs compared afterwards.
- Any `run_*.py` script under `applications/`: these drive model runs.

New checkers added in this revision are listed in `verify_release.py` and below, once
they have been audited in the same way.
