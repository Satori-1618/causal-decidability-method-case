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

`python scripts/verify_release.py` runs each check with `-E -s -B` (ignore environment
variables and the user site, write no bytecode), `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Unlike `-I`, these flags keep a script's own directory
importable, which several application scripts need. It refuses to run outside a full Git
clone (a source ZIP or a `--depth` clone), because the Mixing checker reads the freeze
commit from Git history. It now runs 13 checks; the five added ones are listed below.

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
- `applications/goodfire-mcqa-preflight/run.py --write DIR`: writes a new derivation into DIR
  (never into `results/` itself; refuses to replace differing files) and caches downloads.
  Its default mode (no flag; compares and writes nothing) is on the positive list below.
- Any `run_*.py` script under `applications/`: these drive model runs.

## Added in this revision (audited the same way)

| Command | Writes? | Network? | Audit note |
|---|---|---|---|
| `scripts/check_claims.py [--case NAME] [--network]` | no | only with `--network` | runs only the allow-listed scripts in this table and above; stops with a clear message before a check that needs Git history when the checkout is a ZIP or a `--depth` clone |
| `scripts/describe_claim_inputs.py makelov-table1` | no | no | standard library; reads `per_example.jsonl` |
| `scripts/describe_claim_inputs.py mixing-concentration` | no | no | imports the frozen Mixing checker's `measure()` and `distribution()` (standard library) |
| `scripts/describe_claim_inputs.py tracr-confirmation` | only into temporary copies | no | runs the portable verifier on a copy of `confirmation_001` and the frozen `scripts/verify.py` on a copy of the development run whose summary hash the freeze records; neither builds a Tracr model |
| `scripts/describe_claim_inputs.py saphra-native` | only into a temporary directory | no | unpacks the hash-indexed archive and runs `verify_confirmation.audit` there |
| `scripts/describe_claim_inputs.py saphra-averaged004` | only into a temporary directory | no | checks every archived file against `raw_index.json`, then runs the SciPy recount `verify_statistics.py` |
| `scripts/describe_claim_inputs.py saphra-retro` | no | no | calls the retrospective analyzer's `encoded_results()` on the vendored upstream tables |
| `applications/li-saphra-2507.06445/analyze.py --check` | no | no | rebuilds the stored retrospective outputs and compares them |
| `applications/li-saphra-2507.06445/verify_independently.py` | no | no | separate recount from the released tables |
| `applications/makelov-2311.17030/scripts/describe_post_hoc.py` | no | no | standard library; adds the proportional-rule comparison |
| `applications/makelov-2311.17030/scripts/summarize_query_route.py --results …/query_route_confirmation` | no | no | standard library |
| `applications/goodfire-mcqa-preflight/run.py` (default check mode) | no | yes: pinned GitHub raw files, hash-checked, held in memory | standard library; executes no upstream code |
