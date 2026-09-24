# Verify this application

Run all commands below from the repository root. Start with the
[case study](CASE_STUDY.md) for the meaning of the results. This guide distinguishes
rechecking stored evidence, testing software, and executing a pretrained model.

## 1. Check stored evidence: no installation

Use Python 3.9 or later. These commands use the standard library; `-I -S`
disables user/site packages and Python environment injection. They do not download
weights, load a model, or replace results.

**The 3B checker additionally requires Git and the repository history containing
freeze commit `24cc60e`.** Use a full clone of this branch; a GitHub ZIP or shallow
checkout is insufficient for that provenance check. If needed, fetch history once
with `git fetch --unshallow origin` before going offline. The other four commands
below also work in a source export without Git history. CI uses `fetch-depth: 0`
for the stored-evidence job.

```bash
python3 -I -S examples/confirmed_read_source.py
python3 -I -S applications/makelov-2311.17030/scripts/check_query_route_records.py \
  --results applications/makelov-2311.17030/results/query_route_confirmation
python3 -I -S applications/makelov-2311.17030/scripts/check_donor_factor_512_records.py \
  --results applications/makelov-2311.17030/results/donor_factor_confirmation_512
python3 -I -S applications/makelov-2311.17030/scripts/check_role_baseline_records.py \
  applications/makelov-2311.17030/results/role_baseline_development --check-only
python3 -I -S scripts/check_frozen_files.py
```

| Check | Expected scientific result | Verification limit |
|---|---|---|
| Q1 | B better in 64/64 pairs; exact sign p ≈ 1.0842e-19 | Verifies saved controls and four-pair precision reference, not all original tensors. |
| Round 2 | All three 80%-coverage profiles excluded on 192 pairs | Recomputes primary decisions from stored measurements, not hook execution. |
| Round 3A | 301/512 position-only and 0/512 identity-only; both excluded | Standard-library path does not regenerate the seeded bootstrap; see section 2. |
| Round 3B | STOP for competence and geometry; 32 families; no patches | Recomputes point gates; does not regenerate descriptive confidence intervals. |
| Frozen inventory | All 62 archived files unchanged | Hash identity does not prove execution or historical timing. |

A successful 3B checker verifies a recorded **STOP**, not a successful role-transfer
experiment. Repeated conditions are not additional independent cases. Missing optional
upstream tensor inputs are reported by the relevant checkers; stored evidence can be
checked without reconstructing those tensors. See also [the detailed Q1 scope](RECORDS_ONLY.md).

## 2. Analysis and release tests: pytest and NumPy

Use a fresh virtual environment. Package installation may access the package index;
the subsequent tests and record checks run without model or upstream downloads.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest -q -ra
```

This installs pytest and NumPy, not PyTorch or TransformerLens. The default suite
checks the method, analysis code, frozen records and release
integrity. Without PyTorch, tensor-dependent checks are visibly **skipped**; `-ra`
lists the reasons. Two tests of frozen runner safeguards also require PyTorch via
their historical imports and are explicitly skipped. A green analysis run does not
mean the hook or those runner tests ran.

For 3A, also reproduce both frozen bootstrap seeds from the stored records:

```bash
.venv/bin/python applications/makelov-2311.17030/scripts/check_donor_factor_512_records.py \
  --results applications/makelov-2311.17030/results/donor_factor_confirmation_512 \
  --full-bootstrap
```

This is fresh statistical computation on old measurements, not new model inference.

## 3. Tensor and hook tests: optional PyTorch

```bash
.venv/bin/python -m pip install -e '.[test-hooks]'
.venv/bin/python -c 'import torch; print(torch.__version__)'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest -q -ra
```

This reruns the release suite with tensor-dependent tests enabled. Those tests use
small synthetic tensors and fake model interfaces on CPU. They test indexing,
insertion, isolation, replay guards and dtype references. They do not load GPT-2,
download weights, or reproduce the scientific model runs. CI installs the CPU-only
PyTorch wheel on Linux; a platform-appropriate wheel is used by the local command.

The [workflow](../../.github/workflows/offline-method-case.yml) separates stored evidence,
analysis tests without PyTorch, and tests with PyTorch. Dependency installation uses
the network; model access is disabled during the checks.

## 4. Full model reproduction: a separate operation

The [historical reproduction instructions](README.md#reproduce) require additional
pinned dependencies, model weights and upstream inputs. They can download assets and
run inference. They are not part of the preceding paths or CI. The archived full
application test suite also has those additional requirements; it is not claimed to
pass merely because the release suite passes.

Original preregistrations, scorers, raw results and the application's historical README
remain unchanged. The repository-root README is also preserved in this maintenance
change; use this guide and the case study for the current branch-wide execution status.
