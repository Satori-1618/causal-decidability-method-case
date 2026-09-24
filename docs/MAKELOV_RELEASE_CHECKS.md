# Makelov application release checks — 24 September 2026

## A. Executive summary

The stored-evidence and CPU software-test paths work with the documented dependency
split. Validation used a fresh virtual environment and a source export without local
model inputs, followed by a separate full clone for 3B's Git-history check. No pretrained
model was loaded and no scientific outcome was changed. These checks concern usability
and reproducibility of saved evidence, not a new validation of mechanism identification.

## B. Verified facts

Base branch: `codex/plan3-person-position-role`, starting at
`91214c5f1609ef5ee18cf89f8725d6a46124a85b`, with this maintenance change applied.
Environment: macOS 15.7.3 arm64, Python 3.10.10, pytest 9.1.1, NumPy 2.2.6.
The optional tensor phase installed PyTorch 2.14.0 in the same isolated environment,
after the no-PyTorch phase had finished. Packages were installed from the package
index; inference assets were neither downloaded nor accessed.

| Executed check | Outcome |
|---|---|
| Fresh `pip install -e '.[test]'` | Verified; no PyTorch present. |
| `python -m pytest -q -ra`, clean source export | **262 passed, 9 skipped, 306 subtests passed**, 18.71 s. |
| Add `.[test-hooks]`, rerun the same suite/export | **278 passed, 321 subtests passed; no skips**, 25.72 s. |
| Q1 records-only command | Verified: B better in 64/64; exact sign p = 1.0842e-19. |
| Round 2 records-only command | Verified: all three declared coverage profiles excluded. |
| 3A records-only command | Verified: 301/512 and 0/512 profile counts; both excluded. |
| 3A `--full-bootstrap` | Verified: both frozen bootstrap calculations reproduced with NumPy. |
| 3B `--check-only`, full clone | Verified: competence and geometry STOP, numerical pass; no patches. |
| Frozen-file inventory | Verified: all 62 archived files unchanged. |

The nine no-PyTorch skips comprise four whole-module skips (two hook modules and two
historical runner-import modules), four native-runner test cases and one dtype-reference
test. They are not nine scientific tests that passed; the full optional run executes
the additional cases within those modules. A previously silent optional dtype check
is now a separately visible test.

Use [the verification guide](../applications/makelov-2311.17030/VERIFICATION.md) for
the exact commands. All tests were run with model access disabled; synthetic tensors
and fake models are used by hook tests. The records-only checks use `-I -S`, except
the additional NumPy bootstrap recomputation.

## C. Inferences

These checks support the documented paths on this tested environment. They do not
establish compatibility with every allowed dependency version, full pretrained-model
reproduction, or general scientific calibration. The updated hosted Linux/Python 3.11
CI is configured to exercise the same split; its execution is separate from this local
report and was not observed when this report was written.

## D. Failures found and disposition

- The original `test` extra omitted NumPy, and tensor tests imported PyTorch without
  an optional-runtime boundary. NumPy is now declared and optional skips are explicit.
- The first fresh analysis run still failed collection: two frozen runners indirectly
  import PyTorch via the historical read-source module. Their tests now declare that
  requirement and run in the tensor job. The frozen runner code remains unchanged.
- The 3B checker failed in a Git-free source export because it verifies source files
  at the execution commit with `git show`. It passed in a full clone. This provenance
  requirement is retained, documented, and met by CI using full history. ZIP/shallow
  checkouts without that commit are not a supported path for the complete 3B check.

## E. Operational status by check

| Check | Status | Scope |
|---|---|---|
| Dependency installation | Verified | Both extras installed in a fresh environment. |
| CPU default tests | Verified | Analysis and optional tensor paths above. |
| Minimal stored-evidence run | Verified | All four rounds; 3B requires Git history. |
| Paths and I/O | Verified | Source export/full clone; no local model inputs required. |
| Configuration loading | Verified | Editable package and default pytest configuration. |
| Logging and errors | Verified | Optional skips shown; failed provenance check detected. |
| Context or memory logic | Untested | Not part of this application. |
| GPU execution | Untested | No accelerator or pretrained-model run. |
| Full experimental replay | Untested | Distinct from records-only verification. |

## F. Next use and preserved scope

Start with the [four-round case study](../applications/makelov-2311.17030/CASE_STUDY.md),
then run the records-only commands. Install analysis or tensor extras only for their
respective checks. The original application README remains a hash-frozen historical
document; the root README is deliberately unchanged.

All 124 tracked paths covering result directories, historical application source and
scripts, and the root README were checked against their pre-change hashes and were
unchanged. The `main` ref remains `fbd0b9683487aa756791eaac6f9f9761542057d4`; its README
blob remains `d038255339ca1d7654588cdd2d856729d358c827`. No model result, decision threshold,
preregistration or research claim was refitted to make these release checks pass.
