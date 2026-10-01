# Research release 0.3.0 — 1 October 2026

This release brings the method, the small preflight tool and the existing Makelov,
Goodfire and Tracr applications into one checkout. It adds no pretrained-model
experiment and changes no frozen scientific rule, source binding or raw measurement.
Earlier release notes remain available in Git history.

## Start here

- [README](README.md): the question, short commands and supported claims.
- [Preflight](docs/METHOD_PREFLIGHT.md): build a rival table, check separation,
  then assess measurement resolution under explicit assumptions.
- [Applications](applications/README.md): choose a worked case.
- [Evidence map](docs/EVIDENCE_MAP.md): results, canonical artifacts and limits.

## What changed

- Integrated the previously separate Makelov, Goodfire and Tracr branches, retaining
  their recorded histories, protocols, development outcomes and confirmation records.
- Made the root entry points shorter and distinguished structural audits, controlled
  validation, comparative confirmation and qualification stops.
- Rejected malformed prediction rows: each row must be a list, not a string that
  accidentally becomes one prediction per character.
- Replaced the teaching example's internal "no effect" interpretation with preservation
  of the recipient's observed answer. Donor and recipient baselines must both be checked.
- Documented Goodfire's adapted answer-token readout, the narrow scope of our `W_T`
  rival, and its locally recorded freeze rather than an external preregistration.
- Made Goodfire development-reproduction tests portable across the checked runtimes.
  Two named float64 audit diagnostics allow absolute drift up to `1e-14`; named
  planning-power fields allow `1e-12` after measured Python-version differences
  below `2.3e-13`. Decisions, counts, thresholds, hashes and all other fields remain
  exact. These test comparisons do not relax experimental gates or alter results.
- Added one command for the bundled records checks; CI runs it with full Git history.
- Kept Tracr's historical exact verifier intact and added a release adapter for
  cross-platform tail-probability digits. The adapter independently verifies the
  all-success confidence boundary with exact rational arithmetic. Counts, bounds
  and the decision must match exactly; only the nonbinding reported tail has a
  separately checked reproduction allowance.

## Check this release

The small teaching examples and Q1 check use only Python's standard library:

```bash
python3 examples/causal_preflight.py
python3 examples/causal_preflight.py --config examples/data/causal_preflight_choice_example.json --structure-only
python3 examples/confirmed_read_source.py
```

For all bundled result checks, use a full Git clone and a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,verify]'
python scripts/verify_release.py
python -m pytest -q -ra
```

The verification command uses NumPy and Tracr's frozen SciPy 1.15.3. It downloads nothing, loads no model,
and checks saved evidence rather than replaying the original interventions. The
historical Tracr verifier writes a report, so the wrapper gives it a temporary copy
of the result directory. Optional tensor tests and upstream-source tests may skip;
these skips are displayed. Full Tracr/model replay has separate application dependencies.
See [release checks](docs/RELEASE_CHECKS.md) for the environments actually exercised.

## Boundaries of this release

- Results distinguish specified alternatives on particular tasks and interventions;
  they do not establish an exhaustive candidate set or a unique native mechanism.
- The historical resolution forecast scored **88.47% and 88.20%**, below its declared
  **90%** target. The full synthetic benchmark is not bundled here; its failure and
  other limitations are retained in [validation](docs/validation.md). The current
  small planning screen is conditional, not a newly validated power calculator.
- Whether the prompt improves AI-assisted rival generation is **untested**. No new
  AI-rival pilot is presented as evidence for this release.
- Local commit order and run hashes bind recorded protocols to runs; they do not by
  themselves establish independent timestamps or rule out unrecorded earlier work.
  Goodfire's final freeze has no separate human-review record in this repository.
- Beckmann is a separate project and is not a bundled or reproducible case here.
  No claim of measured cost savings depends on that unpublished application.

The original 62-file Makelov inventory remains checked by
`python3 scripts/check_frozen_files.py`. Application-specific verifiers additionally
check their own contracts, measurements and source hashes. Frozen research artifacts
are preserved; a later release of local history does not retroactively preregister it.
