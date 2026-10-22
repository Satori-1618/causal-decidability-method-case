# Verification and reproduction: screen-transfer 003

**Saved-record audits pass. No neural inference was repeated.** The sole run
used public release `4dd5e5c2b81cfb0881bf09d03ab666848f7132ea`, after review of
`23f7f7f6c6e4da8de391b131d7cb7c770ca3a5af`. Its six heads all completed, with no
shortfall, technical failure, replacement or retry. The [result](RESULT.md)
reports every head and the unfavorable fixed-pool cost comparison.

The record audit reconstructs normalization, historical exclusions, first-64
selection, donor bindings, target-node arithmetic, numerical gates, chronology,
24-bound decisions and the attempted/completed forward ledger. It checks 61
bound source files plus all raw output hashes. It does not independently replay
the neural network or reconstruct unarchived whole-layer tensors.

A separate agent recounted raw margins without importing the production
analysis, using SciPy 1.15.3 beta quantiles for the Clopper–Pearson bounds.
Its [script](reporting/independent_recount.py) and
[result](results/independent_recount.json) agree on all counts and decisions;
the intervals agree with the production calculation to floating-point precision.
This is separate-agent verification, not named external human peer review.

## Reproduce without a model

Run these commands from this directory after cloning the repository. The raw
archive contains all 50 unmodified run files (about 22 MB unpacked, 5.5 MB
compressed). Existing local raw files need not be unpacked again.

```bash
python3 -B -S - <<'PY'
from pathlib import Path
import hashlib, json, tarfile
base = Path('results')
index = json.loads((base/'raw_index.json').read_text())
archive = base/index['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == index['archive_sha256']
if not (base/'run_001').exists():
    with tarfile.open(archive) as source:
        members = source.getmembers()
        assert {m.name for m in members} == set(index['files'])
        assert all(m.isfile() and not Path(m.name).is_absolute()
                   and '..' not in Path(m.name).parts for m in members)
        source.extractall(base)
for name, entry in index['files'].items():
    data = (base/name).read_bytes()
    assert len(data) == entry['bytes']
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
print('PASS: archive and all raw-file hashes')
PY

python3 -B -S verify_plan.py
python3 -B -S reporting/analyze_release_compat.py \
  --run results/run_001 --inputs inputs --check-report results/report.json
python3 -B -S -m unittest discover -s reporting -p 'test_*.py'
python3 -B -S -m unittest test_plan_analysis test_prepare_transfer test_run_transfer test_analysis_transfer
```

The compatibility audit is standard-library only. For the separate statistical
recount, use a Python environment with SciPy installed:

```bash
python3 reporting/independent_recount.py --run results/run_001
```

The original 74-test suite passed before release at `23f7f7f`. Three tests in
`test_freeze.py` deliberately assert that original pending authorization state;
reproduce those on that historical commit. The current commands run the other
65 tests plus six new compatibility tests, without any model forward.

## Disclosed post-run analysis correction

The frozen runner requires top-level release status `approved` and nested review
status `PASS`. The frozen analyzer mistakenly required the nested label to be
`approved` too, and stopped before producing the report. The synthetic tests
had not caught that disagreement between the two interfaces.

The [compatibility wrapper](reporting/analyze_release_compat.py) checks the exact
frozen analyzer and release-validator hashes, validates the full authorization,
and changes exactly that one nested status literal in memory. The report exposes
the original, executed-source and wrapper hashes under `analysis_compatibility`.
The original code, source lock, release record, measurements, candidate rules,
numerical tolerances and statistical decisions were not edited. Six new tests
cover valid and invalid release states, tampering and the one-replacement rule.

The shorter-prefix exception was disclosed and accepted **before** execution;
its corrected counts are in [the review](REVIEW.md) and [result](RESULT.md).
It is separate from this metadata correction.

**The one authorized execution is complete.** These commands verify saved data;
they do not authorize a new neural run or the later mechanistic experiment.
