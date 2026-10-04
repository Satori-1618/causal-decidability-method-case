# Recompute without rerunning the model

The single authorized execution is complete. Keep the release and plan as
historical pre-run records; they do **not** authorize another execution.
The original producer sources and inputs remain unchanged after public release
`5cb6f770494a2e42bfdfeaaa3206bc9beab3f75a`.

## Records

`results/run_001.tar.gz` contains all 17 original output files. Only archive
header times were normalized; file contents are unchanged. `results/raw_index.json`
binds the archive and every file with SHA-256 and size. The manifest additionally
binds 16 other output files and 88 source bindings. The three convenient manifest,
primary and descriptive JSON copies beside the archive are byte-identical to
their archived originals.

From this directory, extract and check the recorded bytes:

```bash
tar -xzf results/run_001.tar.gz -C results
python3 - <<'PY'
from pathlib import Path
import hashlib, json
p = Path('results'); index = json.loads((p/'raw_index.json').read_text())
assert hashlib.sha256((p/index['archive']).read_bytes()).hexdigest() == index['archive_sha256']
for name, entry in index['files'].items():
    data = (p/'run_001'/name).read_bytes()
    assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
print('Archive and all raw file hashes match')
PY
```

## Independent audits

The statistical recount requires SciPy, but **no PyTorch, checkpoint or model
call**. It was checked with Python 3.10.10, SciPy 1.15.3 and NumPy 2.2.6.
The provenance audit uses the standard library and a Git clone containing the
review/release commits. The statistical audit also works from a clean export.

```bash
python3 -B reporting/verify_statistics.py results/run_001 > /tmp/averaged004-statistics.json
python3 -B -S reporting/verify_provenance.py --repo ../../../.. --run results/run_001 > /tmp/averaged004-provenance.json
```

The first independently implements the predictions and arithmetic, compares
raw counts and diagnostic summaries with the original outputs, and uses SciPy
for Clopper–Pearson and binomial calculations. Tiny cross-version float
differences are checked with a numerical tolerance, not exact decimal equality.
It does not import the producer analysis.

The second checks review-to-release changes, all source/output hashes, recipient
selection, calibration-only forecasts, ordering of receipts and forward events,
complete cells and forward counts. It reconstructs dtype-rounded intended and
delivered nodes from compact raw snapshots. Full attention and off-target tensors
were not serialized, so their preservation checks remain producer records.
Both audits were performed by separate agents; this is not an independent human
review or a second model replication.

The original 67 synthetic tests passed before release. No original test, input,
scientific rule, threshold or primary source was changed after review. New files
under `reporting/` and `results/` are post-run audit/report artifacts; they are not
silently added to the historical producer lock.
