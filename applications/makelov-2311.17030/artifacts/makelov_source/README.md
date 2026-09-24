# Upstream inputs, fetched rather than redistributed

`python3 scripts/fetch_upstream.py` downloads every file listed in `source_manifest.json`
from amakelov/activation-patching-illusion at revision
`e0c465b74561d9c3dd1f2afa770974bf5fcaee01`, plus `das_resid_mid.joblib`, and verifies each
sha256 before writing it. The upstream repository carries no licence, so none of its
files are committed here.

The two `.joblib` directions are read as raw float32 payloads with a byte-checked header;
no downloaded pickle is ever executed.
