"""Fetch the upstream inputs at the pinned revision and verify every hash.

    python3 scripts/fetch_upstream.py

The upstream repository (amakelov/activation-patching-illusion) carries no licence, so
its files are not redistributed here. This script downloads each file listed in
``artifacts/makelov_source/source_manifest.json``, plus the ``das_resid_mid.joblib``
direction pinned by the resid_mid.8 preregistration, into ``artifacts/makelov_source/``.
It refuses to write any file whose sha256 differs from its pin.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / 'artifacts' / 'makelov_source'
RAW = 'https://raw.githubusercontent.com/amakelov/activation-patching-illusion/{rev}/{path}'
#: pinned in src/resid_mid8_pilot.py and in the preregistration (§13, item 5)
EXTRA = {'das_resid_mid.joblib':
         '4e69edf9361c3216d9e6bb753d75de440128858757f93028a2293db0488124e3'}


def main():
    manifest = json.loads((TARGET / 'source_manifest.json').read_text())
    pins = dict(manifest['files'], **EXTRA)
    failures = []
    for path, sha in sorted(pins.items()):
        out = TARGET / path
        if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == sha:
            print('present ', path)
            continue
        url = RAW.format(rev=manifest['revision'], path=path)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            failures.append(f'{path}: expected {sha}, got {got}')
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print('fetched ', path)
    if failures:
        sys.exit('hash mismatch, nothing written for:\n  ' + '\n  '.join(failures))


if __name__ == '__main__':
    main()
