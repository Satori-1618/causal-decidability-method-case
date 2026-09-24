"""Check archived file identity without git, downloads or model execution."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'release/FROZEN_FILES.json').read_text())
    if manifest.get('schema_version') != 1 or not manifest.get('files'):
        raise ValueError('Missing or unsupported frozen-file inventory')
    for name, expected in manifest['files'].items():
        location = (root / name).resolve()
        if not location.is_relative_to(root):
            raise ValueError(f'Frozen path escapes repository: {name}')
        if not location.is_file():
            raise ValueError(f'Missing frozen file: {name}')
        actual = hashlib.sha256(location.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Changed frozen file: {name}')
    return {'status': 'unchanged', 'files': len(manifest['files']),
            'source_commit': manifest['source_commit']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = check()
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'FAILED: {error}\n')
    print(f"PASS: all {result['files']} archived files match {result['source_commit']}.")
    print('This checks file identity, not execution or an external freeze timestamp.')


if __name__ == '__main__':
    main()
