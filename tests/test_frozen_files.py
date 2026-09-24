import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('frozen_check', ROOT / 'scripts/check_frozen_files.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_archive_matches_development_snapshot():
    assert checker.check()['files'] == 62


def test_modified_and_missing_archived_record_fail(tmp_path):
    manifest = json.loads((ROOT / 'release/FROZEN_FILES.json').read_text())
    name = 'applications/makelov-2311.17030/results/makelov_read_source_q1/records.jsonl'
    manifest['files'] = {name: manifest['files'][name]}
    (tmp_path / 'release').mkdir()
    (tmp_path / 'release/FROZEN_FILES.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Missing frozen file'):
        checker.check(tmp_path)
    record = tmp_path / name
    record.parent.mkdir(parents=True)
    record.write_bytes((ROOT / name).read_bytes())
    assert checker.check(tmp_path)['files'] == 1
    record.write_bytes(record.read_bytes().replace(b'baseline', b'changed_', 1))
    with pytest.raises(ValueError, match='Changed frozen file'):
        checker.check(tmp_path)


def test_inventory_cannot_read_outside_repository(tmp_path):
    (tmp_path / 'release').mkdir()
    manifest = {'schema_version': 1, 'source_commit': 'test',
                'files': {'../outside': '0' * 64}}
    (tmp_path / 'release/FROZEN_FILES.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='escapes repository'):
        checker.check(tmp_path)
