"""Integration checks for the published, model-free Q1 entry point."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_confirmed_example_without_site_packages():
    completed = subprocess.run(
        [sys.executable, '-S', str(ROOT / 'examples' / 'confirmed_read_source.py'), '--json'],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    report = json.loads(completed.stdout)
    result = report['current_method_comparison']
    assert result['units'] == 64
    assert result['units_favouring'] == {'A_visible_read': 0, 'B_null_read': 64}
    assert result['ties'] == 0
    assert result['sign_test_p'] == 2.0 ** -63
    assert 'no adequacy' in report['claim']
