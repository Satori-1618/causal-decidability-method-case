"""Verify the frozen 64-pair Q1 records, then compare them with the current method.

    python3 examples/confirmed_read_source.py [--json]

Standard library only. No installation, downloads, model loading or file writes.
This checks stored experimental evidence, not a fresh execution of the intervention.
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

# Keep the records-only entry point from creating local import caches.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'applications' / 'makelov-2311.17030'
sys.path.insert(0, str(ROOT / 'src'))

from causal_decidability.evaluate import compare, load_rows  # noqa: E402

CANDIDATES = {
    'A_visible_read': {'read_row': 'same_as:full', 'read_null': 'same_as:baseline'},
    'B_null_read': {'read_row': 'same_as:baseline', 'read_null': 'same_as:full'},
}


def run(application_root=APP):
    application_root = Path(application_root)
    checker_path = application_root / 'scripts' / 'check_read_source_q1_records.py'
    spec = importlib.util.spec_from_file_location('q1_records_check', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checked = checker.check(application_root)

    records_path = application_root / 'results' / 'makelov_read_source_q1' / 'records.jsonl'
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows, repeats = [], {}
    # The verifier has already required exactly the two reciprocal records per case.
    for record in records:
        unit = record['case_id']
        repeat = str(repeats.get(unit, 0))
        repeats[unit] = repeats.get(unit, 0) + 1
        rows.extend((unit, repeat, condition, record['margins'][condition])
                    for condition in ('baseline', 'full', 'read_row', 'read_null'))
    comparison = compare(load_rows(rows), CANDIDATES, ['read_row', 'read_null'],
                         scope='pooled', seed=0)[0]
    primary = checked['primary']
    if (comparison['units'] != 64 or comparison['units_favouring'] != primary['wins']
            or comparison['ties'] != primary['ties']
            or not math.isclose(comparison['sign_test_p'], primary['p'], rel_tol=1e-12,
                                abs_tol=0.0)):
        raise ValueError('Current method disagrees with the verified frozen Q1 primary result')
    return {'archived_evidence': checked, 'current_method_comparison': comparison,
            'claim': 'Relative prediction comparison under the declared interventions; '
                     'no adequacy or unique/native mechanism claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--json', action='store_true', help='print the complete verification report')
    args = parser.parse_args()
    try:
        result = run()
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'FAILED: {error}\n')
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    comparison = result['current_method_comparison']
    wins = comparison['units_favouring']
    print('PASS: frozen Q1 records verified; current method agrees with the primary result.')
    print(f"64 fresh base pairs: B wins {wins['B_null_read']}, A wins {wins['A_visible_read']}, "
          f"ties {comparison['ties']}.")
    print(f"Exact two-sided sign test: p = {comparison['sign_test_p']:.6g}.")
    print('Baseline/full are measured anchors; only the two read-source interventions test the candidates.')
    print(result['claim'])
    print('Verified: archived records and their declared checks. Not replayed: model or live tensor interventions.')
    print('Bootstrap intervals and sign-flip diagnostics in --json are secondary, not the frozen primary test.')


if __name__ == '__main__':
    main()
