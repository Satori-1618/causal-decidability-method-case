#!/usr/bin/env python3
"""Check every public claim listed in applications/*/CLAIMS.json against stored evidence.

Each claim names the check that supports it:

* ``recompute``: an allow-listed, model-free script recomputes the value from stored
  records (level N) or from design inputs (level D);
* ``stored``: the value is read from a stored summary file. It is reported separately
  because reading a summary is not a recomputation. With ``verified_by``, an allow-listed
  checker that recomputes that summary from raw records must pass first;
* ``logged``: the claim rests on a protocol or log entry that cannot be recomputed here,
  for example a model forward or a timestamp. It is listed, not checked.

Nothing is written. No script on the allow-list loads a model, tokenizer or weights; see
docs/sendbar/MODEL_FREE_COMMANDS.md. The run fails if any claim fails, if a command is not
allow-listed, or if a claims file is malformed.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKELOV = 'applications/makelov-2311.17030'
MIXING = 'applications/gur-arieh-2510.06182'
SAPHRA = 'applications/li-saphra-2507.06445'

# Positive list of model-free scripts (U05). Adding one requires the same call-chain audit.
ALLOWED = {
    'examples/confirmed_read_source.py',
    'examples/causal_preflight.py',
    f'{MAKELOV}/scripts/check_query_route_records.py',
    f'{MAKELOV}/scripts/check_donor_factor_512_records.py',
    f'{MAKELOV}/scripts/check_role_baseline_records.py',
    f'{MAKELOV}/scripts/describe_post_hoc.py',
    f'{MAKELOV}/scripts/summarize_query_route.py',
    f'{MIXING}/scripts/check_mixing_round1_records.py',
    'scripts/describe_claim_inputs.py',
    'applications/goodfire-mcqa-preflight/run.py',
    f'{SAPHRA}/analyze.py',
    f'{SAPHRA}/verify_independently.py',
}
# Scripts on the list above that download pinned public inputs; run only with --network.
NEEDS_NETWORK = {'applications/goodfire-mcqa-preflight/run.py'}
LEVELS = {'N', 'D', 'logged'}
STATUSES = {'confirmation', 'exploration', 'design_analysis', 'descriptive', 'development',
            'qualification_stop', 'future'}
KINDS = {'recompute', 'stored', 'logged'}


def pointer(document, path):
    """Resolve an RFC 6901 JSON pointer."""
    if path == '':
        return document
    for token in path.lstrip('/').split('/'):
        token = token.replace('~1', '/').replace('~0', '~')
        document = document[int(token)] if isinstance(document, list) else document[token]
    return document


def run_command(argv, cache):
    key = tuple(argv)
    if key not in cache:
        if argv[0] not in ALLOWED:
            raise ValueError(f'command not on the model-free allow-list: {argv[0]}')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        # -E -s -B: ignore environment and user site, write no bytecode; unlike -I, keep the
        # script's own directory importable, which several application scripts rely on.
        result = subprocess.run([sys.executable, '-E', '-s', '-B', *argv], cwd=ROOT, env=env,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'{" ".join(argv)} failed:\n{result.stdout}{result.stderr}')
        cache[key] = result.stdout
    return cache[key]


def compare(actual, expectation):
    if 'equals' in expectation:
        return actual == expectation['equals']
    if 'approx' in expectation:
        return abs(actual - expectation['approx']) <= expectation['abs_tol']
    raise ValueError(f'unknown expectation: {expectation}')


def check_claim(claim, cache, network):
    check = claim['check']
    kind = check['kind']
    if kind == 'logged':
        return True, 'listed; rests on: ' + check['evidence']
    argv = check.get('argv') or check.get('verified_by')
    if argv and argv[0] in NEEDS_NETWORK and not network:
        return None, 'skipped: downloads pinned public inputs; rerun with --network'
    if kind == 'recompute':
        output = run_command(check['argv'], cache)
        if check.get('stdout') == 'text':
            missing = [s for s in check['contains'] if s not in output]
            return not missing, 'missing text: ' + '; '.join(missing) if missing else 'ok'
        # Some checkers print a JSON report followed by a PASS line; read the first document.
        document, _ = json.JSONDecoder().raw_decode(output.lstrip())
    else:
        if 'verified_by' in check:
            run_command(check['verified_by'], cache)
        document = json.loads((ROOT / check['file']).read_text())
    failures = []
    for expectation in check['expect']:
        actual = pointer(document, expectation['pointer'])
        if not compare(actual, expectation):
            failures.append(f"{expectation['pointer']}: got {actual!r}, expected {expectation}")
    return not failures, '; '.join(failures) or 'ok'


def validate(claims_file, data):
    for field in ('case', 'claims'):
        if field not in data:
            raise ValueError(f'{claims_file}: missing {field}')
    seen = set()
    for claim in data['claims']:
        for field in ('id', 'text', 'appears_in', 'status', 'level', 'population', 'sources',
                      'check'):
            if field not in claim:
                raise ValueError(f"{claims_file}: claim {claim.get('id')} lacks {field}")
        if claim['id'] in seen:
            raise ValueError(f"{claims_file}: duplicate id {claim['id']}")
        seen.add(claim['id'])
        if claim['status'] not in STATUSES or claim['level'] not in LEVELS:
            raise ValueError(f"{claims_file}: {claim['id']} has an unknown status or level")
        if claim['check']['kind'] not in KINDS:
            raise ValueError(f"{claims_file}: {claim['id']} has an unknown check kind")
        if (claim['level'] == 'logged') != (claim['check']['kind'] == 'logged'):
            raise ValueError(f"{claims_file}: {claim['id']} mixes logged and checked")
        for source in claim['sources']:
            if not (ROOT / source.split('#')[0]).exists():
                raise ValueError(f"{claims_file}: {claim['id']} cites missing {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', help='only check this case (folder name under applications/)')
    parser.add_argument('--json', action='store_true', help='print the full report as JSON')
    parser.add_argument('--network', action='store_true',
                        help='also run checks that download pinned, hash-checked public inputs')
    args = parser.parse_args()
    files = sorted(ROOT.glob('applications/*/CLAIMS.json'))
    if args.case:
        files = [f for f in files if f.parent.name == args.case]
    if not files:
        raise SystemExit('no CLAIMS.json found')
    cache, report, failed = {}, [], 0
    for claims_file in files:
        data = json.loads(claims_file.read_text())
        validate(claims_file.relative_to(ROOT), data)
        for claim in data['claims']:
            ok, detail = check_claim(claim, cache, args.network)
            failed += ok is False
            kind = claim['check']['kind']
            if kind == 'stored' and 'verified_by' in claim['check']:
                kind = 'stored+verified'
            report.append({'case': data['case'], 'id': claim['id'], 'level': claim['level'],
                           'kind': kind, 'status': claim['status'],
                           'passed': ok, 'detail': detail})
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for row in report:
            mark = {True: 'PASS', False: 'FAIL', None: 'SKIP'}[row['passed']]
            if row['kind'] == 'logged':
                mark = 'LIST'
            print(f"{mark} [{row['level']}/{row['kind']}] {row['case']}: {row['id']}"
                  + ('' if row['passed'] else f" -- {row['detail']}"))
        skipped = sum(row['passed'] is None for row in report)
        totals = {}
        for row in report:
            totals[row['kind']] = totals.get(row['kind'], 0) + 1
        print(f"{len(report)} claims in {len(files)} files: "
              + ', '.join(f'{k} {v}' for k, v in sorted(totals.items()))
              + f'; failed {failed}; skipped {skipped}'
              + (' (network checks; rerun with --network)' if skipped else '') + '.')
        print('recompute = recalculated from stored records or design inputs; stored = read '
              'from a stored summary (+verified: a checker first recomputed that summary from '
              'raw records); logged = listed only, not recomputable here.')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
