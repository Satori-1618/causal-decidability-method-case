"""Independently verify Stage A records with the Python standard library only.

No model, tokenizer, producer analyzer or geometry module is imported. Bootstrap
and Clopper--Pearson intervals are deliberately outside the verification scope.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[3]
ROLES = ('giver', 'receiver', 'observer')
WORDINGS = ('fit', 'heldout')
PRECISIONS = ('float32', 'float64')
CONTEXTS = {'d0': ((0, 1, 2), 'gave'), 'd1': ((1, 0, 2), 'received')}
CONTEXTS.update({f'b{i}_{form}': ((i, (i+1) % 3, (i+2) % 3), form)
                 for i in range(3) for form in ('gave', 'watched')})
GROUPS = [(q, w, f) for q in ROLES for w in WORDINGS for f in ('gave', 'received', 'watched')]
DIAGNOSTICS = {
    'name': ('d1/b1_gave/observer/receiver',),
    'position': ('d1/b1_gave/observer/receiver',),
    'switch': ('d0/b1_gave/observer/giver', 'd0/b1_gave/observer/receiver'),
    'no_op': ('d0/b1_gave/observer/giver', 'd0/b1_gave/observer/receiver'),
}
CONTRACT = {'n': 32, 'sampling_seed': 24092441, 'competence_threshold': .90,
            'geometry_coverage': .90, 'geometry_gap_nats': .52,
            'optimistic_role_gain': 1., 'numerical_budget_nats': .01,
            'bootstrap_draws': 20000, 'bootstrap_seed': 320260940, 'batch_size': 16}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float), 'expected a number, not a boolean')
    value = float(value)
    require(math.isfinite(value), 'nonfinite number')
    return value


def integer(value):
    require(type(value) is int and value >= 0, 'expected nonnegative integer')
    return value


def vector(values):
    require(isinstance(values, list) and len(values) == 3, 'expected three coordinates')
    return [number(x) for x in values]


def close(actual, expected, label='value'):
    """Compare a recomputed subset, retaining exact categorical/threshold checks."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f'{label}: expected a mapping')
        for key, value in expected.items():
            require(key in actual, f'{label}: missing {key}')
            close(actual[key], value, f'{label}.{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'{label}: wrong list size')
        for i, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, f'{label}[{i}]')
    elif isinstance(expected, float):
        require(math.isclose(number(actual), expected, rel_tol=1e-10, abs_tol=1e-11),
                f'{label}: arithmetic mismatch: {actual} != {expected}')
    else:
        require(type(actual) is type(expected) and actual == expected, f'{label}: value mismatch')


def margins(logits):
    a, b, c = vector(logits)
    return [a-b, a-c, b-c]


def subtract(a, b):
    return [x-y for x, y in zip(a, b)]


def distance(a, b):
    return math.hypot(*subtract(a, b))/math.sqrt(3.)


def projection(point, direction):
    # Normalize first: the zero vector denotes the singleton {0}, not any line.
    norm = math.hypot(*direction)
    if norm == 0:
        return [0., 0., 0.]
    unit = [x/norm for x in direction]
    amplitude = math.fsum(x*y for x, y in zip(point, unit))
    return [amplitude*x for x in unit]


def mention_order(context):
    (g, r, o), form = CONTEXTS[context]
    return {'gave': (g, r, o), 'received': (r, g, o), 'watched': (o, r, g)}[form]


def geometry(rows, wording):
    def endpoint(context, query):
        return margins(rows[f'{wording}/{context}/{query}']['name_logits'])

    def role_response(cell):
        _, recipient, qr, qd = cell.split('/')
        return subtract(endpoint(recipient, qd), endpoint(recipient, qr))

    switch = [role_response(cell) for cell in DIAGNOSTICS['switch']]
    center = [(a+b)/2 for a, b in zip(*switch)]
    diagnostics = {}
    for rival, cells in DIAGNOSTICS.items():
        diagnostics[rival] = []
        for cell in cells:
            donor, recipient, qr, qd = cell.split('/')
            response = role_response(cell)
            if rival in ('name', 'position'):
                target = CONTEXTS[donor][0][ROLES.index(qd)]
                if rival == 'position':
                    target = mention_order(recipient)[mention_order(donor).index(target)]
                target_role = ROLES[CONTEXTS[recipient][0].index(target)]
                line = subtract(endpoint(recipient, target_role), endpoint(recipient, qr))
                fitted = projection(response, line)
            else:
                fitted = center if rival == 'switch' else [0., 0., 0.]
            gap = distance(response, fitted)
            diagnostics[rival].append({'cell_id': cell, 'maximum_gain_role_prediction': response,
                                      'oracle_projection': fitted, 'gap': gap, 'passes': gap > .52})
    minimum = min(d['gap'] for ds in diagnostics.values() for d in ds)
    return {'maximum_gain': 1., 'diagnostics': diagnostics, 'minimum_gap': minimum,
            'potential_separation': minimum > .52}


def validate_case(case):
    require(isinstance(case['case_id'], str) and case['case_id'], 'missing case ID')
    names, answer_ids = case['names'], case['answer_token_ids']
    require(len(names) == len(set(names)) == 3 and all(isinstance(n, str) and n for n in names),
            'three distinct names required')
    require(len(answer_ids) == len(set(answer_ids)) == 3, 'three answer token IDs required')
    for token in answer_ids:
        integer(token)
    expected = [f'{w}/{c}/{q}' for w in WORDINGS for c in CONTEXTS for q in ROLES]
    require([r['row_id'] for r in case['rows']] == expected, 'frozen 48-row grid differs')
    for row in case['rows']:
        w, c, q = row['row_id'].split('/')
        target = CONTEXTS[c][0][ROLES.index(q)]
        close(row, {'wording': w, 'context_id': c, 'form': CONTEXTS[c][1], 'query': q,
                    'correct_name_index': target, 'correct_answer_token_id': answer_ids[target]})
        tokens = row['token_ids']
        require(tokens and all(type(t) is int and t >= 0 for t in tokens), 'invalid frozen tokens')
        require(tokens[0] == integer(case['bos_token_id']) and row['position'] == len(tokens)-1,
                'invalid BOS or absolute last position')
        positions = row['name_token_positions']
        require(len(positions) == 3 and positions == sorted(set(positions))
                and all(type(p) is int and 0 < p < len(tokens) for p in positions), 'invalid name slots')
        require([tokens[p] for p in positions] == [answer_ids[i] for i in mention_order(c)],
                'name slots do not match frozen role bindings')
    return {r['row_id']: r for r in case['rows']}


def inspect_precision(case, fixed, raw, precision):
    require(len(raw['rows']) == 48, 'incomplete native rows')
    rows = {r['row_id']: r for r in raw['rows']}
    require(len(rows) == 48 and set(rows) == set(fixed), 'duplicate or missing native row')
    derived = {}
    for row_id, row in rows.items():
        close(row, {key: fixed[row_id][key] for key in
                    ('wording', 'context_id', 'form', 'query', 'correct_name_index')})
        logits = vector(row['name_logits'])
        mass = number(row['name_probability_mass'])
        require(0 <= mass <= 1, 'name mass outside [0,1]')
        argmax = integer(row['full_vocab_argmax_id'])
        require(argmax < 50257, 'GPT-2 vocabulary argmax out of range')
        winners = [i for i, x in enumerate(logits) if x == max(logits)]
        derived[row_id] = {'pairwise_margins': margins(logits),
                          'three_name_correct': winners == [row['correct_name_index']],
                          'three_name_top_tie': len(winners) > 1,
                          'three_name_prediction': winners[0] if len(winners) == 1 else None,
                          'full_vocab_argmax_is_correct_name': argmax == case['answer_token_ids'][row['correct_name_index']]}
    control = raw['controls']
    first = case['rows'][0]['row_id']
    close(control, {'passed': True, 'token_replay_passed': True, 'no_padding': True, 'replay_row_id': first})
    replay = vector(control['replay_name_logits'])
    error = max(abs(a-b) for a, b in zip(replay, rows[first]['name_logits']))
    epsilon = 2.**(-23 if precision == 'float32' else -52)
    tolerance = 64*epsilon*max(1., *map(abs, replay))
    require(number(control['batch_vs_single_max_name_logit_error']) == error,
            'recorded replay error differs from saved replay logits')
    require(number(control['batch_vs_single_tolerance']) == tolerance,
            'recorded replay tolerance differs from the dtype formula')
    require(error <= tolerance, 'native replay discrepancy exceeds fixed tolerance')
    lengths = Counter(len(r['token_ids']) for r in fixed.values())
    close(raw, {'prompt_evaluations': 49,
                'model_forward_calls': 1+sum((n+15)//16 for n in lengths.values())})
    scores = {}
    for q, w, f in GROUPS:
        ids = [i for i, r in rows.items() if (r['query'], r['wording'], r['form']) == (q, w, f)]
        count = sum(derived[i]['three_name_correct'] for i in ids)
        scores['|'.join((q, w, f))] = {'correct': count, 'rows': len(ids), 'family_accuracy': count/len(ids)}
    return {'rows': rows, 'derived_rows': derived, 'group_scores': scores,
            'potential_geometry': {w: geometry(rows, w) for w in WORDINGS}}


def recompute(cases, records):
    require(len(cases) == len(records) == 32, 'exactly 32 families required')
    case_ids = [c['case_id'] for c in cases]
    require(len(set(case_ids)) == 32, 'duplicate frozen family')
    by_id = {r['case_id']: r for r in records}
    require(len(by_id) == 32 and set(by_id) == set(case_ids), 'record family coverage differs')
    families, unresolved = {}, []
    for case in cases:
        cid, fixed = case['case_id'], validate_case(case)
        raw = by_id[cid]['precisions']
        require(set(raw) == set(PRECISIONS), 'both precisions required')
        parsed = {p: inspect_precision(case, fixed, raw[p], p) for p in PRECISIONS}
        a, b = parsed['float32'], parsed['float64']
        discrepancies = {}
        for rid in fixed:
            discrepancies[f'row.{rid}.margins'] = max(abs(x-y) for x, y in zip(
                a['derived_rows'][rid]['pairwise_margins'], b['derived_rows'][rid]['pairwise_margins']))
        for w in WORDINGS:
            for rival in DIAGNOSTICS:
                for left, right in zip(a['potential_geometry'][w]['diagnostics'][rival],
                                       b['potential_geometry'][w]['diagnostics'][rival]):
                    prefix = f"{w}.{rival}.{left['cell_id']}"
                    discrepancies[prefix+'.gap'] = abs(left['gap']-right['gap'])
                    for field in ('maximum_gain_role_prediction', 'oracle_projection'):
                        discrepancies[prefix+'.'+field] = max(abs(x-y) for x, y in zip(left[field], right[field]))
        maximum = max(discrepancies.values())
        require(math.isfinite(maximum), 'nonfinite derived discrepancy')
        resolved = maximum <= .01
        if not resolved:
            unresolved.append(cid)
        success = resolved and all(parsed[p]['potential_geometry'][w]['potential_separation']
                                   for p in PRECISIONS for w in WORDINGS)
        families[cid] = {'precisions': parsed, 'resolved': resolved, 'potential_geometry_success': success,
                         'numerical_resolution': {'passed': resolved, 'maximum_discrepancy_nat': maximum,
                                                  'discrepancies_nat': discrepancies}}
    groups, failed = {}, []
    for q, w, f in GROUPS:
        key, values = '|'.join((q, w, f)), {}
        for p in PRECISIONS:
            scores = [families[cid]['precisions'][p]['group_scores'][key] for cid in case_ids]
            mean = math.fsum(s['family_accuracy'] for s in scores)/32
            values[p] = {'mean_family_accuracy': mean, 'family_scores': [s['family_accuracy'] for s in scores],
                         'correct_rows': sum(s['correct'] for s in scores), 'rows_per_family': scores[0]['rows'],
                         'n_families': 32, 'passes': mean >= .90}
        passed = all(v['passes'] for v in values.values())
        groups[key] = {'query': q, 'wording': w, 'form': f, 'precisions': values, 'passes': passed}
        if not passed:
            failed.append(key)
    successes = sum(f['potential_geometry_success'] for f in families.values())
    reasons = (['INVALIDNUMERICS'] if unresolved else []) + (['STOP_COMPETENCE'] if failed else [])
    if successes < 29:
        reasons.append('STOP_GEOMETRY')
    return {'status': reasons[0] if reasons else 'PASS', 'failure_reasons': reasons,
            'n_units': 32, 'n_rows_per_precision_per_family': 48, 'competence_groups': groups,
            'failed_competence_groups': failed,
            'potential_geometry': {'successes': successes, 'n': 32, 'fraction': successes/32,
                                   'minimum_required_families': 29, 'passes': successes >= 29, 'maximum_gain_only': True},
            'numerical_resolution': {'passed': not unresolved, 'unresolved_case_ids': unresolved,
                'maximum_discrepancy_nat': max(f['numerical_resolution']['maximum_discrepancy_nat'] for f in families.values())},
            'families': families, 'relational_response_supported': False, 'confirmation_authorized': False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_directory(directory, repo=REPO):
    directory, repo = Path(directory).resolve(), Path(repo).resolve()
    require(not (directory/'FAILED.json').exists(), 'failed execution marker is present')
    names = ('manifest.json', 'cases.json', 'records.jsonl', 'summary.json', 'RUN_STARTED.json',
             'measurements_float32.jsonl', 'measurements_float64.jsonl')
    listed = read_json(directory/'artifact_hashes.json')
    require(set(names) <= set(listed), 'artifact inventory lacks required files')
    for name, checksum in listed.items():
        require(Path(name).name == name, 'artifact inventory paths must be local filenames')
        require(digest(directory/name) == checksum, f'artifact hash mismatch: {name}')
    manifest, summary, started = [read_json(directory/name) for name in
                                  ('manifest.json', 'summary.json', 'RUN_STARTED.json')]
    require(manifest['contract'] == CONTRACT, 'frozen contract differs from checker')
    close(manifest, {'experiment': 'role_baseline_3b_stage_a', 'confirmation': False,
                     'n_families': 32, 'native_rows_per_family_per_precision': 48,
                     'precisions': list(PRECISIONS)})
    require(manifest['cases_sha256'] == digest(directory/'cases.json'), 'case hash mismatch')
    for obj in (summary, started):
        require(obj['manifest_sha256'] == digest(directory/'manifest.json'), 'manifest digest mismatch')
    require(summary['records_sha256'] == digest(directory/'records.jsonl'), 'record digest mismatch')
    commit = started['execution_git_head']
    require(isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit), 'invalid execution commit')
    frozen = {**manifest['code_files_sha256'], **manifest['documents_sha256']}
    for name in ('manifest.json', 'cases.json'):
        frozen[str((directory/name).relative_to(repo))] = digest(directory/name)
    for name, checksum in frozen.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'invalid frozen path')
        content = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=repo)
        require(hashlib.sha256(content).hexdigest() == checksum, f'execution commit differs: {name}')
    cases, records = read_json(directory/'cases.json'), read_jsonl(directory/'records.jsonl')
    for precision in PRECISIONS:
        separate = read_jsonl(directory/f'measurements_{precision}.jsonl')
        require(separate == [{'case_id': r['case_id'], **r['precisions'][precision]} for r in records],
                f'{precision} measurement log differs from combined records')
    checked = recompute(cases, records)
    close(summary, {key: value for key, value in checked.items() if key != 'families'}, 'summary')
    saved = {r['case_id']: r for r in summary['records']}
    require(len(saved) == 32 and set(saved) == set(checked['families']), 'summary family coverage differs')
    for cid, family in checked['families'].items():
        close(saved[cid], {k: v for k, v in family.items() if k != 'precisions'}, cid)
        for p, expected in family['precisions'].items():
            actual = saved[cid]['precisions'][p]
            close(actual, {k: expected[k] for k in ('group_scores', 'potential_geometry')}, f'{cid}.{p}')
            rows = {r['row_id']: r for r in actual['rows']}
            require(len(rows) == 48 and set(rows) == set(expected['rows']), 'summary row coverage differs')
            for rid in rows:
                close(rows[rid], expected['rows'][rid], f'{cid}.{p}.{rid}.raw')
                close(rows[rid], expected['derived_rows'][rid], f'{cid}.{p}.{rid}.derived')
    calls = sum(p['model_forward_calls'] for r in records for p in r['precisions'].values())
    close(summary, {'confirmation': False, 'patch_forwards': 0,
                    'prompt_evaluations': 32*2*49, 'model_forward_calls': calls})
    return {'schema_version': 'role-baseline-independent-check-v1', 'verified': True,
            'independent_implementation': 'Python standard library; no producer analyzer or geometry imports',
            'execution_git_head': commit, 'checker_sha256': digest(__file__),
            'input_sha256': {name: digest(directory/name) for name in (*names, 'artifact_hashes.json')},
            **{k: v for k, v in checked.items() if k != 'families'},
            'verified_scope': ['artifact hashes and frozen code/documents/cases/manifest at execution commit',
                'all 32 families, both precisions, all 48 rows and independent role-label/token-slot checks',
                'strict-tie correctness, family-first group accuracy and every declared competence gate',
                'every maximum-gain diagnostic using independent line projection arithmetic',
                'native and derived cross-precision discrepancies, replay logits/error/tolerance and run counts',
                'all stop reasons and the joint 29-of-32 geometry rule'],
            'not_verified': ['bootstrap and Clopper-Pearson confidence intervals',
                'model weights, runtime execution or tokenizer semantics independently of frozen producer records',
                'full-vocabulary logits and probability mass, which are not saved in full',
                'any intervention, learned gain, relational-transfer or confirmation claim']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--check-only', action='store_true',
                        help='verify published artifacts without writing or replacing verification.json')
    args = parser.parse_args()
    result = verify_directory(args.results)
    if not args.check_only:
        with (args.results/'verification.json').open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
    print(json.dumps({k: result[k] for k in ('verified', 'status', 'failure_reasons', 'potential_geometry')}, indent=2))


if __name__ == '__main__':
    main()
