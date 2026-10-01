"""Independent raw-artifact audit; imports neither runner nor either analyzer.

The audit checks local provenance, not an externally timestamped preregistration.
Saved tensor-control records can be checked for consistency; unsaved tensors
cannot be independently reconstructed without another model execution.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ('native', 'uniform_all_queries', 'both', 'routing_only', 'gate_only',
        'within_token', 'token_mass')
STAGES = {
    'stage1': ('both', 'routing_only', 'gate_only', 'H_R', 'H_G', 'routing', 'gating'),
    'stage2': ('routing_only', 'within_token', 'token_mass', 'H_W', 'H_T', 'within_token', 'token_mass'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def average(values):
    return math.fsum(values) / len(values)


def compare(actual, expected, path='report'):
    """Compare all expected fields; allow additional descriptive report fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), path + ': expected mapping')
        for key, value in expected.items():
            require(key in actual, path + ': missing ' + key)
            compare(actual[key], value, path + '.' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), path + ': list length')
        for i, value in enumerate(expected):
            compare(actual[i], value, path + '[' + str(i) + ']')
    elif isinstance(expected, float):
        require(math.isfinite(actual) and math.isclose(actual, expected, abs_tol=2e-12, rel_tol=2e-12), path + ': numeric mismatch')
    else:
        require(actual == expected, path + ': value mismatch')


def likelihood_interval(hits, n, alpha):
    """Invert a likelihood-level set, independently of producer's KL function."""
    require(n > 0 and 0 <= hits <= n and 0 < alpha < 1, 'Invalid confidence input')
    q = hits / n
    if hits == 0:
        return 0.0, -math.expm1(math.log(alpha) / n)
    if hits == n:
        return math.exp(math.log(alpha) / n), 1.0
    level = hits * math.log(q) + (n - hits) * math.log1p(-q) + math.log(alpha)

    def inside(p):
        return hits * math.log(p) + (n - hits) * math.log1p(-p) >= level

    left, right = 0.0, q
    for _ in range(100):
        middle = (left + right) / 2
        if inside(middle):
            right = middle
        else:
            left = middle
    lower = left
    left, right = q, 1.0
    for _ in range(100):
        middle = (left + right) / 2
        if inside(middle):
            left = middle
        else:
            right = middle
    return lower, right


def verify_prediction_rows(predictions, observations, stage):
    endpoint, arm_a, arm_b, _, _, alias_a, alias_b = STAGES[stage]
    require(len(predictions) == len(observations), stage + ': missing forecast')
    for predicted, measured in zip(predictions, observations):
        for key in ('family_id', 'condition'):
            require(predicted[key] == measured[key], stage + ': forecast row misalignment')
        baseline = measured['float64_native']
        target = measured['float64_' + endpoint]
        expected = {alias_a: {arm_a: target, arm_b: baseline},
                    alias_b: {arm_a: baseline, arm_b: target}}
        require(predicted['predictions'] == expected, stage + ': forecast differs from its endpoint rule')


def recompute_stage(rows, stage, freeze):
    endpoint, arm_a, arm_b, candidate_a, candidate_b, _, _ = STAGES[stage]
    groups = defaultdict(list)
    for row in rows:
        groups[row['family_id']].append(row)
    eligible, member_counts, errors = [], Counter(), {candidate_a: {}, candidate_b: {}}
    precision = 0.0
    for family, members in groups.items():
        require(len(members) == 3 and {r['condition'] for r in members} == {'valid', 'invalid_open', 'invalid_close'}, 'Incomplete family')
        gaps = [abs(r['float64_' + endpoint] - r['float64_native']) for r in members]
        if max(gaps) > freeze['gap_threshold']:
            eligible.append(family)
        for row, gap in zip(members, gaps):
            if gap > freeze['gap_threshold']:
                member_counts[row['condition']] += 1
        for candidate, targets in ((candidate_a, (endpoint, 'native')), (candidate_b, ('native', endpoint))):
            by_dtype = {}
            for dtype in ('float64', 'float32'):
                deviations = [abs(row[dtype + '_' + arm] - row[dtype + '_' + reference])
                              for row in members for arm, reference in zip((arm_a, arm_b), targets)]
                by_dtype[dtype] = max(deviations)
            errors[candidate][family] = by_dtype['float64']
            precision = max(precision, abs(by_dtype['float64'] - by_dtype['float32']))
    decisions = {}
    for candidate in (candidate_a, candidate_b):
        values = [errors[candidate][family] for family in eligible]
        n = len(values)
        definite = sum(e <= freeze['prediction_tolerance'] - freeze['numerical_tolerance'] for e in values)
        possible = sum(e <= freeze['prediction_tolerance'] + freeze['numerical_tolerance'] for e in values)
        alpha = freeze['alpha'] / freeze['one_sided_bound_count']
        lower = likelihood_interval(definite, n, alpha)[0] if n else 0.0
        upper = likelihood_interval(possible, n, alpha)[1] if n else 1.0
        if precision > freeze['numerical_tolerance']:
            status = 'technical_invalidity'
        elif n < freeze['minimum_eligible_families']:
            status = 'insufficient_eligible_families'
        elif lower >= freeze['adequacy']:
            status = 'adequate'
        elif upper < freeze['adequacy']:
            status = 'excluded'
        else:
            status = 'unresolved'
        decisions[candidate] = {'definite_hits': definite, 'possible_hits': possible, 'n': n,
                               'lower': lower, 'upper': upper, 'status': status,
                               'mean_family_max_error_eligible': average(values) if n else None,
                               'mean_family_max_error_all': average(list(errors[candidate].values()))}
    return {'eligible_families': len(eligible), 'eligible_members_by_condition': dict(member_counts),
            'maximum_prediction_error_precision_difference': precision, 'candidates': decisions,
            'not_excluded': [c for c, d in decisions.items() if d['status'] != 'excluded'],
            'adequate': [c for c, d in decisions.items() if d['status'] == 'adequate']}


def verify_controls(controls, rows, freeze):
    require(controls['precision_gate_passed'] is True, 'Failed precision gate')
    for dtype, floor, eps in [('float32', 1e-5, 2**-23), ('float64', 1e-10, 2**-52)]:
        c = controls[dtype]
        for key in ['identity_max_margin_error', 'identity_max_node_error', 'last_layer_eos_vs_all_queries_max_error', 'stage2_joint_vs_routing_max_margin_error']:
            require(0 <= c[key] <= floor, 'Failed stored control: ' + dtype + '/' + key)
        require(c['gate_sign_pattern_mismatches'] == 0, 'Failed gate-pattern control')
        require(c['minimum_native_bracket_mass'] > 0 and c['minimum_native_symbol_mass'] > 0, 'Undefined conditional routing')
        require(c['stage2_joint_vs_routing_max_attention_error'] <= 64 * eps, 'Failed joint endpoint attention control')
        require(c['offtarget_and_current_value_checks'] == 'passed_all_eos_arms', 'Missing off-target control')
        for arm in ('within_token', 'token_mass'):
            operator = c[arm + '_delivered_operator']
            require(operator['dtype_tolerance'] == 64 * eps, 'Wrong dtype-derived operator tolerance')
            require(operator['injected_attention_exactly_requested'] is True, 'Tensor delivery failure')
            require(operator['maximum_special_token_error'] == 0, 'Special tokens changed')
            require(max(operator['maximum_symbol_mass_error'], operator['maximum_conditional_ratio_error']) <= 64 * eps, 'Symbol operator control failed')
    arm_errors = {arm: max(abs(r['float32_' + arm] - r['float64_' + arm]) for r in rows) for arm in ARMS}
    compare(controls['fp32_fp64_max_error_by_arm'], arm_errors, 'arm precision')
    compare(controls['fp32_fp64_max_error_overall'], max(arm_errors.values()), 'overall precision')
    differences = {}
    for endpoint, arm_a, arm_b, *_ in STAGES.values():
        for arm, reference in ((arm_a, endpoint), (arm_b, 'native'), (arm_a, 'native'), (arm_b, endpoint)):
            differences[arm + ' minus ' + reference] = max(abs((r['float32_' + arm] - r['float32_' + reference]) -
                                                                (r['float64_' + arm] - r['float64_' + reference])) for r in rows)
    compare(controls['fp32_fp64_prediction_error_difference_by_contrast'], differences, 'contrast precision')
    require(max(list(arm_errors.values()) + list(differences.values())) <= freeze['numerical_tolerance'], 'Precision allowance exceeded')


def check_scope(freeze):
    with (HERE.parent / 'results/heads.csv').open() as handle:
        source = list(csv.DictReader(handle))
    expected = {}
    for row in source:
        if row['head_type'] != 'sign-matching' or int(row['n_layer']) < 2 or row['layer'] != row['n_layer']:
            continue
        key = (row['model_id'], int(row['head']))
        pair = (int(row['initialization_seed']), int(row['shuffle_seed']))
        focus = key == ('1aez5d6p', 2)
        if pair != (365, 220) or focus:
            expected[key] = (row, pair, focus)
    actual = {(t['model_id'], t['head']): t for t in freeze['tasks']}
    require(len(actual) == len(freeze['tasks']) and actual.keys() == expected.keys(), 'Changed or duplicate frozen cohort')
    for key, task in actual.items():
        row, pair, focus = expected[key]
        require((task['init_seed'], task['shuffle_seed']) == pair, 'Wrong training seeds')
        require(task['n_layer'] == int(row['n_layer']) and task['n_head'] == int(row['n_head']), 'Wrong architecture')
        require(task['role'] == ('development_checkpoint_fresh_inputs' if focus else 'transfer_cohort'), 'Wrong development/transfer role')
    require(freeze['one_sided_bound_count'] == 8 * len(actual), 'Multiplicity count changed')


def verify_case_population(rows, freeze):
    families = defaultdict(list)
    seen = set()
    for row in rows:
        text = row['string']
        require(len(text) == 32 and text.count('(') == text.count(')') == 16, 'Wrong bracket population')
        require(type(row['valid']) is bool and row['valid'] == all(text[:i].count('(') >= text[:i].count(')') for i in range(1, 33)), 'Wrong validity label')
        require(text not in seen, 'Repeated input string')
        seen.add(text)
        families[row['family_id']].append(row)
    orbits = set()
    for family, members in families.items():
        require(len(members) == 3 and {r['condition'] for r in members} == {'valid', 'invalid_open', 'invalid_close'}, 'Incorrect family membership')
        canonical = {min(r['string'][k:] + r['string'][:k] for k in range(32)) for r in members}
        require(len(canonical) == 1 and not canonical & orbits, 'Repeated or mixed cyclic orbits')
        representative = next(iter(canonical))
        require(hashlib.sha256(representative.encode()).hexdigest()[:20] == family, 'Wrong orbit identifier')
        orbits.update(canonical)
        for row in members:
            require(row['valid'] == (row['condition'] == 'valid'), 'Condition/label mismatch')
            if row['condition'] != 'valid':
                require(row['string'][0] == ('(' if row['condition'] == 'invalid_open' else ')'), 'Wrong initial-symbol stratum')
    require(len(orbits) == freeze['generator']['families'], 'Wrong family count')
    checked_public = 0
    for name in ('indist', 'ood'):
        source = HERE / 'cache/data/model_preds' / (name + '_data_preds.csv')
        if not source.exists():
            continue
        with source.open() as handle:
            for row in csv.DictReader(handle):
                text = row['string']
                if len(text) == 32 and text.count('(') == 16:
                    require(min(text[k:] + text[:k] for k in range(32)) not in orbits, 'Public evaluation orbit leaked into confirmation')
        checked_public += 1
    return checked_public


def audit(freeze_path, run, report_path=None):
    freeze = read_json(freeze_path)
    manifest = read_json(run / 'manifest.json')
    require(manifest['status'] in {'completed', 'completed_with_invalid_tasks'}, 'Run is not complete')
    require(manifest['freeze_sha256'] == digest(freeze_path), 'Manifest uses a different freeze')
    check_scope(freeze)
    for key, filename in [('protocol_sha256', 'CONFIRMATION.md'), ('analyzer_sha256', 'analyze_confirmation.py'), ('generator_sha256', 'prepare_confirmation.py')]:
        require(freeze[key] == digest(HERE / filename), 'Changed frozen source: ' + filename)
    for relative, expected in manifest.get('source_hashes', {}).items():
        require(digest(ROOT / relative) == expected, 'Changed execution source: ' + relative)
    cases_path = freeze_path.parent / freeze['cases_file']
    require(digest(cases_path) == freeze['cases_sha256'], 'Changed frozen cases')
    lock_path = freeze_path.parent / freeze['asset_lock_file']
    require(digest(lock_path) == freeze['asset_lock_sha256'], 'Changed checkpoint lock')
    cached_checked, cached_missing = 0, []
    for lock in (read_json(lock_path), read_json(HERE / 'ASSET_LOCK.json')):
        for relative, binding in lock['files'].items():
            cached = HERE / 'cache' / relative
            if cached.exists():
                require(digest(cached) == binding['sha256'], 'Changed cached asset: ' + relative)
                cached_checked += 1
            else:
                cached_missing.append(relative)
    for relative, binding in read_json(HERE.parent / 'SOURCE_LOCK.json')['files'].items():
        require(digest(HERE.parent / 'upstream' / relative) == binding['sha256'], 'Changed vendored source: ' + relative)
    base_rows = read_rows(cases_path)
    base_keys = [(r['family_id'], r['condition']) for r in base_rows]
    require(len(set(base_keys)) == len(base_keys) == freeze['generator']['families'] * 3, 'Invalid frozen case count')
    checked_public = verify_case_population(base_rows, freeze)
    original = None
    if 'original_run' in manifest:
        recorded_original = Path(manifest['original_run'])
        # Prefer the sibling carried with an export/bundle; do not silently
        # borrow a matching absolute source directory on the author's machine.
        original = run.parent / recorded_original.name
        require(digest(original / 'manifest.json') == manifest['original_manifest_sha256'], 'Original failed run changed')
        require(read_json(original / 'manifest.json')['status'] == 'failed', 'Continuation must preserve original failed status')
        require(manifest['fixed_error_family_K'] == freeze['one_sided_bound_count'], 'Continuation changed multiplicity')
        require(digest(HERE / 'CONTINUATION.md') == manifest['amendment_sha256'], 'Changed continuation amendment')
        require(digest(HERE / 'continue_confirmation.py') == manifest['continuation_code_sha256'], 'Changed continuation runner')
    entries = {(e['task']['model_id'], e['task']['head']): e for e in manifest['tasks']}
    require(len(entries) == len(manifest['tasks']) == len(freeze['tasks']), 'Incomplete or duplicate task accounting')
    tasks = []
    for task in freeze['tasks']:
        key = (task['model_id'], task['head'])
        entry = entries[key]
        require(entry['task'] == task, 'Task specification changed')
        folder = run / 'tasks' / (key[0] + '_head' + str(key[1]))
        for filename, expected in entry['files'].items():
            require(digest(folder / filename) == expected, 'Changed measured artifact: ' + str(folder / filename))
            if 'carried_from' in entry:
                require(original is not None and digest(original / 'tasks' / folder.name / filename) == expected, 'Carried original artifact changed')
        status = entry.get('status', 'completed')
        require(status in {'completed', 'technical_invalidity'}, 'Undeclared task outcome')
        if status == 'technical_invalidity':
            require(isinstance(entry.get('error'), str) and entry['error'], 'Invalid task needs a recorded error')
            tasks.append({'task': task, 'execution_status': status, 'error': entry['error'], 'stages': None})
            continue
        require({'cases.jsonl', 'controls.json', 'stage1_predictions.jsonl', 'stage2_predictions.jsonl', 'stage1_receipt.json', 'stage2_receipt.json'} <= entry['files'].keys(), 'Missing completed-task artifact')
        rows = read_rows(folder / 'cases.jsonl')
        require(len(rows) == len(base_rows), 'Missing observations')
        for row, base in zip(rows, base_rows):
            require(all(row[k] == v for k, v in base.items()), 'Measured case differs from frozen case')
            require(row['model_id'] == key[0] and row['head'] == key[1], 'Wrong measured checkpoint')
            require(all(isinstance(row[d + '_' + a], (float, int)) and math.isfinite(row[d + '_' + a]) for d in ('float32', 'float64') for a in ARMS), 'Nonfinite margin')
        times = []
        for stage, (endpoint, _, _, _, _, alias_a, alias_b) in STAGES.items():
            predictions_path = folder / (stage + '_predictions.jsonl')
            verify_prediction_rows(read_rows(predictions_path), rows, stage)
            receipt = read_json(folder / (stage + '_receipt.json'))
            require(receipt['prediction_file_sha256'] == digest(predictions_path), 'Forecast receipt hash mismatch')
            require(receipt['preflight_code_sha256'] == digest(ROOT / 'examples/causal_preflight.py'), 'Different preflight implementation')
            differs = any(r['float64_' + endpoint] != r['float64_native'] for r in rows)
            expected_groups = {frozenset([alias_a]), frozenset([alias_b])} if differs else {frozenset([alias_a, alias_b])}
            require({frozenset(g) for g in receipt['preflight']['identical_mean_groups']} == expected_groups, 'Preflight grouping disagrees with forecasts')
            pairs = receipt['preflight']['pairs']
            require(len(pairs) == 1 and pairs[0]['status'] == ('different_declared_means' if differs else 'identical_declared_means'), 'Wrong structural preflight verdict')
            times.append(datetime.fromisoformat(receipt['written_before_target_arms_at']))
        require(times[0] <= times[1], 'Prediction-stage chronology reversed')
        verify_controls(read_json(folder / 'controls.json'), rows, freeze)
        record = {'task': task, 'execution_status': 'completed', 'stages': {s: recompute_stage(rows, s, freeze) for s in STAGES}, 'by_condition': {}}
        for condition in ('valid', 'invalid_open', 'invalid_close'):
            selected = [r for r in rows if r['condition'] == condition]
            record['by_condition'][condition] = {a: {'correct': sum((r['float64_' + a] < 0) == r['valid'] for r in selected), 'n': len(selected),
                'mean_margin_effect': average([r['float64_' + a] - r['float64_native'] for r in selected])} for a in ARMS if a != 'uniform_all_queries'}
        record['interactions'] = {}
        for stage, (endpoint, arm_a, arm_b, *_) in STAGES.items():
            interaction = [r['float64_' + endpoint] - r['float64_' + arm_a] - r['float64_' + arm_b] + r['float64_native'] for r in rows]
            record['interactions'][stage] = {'mean': average(interaction),
                                            'mean_absolute': average([abs(v) for v in interaction]),
                                            'max_absolute': max(abs(v) for v in interaction)}
        tasks.append(record)
    summary = {}
    for role in ('development_checkpoint_fresh_inputs', 'transfer_cohort'):
        subset = [t for t in tasks if t['task']['role'] == role]
        summary[role] = {'tasks': len(subset), 'unique_models': len({t['task']['model_id'] for t in subset}), 'execution': dict(Counter(t['execution_status'] for t in subset))}
        for stage, (_, _, _, a, b, _, _) in STAGES.items():
            summary[role][stage] = {c: dict(Counter(t['stages'][stage]['candidates'][c]['status'] if t['stages'] else 'technical_invalidity' for t in subset)) for c in (a, b)}
    if report_path:
        report = read_json(report_path)
        compare(report['summary'], summary)
        compare(report['tasks'], tasks)
        if 'run_manifest_sha256' in report:
            require(report['run_manifest_sha256'] == digest(run / 'manifest.json'), 'Report manifest mismatch')
    task_audits = []
    for record in tasks:
        key = (record['task']['model_id'], record['task']['head'])
        item = {'model_id': key[0], 'head': key[1], 'role': record['task']['role'],
                'execution_status': record['execution_status'], 'verified_artifact_hashes': entries[key]['files']}
        if record['stages'] is not None:
            item['stages'] = {stage: {'eligible_families': result['eligible_families'],
                'candidates': {candidate: {k: result['candidates'][candidate][k] for k in ('status', 'n', 'definite_hits', 'possible_hits', 'lower', 'upper')}
                               for candidate in result['candidates']}} for stage, result in record['stages'].items()}
        else:
            item['error'] = record['error']
        task_audits.append(item)
    return {'verified': True, 'tasks': len(tasks), 'completed': sum(t['execution_status'] == 'completed' for t in tasks),
            'technical_invalidity': sum(t['execution_status'] == 'technical_invalidity' for t in tasks),
            'report_compared': bool(report_path), 'summary': summary,
            'task_audits': task_audits,
            'provenance': {'verifier_sha256': digest(__file__), 'freeze_sha256': digest(freeze_path),
                           'run_manifest_sha256': digest(run / 'manifest.json'),
                           'analyzer_report_sha256': digest(report_path) if report_path else None},
            'cached_asset_hash_checks': cached_checked, 'missing_cached_assets': cached_missing,
            'public_input_tables_checked_for_orbit_overlap': checked_public,
            'limitations': ['Local hashes and timestamps are not external preregistration.', 'Saved tensor-control records checked; unsaved tensors not independently rerun.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise SystemExit('Refusing to overwrite an existing verification artifact')
    result = audit(args.freeze.resolve(), args.run.resolve(), args.report)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: result[k] for k in ('verified', 'tasks', 'completed', 'technical_invalidity', 'summary')}, indent=2))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
