"""Execute frozen, sequentially recorded factorial forecasts on fresh cases.

This runner measures and checks fidelity; the frozen statistical decision is
implemented separately. Forecast files precede their target intervention calls.
Local hashes/commit records establish local provenance, not a public timestamp.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import time

import torch

from qualify import digest, dump
from runtime import DyckRuntime, HERE, make_model
from token_split import token_factorized_weights


ARMS = ['native', 'uniform_all_queries', 'both', 'routing_only', 'gate_only', 'within_token', 'token_mass']
FORECAST_DIFFERENCES = [('routing_only','both'), ('gate_only','native'),
                        ('routing_only','native'), ('gate_only','both'),
                        ('within_token','routing_only'), ('token_mass','native'),
                        ('within_token','native'), ('token_mass','routing_only')]
ROOT = HERE.parents[2]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def jsonlines(path, rows):
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        handle.flush()


def resolve_path(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def assert_committed(path):
    relative = path.resolve().relative_to(ROOT).as_posix()
    saved = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT)
    if saved != path.read_bytes():
        raise ValueError('File must equal its committed HEAD version before execution: ' + relative)


def load_cases(freeze, freeze_path):
    path = resolve_path(freeze_path.parent, freeze.get('cases_file', freeze.get('cases_path', 'cases.jsonl')))
    if digest(path) != freeze['cases_sha256']:
        raise ValueError('Case file hash differs from freeze')
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    if len(cases) != 1536:
        raise ValueError('Frozen design requires 1536 cases')
    families = {}
    seen = set()
    seen_strings = set()
    for row in cases:
        if type(row['valid']) is not bool or len(row['string']) != 32:
            raise ValueError('Cases must have Boolean validity and length 32')
        if set(row['string']) != {'(', ')'} or row['string'].count('(') != 16:
            raise ValueError('Every case must have equal bracket counts')
        key = (row['family_id'], row['condition'])
        if key in seen:
            raise ValueError('Duplicate family/condition')
        seen.add(key)
        if row['string'] in seen_strings:
            raise ValueError('Duplicate input string')
        seen_strings.add(row['string'])
        families.setdefault(row['family_id'], []).append(row)
        depth = 0
        true_valid = True
        for char in row['string']:
            depth += 1 if char == '(' else -1
            true_valid &= depth >= 0
        if bool(true_valid and depth == 0) != row['valid']:
            raise ValueError('Validity label is inconsistent with the string')
    if len(families) != 512:
        raise ValueError('Expected 512 independent orbit families')
    orbits = set()
    for rows in families.values():
        invalid = [r for r in rows if not r['valid']]
        if len(rows) != 3 or len(invalid) != 2 or {r['string'][0] for r in invalid} != {'(', ')'}:
            raise ValueError('Each family needs one valid and two invalid strings with different first symbols')
        representatives = {min(r['string'][k:] + r['string'][:k] for k in range(32)) for r in rows}
        if len(representatives) != 1 or representatives & orbits:
            raise ValueError('Families must be distinct rotation orbits')
        orbits.update(representatives)
    return path, cases


def native_sign_pattern(text, row):
    depth = 0
    negative, nonnegative = [], []
    for index, symbol in enumerate(text, 1):
        depth += 1 if symbol == '(' else -1
        (negative if depth < 0 else nonnegative).append(float(row[index]))
    if not negative or not nonnegative:
        return None
    return min(negative) > max(nonnegative) if depth < 0 else max(negative) < min(nonnegative)


def check_offtargets(native, result, runtime):
    if not torch.equal(native.values, result.values):
        raise AssertionError('A final-layer attention intervention changed upstream values')
    delta = result.preprojection - native.preprojection
    head_dim = runtime.module.n_embd // runtime.module.n_head
    index = torch.arange(len(native.margins))
    start = runtime.head * head_dim
    delta[index, native.eos_positions, start:start + head_dim] = 0
    if torch.count_nonzero(delta):
        raise AssertionError('An off-target head or query changed')


def check_token_operator(native, result, intended, strings, mode):
    """Check delivered tensors, including after dtype conversion, per row."""
    actual = result.attention
    if not torch.equal(actual, intended):
        raise AssertionError('Injected attention differs from the requested tensor')
    tolerance = 64 * torch.finfo(actual.dtype).eps
    index = torch.arange(len(strings))
    special_error = max(float((actual[:,0]-native.attention[:,0]).abs().max()),
                        float((actual[index,native.eos_positions]-native.attention[index,native.eos_positions]).abs().max()))
    if special_error != 0:
        raise AssertionError('Stage-2 operator changed a special-token weight')
    mass_error = ratio_error = 0.
    for symbol in '()':
        mask = torch.zeros_like(actual,dtype=torch.bool)
        for row,text in enumerate(strings):
            mask[row,[i+1 for i,char in enumerate(text) if char==symbol]] = True
        original = native.attention * mask
        delivered = actual * mask
        original_mass = original.sum(1,keepdim=True)
        delivered_mass = delivered.sum(1,keepdim=True)
        if torch.any(original_mass <= 0) or torch.any(delivered_mass <= 0):
            raise AssertionError('Symbol conditional distribution is undefined')
        if mode == 'within_token':
            mass_error = max(mass_error,float((original_mass-delivered_mass).abs().max()))
            desired = mask.to(actual.dtype) / mask.sum(1,keepdim=True)
        elif mode == 'token_mass':
            desired = original / original_mass
            bracket_mass = torch.stack([row[1:eos].sum() for row,eos in zip(native.attention,native.eos_positions)])[:,None]
            target_mass = bracket_mass * mask.sum(1,keepdim=True) / torch.tensor([len(s) for s in strings])[:,None]
            mass_error = max(mass_error,float((target_mass-delivered_mass).abs().max()))
        else:
            raise ValueError('Unknown token contract')
        ratio_error = max(ratio_error,float((delivered/delivered_mass-desired).abs().max()))
    if max(mass_error,ratio_error) > tolerance:
        raise AssertionError('Delivered symbol masses or conditional ratios violate the dtype-derived gate')
    return {'maximum_symbol_mass_error':mass_error,'maximum_conditional_ratio_error':ratio_error,
            'maximum_special_token_error':special_error,'dtype_tolerance':tolerance,
            'injected_attention_exactly_requested':True}


def prediction_error_precision(results):
    return {actual+' minus '+endpoint:
            float(((results['float32'][actual].margins.double()-results['float32'][endpoint].margins.double())-
                   (results['float64'][actual].margins-results['float64'][endpoint].margins)).abs().max())
            for actual,endpoint in FORECAST_DIFFERENCES}


def preflight_receipt(path, rows, candidates, arms):
    # Execute the released preflight implementation, using all casewise forecasts
    # rather than their potentially cancelling aggregate means.
    spec = importlib.util.spec_from_file_location('cohort_causal_preflight', ROOT / 'examples/causal_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {'cells': [], 'predictions': {name: [] for name in candidates},
              'prediction_source': 'Previously measured reference-precision endpoints, recorded before target intervention execution'}
    for index, row in enumerate(rows):
        for arm in arms:
            config['cells'].append({'id': str(index) + ':' + arm})
            for name in candidates:
                config['predictions'][name].append(row['predictions'][name][arm])
    result = module.check(config, structure_only=True)
    return {'prediction_file_sha256': digest(path), 'written_before_target_arms_at': utcnow(),
            'preflight_code_sha256': digest(ROOT / 'examples/causal_preflight.py'),
            'preflight': {key: result[key] for key in ['scope', 'mode', 'identical_mean_groups', 'pairs', 'calibration']}}


def load_runtime(task, lock, dtype):
    model_id = task['model_id']
    relative = 'data/model_weights/run_' + model_id + '/run_' + model_id + '_checkpoint_5.pt'
    path = HERE / 'cache' / relative
    expected = lock['files'][relative]
    if digest(path) != expected['sha256']:
        raise ValueError('Checkpoint hash mismatch: ' + model_id)
    model = make_model(n_layer=int(task['n_layer']), n_head=int(task['n_head']), dtype=dtype)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return DyckRuntime(model, layer=int(task['n_layer']) - 1, head=int(task['head']) - 1)


def run_task(task, cases, lock, out, numerical_tolerance):
    out.mkdir(parents=True)
    strings = [row['string'] for row in cases]
    runtimes, results, controls = {}, {}, {}
    for dtype in [torch.float32, torch.float64]:
        tag = str(dtype).split('.')[-1]
        runtime = load_runtime(task, lock, dtype)
        runtimes[tag] = runtime
        results[tag] = {}
        native = runtime.run(strings)
        identity = runtime.run(strings, mode='identity')
        uniform = runtime.run(strings, mode='uniform_all_queries')
        both = runtime.run(strings, mode='both')
        check_offtargets(native, identity, runtime)
        check_offtargets(native, both, runtime)
        floor = 1e-5 if dtype == torch.float32 else 1e-10
        ctl = {
            'identity_max_margin_error': float((native.margins - identity.margins).abs().max()),
            'identity_max_node_error': float((native.node - identity.node).abs().max()),
            'last_layer_eos_vs_all_queries_max_error': float((uniform.margins - both.margins).abs().max()),
            'minimum_native_bracket_mass': min(float(a[1:len(s)+1].sum()) for s, a in zip(strings, native.attention)),
            'minimum_native_symbol_mass': min(float(a[[i+1 for i,c in enumerate(s) if c == symbol]].sum())
                                              for s,a in zip(strings,native.attention) for symbol in '()'),
        }
        if max(ctl[key] for key in ['identity_max_margin_error', 'identity_max_node_error', 'last_layer_eos_vs_all_queries_max_error']) > floor:
            raise AssertionError('Identity or EOS scope reconstruction exceeds precision-specific tolerance')
        controls[tag] = ctl
        results[tag].update(native=native, uniform_all_queries=uniform, both=both)
    stage1 = []
    ref = results['float64']
    for i, case in enumerate(cases):
        n, u = float(ref['native'].margins[i]), float(ref['both'].margins[i])
        stage1.append({'family_id': case['family_id'], 'condition': case['condition'],
                       'predictions': {'routing': {'routing_only': u, 'gate_only': n},
                                       'gating': {'routing_only': n, 'gate_only': u}}})
    prediction_path = out / 'stage1_predictions.jsonl'
    jsonlines(prediction_path, stage1)
    dump(out / 'stage1_receipt.json', preflight_receipt(prediction_path, stage1, ['routing', 'gating'], ['routing_only', 'gate_only']))
    for tag, runtime in runtimes.items():
        native = results[tag]['native']
        for mode in ['routing_only', 'gate_only']:
            result = runtime.run(strings, mode=mode)
            check_offtargets(native, result, runtime)
            results[tag][mode] = result
        mismatches = sum(native_sign_pattern(s,a) != native_sign_pattern(s,b)
                         for s,a,b in zip(strings,native.attention,results[tag]['gate_only'].attention))
        controls[tag]['gate_sign_pattern_mismatches'] = mismatches
        if mismatches:
            raise AssertionError('Gate-only intervention altered native within-bracket sign pattern')
    stage2 = []
    for i, case in enumerate(cases):
        n, r = float(ref['native'].margins[i]), float(ref['routing_only'].margins[i])
        stage2.append({'family_id': case['family_id'], 'condition': case['condition'],
                       'predictions': {'within_token': {'within_token': r, 'token_mass': n},
                                       'token_mass': {'within_token': n, 'token_mass': r}}})
    prediction_path = out / 'stage2_predictions.jsonl'
    jsonlines(prediction_path, stage2)
    dump(out / 'stage2_receipt.json', preflight_receipt(prediction_path, stage2, ['within_token', 'token_mass'], ['within_token', 'token_mass']))
    for tag, runtime in runtimes.items():
        native = results[tag]['native']
        for mode in ['within_token', 'token_mass']:
            weights = token_factorized_weights(native.attention, strings, mode)
            result = runtime.run(strings, mode='custom_attention', attention_override=weights)
            check_offtargets(native, result, runtime)
            controls[tag][mode+'_delivered_operator'] = check_token_operator(native,result,weights,strings,mode)
            results[tag][mode] = result
        # Execute the second decomposition's joint endpoint as a control;
        # equality of constructor outputs alone is not enough.
        weights = token_factorized_weights(native.attention,strings,'both')
        endpoint = runtime.run(strings,mode='custom_attention',attention_override=weights)
        check_offtargets(native,endpoint,runtime)
        if not torch.equal(endpoint.attention,weights):
            raise AssertionError('Stage-2 joint endpoint tensor was not delivered')
        routing = results[tag]['routing_only']
        endpoint_error = float((endpoint.margins-routing.margins).abs().max())
        weight_error = float((endpoint.attention-routing.attention).abs().max())
        controls[tag]['stage2_joint_vs_routing_max_margin_error'] = endpoint_error
        controls[tag]['stage2_joint_vs_routing_max_attention_error'] = weight_error
        if endpoint_error > (1e-5 if tag=='float32' else 1e-10) or weight_error > 64*torch.finfo(weights.dtype).eps:
            raise AssertionError('Executed stage-2 joint endpoint differs from routing-only endpoint')
        controls[tag]['offtarget_and_current_value_checks'] = 'passed_all_eos_arms'
    precision = {arm: float((results['float32'][arm].margins.double() - results['float64'][arm].margins).abs().max()) for arm in ARMS}
    controls['fp32_fp64_max_error_by_arm'] = precision
    controls['fp32_fp64_max_error_overall'] = max(precision.values())
    controls['fp32_fp64_tolerance'] = numerical_tolerance
    prediction_precision = prediction_error_precision(results)
    controls['fp32_fp64_prediction_error_difference_by_contrast'] = prediction_precision
    controls['fp32_fp64_prediction_error_difference_maximum'] = max(prediction_precision.values())
    controls['precision_gate_passed'] = max(precision.values()) <= numerical_tolerance and max(prediction_precision.values()) <= numerical_tolerance
    # Preserve measured outcomes before raising a numerical-gate failure.
    case_records = []
    for i, case in enumerate(cases):
        record = {**case, 'model_id': task['model_id'], 'head': int(task['head'])}
        for tag in results:
            for arm in ARMS:
                record[tag + '_' + arm] = float(results[tag][arm].margins[i])
        case_records.append(record)
    jsonlines(out / 'cases.jsonl', case_records)
    dump(out / 'controls.json', controls)
    if not controls['precision_gate_passed']:
        raise AssertionError('Paired fp32/fp64 error exceeds frozen numerical tolerance')
    return {'task': task, 'files': {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()},
            'precision_max_error': controls['fp32_fp64_max_error_overall']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = json.loads(freeze_path.read_text())
    cases_path, cases = load_cases(freeze, freeze_path)
    lock_path = resolve_path(freeze_path.parent, freeze.get('asset_lock_file', freeze.get('asset_lock_path', 'COHORT_ASSET_LOCK.json')))
    if 'asset_lock_sha256' in freeze and digest(lock_path) != freeze['asset_lock_sha256']:
        raise ValueError('Cohort asset lock differs from freeze')
    lock = json.loads(lock_path.read_text())
    bound_paths = [freeze_path,cases_path,lock_path,HERE/'confirm.py',HERE/'runtime.py',HERE/'token_split.py',
                   HERE/'CONFIRMATION.md',HERE/'analyze_confirmation.py',HERE/'prepare_confirmation.py',
                   HERE/'ASSET_LOCK.json',HERE.parent/'SOURCE_LOCK.json',ROOT/'examples/causal_preflight.py',
                   ROOT/'src/causal_decidability/design.py',HERE/'qualify.py']
    for path in bound_paths:
        assert_committed(path)
    original_lock = json.loads((HERE.parent/'SOURCE_LOCK.json').read_text())
    for relative, expected in original_lock['files'].items():
        if digest(HERE.parent/'upstream'/relative) != expected['sha256']:
            raise ValueError('Original source hash mismatch: '+relative)
    dependency_lock = json.loads((HERE/'ASSET_LOCK.json').read_text())
    dependency = 'utils/minGPT/utils.py'
    if digest(HERE/'cache'/dependency) != dependency_lock['files'][dependency]['sha256']:
        raise ValueError('Cached upstream model dependency hash mismatch')
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing run')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    manifest = {
        'status': 'running', 'started_at': utcnow(),
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'freeze_sha256': digest(freeze_path), 'cases_sha256': digest(cases_path),
        'asset_lock_sha256': digest(lock_path),
        'source_hashes': {str(p.relative_to(ROOT)):digest(p) for p in bound_paths},
        'environment': {'torch': torch.__version__, 'python': platform.python_version(), 'platform': platform.platform(), 'device':'cpu','threads':1},
        'tasks': [],
    }
    dump(args.output/'manifest.json', manifest)
    try:
        for task in freeze['tasks']:
            name = task['model_id'] + '_head' + str(task['head'])
            print('Measuring',name,flush=True)
            result = run_task(task,cases,lock,args.output/'tasks'/name,float(freeze.get('numerical_tolerance',.001)))
            manifest['tasks'].append(result)
            manifest['elapsed_seconds'] = time.monotonic()-started
            dump(args.output/'manifest.json',manifest)
            print('Completed',name,flush=True)
        manifest['status'] = 'completed'
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = repr(error)
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic()-started
        manifest['finished_at'] = utcnow()
        dump(args.output/'manifest.json',manifest)


if __name__ == '__main__':
    main()
