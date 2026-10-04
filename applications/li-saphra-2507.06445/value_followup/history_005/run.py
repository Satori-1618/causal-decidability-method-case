"""A gated history-grid adapter over the unchanged value-transfer machinery.

Only 004's context-driven screen, eight-job transfer stage, raw-margin accessor
and atomic writers are reused. The scientific analysis and release are 005's.
Importing this module does not import torch or construct a neural model.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
VALUE = HERE.parent
NATIVE = VALUE.parent / 'native_followup'
ROOT = HERE.parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shared = load_module('history_005_shared_value_runner', VALUE / 'averaged_anchors_004/run.py')


def scientific_plan(plan):
    return {key: value for key, value in plan.items() if key not in ('status', 'execution_authorized', 'provenance')}


def execution_git_gate(plan, release, sources):
    reviewed = release['final_review']['reviewed_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', reviewed, 'HEAD'], cwd=ROOT, check=True)
    for relative in sources:
        path = ROOT / relative
        payload = path.read_bytes()
        if subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT) != payload:
            raise ValueError('Uncommitted execution dependency: ' + relative)
        if path.suffix == '.md' or path in (HERE / 'plan.json', HERE / 'EXECUTION_RELEASE.json', HERE / 'SOURCE_LOCK.json'):
            continue
        if subprocess.check_output(['git', 'show', reviewed + ':' + relative], cwd=ROOT) != payload:
            raise ValueError('Computational source/input changed since review: ' + relative)
    previous = json.loads(subprocess.check_output(
        ['git', 'show', reviewed + ':' + str((HERE / 'plan.json').relative_to(ROOT))], cwd=ROOT))
    if scientific_plan(previous) != scientific_plan(plan):
        raise ValueError('Scientific plan changed since final review')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', '--heads', 'origin', 'refs/heads/' + branch], cwd=ROOT, text=True).split()
    if not branch or not remote or remote[0] != head:
        raise ValueError('Execution-release commit is not the published branch head')


def preflight(inputs, execute=False):
    verifier = load_module('history_005_freeze_verifier', HERE / 'verify_freeze.py')
    plan = json.loads((HERE / 'plan.json').read_text())
    release = json.loads((HERE / 'EXECUTION_RELEASE.json').read_text())
    verifier.verify_contract(plan)
    if execute:
        verifier.require_release(plan, release)
    sources = verifier.verify_source_lock()
    required = list(HERE.glob('*.py')) + [HERE / name for name in ('plan.json', 'PROTOCOL.md', 'EXECUTION_RELEASE.json')]
    required += [path for path in inputs.rglob('*') if path.is_file()]
    required.append(VALUE / 'averaged_anchors_004/run.py')
    for path in required:
        if str(path.resolve().relative_to(ROOT)) not in sources:
            raise ValueError('Unbound source/input: ' + str(path))
    cohort = json.loads((NATIVE / 'frozen/confirmation_001/COHORT_ASSET_LOCK.json').read_text())
    model = plan['model']
    if cohort['files'][model['checkpoint']]['sha256'] != model['sha256']:
        raise ValueError('Checkpoint differs from inherited lock')
    if execute:
        execution_git_gate(plan, release, sources)
        for relative, entry in json.loads((VALUE.parent / 'SOURCE_LOCK.json').read_text())['files'].items():
            if shared.sha(VALUE.parent / 'upstream' / relative) != entry['sha256']:
                raise ValueError('Upstream source integrity failure: ' + relative)
        assets = json.loads((NATIVE / 'ASSET_LOCK.json').read_text())
        dependency = 'utils/minGPT/utils.py'
        if shared.sha(NATIVE / 'cache' / dependency) != assets['files'][dependency]['sha256']:
            raise ValueError('Cached upstream Python dependency changed')
        if shared.sha(NATIVE / 'cache' / model['checkpoint']) != model['sha256']:
            raise ValueError('Checkpoint integrity failure')
    preparer = load_module('history_005_input_preparer', HERE / 'prepare.py')
    preparation, candidates, families = preparer.validate_inputs(plan, inputs)
    if shared.sha(inputs / 'preparation.json') != plan['provenance']['prepared_inputs_sha256']:
        raise ValueError('Prepared input manifest differs from the frozen plan')
    return plan, release, sources, cohort, preparation, candidates, families


def freeze_forecasts(context, analysis):
    results, rows = [], []
    for record in context['records']:
        result = analysis.calibration_result(shared.role_margins(record, 'calibration'))
        results.append(result)
        record['calibration_analysis'] = result
        record['forecasts'] = result['forecasts']
        record['separating'] = result['separating']
        rows.append({'family_id': record['family_id'], **result})
    gate = analysis.start_rule(results)
    directory = context['directory']
    shared.write_rows(directory / 'forecasts.jsonl', rows)
    shared.write_rows(directory / 'calibration_cases.jsonl', context['records'])
    context['controls']['forecast_precision'] = {
        'maximum_signed_forecast_contrast_dtype_difference': max(r['forecast_contrast_dtype_error'] for r in results),
        'per_family_pair_states': [{'family_id': row['family_id'], 'pair_separation': row['pair_separation']} for row in rows]}
    shared.write_json(directory / 'controls.json', context['controls'])
    context['manifest'].update(separating_families=gate['separating'], target_start_passed=gate['start_targets'])
    shared.write_json(directory / 'forecast_receipt.json', {'written_before_targets_at': shared.now(),
        'forecasts_sha256': shared.sha(directory / 'forecasts.jsonl'),
        'calibration_cases_sha256': shared.sha(directory / 'calibration_cases.jsonl'),
        'screening_receipt_sha256': shared.sha(directory / 'screening_receipt.json'),
        'source_hashes': context['sources'], 'start_rule': gate})
    if not gate['start_targets']:
        context['manifest']['status'] = 'insufficient_design_yield'
    return gate['start_targets']


def analyze_targets(context, analysis):
    results = []
    for record in context['records']:
        result = analysis.family_result(shared.role_margins(record, 'calibration'), shared.role_margins(record, 'target'))
        if result['forecasts'] != record['forecasts'] or result['separating'] != record['separating']:
            raise AssertionError('Pre-target centered forecasts changed')
        record['analysis'] = result
        results.append(result)
    shared.write_rows(context['directory'] / 'cases.jsonl', context['records'])
    shared.write_json(context['directory'] / 'analysis_summary.json', analysis.cohort_result(results))


def execution_dependencies():
    """Load the same low-level operator only after the 005 release passes."""
    import torch
    sys.path.insert(0, str(NATIVE))
    sys.path.insert(0, str(VALUE))
    from confirm import load_runtime
    from value_runtime import run_value_transfers
    original = load_module('history_005_original_value_runner', VALUE / 'run.py')
    directory = VALUE / 'screen_transfer_003'
    sys.path.insert(0, str(directory))
    ledger_source = load_module('history_005_ledger_source', directory / 'run_transfer.py')
    analysis = load_module('history_005_analysis', HERE / 'analysis.py')
    return torch, load_runtime, run_value_transfers, original.choose_recipient_positions, ledger_source.ForwardLedger, analysis


def execute(output, plan, release, sources, cohort, preparation, candidates, templates):
    torch, load_runtime, transfer, choose_positions, Ledger, analysis = execution_dependencies()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(output / 'forward_events.jsonl')
    ledger.data.pop('anchor_jobs_completed_by_dtype')
    for role in ('calibration', 'target'):
        ledger.data[role + '_jobs_completed_by_dtype'] = {tag: 0 for tag in shared.DTYPES}
    manifest = {'round': plan['round'], 'status': 'running', 'started_at': shared.now(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'reviewed_commit': release['final_review']['reviewed_commit'], 'source_hashes': sources,
        'plan_sha256': shared.sha(HERE / 'plan.json'), 'execution_release_sha256': shared.sha(HERE / 'EXECUTION_RELEASE.json'),
        'preparation': preparation, 'model': plan['model'],
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu', 'threads': 1}}
    shared.write_json(output / 'manifest.json', manifest)
    context = {'plan': plan, 'directory': output, 'sources': sources, 'manifest': manifest,
               'controls': {}, 'ledger': ledger, 'runtimes': {}}
    handles, start = [], time.monotonic()
    try:
        model = plan['model']
        task = {'model_id': model['model'], 'n_layer': model['layer_one_based'],
                'n_head': model['model_heads'], 'head': model['head_one_based']}
        for tag in shared.DTYPES:
            runtime = load_runtime(task, cohort, getattr(torch, tag))
            context['runtimes'][tag] = runtime
            handles.extend(ledger.attach(runtime.model, tag))
        if not shared.screen(context, candidates, templates, choose_positions):
            return
        shared.transfer_stage(context, 'calibration', transfer)
        if not freeze_forecasts(context, analysis):
            return
        shared.transfer_stage(context, 'target', transfer)
        analyze_targets(context, analysis)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='technical_failure', error_type=type(error).__name__, error=str(error))
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        manifest.update(finished_at=shared.now(), elapsed_seconds=time.monotonic() - start, cost=ledger.snapshot(),
            output_hashes={p.name: shared.sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'})
        shared.write_json(output / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=HERE / 'inputs')
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.execute and (args.output is None or args.output.exists()):
        raise ValueError('Execution requires a new --output directory')
    frozen = preflight(args.inputs, execute=args.execute)
    if not args.execute:
        print('PASS: history005 frozen sources and regenerated inputs; no model imported or measured')
        return
    execute(args.output, *frozen)


if __name__ == '__main__':
    main()
