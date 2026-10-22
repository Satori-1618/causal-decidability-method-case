"""Validate the averaged-anchor freeze; measure only after explicit public release."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
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
DTYPES = ('float32', 'float64')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    temporary.replace(path)


def execution_git_gate(plan, release, sources):
    reviewed = release['final_review']['reviewed_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', reviewed, 'HEAD'], cwd=ROOT, check=True)
    for relative in sources:
        path = ROOT / relative
        if subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT) != path.read_bytes():
            raise ValueError('Execution dependency is not committed: ' + relative)
        if path.suffix == '.md' or path in (HERE / 'plan.json', HERE / 'EXECUTION_RELEASE.json', HERE / 'SOURCE_LOCK.json'):
            continue
        if subprocess.check_output(['git', 'show', reviewed + ':' + relative], cwd=ROOT) != path.read_bytes():
            raise ValueError('Computational source/input changed after final review: ' + relative)
    previous = json.loads(subprocess.check_output(
        ['git', 'show', reviewed + ':' + str((HERE / 'plan.json').relative_to(ROOT))], cwd=ROOT))
    for key in plan:
        if key not in ('status', 'execution_authorized', 'provenance') and previous.get(key) != plan[key]:
            raise ValueError('Scientific plan changed after review: ' + key)
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    published = subprocess.check_output(['git', 'ls-remote', '--heads', 'origin', 'refs/heads/' + branch],
                                        cwd=ROOT, text=True).split()
    if not branch or not published or published[0] != current:
        raise ValueError('Released execution commit is not the published branch head')


def preflight(inputs, execute=False):
    """No torch/cache deserialization or output creation in validation mode."""
    verifier = load_module('averaged_anchor_freeze_verifier', HERE / 'verify_freeze.py')
    plan = json.loads((HERE / 'plan.json').read_text())
    release = json.loads((HERE / 'EXECUTION_RELEASE.json').read_text())
    verifier.verify_contract(plan)
    if execute:
        verifier.require_release(plan, release)
    sources = verifier.verify_source_lock()
    model = plan['model']
    if (model['model'], model['layer_one_based'], model['head_one_based'], model['model_heads'],
            model['hidden_width'], model['head_width']) != ('a9g0io1r', 2, 1, 4, 64, 16):
        raise ValueError('This reference checkpoint requires four heads of width16')
    required = list(HERE.glob('*.py')) + [HERE / name for name in ('plan.json', 'PROTOCOL.md', 'EXECUTION_RELEASE.json')]
    required += [path for path in inputs.rglob('*') if path.is_file()]
    for path in required:
        if str(path.resolve().relative_to(ROOT)) not in sources:
            raise ValueError('Source/input is absent from the freeze: ' + str(path))
    cohort = json.loads((NATIVE / 'frozen/confirmation_001/COHORT_ASSET_LOCK.json').read_text())
    if cohort['files'][model['checkpoint']]['sha256'] != model['sha256']:
        raise ValueError('Reference checkpoint differs from inherited source lock')
    if execute:
        execution_git_gate(plan, release, sources)
        for relative, entry in json.loads((VALUE.parent / 'SOURCE_LOCK.json').read_text())['files'].items():
            if sha(VALUE.parent / 'upstream' / relative) != entry['sha256']:
                raise ValueError('Upstream source integrity failure: ' + relative)
        assets = json.loads((NATIVE / 'ASSET_LOCK.json').read_text())
        dependency = 'utils/minGPT/utils.py'
        if sha(NATIVE / 'cache' / dependency) != assets['files'][dependency]['sha256']:
            raise ValueError('Cached upstream Python dependency changed')
        if sha(NATIVE / 'cache' / model['checkpoint']) != model['sha256']:
            raise ValueError('Checkpoint integrity failure')
    preparer = load_module('averaged_anchor_input_preparer', HERE / 'prepare.py')
    preparation, candidates, families = preparer.validate_inputs(plan, inputs)
    return plan, release, sources, cohort, preparation, candidates, families


def selected_indices(margins, cutoff, quota):
    return [i for i, value in enumerate(margins) if value < cutoff][:quota]


def screen(context, candidates, templates, choose_positions):
    plan, directory = context['plan'], context['directory']
    native, margins = {}, {}
    strings = [row['recipient'] for row in candidates]
    for tag, runtime in context['runtimes'].items():
        result = runtime.run(strings)
        native[tag] = result
        margins[tag] = result.margins.tolist()
        context['ledger'].data['baseline_candidates_completed_by_dtype'][tag] = len(strings)
        write_rows(directory / ('native_' + tag + '.jsonl'),
                   [{**row, 'margin': margin} for row, margin in zip(candidates, margins[tag])])
    if any(not math.isfinite(v) for values in margins.values() for v in values):
        raise AssertionError('Non-finite native margin')
    cutoff = plan['screen']['strict_cutoff_nat']
    labels = {tag: [value < cutoff for value in values] for tag, values in margins.items()}
    selected = selected_indices(margins['float64'], cutoff, plan['sampling']['families'])
    chosen = choose_positions(strings, native['float64'])
    chosen_set = set(selected)
    rows = [{**row, 'margin_float32': margins['float32'][i], 'margin_float64': margins['float64'][i],
             'screen_accept_float32': labels['float32'][i], 'screen_accept_float64': labels['float64'][i],
             'selected': i in chosen_set, 'recipient_position': chosen[i]} for i, row in enumerate(candidates)]
    write_rows(directory / 'screening.jsonl', rows)
    error = max(abs(a - b) for a, b in zip(margins['float32'], margins['float64']))
    mismatches = sum(a != b for a, b in zip(labels['float32'], labels['float64']))
    context['controls']['native_screen'] = {'max_dtype_margin_difference': error,
        'dtype_classification_mismatches': mismatches, 'cases_per_dtype': len(candidates),
        'accepted_pool_count': sum(labels['float64']), 'selected_count': len(selected)}
    write_json(directory / 'controls.json', context['controls'])
    if error > plan['intervention']['numerical_allowance_nat'] or mismatches:
        raise AssertionError('Native dtype or screen-label gate failed; no case replacement')
    if len(selected) < plan['sampling']['families']:
        context['manifest']['status'] = 'insufficient_screening_yield'
        return False
    families = []
    for template, index in zip(templates, selected):
        family = copy.deepcopy(template)
        family.update({**{key: rows[index][key] for key in ('candidate_id', 'candidate_index', 'recipient',
                       'recipient_position', 'margin_float32', 'margin_float64')},
                       'family_id': rows[index]['candidate_id'], 'donor_template_id': template['family_id']})
        families.append(family)
    context['families'] = families
    write_rows(directory / 'selected_families.jsonl', families)
    write_json(directory / 'recipient_selection.json', [
        {'family_id': f['family_id'], 'recipient': f['recipient'], 'recipient_position': f['recipient_position'],
         'native_margin': margins['float64'][f['candidate_index']],
         'native_eos_attention': native['float64'].attention[f['candidate_index']].tolist()} for f in families])
    write_json(directory / 'screening_receipt.json', {'written_before_calibration_at': now(),
        'screening_sha256': sha(directory / 'screening.jsonl'),
        'selected_families_sha256': sha(directory / 'selected_families.jsonl'),
        'recipient_selection_sha256': sha(directory / 'recipient_selection.json'),
        'source_hashes': context['sources']})
    return True


def transfer_stage(context, role, transfer):
    """All families, including nonseparating families, receive the stage."""
    directory, families = context['directory'], context['families']
    context['ledger'].stage = role
    context['manifest'][role + '_measurement_started_at'] = now()
    write_json(directory / 'manifest.json', context['manifest'])
    if 'records' not in context:
        keys = ('family_id', 'candidate_id', 'candidate_index', 'recipient', 'recipient_position',
                'donor_template_id', 'margin_float32', 'margin_float64')
        context['records'] = [{**{key: f[key] for key in keys}, 'cells': {}} for f in families]
    jobs = [(i, donor) for i, family in enumerate(families) for donor in family['donors'] if donor['role'] == role]
    if len(jobs) != len(families) * 8:
        raise ValueError('Incomplete frozen stage grid')
    context['controls'][role] = {}
    for tag, runtime in context['runtimes'].items():
        result = transfer(runtime, [families[i]['recipient'] for i, _ in jobs],
                          [d['string'] for _, d in jobs], [families[i]['recipient_position'] for i, _ in jobs],
                          [d['position'] for _, d in jobs])
        snapshots = result.snapshots()
        if len(snapshots) != len(jobs):
            raise AssertionError('Transfer result dropped a job')
        context['ledger'].data[role + '_jobs_completed_by_dtype'][tag] = len(jobs)
        context['controls'][role][tag] = result.controls
        for (i, donor), snapshot in zip(jobs, snapshots):
            context['records'][i]['cells'].setdefault(donor['cell'], {'donor_metadata': donor})[tag] = snapshot
        write_rows(directory / ('calibration_cases.jsonl' if role == 'calibration' else 'cases.jsonl'), context['records'])
        write_json(directory / 'controls.json', context['controls'])


def role_margins(row, role):
    return {tag: {name: cell[tag]['patched_margin'] for name, cell in row['cells'].items()
                  if cell['donor_metadata']['role'] == role} for tag in DTYPES}


def freeze_forecasts(context, analysis):
    directory = context['directory']
    results, forecasts = [], []
    for row in context['records']:
        result = analysis.calibration_result(role_margins(row, 'calibration'))
        results.append(result)
        predictions = result['forecasts']
        gaps = {tag: max(abs(values['B_avg'][name] - values['P_avg'][name]) for name in values['B_avg'])
                for tag, values in predictions.items()}
        row.update(forecasts=predictions, separation_gap_float32=gaps['float32'],
                   separation_gap_float64=gaps['float64'], separating=result['separating'])
        forecasts.append({key: row[key] for key in ('family_id', 'forecasts', 'separation_gap_float32',
                                                    'separation_gap_float64', 'separating')})
    # All candidate arithmetic and its gates come from the shipped analysis.
    gate = analysis.start_rule(results)
    write_rows(directory / 'forecasts.jsonl', forecasts)
    write_rows(directory / 'calibration_cases.jsonl', context['records'])
    # Amendment 002: a separate calibration-only diagnostic, never an extra
    # candidate in the frozen start rule or inferential error family.
    descriptive = load_module('averaged_004_descriptives', HERE / 'descriptives.py')
    context['descriptive_forecasts'] = [
        {'family_id': row['family_id'],
         'forecasts': descriptive.forecasts(role_margins(row, 'calibration'))}
        for row in context['records']]
    write_rows(directory / 'descriptive_forecasts.jsonl', context['descriptive_forecasts'])
    context['controls']['forecast_precision'] = {
        'maximum_signed_contrast_dtype_difference': max(r['forecast_contrast_dtype_error'] for r in results),
        'separation_boundary_straddles': 0}
    write_json(directory / 'controls.json', context['controls'])
    context['manifest']['separating_families'] = gate['separating']
    context['manifest']['target_start_passed'] = gate['start_targets']
    write_json(directory / 'forecast_receipt.json', {'written_before_targets_at': now(),
        'forecasts_sha256': sha(directory / 'forecasts.jsonl'),
        'descriptive_forecasts_sha256': sha(directory / 'descriptive_forecasts.jsonl'),
        'calibration_cases_sha256': sha(directory / 'calibration_cases.jsonl'),
        'screening_receipt_sha256': sha(directory / 'screening_receipt.json'),
        'source_hashes': context['sources'], 'start_rule': gate})
    if not gate['start_targets']:
        context['manifest']['status'] = 'insufficient_design_yield'
    return gate['start_targets']


def target_precision(context, analysis):
    results = []
    for row in context['records']:
        result = analysis.family_result(role_margins(row, 'calibration'), role_margins(row, 'target'))
        if result['forecasts'] != row['forecasts'] or result['separating'] != row['separating']:
            raise AssertionError('A frozen pre-target forecast changed')
        row['analysis'] = result
        results.append(result)
    context['controls']['target_precision'] = {
        'maximum_prediction_error_dtype_difference': max(max(r['prediction_error_dtype_discrepancy'].values()) for r in results),
        'maximum_same_cell_signed_gap_dtype_difference': max(r['within_cell_dtype_discrepancy'] for r in results),
        'same_cell_boundary_straddles': sum(len(r['within_cell_numerically_unresolved']) for r in results),
        'same_cell_boundary_straddle_families': sum(bool(r['within_cell_numerically_unresolved']) for r in results)}
    write_rows(context['directory'] / 'cases.jsonl', context['records'])
    write_json(context['directory'] / 'controls.json', context['controls'])


def descriptive_outputs(context):
    """Add-on arithmetic only; cannot revise the primary decisions or gates."""
    descriptive = load_module('averaged_004_descriptives', HERE / 'descriptives.py')
    rows = []
    for row, frozen in zip(context['records'], context['descriptive_forecasts']):
        result = descriptive.family_result(role_margins(row, 'calibration'), role_margins(row, 'target'))
        if frozen['family_id'] != row['family_id'] or result['forecasts'] != frozen['forecasts']:
            raise AssertionError('A descriptive pre-target forecast changed')
        rows.append(dict(result, family_id=row['family_id'], separating=row['separating']))
    write_rows(context['directory'] / 'descriptive_cases.jsonl', rows)
    write_json(context['directory'] / 'descriptive_summary.json', descriptive.cohort_result(rows))


def execution_dependencies():
    """Only called after the public execution-release gate."""
    import torch
    sys.path.insert(0, str(NATIVE))
    sys.path.insert(0, str(VALUE))
    from confirm import load_runtime
    from value_runtime import run_value_transfers
    previous = load_module('averaged_004_previous_value_runner', VALUE / 'run.py')
    transfer_dir = VALUE / 'screen_transfer_003'
    sys.path.insert(0, str(transfer_dir))
    transfer = load_module('averaged_004_previous_transfer_runner', transfer_dir / 'run_transfer.py')
    analysis = load_module('averaged_004_analysis', HERE / 'analysis.py')
    return torch, load_runtime, run_value_transfers, previous.choose_recipient_positions, transfer.ForwardLedger, analysis


def execute(output, plan, release, sources, cohort, preparation, candidates, templates):
    torch, load_runtime, transfer, choose_positions, Ledger, analysis = execution_dependencies()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(output / 'forward_events.jsonl')
    ledger.data.pop('anchor_jobs_completed_by_dtype')
    for role in ('calibration', 'target'):
        ledger.data[role + '_jobs_completed_by_dtype'] = {tag: 0 for tag in DTYPES}
    manifest = {'round': plan['round'], 'status': 'running', 'started_at': now(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'reviewed_commit': release['final_review']['reviewed_commit'], 'source_hashes': sources,
        'plan_sha256': sha(HERE / 'plan.json'), 'execution_release_sha256': sha(HERE / 'EXECUTION_RELEASE.json'),
        'preparation': preparation, 'model': plan['model'],
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu', 'threads': 1}}
    write_json(output / 'manifest.json', manifest)
    context = {'plan': plan, 'directory': output, 'sources': sources, 'manifest': manifest,
               'controls': {}, 'ledger': ledger, 'runtimes': {}}
    handles = []
    start = time.monotonic()
    try:
        model = plan['model']
        task = {'model_id': model['model'], 'n_layer': model['layer_one_based'],
                'n_head': model['model_heads'], 'head': model['head_one_based']}
        for tag in DTYPES:
            runtime = load_runtime(task, cohort, getattr(torch, tag))
            context['runtimes'][tag] = runtime
            handles.extend(ledger.attach(runtime.model, tag))
        if not screen(context, candidates, templates, choose_positions):
            return
        transfer_stage(context, 'calibration', transfer)
        if not freeze_forecasts(context, analysis):
            return
        transfer_stage(context, 'target', transfer)
        target_precision(context, analysis)
        write_json(output / 'analysis_summary.json', analysis.cohort_result([row['analysis'] for row in context['records']]))
        descriptive_outputs(context)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='technical_failure', error_type=type(error).__name__, error=str(error))
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        manifest.update(finished_at=now(), elapsed_seconds=time.monotonic() - start, cost=ledger.snapshot(),
                        output_hashes={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'})
        write_json(output / 'manifest.json', manifest)


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
        print('PASS: frozen sources and regenerated inputs; no model imported or measured')
        return
    execute(args.output, *frozen)


if __name__ == '__main__':
    main()
