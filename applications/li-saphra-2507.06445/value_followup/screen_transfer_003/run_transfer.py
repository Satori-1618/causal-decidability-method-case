"""Validate by default; execute the six-head screen only after recorded release.

No torch import, checkpoint deserialization, model creation or forward occurs
unless --execute passes the committed plan/review/user-release checks first.
"""
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

from prepare_transfer import HERE, VALUE, NATIVE, ROOT, head_key, sha, read_rows, validate_inputs
from verify_plan import verify_contract, verify_hashes, require_execution_release

DTYPES = ('float32', 'float64')
ANCHORS = ('neg_20_0', 'pos_28_0')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    temporary.replace(path)


def committed(path):
    relative = str(path.resolve().relative_to(ROOT))
    payload = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT)
    if payload != path.read_bytes():
        raise ValueError('Uncommitted execution dependency: ' + relative)


def preflight(inputs, execute=False):
    """Source/asset/input checks are shared failures, before output or models."""
    plan = json.loads((HERE / 'plan.json').read_text())
    release = json.loads((HERE / 'EXECUTION_RELEASE.json').read_text())
    verify_contract(plan)
    if execute:
        require_execution_release(plan, release)
    lock = json.loads((HERE / 'SOURCE_LOCK.json').read_text())
    verify_hashes(ROOT, lock['files'])
    required = list(HERE.glob('*.py')) + [HERE / name for name in
               ('plan.json', 'PROTOCOL.md', 'EXECUTION_RELEASE.json')]
    required += [p for p in inputs.rglob('*') if p.is_file()]
    for path in required:
        relative = str(path.resolve().relative_to(ROOT))
        if relative not in lock['files']:
            raise ValueError('Execution/input source absent from SOURCE_LOCK: ' + relative)
    sources = dict(lock['files'])
    sources[str((HERE / 'SOURCE_LOCK.json').relative_to(ROOT))] = sha(HERE / 'SOURCE_LOCK.json')
    if execute:
        for relative in sources:
            committed(ROOT / relative)
        reviewed = release['final_review']['reviewed_commit']
        subprocess.run(['git', 'merge-base', '--is-ancestor', reviewed, 'HEAD'], cwd=ROOT, check=True)
        for relative in sources:
            path = ROOT / relative
            if path.suffix == '.md' or path in (HERE / 'plan.json', HERE / 'EXECUTION_RELEASE.json', HERE / 'SOURCE_LOCK.json'):
                continue
            reviewed_bytes = subprocess.check_output(['git', 'show', reviewed + ':' + relative], cwd=ROOT)
            if reviewed_bytes != path.read_bytes():
                raise ValueError('Computational source/input changed since final review: ' + relative)
        reviewed_plan = json.loads(subprocess.check_output(
            ['git', 'show', reviewed + ':' + str((HERE / 'plan.json').relative_to(ROOT))], cwd=ROOT))
        for key in ('cohort', 'reference', 'screen', 'intervention', 'inference', 'cost'):
            if reviewed_plan[key] != plan[key]:
                raise ValueError('Scientific plan changed since final review: ' + key)
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        remote = subprocess.check_output(['git', 'ls-remote', '--heads', 'origin', 'refs/heads/' + branch],
                                         cwd=ROOT, text=True).split()
        if not branch or not remote or remote[0] != head:
            raise ValueError('Execution release commit is not the published branch head')
    # Cached Python dependencies must be checked before their eventual import.
    application_lock = json.loads((VALUE.parent / 'SOURCE_LOCK.json').read_text())
    for relative, entry in application_lock['files'].items():
        if sha(VALUE.parent / 'upstream' / relative) != entry['sha256']:
            raise ValueError('Vendored upstream changed: ' + relative)
    asset_lock = json.loads((NATIVE / 'ASSET_LOCK.json').read_text())
    dependency = 'utils/minGPT/utils.py'
    if execute and sha(NATIVE / 'cache' / dependency) != asset_lock['files'][dependency]['sha256']:
        raise ValueError('Cached upstream Python dependency changed')
    cohort_lock = json.loads((NATIVE / 'frozen/confirmation_001/COHORT_ASSET_LOCK.json').read_text())
    for task in plan['cohort']:
        expected = cohort_lock['files'][task['checkpoint']]['sha256']
        if task['sha256'] != expected or (execute and sha(NATIVE / 'cache' / task['checkpoint']) != expected):
            raise ValueError('Shared checkpoint integrity failure: ' + task['model'])
    reference = 'data/model_weights/run_{0}/run_{0}_checkpoint_5.pt'.format(plan['reference']['model'])
    if execute and sha(NATIVE / 'cache' / reference) != plan['reference']['checkpoint_sha256']:
        raise ValueError('Reference checkpoint changed')
    preparation, payloads = validate_inputs(plan, inputs)
    return plan, release, preparation, payloads, sources, cohort_lock


def normalization(vectors, normalized_cutoff):
    """Float64 scalar algebra on stored weights; never a forward-derived fit."""
    if set(vectors) != {'w_false', 'w_true', 'ln_gamma', 'ln_beta'}:
        raise ValueError('Unexpected normalization inputs')
    if any(len(v) != 64 or not all(math.isfinite(x) for x in v) for v in vectors.values()):
        raise ValueError('Readout normalization requires finite width-64 vectors')
    w = [a - b for a, b in zip(vectors['w_false'], vectors['w_true'])]
    center = sum(a * b for a, b in zip(w, vectors['ln_beta']))
    weighted = [a * b for a, b in zip(w, vectors['ln_gamma'])]
    mean = sum(weighted) / 64
    radius = math.sqrt(64) * math.sqrt(sum((x - mean) ** 2 for x in weighted))
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Degenerate model LayerNorm range')
    return {**vectors, 'center': center, 'radius': radius, 'normalized_cutoff': normalized_cutoff,
            'raw_cutoff': center + radius * normalized_cutoff}


def load_normalizations(plan, torch):
    def one(model, digest):
        path = NATIVE / 'cache/data/model_weights' / ('run_' + model) / ('run_' + model + '_checkpoint_5.pt')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if ('lm_head.bias' in state or tuple(state['lm_head.weight'].shape) != (5, 64)
                or tuple(state['transformer.ln_f.weight'].shape) != (64,)
                or tuple(state['transformer.ln_f.bias'].shape) != (64,)):
            raise ValueError('Unsupported normalization architecture: ' + model)
        # Cast each stored vector before subtraction, never after.
        vectors = {'w_false': state['lm_head.weight'][0].double().tolist(),
                   'w_true': state['lm_head.weight'][1].double().tolist(),
                   'ln_gamma': state['transformer.ln_f.weight'].double().tolist(),
                   'ln_beta': state['transformer.ln_f.bias'].double().tolist()}
        return {**normalization(vectors, plan['reference']['normalized_cutoff']),
                'model_id': model, 'checkpoint_sha256': digest}
    reference = one(plan['reference']['model'], plan['reference']['checkpoint_sha256'])
    for key in ('center', 'radius'):
        if not math.isclose(reference[key], plan['reference'][key], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Reference static algebra differs from the plan')
    return {'written_before_native_at': now(), 'reference': reference,
            'heads': {head_key(t): one(t['model'], t['sha256']) for t in plan['cohort']}}


def select(margins, norm, quota):
    z = [(m - norm['center']) / norm['radius'] for m in margins]
    labels = ['accepted' if value < norm['normalized_cutoff'] else 'rejected' for value in z]
    selected = {s: [i for i, value in enumerate(labels) if value == s][:quota]
                for s in ('accepted', 'rejected')}
    return z, labels, selected


class ForwardLedger:
    """Record actual started and returned GPT forwards, including failed calls."""
    def __init__(self, path):
        self.path = path
        self.stage = 'native'
        self.data = {'sequence_forwards_attempted': 0, 'sequence_forwards_completed': 0,
                     'forward_batches_attempted': 0, 'forward_batches_completed': 0,
                     'baseline_candidates_completed_by_dtype': {t: 0 for t in DTYPES},
                     'anchor_jobs_completed_by_dtype': {t: 0 for t in DTYPES},
                     'failed_forward_internal_progress_unknown': False}
        self.pending = {}

    def event(self, value):
        with self.path.open('a') as handle:
            handle.write(json.dumps({**value, 'at': now()}) + '\n')
            handle.flush()

    def attach(self, model, tag):
        def before(module, args):
            size = int(args[0].shape[0])
            self.pending[tag] = size
            self.data['sequence_forwards_attempted'] += size
            self.data['forward_batches_attempted'] += 1
            self.event({'event': 'started', 'dtype': tag, 'stage': self.stage, 'sequences': size})
        def after(module, args, output):
            size = self.pending.pop(tag)
            self.data['sequence_forwards_completed'] += size
            self.data['forward_batches_completed'] += 1
            self.event({'event': 'completed', 'dtype': tag, 'stage': self.stage, 'sequences': size})
        return [model.register_forward_pre_hook(before), model.register_forward_hook(after)]

    def snapshot(self):
        result = copy.deepcopy(self.data)
        result['failed_forward_internal_progress_unknown'] = bool(self.pending)
        return result


def save_status(context):
    directory, status = context['directory'], context['status']
    status['cost'] = context['ledger'].snapshot()
    status['output_hashes'] = {p.name: sha(p) for p in sorted(directory.iterdir())
                              if p.is_file() and p.name != 'status.json'}
    write_json(directory / 'status.json', status)


def failed(context, error):
    context['status'].update(status='technical_failure', finished_at=now(),
                             error_type=type(error).__name__, error=str(error))
    (context['directory'] / 'failure.txt').write_text(traceback.format_exc())
    save_status(context)


def screen_head(context, payload, norm, plan, load_runtime, torch, choose_positions):
    candidates, templates = payload
    directory, status, ledger = context['directory'], context['status'], context['ledger']
    task = status['task']
    runtime_task = {'model_id': task['model_id'], 'n_layer': task['n_layer'],
                    'n_head': task['model_heads'], 'head': task['head']}
    runtimes, native = {}, {}
    context['runtimes'] = runtimes
    for tag in DTYPES:
        runtime = load_runtime(runtime_task, context['cohort_lock'], getattr(torch, tag))
        runtimes[tag] = runtime
        context['handles'].extend(ledger.attach(runtime.model, tag))
        native[tag] = runtime.run([r['recipient'] for r in candidates])
        ledger.data['baseline_candidates_completed_by_dtype'][tag] = len(candidates)
    margins = {tag: native[tag].margins.tolist() for tag in DTYPES}
    if any(not math.isfinite(value) for values in margins.values() for value in values):
        raise AssertionError('Non-finite native margin')
    selections = {tag: select(margins[tag], norm, plan['screen']['families_per_stratum']) for tag in DTYPES}
    z64, labels, selected = selections['float64']
    chosen = choose_positions([r['recipient'] for r in candidates], native['float64'])
    selected_set = set(selected['accepted'] + selected['rejected'])
    screening = [{**row, 'margin_float32': margins['float32'][i], 'margin_float64': margins['float64'][i],
                  'normalized_margin_float32': selections['float32'][0][i], 'normalized_margin_float64': z64[i],
                  'screen_accept_float32': selections['float32'][1][i] == 'accepted',
                  'screen_accept_float64': labels[i] == 'accepted', 'stratum': labels[i],
                  'selected': i in selected_set, 'recipient_position': chosen[i]}
                 for i, row in enumerate(candidates)]
    write_rows(directory / 'screening.jsonl', screening)
    error = max(abs(a - b) for a, b in zip(margins['float32'], margins['float64']))
    mismatches = sum(a != b for a, b in zip(selections['float32'][1], labels))
    controls = {'native_screen': {'max_dtype_margin_difference': error,
                'dtype_classification_mismatches': mismatches,
                'cases_per_dtype': len(candidates),
                'pool_counts': {s: labels.count(s) for s in selected},
                'selected_counts': {s: len(v) for s, v in selected.items()}}}
    context['controls'] = controls
    write_json(directory / 'controls.json', controls)
    if error > plan['intervention']['numerical_allowance_nat'] or mismatches:
        raise AssertionError('Native precision/classification failed; no case deletion')
    if any(len(indices) != plan['screen']['families_per_stratum'] for indices in selected.values()):
        status.update(status='insufficient_yield', finished_at=now())
        save_status(context)
        return
    families = []
    for template, (stratum, i) in zip(templates, [(s, i) for s in ('accepted', 'rejected') for i in selected[s]]):
        family = copy.deepcopy(template)
        family.update({**{k: screening[i][k] for k in ('candidate_id', 'candidate_index', 'stratum',
                       'recipient', 'recipient_position', 'margin_float32', 'margin_float64')},
                       'family_id': candidates[i]['candidate_id'], 'donor_template_id': template['family_id']})
        families.append(family)
    context['families'] = families
    write_rows(directory / 'selected_families.jsonl', families)
    write_json(directory / 'recipient_selection.json', [
        {'family_id': f['family_id'], 'recipient': f['recipient'], 'recipient_position': f['recipient_position'],
         'native_margin': margins['float64'][f['candidate_index']],
         'native_eos_attention': native['float64'].attention[f['candidate_index']].tolist()}
        for f in families])
    write_json(directory / 'screening_receipt.json', {'written_before_transfers_at': now(),
        'screening_sha256': sha(directory / 'screening.jsonl'),
        'selected_families_sha256': sha(directory / 'selected_families.jsonl'),
        'recipient_selection_sha256': sha(directory / 'recipient_selection.json'),
        'normalization': norm, 'source_hashes': context['sources'],
        'selection_rule': 'First64 per frozen stratum; accepted templates0..63, rejected64..127'})
    status['status'] = 'screened'
    save_status(context)


def anchor_head(context, plan, transfer):
    directory, status = context['directory'], context['status']
    families, ledger = context['families'], context['ledger']
    ledger.stage = 'anchors'
    status['anchors_measurement_started_at'] = now()
    save_status(context)
    keys = ('family_id', 'candidate_id', 'candidate_index', 'stratum', 'recipient', 'recipient_position',
            'donor_template_id', 'margin_float32', 'margin_float64')
    records = [{**{k: f[k] for k in keys}, 'cells': {}} for f in families]
    jobs = [(i, donor) for i, f in enumerate(families) for donor in f['donors'] if donor['cell'] in ANCHORS]
    context['controls']['anchors'] = {}
    for tag, runtime in context['runtimes'].items():
        result = transfer(runtime, [families[i]['recipient'] for i, _ in jobs],
                          [d['string'] for _, d in jobs], [families[i]['recipient_position'] for i, _ in jobs],
                          [d['position'] for _, d in jobs])
        ledger.data['anchor_jobs_completed_by_dtype'][tag] = len(jobs)
        context['controls']['anchors'][tag] = result.controls
        for (i, donor), snapshot in zip(jobs, result.snapshots()):
            records[i]['cells'].setdefault(donor['cell'], {'donor_metadata': donor})[tag] = snapshot
        write_rows(directory / 'cases.jsonl', records)
        write_json(directory / 'controls.json', context['controls'])
    maximum = 0.
    mismatches = 0
    for row in records:
        differences = {tag: row['cells'][ANCHORS[1]][tag]['patched_margin'] -
                            row['cells'][ANCHORS[0]][tag]['patched_margin'] for tag in DTYPES}
        if not all(math.isfinite(value) for value in differences.values()):
            raise AssertionError('Non-finite anchor contrast')
        discrepancy = abs(differences['float32'] - differences['float64'])
        maximum = max(maximum, discrepancy)
        labels = {tag: abs(value) > plan['intervention']['strict_gap_nat'] for tag, value in differences.items()}
        mismatches += labels['float32'] != labels['float64']
        row.update(anchor_gap_float32=abs(differences['float32']), anchor_gap_float64=abs(differences['float64']),
                   anchor_separating=labels['float64'], signed_anchor_dtype_discrepancy=discrepancy)
    write_rows(directory / 'cases.jsonl', records)
    context['controls']['anchor_precision'] = {'maximum_signed_contrast_dtype_difference': maximum,
                                              'separation_boundary_straddles': mismatches}
    write_json(directory / 'controls.json', context['controls'])
    if maximum > plan['intervention']['numerical_allowance_nat'] or mismatches:
        raise AssertionError('Anchor precision/boundary failed; no replacements')
    status.update(status='completed', finished_at=now())
    save_status(context)


def execute(output, plan, release, preparation, payloads, sources, cohort_lock):
    """Only called after preflight(..., execute=True); no runtime overrides."""
    import torch
    sys.path.insert(0, str(NATIVE))
    sys.path.insert(0, str(VALUE))
    from confirm import load_runtime
    from value_runtime import run_value_transfers
    spec = importlib.util.spec_from_file_location('transfer_previous_runner', VALUE / 'run.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = {'round': plan['round'], 'status': 'running', 'started_at': now(),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'reviewed_commit': release['final_review']['reviewed_commit'], 'source_hashes': sources,
                'preparation': preparation, 'plan_sha256': sha(HERE / 'plan.json'),
                'execution_release_sha256': sha(HERE / 'EXECUTION_RELEASE.json'), 'heads': [],
                'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                'device': 'cpu', 'threads': 1}}
    write_json(output / 'manifest.json', manifest)
    contexts = []
    try:
        norms = load_normalizations(plan, torch)
        write_json(output / 'normalization.json', norms)
        manifest['normalization_sha256'] = sha(output / 'normalization.json')
        write_json(output / 'manifest.json', manifest)
        # Finish and persist every screen before measuring the first anchor.
        for task in plan['cohort']:
            key = head_key(task)
            directory = output / 'heads' / key
            directory.mkdir(parents=True)
            status = {'head_key': key, 'status': 'running', 'started_at': now(),
                      'task': {'model_id': task['model'], 'head': task['head_one_based'],
                               'model_heads': task['model_heads'], 'n_layer': task['layer_one_based']},
                      'normalization': norms['heads'][key], 'source_hashes': sources}
            context = {'directory': directory, 'status': status, 'ledger': ForwardLedger(directory / 'forward_events.jsonl'),
                       'cohort_lock': cohort_lock, 'sources': sources, 'handles': []}
            contexts.append(context)
            save_status(context)
            try:
                screen_head(context, payloads[key], norms['heads'][key], plan, load_runtime, torch,
                            previous.choose_recipient_positions)
            except Exception as error:
                failed(context, error)
        manifest['all_screening_completed_at'] = now()
        write_json(output / 'manifest.json', manifest)
        for context in contexts:
            if context['status']['status'] == 'screened':
                try:
                    anchor_head(context, plan, run_value_transfers)
                except Exception as error:
                    failed(context, error)
        manifest['status'] = ('completed' if all(c['status']['status'] == 'completed' for c in contexts)
                              else 'completed_with_unavailable_heads')
    except BaseException as error:
        manifest.update(status='batch_failure', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for context in contexts:
            for handle in context['handles']:
                handle.remove()
            save_status(context)
        manifest['heads'] = [{'head_key': c['status']['head_key'], 'status': c['status']['status'],
                              'status_sha256': sha(c['directory'] / 'status.json')} for c in contexts]
        manifest.update(finished_at=now(), elapsed_seconds=time.monotonic() - start)
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
        print('PASS: locked sources and six regenerated input sets; no model imported or measured; cached assets checked only on execution')
        return
    execute(args.output, *frozen)


if __name__ == '__main__':
    main()
