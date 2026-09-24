"""Prospective 512-family amendment; immutable original 3A evidence is retained.

The inference implementation is versioned from run_donor_factor.py. Only the
planning grid and its provenance/authorization change, not the instrument or analyzer.
"""
import argparse
import copy
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from run_query_route import configure, load_model, qualify, save, validate_source
from makelov_read_source import sha256
from run_donor_factor import CONTRACT as ORIGINAL_CONTRACT, code_files as original_code_files

CONTRACT = copy.deepcopy(ORIGINAL_CONTRACT)
CONTRACT['planning']['n_grid'].append(512)
AMENDMENT_ID = 'round3a-512-v1'
ORIGINAL_STOP_SHA256 = 'fa30e74a514e91a9eeac28930384addfa1ca9ae73b871d9423defd2312a40a28'


def amendment_binding():
    protocol = ROOT/'DONOR_FACTOR_512_AMENDMENT.md'
    original = ROOT/'results/donor_factor_planning/planning.json'
    if sha256(original) != ORIGINAL_STOP_SHA256:
        raise RuntimeError('original planning STOP changed')
    old = json.loads(original.read_text())
    if old['status'] != 'STOP' or old['selected_n'] is not None:
        raise RuntimeError('amendment requires the preserved original STOP')
    return {'id': AMENDMENT_ID,
            'protocol': {'path': str(protocol.relative_to(REPO)), 'sha256': sha256(protocol)},
            'original_stop': {'path': str(original.relative_to(REPO)),
                              'sha256': ORIGINAL_STOP_SHA256}}


def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def code_files():
    return [*original_code_files(), Path(__file__).resolve(),
            ROOT/'src/donor_factor_planning_512.py', ROOT/'DONOR_FACTOR_512_AMENDMENT.md']


def validate_planning(decision, pilot_cases_path):
    n = decision.get('selected_n')
    if (decision.get('schema_version') != 'donor-factor-planning-512-v1'
            or decision.get('amendment') != amendment_binding()
            or decision.get('settings', {}).get('candidate_sizes') != CONTRACT['planning']['n_grid']
            or decision.get('status') != 'SELECTED'
            or decision.get('protocol_matches_frozen_simulation_counts') is not True
            or decision.get('protocol_matches_development_size') is not True
            or decision.get('n_development_units') != 32
            or n != 512):
        raise RuntimeError('planning did not pass the frozen selection rule')
    candidate = next((c for c in decision.get('candidates', []) if c.get('n') == n), None)
    if (not candidate or candidate.get('passes') is not True
            or candidate['coverage'].get('passes') is not True
            or candidate['normal_reference'].get('passes') is not True
            or not all(candidate['bootstrap_checks'][d].get('passes') is True
                       for d in ('gaussian', 'empirical'))):
        raise RuntimeError('selected n lacks all declared power checks')
    binding = decision.get('development_binding', {})
    original = json.loads((ROOT/'results/donor_factor_planning/planning.json').read_text())
    if (binding != original['development_binding']
            or decision.get('candidates', [])[:4] != original['candidates']):
        raise RuntimeError('amendment changed the development evidence or original size calculations')
    directory = pilot_cases_path.resolve().parent
    for name in ('cases.json', 'records.jsonl', 'summary.json', 'manifest.json'):
        if binding.get(name) != sha256(directory/name):
            raise RuntimeError(f'planning is not bound to this development {name}')
    if decision.get('planner_code_sha256') != sha256(ROOT/'src/donor_factor_planning_512.py'):
        raise RuntimeError('planner code differs from planning result')
    return n


def prepare(args):
    import numpy as np
    from makelov_read_source import decompose_direction, load_published_vector
    from donor_factor_sampling import historical_exclusions, sample_iid_families
    if args.stage != 'confirmation':
        raise RuntimeError('this amendment only authorizes fresh confirmation after planning')
    if args.output.exists():
        raise RuntimeError('output exists; preserve old artifacts')
    qualification = ROOT/'results/donor_factor_environment/qualification.json'
    if not qualification.exists() or not json.loads(qualification.read_text())['passed']:
        raise RuntimeError('historical runtime qualification must pass first')
    source = validate_source(args.source)
    development_cases, planning_input = [], None
    if args.stage == 'development':
        n, seed = 32, 24092431
    else:
        if not args.pilot_cases or not args.planning:
            raise RuntimeError('confirmation needs pilot cases and its planning decision')
        development_cases = json.loads(args.pilot_cases.read_text())
        if len(development_cases) != 32:
            raise RuntimeError('expected all 32 development families')
        decision = json.loads(args.planning.read_text())
        n = validate_planning(decision, args.pilot_cases)
        seed = 24092432
        planning_input = {'path': str(args.planning.relative_to(REPO)),
                          'sha256': sha256(args.planning), 'selected_n': n}
    model, tok, snapshot = load_model()
    excluded, provenance = historical_exclusions(ROOT, development_cases=development_cases)
    cases, sampling = sample_iid_families(
        args.source, tok, n=n, seed=seed, stage=args.stage,
        excluded_prompts=excluded, exclusion_provenance=provenance)
    directions, decomposition = decompose_direction(
        load_published_vector(args.source/'das_mlp8.joblib'),
        model.W_out[8].detach().cpu().numpy())
    args.output.mkdir(parents=True)
    save(args.output/'cases.json', cases)
    np.savez(args.output/'directions.npz', **directions)
    manifest = {
        'schema_version': 1, 'experiment': 'donor_factor_3a', 'stage': args.stage,
        'amendment': amendment_binding(),
        'status': 'prepared_before_new_model_forwards', 'created_unix': time.time(),
        'n_families': n, 'seed': seed, 'sampling': sampling, 'source': source,
        'code_git_head': git_head(),
        'code_files_sha256': {str(p.relative_to(REPO)): sha256(p) for p in code_files()},
        'cases_sha256': sha256(args.output/'cases.json'),
        'directions_sha256': sha256(args.output/'directions.npz'),
        'qualification_sha256': sha256(qualification), 'planning_input': planning_input,
        'development_input': ({'path': str(args.pilot_cases.resolve().relative_to(REPO)),
                               'sha256': sha256(args.pilot_cases)} if args.pilot_cases else None),
        'model_snapshot_revision': snapshot.name,
        'model_files_sha256': {p.name: sha256(p) for p in snapshot.iterdir()
                               if p.suffix in ('.json', '.safetensors', '.bin')},
        'model_transforms': {'center_unembed': True, 'center_writing_weights': True,
                             'fold_ln': True, 'refactor_factored_attn_matrices': True},
        'decomposition': decomposition, 'site': 'blocks.8.mlp.hook_post',
        'operator': '(h_d-h_r) dot v_null, multiplied by unchanged v_full',
        'precisions': ['float32', 'float64'],
        'precision_scope': 'same processed float32 weights promoted to float64; independently recomputed caches',
        'contract': CONTRACT,
        'runtime': {k: importlib.metadata.version(k) for k in
                    ('torch', 'transformers', 'transformer-lens', 'numpy')},
        'runtime_dependency_override': 'historical TL2.17.0 / torch2.5.1 qualified despite metadata torch>=2.6',
        'platform': platform.platform(), 'num_threads': 4,
        'scope': 'one held-out template and one prefix; names, objects and places vary',
        'stop_rule': 'technical failure blocks interpretation; no interim scientific stopping or replacing cases',
    }
    save(args.output/'manifest.json', manifest)
    print(json.dumps({'output': str(args.output), 'n': n,
                      'manifest_sha256': sha256(args.output/'manifest.json'),
                      'sampling_scope': manifest['scope']}), flush=True)


def run(args):
    import numpy as np
    import torch
    from donor_factor import run_family
    from donor_factor_analysis import analyze_records
    manifest_path = args.output/'manifest.json'
    if not args.manifest_sha256 or sha256(manifest_path) != args.manifest_sha256:
        raise RuntimeError('explicit frozen manifest hash required')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['contract'] != CONTRACT or manifest['experiment'] != 'donor_factor_3a'
            or manifest.get('amendment') != amendment_binding()
            or manifest['stage'] != 'confirmation' or manifest['n_families'] != 512):
        raise RuntimeError('frozen contract/instrument mismatch')
    for name, digest in manifest['code_files_sha256'].items():
        if sha256(REPO/name) != digest:
            raise RuntimeError(f'frozen code changed: {name}')
    for stem, suffix in (('cases', '.json'), ('directions', '.npz')):
        if sha256(args.output/(stem+suffix)) != manifest[stem+'_sha256']:
            raise RuntimeError(f'frozen {stem} changed')
    for name, version in manifest['runtime'].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'runtime changed: {name}')
    if args.output.joinpath('RUN_STARTED.json').exists():
        raise RuntimeError('run already started; no overwrite or repeat')
    cases = json.loads((args.output/'cases.json').read_text())
    if len(cases) != manifest['n_families'] or len({c['case_id'] for c in cases}) != len(cases):
        raise RuntimeError('case count/unique draw IDs mismatch')
    start = time.time()
    save(args.output/'RUN_STARTED.json', {'manifest_sha256': args.manifest_sha256,
                                        'started_unix': start, 'execution_git_head': git_head()})
    try:
        model, _, snapshot = load_model()
        for name, digest in manifest['model_files_sha256'].items():
            if sha256(snapshot/name) != digest:
                raise RuntimeError(f'model input changed: {name}')
        directions = dict(np.load(args.output/'directions.npz', allow_pickle=False))
        records = [{'case_id': c['case_id'], 'content_family_id': c['content_family_id'],
                    'precisions': {}} for c in cases]
        inference_start = time.time()
        for name, dtype in (('float32', torch.float32), ('float64', torch.float64)):
            model.to(dtype).eval()
            with (args.output/f'measurements_{name}.jsonl').open('x') as handle:
                for i, case in enumerate(cases):
                    result = run_family(model, case, directions)
                    records[i]['precisions'][name] = result
                    handle.write(json.dumps({'case_id': case['case_id'], **result}, allow_nan=False)+'\n')
                    handle.flush()
                    if (i+1) % 8 == 0:
                        print(json.dumps({'stage': manifest['stage'], 'dtype': name,
                                          'done': i+1, 'total': len(cases),
                                          'seconds': round(time.time()-inference_start, 2)}), flush=True)
        inference_seconds = time.time()-inference_start
        with (args.output/'records.jsonl').open('x') as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
        summary = analyze_records(records)
        summary.update({'stage': manifest['stage'], 'confirmation': manifest['stage']=='confirmation',
                        'inference_seconds': inference_seconds, 'elapsed_seconds': time.time()-start,
                        'manifest_sha256': args.manifest_sha256,
                        'records_sha256': sha256(args.output/'records.jsonl'),
                        'model_forward_calls': 12*len(cases),
                        'model_prompt_evaluations': 48*len(cases)})
        save(args.output/'summary.json', summary)
        save(args.output/'artifact_hashes.json', {p.name: sha256(p) for p in args.output.iterdir()
                                                 if p.is_file() and p.name!='artifact_hashes.json'})
        print(json.dumps({'completed': str(args.output), 'n': len(cases),
                          'inference_seconds': inference_seconds,
                          'summary_sha256': sha256(args.output/'summary.json')}), flush=True)
    except Exception as exc:
        save(args.output/'FAILED.json', {'type': type(exc).__name__, 'message': str(exc),
                                         'elapsed_seconds': time.time()-start})
        raise


def plan(args):
    from donor_factor_planning_512 import plan_confirmation
    if not args.development or args.output.exists():
        raise RuntimeError('planning requires a development path and a new output directory')
    directory = args.development.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['stage'] != 'development' or manifest['n_families'] != 32:
        raise RuntimeError('planning needs the frozen 32-family development run')
    if manifest['contract'] != ORIGINAL_CONTRACT:
        raise RuntimeError('development contract differs')
    for name, digest in manifest['code_files_sha256'].items():
        if sha256(REPO/name) != digest:
            raise RuntimeError(f'frozen producer changed before planning: {name}')
    summary = json.loads((directory/'summary.json').read_text())
    if (summary['records_sha256'] != sha256(directory/'records.jsonl')
            or summary['manifest_sha256'] != sha256(directory/'manifest.json')):
        raise RuntimeError('development evidence hashes mismatch')
    records = [json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    original = json.loads((ROOT/'results/donor_factor_planning/planning.json').read_text())
    for name, digest in original['development_binding'].items():
        if sha256(directory/name) != digest:
            raise RuntimeError(f'amendment must use original development {name}')
    start = time.time()
    print(json.dumps({'planning': 'started', 'n_development': len(records)}), flush=True)
    amendment = amendment_binding()
    result = plan_confirmation(records)
    if result['candidates'][:4] != original['candidates']:
        raise RuntimeError('original four size calculations did not reproduce exactly')
    result['amendment'] = amendment
    result['development_binding'] = {name: sha256(directory/name) for name in
                                    ('cases.json', 'records.jsonl', 'summary.json', 'manifest.json')}
    result['planner_code_sha256'] = sha256(ROOT/'src/donor_factor_planning_512.py')
    result['planning_elapsed_seconds'] = time.time()-start
    result['created_unix'] = time.time()
    args.output.mkdir(parents=True)
    save(args.output/'planning.json', result)
    print(json.dumps({'status': result['status'], 'selected_n': result['selected_n'],
                      'seconds': result['planning_elapsed_seconds'],
                      'planning_sha256': sha256(args.output/'planning.json')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run', 'qualify', 'plan'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT/'artifacts/makelov_source')
    p.add_argument('--stage', choices=('confirmation',), default='confirmation')
    p.add_argument('--pilot-cases', type=Path)
    p.add_argument('--planning', type=Path)
    p.add_argument('--development', type=Path)
    p.add_argument('--manifest-sha256')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.planning:
        args.planning = args.planning.resolve()
    configure()
    {'prepare': prepare, 'run': run, 'qualify': qualify, 'plan': plan}[args.command](args)


if __name__ == '__main__':
    main()
