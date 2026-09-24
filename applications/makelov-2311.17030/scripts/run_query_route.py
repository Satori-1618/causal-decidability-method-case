"""Prepare, qualify, run and reproduce the fixed round-2 query-route experiment."""
import argparse
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


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def code_files():
    return [Path(__file__).resolve(), *(ROOT / 'src' / name for name in (
        'makelov_read_source.py', 'query_route.py', 'query_route_analysis.py',
        'query_route_sampling.py'))]


def configure():
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(240924)


def load_model():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from transformer_lens import HookedTransformer
    snapshot = Path(snapshot_download('gpt2', local_files_only=True))
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    hf = GPT2LMHeadModel.from_pretrained(snapshot, local_files_only=True,
                                       attn_implementation='eager').eval()
    model = HookedTransformer.from_pretrained(
        'gpt2-small', hf_model=hf, tokenizer=tok, device='cpu', dtype=torch.float32,
        center_unembed=True, center_writing_weights=True, fold_ln=True,
        refactor_factored_attn_matrices=True)
    model.eval().requires_grad_(False)
    if model.cfg.n_layers != 12 or model.cfg.d_model != 768 or model.cfg.d_head != 64:
        raise RuntimeError('unexpected model architecture')
    if (model.cfg.positional_embedding_type != 'standard'
            or getattr(model.cfg, 'use_qk_norm', False)):
        raise RuntimeError('query semantics require standard GPT-2 positions without QK normalization')
    return model, tok, snapshot


def validate_source(source):
    from makelov_read_source import sha256
    manifest = json.loads((source / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha256(source / name) != digest:
            raise RuntimeError(f'upstream input hash mismatch: {name}')
    return manifest


def prepare(args):
    import numpy as np
    from makelov_read_source import decompose_direction, load_published_vector, sha256
    from query_route import CELL_MEANINGS, QUERY_HEADS
    from query_route_sampling import historical_exclusions, sample_iid_cases
    if args.output.exists():
        raise RuntimeError('output already exists; preserve previous artifacts')
    source = validate_source(args.source)
    model, tok, snapshot = load_model()
    pilot_cases = json.loads(args.pilot_cases.read_text()) if args.pilot_cases else []
    if args.stage == 'confirmation' and len(pilot_cases) != 32:
        raise RuntimeError('confirmation requires the 32 declared development cases')
    excluded, provenance = historical_exclusions(ROOT, pilot_cases=pilot_cases)
    n, seed = (32, 24092401) if args.stage == 'development' else (192, 24092402)
    cases, sampling = sample_iid_cases(args.source, tok, n=n, seed=seed, stage=args.stage,
                                       excluded_prompts=excluded, exclusion_provenance=provenance)
    directions, decomposition = decompose_direction(
        load_published_vector(args.source / 'das_mlp8.joblib'),
        model.W_out[8].detach().cpu().numpy())
    args.output.mkdir(parents=True)
    save(args.output / 'cases.json', cases)
    np.savez(args.output / 'directions.npz', **directions)
    manifest = {
        'schema_version': 1, 'stage': args.stage, 'status': 'prepared_before_model_forwards',
        'created_unix': time.time(), 'source': source,
        'n_base_pairs': n, 'seed': seed, 'sampling': sampling,
        'code_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'code_files_sha256': {str(p.relative_to(REPO)): sha256(p) for p in code_files()},
        'cases_sha256': sha256(args.output / 'cases.json'),
        'directions_sha256': sha256(args.output / 'directions.npz'),
        'model_snapshot_revision': snapshot.name,
        'model_files_sha256': {p.name: sha256(p) for p in snapshot.iterdir()
                               if p.suffix in ('.json', '.safetensors', '.bin')},
        'model_transforms': {'center_unembed': True, 'center_writing_weights': True,
                             'fold_ln': True, 'refactor_factored_attn_matrices': True},
        'decomposition': decomposition, 'query_heads': {k: list(v) for k, v in QUERY_HEADS.items()},
        'mlp_site': 'blocks.8.mlp.hook_post', 'patch': 'read_null with full write vector',
        'position': 'absolute last prompt token P-1', 'cells': CELL_MEANINGS,
        'precisions': ['float32', 'float64'],
        'precision_scope': 'same processed float32 weights promoted to float64; separate caches; reference not exact truth',
        'runtime': {k: importlib.metadata.version(k) for k in
                    ('torch', 'transformers', 'transformer-lens', 'numpy')},
        'runtime_dependency_override': 'TL 2.17.0 declares torch>=2.6; retain historical torch2.5.1 and qualify empirically',
        'platform': platform.platform(), 'num_threads': 4, 'deterministic_algorithms': True,
        'scientific_contract': {'kappa': 0.25, 'numerical_fraction': 0.025,
                                'coverage': 0.8, 'alpha': 0.05, 'profiles': 3,
                                'family': 'three two-sided CP intervals; alpha/6 per tail',
                                'success': 'both directions, both precisions, resolved anchor; no outcome exclusions'},
        'identity_tolerance': '64 * dtype epsilon * max(1, selected answer logit absmax); margin budget twice this',
        'stop_rule': 'technical failure invalidates; no interim outcome stopping or sample enlargement',
    }
    save(args.output / 'manifest.json', manifest)
    print(json.dumps({'prepared': str(args.output), 'n': n,
                      'manifest_sha256': sha256(args.output / 'manifest.json'),
                      'sampling': sampling}, indent=2), flush=True)


def run(args):
    import numpy as np
    import torch
    from makelov_read_source import sha256
    from query_route import run_route_case, QUERY_HEADS
    from query_route_analysis import analyze_records
    manifest_path = args.output / 'manifest.json'
    if not args.manifest_sha256 or sha256(manifest_path) != args.manifest_sha256:
        raise RuntimeError('explicit frozen manifest hash is required and must match')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['stage'] not in ('development', 'confirmation')
            or manifest['precisions'] != ['float32', 'float64']
            or manifest['query_heads'] != {k: list(v) for k, v in QUERY_HEADS.items()}):
        raise RuntimeError('manifest instrument semantics do not match this runner')
    cases = json.loads((args.output / 'cases.json').read_text())
    if (len(cases) != manifest['n_base_pairs']
            or len({c['case_id'] for c in cases}) != len(cases)):
        raise RuntimeError('manifest sample count or unique draw IDs do not match')
    for name, digest in manifest['code_files_sha256'].items():
        if sha256(REPO / name) != digest:
            raise RuntimeError(f'frozen code changed: {name}')
    for name in ('cases', 'directions'):
        ext = '.json' if name == 'cases' else '.npz'
        if sha256(args.output / (name+ext)) != manifest[name+'_sha256']:
            raise RuntimeError(f'frozen {name} changed')
    for name, expected in manifest['runtime'].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f'runtime changed: {name}')
    started = args.output / 'RUN_STARTED.json'
    if started.exists():
        raise RuntimeError('this frozen run was already started; no overwrite or second scoring')
    start = time.time()
    save(started, {'manifest_sha256': args.manifest_sha256, 'started_unix': start,
                   'execution_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()})
    model, _, snapshot = load_model()
    for name, digest in manifest['model_files_sha256'].items():
        if sha256(snapshot / name) != digest:
            raise RuntimeError(f'model input changed: {name}')
    directions = dict(np.load(args.output / 'directions.npz', allow_pickle=False))
    records = [{'pair_id': c['case_id'], 'content_pair_id': c.get('content_pair_id'),
                'precisions': {}} for c in cases]
    try:
        for dtype_name, dtype in (('float32', torch.float32), ('float64', torch.float64)):
            model.to(dtype)
            model.eval()
            with (args.output / f'measurements_{dtype_name}.jsonl').open('x') as handle:
                for i, case in enumerate(cases):
                    result = run_route_case(model, case, directions)
                    records[i]['precisions'][dtype_name] = result
                    handle.write(json.dumps({'pair_id': case['case_id'], **result}, allow_nan=False)+'\n')
                    handle.flush()
                    if (i+1) % 8 == 0:
                        print(json.dumps({'stage': manifest['stage'], 'dtype': dtype_name,
                                          'done': i+1, 'total': len(cases),
                                          'elapsed_seconds': round(time.time()-start, 2)}), flush=True)
        with (args.output / 'records.jsonl').open('x') as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
        contract = manifest['scientific_contract']
        result = analyze_records(records, **{k: contract[k] for k in
                                  ('kappa', 'numerical_fraction', 'coverage', 'alpha')})
        result.update({'stage': manifest['stage'], 'elapsed_seconds': time.time()-start,
                       'model_forward_calls': 14*len(cases), 'manifest_sha256': args.manifest_sha256,
                       'records_sha256': sha256(args.output / 'records.jsonl'),
                       'confirmation': manifest['stage'] == 'confirmation'})
        save(args.output / 'summary.json', result)
        save(args.output / 'artifact_hashes.json', {p.name: sha256(p) for p in args.output.iterdir()
                                                   if p.is_file() and p.name != 'artifact_hashes.json'})
        print(json.dumps({k: v for k, v in result.items() if k not in ('pairs', 'per_pair')}, indent=2), flush=True)
    except Exception as exc:
        save(args.output / 'FAILED.json', {'type': type(exc).__name__, 'message': str(exc),
                                          'elapsed_seconds': time.time()-start})
        raise


def qualify(args):
    """Replay two old Q1 cases, not any fresh round-2 outcomes."""
    import numpy as np
    from makelov_read_source import decompose_direction, load_published_vector, run_case
    model, _, _ = load_model()
    validate_source(args.source)
    directions, decomposition = decompose_direction(load_published_vector(args.source/'das_mlp8.joblib'),
                                                    model.W_out[8].detach().cpu().numpy())
    prior = ROOT/'results/makelov_read_source_q1'
    cases = json.loads((prior/'cases.json').read_text())[:2]
    known = {(r['case_id'], r['receiver_pattern']): r for r in
             (json.loads(line) for line in (prior/'records.jsonl').read_text().splitlines())}
    actual = [r for c in cases for r in run_case(model, c, directions)]
    error = max(abs(v-known[(r['case_id'], r['receiver_pattern'])]['margins'][arm])
                for r in actual for arm, v in r['margins'].items())
    result = {'historical_cases': [c['case_id'] for c in cases],
              'max_answer_margin_discrepancy_vs_stored_q1': error,
              'passed': error <= 0.0001, 'decomposition': decomposition,
              'runtime': {k: importlib.metadata.version(k) for k in
                          ('torch', 'transformers', 'transformer-lens', 'numpy')},
              'config': {'positional_embedding_type': model.cfg.positional_embedding_type,
                         'n_layers': model.cfg.n_layers, 'n_heads': model.cfg.n_heads,
                         'd_head': model.cfg.d_head},
              'scope': 'two historical Q1 cases; validates replay, not new hook or scientific outcome'}
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'qualification.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise RuntimeError('historical replay discrepancy exceeds declared 1e-4 nat qualification tolerance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run', 'qualify'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT/'artifacts/makelov_source')
    p.add_argument('--stage', choices=('development', 'confirmation'), default='development')
    p.add_argument('--pilot-cases', type=Path)
    p.add_argument('--manifest-sha256')
    args = p.parse_args()
    configure()
    {'prepare': prepare, 'run': run, 'qualify': qualify}[args.command](args)


if __name__ == '__main__':
    main()
