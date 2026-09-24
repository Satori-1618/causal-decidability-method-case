"""Freeze and run Round 3B Stage A: native competence and geometry, no patches."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
PROTOCOL = ROOT / 'ROLE_BASELINE_PROTOCOL.md'
PACKAGES = ('torch', 'transformers', 'transformer-lens', 'numpy')
TRANSFORMS = {'center_unembed': True, 'center_writing_weights': True,
              'fold_ln': True, 'refactor_factored_attn_matrices': True}
CONTRACT = {
    'n': 32, 'sampling_seed': 24092441, 'competence_threshold': .90,
    'geometry_coverage': .90, 'geometry_gap_nats': .52,
    'optimistic_role_gain': 1.0, 'numerical_budget_nats': .01,
    'bootstrap_draws': 20000, 'bootstrap_seed': 320260940,
    'batch_size': 16,
}


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def hashes(paths):
    from makelov_read_source import sha256
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


def frozen_code():
    # Include the reused loader and all local source dependencies, not just this entry point.
    return [Path(__file__).resolve(), ROOT / 'scripts/run_query_route.py',
            *sorted((ROOT / 'src').glob('*.py'))]


def cached_snapshot():
    from huggingface_hub import snapshot_download
    return Path(snapshot_download('gpt2', local_files_only=True))


def model_files(snapshot):
    # Pin weights AND tokenizer vocabulary/merges, including symlink targets via sha256.
    return sorted(p for p in snapshot.iterdir() if p.is_file())


def prepare(args):
    from transformers import AutoTokenizer
    from makelov_read_source import sha256
    from role_baseline_sampling import historical_exclusions, sample_families
    from run_query_route import validate_source
    if args.output.exists():
        raise RuntimeError('output exists; preserve earlier artifacts')
    source = validate_source(args.source)
    snapshot = cached_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    inputs = {p.name: sha256(p) for p in model_files(snapshot)}
    excluded, exclusion_provenance = historical_exclusions(ROOT)
    cases, sampling = sample_families(
        args.source, tokenizer, n=CONTRACT['n'], seed=CONTRACT['sampling_seed'],
        excluded_prompts=excluded, exclusion_provenance=exclusion_provenance,
        tokenizer_fingerprint=json.dumps({'snapshot_revision': snapshot.name, 'files': inputs},
                                         sort_keys=True))
    args.output.mkdir(parents=True)
    save(args.output / 'cases.json', cases)
    manifest = {
        'schema_version': 1, 'experiment': 'role_baseline_3b_stage_a',
        'status': 'prepared_before_any_3b_model_forwards', 'confirmation': False,
        'created_unix': time.time(), 'code_git_head_before_freeze': head(),
        'code_files_sha256': hashes(frozen_code()),
        'documents_sha256': hashes([PROTOCOL, REPO / 'docs/GOODFIRE_CROSSWALK.md',
                                     REPO / 'docs/ROUND3B_POSITIVE_IDENTIFICATION.md']),
        'cases_sha256': sha256(args.output / 'cases.json'),
        'sampling': sampling, 'source': source, 'contract': CONTRACT,
        'model_snapshot_revision': snapshot.name, 'model_files_sha256': inputs,
        'model_transforms': TRANSFORMS,
        'precisions': ['float32', 'float64'],
        'precision_scope': 'same processed fp32 weights promoted to fp64; reference not exact truth',
        'instrument': 'native model forwards only; no intervention, donor vectors or hook installation',
        'position': 'explicit BOS; each prompt absolute last position; no padding',
        'n_families': len(cases), 'native_rows_per_family_per_precision': 48,
        'runtime': {p: importlib.metadata.version(p) for p in PACKAGES},
        'runtime_dependency_override': 'historical torch2.5.1 retained with TL2.17.0 (declares torch>=2.6)',
        'platform': platform.platform(), 'device': 'cpu', 'num_threads': 4,
        'controls': 'finite scores; frozen token replay; one batch-vs-single native replay per family/precision',
        'replay_tolerance': '64*dtype_epsilon*max(1, absolute selected name logit); diagnostic, not accuracy proof',
        'stop_rule': 'no selection of correct families; no prompt repair, patch run or replacement after failure',
    }
    save(args.output / 'manifest.json', manifest)
    print(json.dumps({'prepared': str(args.output), 'families': len(cases),
                      'manifest_sha256': sha256(args.output / 'manifest.json'),
                      'proposal_count': sampling['proposal_count'],
                      'rejection_counts': sampling['rejection_counts'],
                      'unique_content_families': sampling['unique_content_families']},
                     indent=2), flush=True)


def measure_family(model, tokenizer, case, batch_size=16):
    """Only ordinary unhooked forwards. Equal-length batches preserve absolute positions."""
    import torch
    from role_baseline_sampling import validate_family
    validate_family(case)
    groups = defaultdict(list)
    for row in case['rows']:
        expected = [case['bos_token_id']] + tokenizer.encode(row['prompt'], add_special_tokens=False)
        if expected != row['token_ids']:
            raise RuntimeError('frozen tokenizer replay mismatch')
        groups[len(row['token_ids'])].append(row)
    answers = case['answer_token_ids']
    rows, calls, evaluations = {}, 0, 0
    with torch.inference_mode():
        for length in sorted(groups):
            group = groups[length]
            for start in range(0, len(group), batch_size):
                batch = group[start:start+batch_size]
                tokens = torch.tensor([r['token_ids'] for r in batch], dtype=torch.long)
                output = model(tokens)
                calls += 1
                evaluations += len(batch)
                # All positions equal length-1 in an unpadded batch, validated above.
                logits = output[:, length-1, :].detach().cpu().double()
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError('nonfinite native logits')
                for index, row in enumerate(batch):
                    selected = logits[index, answers]
                    mass = torch.exp(torch.logsumexp(selected, dim=0)
                                     - torch.logsumexp(logits[index], dim=0)).item()
                    argmax = int(logits[index].argmax().item())
                    if not math.isfinite(mass) or not 0 <= mass <= 1:
                        raise RuntimeError('invalid native candidate probability mass')
                    result = {k: row[k] for k in ('row_id', 'wording', 'context_id', 'form',
                                                   'query', 'correct_name_index')}
                    result.update(name_logits=selected.tolist(), name_probability_mass=mass,
                                  full_vocab_argmax_id=argmax,
                                  full_vocab_argmax_token=tokenizer.decode([argmax]))
                    rows[row['row_id']] = result
                del output, logits
        first = case['rows'][0]
        replay = model(torch.tensor([first['token_ids']], dtype=torch.long))
        calls += 1
        evaluations += 1
        actual = replay[0, first['position'], answers].detach().cpu().double()
        observed = torch.tensor(rows[first['row_id']]['name_logits'], dtype=torch.float64)
        error = float((actual-observed).abs().max())
        tolerance = 64*torch.finfo(replay.dtype).eps*max(1., float(actual.abs().max()))
        controls = {'passed': math.isfinite(error) and error <= tolerance,
                    'replay_name_logits': actual.tolist(),
                    'batch_vs_single_max_name_logit_error': error,
                    'batch_vs_single_tolerance': tolerance, 'token_replay_passed': True,
                    'no_padding': True, 'replay_row_id': first['row_id']}
        if not controls['passed']:
            raise RuntimeError(f'native replay discrepancy {error} exceeds {tolerance}')
    return {'rows': [rows[r['row_id']] for r in case['rows']], 'controls': controls,
            'model_forward_calls': calls, 'prompt_evaluations': evaluations}


def validate_manifest(output, expected_hash):
    from makelov_read_source import sha256
    path = output / 'manifest.json'
    if not expected_hash or sha256(path) != expected_hash:
        raise RuntimeError('explicit frozen manifest hash is required')
    manifest = json.loads(path.read_text())
    if (manifest['experiment'] != 'role_baseline_3b_stage_a'
            or manifest['contract'] != CONTRACT or manifest['model_transforms'] != TRANSFORMS
            or manifest['precisions'] != ['float32', 'float64']):
        raise RuntimeError('manifest semantics differ from the frozen runner')
    for name, digest in {**manifest['code_files_sha256'], **manifest['documents_sha256']}.items():
        if sha256(REPO / name) != digest:
            raise RuntimeError(f'frozen input changed: {name}')
    if sha256(output / 'cases.json') != manifest['cases_sha256']:
        raise RuntimeError('frozen cases changed')
    for package, version in manifest['runtime'].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'runtime changed: {package}')
    cases = json.loads((output / 'cases.json').read_text())
    if len(cases) != CONTRACT['n'] or len({c['case_id'] for c in cases}) != len(cases):
        raise RuntimeError('wrong number of independent draws')
    return manifest, cases


def run(args):
    import torch
    from makelov_read_source import sha256
    from role_baseline_analysis import analyze_baselines
    from run_query_route import configure, load_model
    manifest, cases = validate_manifest(args.output, args.manifest_sha256)
    # Freeze tracked inputs before execution; do not call a dirty checkout preregistered.
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise RuntimeError('commit the frozen run and leave a clean tree before execution')
    start = time.time()
    save(args.output / 'RUN_STARTED.json', {'manifest_sha256': args.manifest_sha256,
                                          'execution_git_head': head(), 'started_unix': start})
    records = [{'case_id': c['case_id'], 'precisions': {}} for c in cases]
    try:
        configure()
        model, tokenizer, snapshot = load_model()
        if snapshot.name != manifest['model_snapshot_revision']:
            raise RuntimeError('cached snapshot revision changed')
        for name, digest in manifest['model_files_sha256'].items():
            if sha256(snapshot / name) != digest:
                raise RuntimeError(f'model input changed: {name}')
        for name, dtype in (('float32', torch.float32), ('float64', torch.float64)):
            model.to(dtype)
            model.eval()
            with (args.output / f'measurements_{name}.jsonl').open('x') as handle:
                for index, case in enumerate(cases):
                    result = measure_family(model, tokenizer, case, CONTRACT['batch_size'])
                    records[index]['precisions'][name] = result
                    handle.write(json.dumps({'case_id': case['case_id'], **result}, allow_nan=False)+'\n')
                    handle.flush()
                    if (index+1) % 8 == 0:
                        print(json.dumps({'precision': name, 'done': index+1, 'of': len(cases),
                                          'elapsed_seconds': round(time.time()-start, 2)}), flush=True)
        with (args.output / 'records.jsonl').open('x') as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
        result = analyze_baselines(cases, records, bootstrap_draws=CONTRACT['bootstrap_draws'],
                                   seed=CONTRACT['bootstrap_seed'])
        result.update(manifest_sha256=args.manifest_sha256,
                      records_sha256=sha256(args.output / 'records.jsonl'),
                      elapsed_seconds=time.time()-start, confirmation=False, patch_forwards=0,
                      model_forward_calls=sum(p['model_forward_calls'] for r in records
                                              for p in r['precisions'].values()),
                      prompt_evaluations=sum(p['prompt_evaluations'] for r in records
                                             for p in r['precisions'].values()))
        save(args.output / 'summary.json', result)
        save(args.output / 'artifact_hashes.json', {p.name: sha256(p) for p in args.output.iterdir()
                                                    if p.is_file() and p.name != 'artifact_hashes.json'})
        print(json.dumps({k: v for k, v in result.items() if k not in ('families', 'per_family', 'records')},
                         indent=2), flush=True)
    except Exception as exc:
        save(args.output / 'FAILED.json', {'type': type(exc).__name__, 'message': str(exc),
                                          'elapsed_seconds': time.time()-start})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/makelov_source')
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
