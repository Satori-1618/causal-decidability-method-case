"""Small external development pilot; no training, no confirmatory claim.

python scripts/run_makelov_read_source.py --output results/makelov_read_source_001
Requires torch, transformers and transformer-lens. Uses locally cached GPT-2.
"""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import makelov_read_source as experiment


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def estimate(values, seed=81023):
    """Descriptive paired bootstrap over BASE PAIRS, not the two swap directions."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (5000, len(x)))].mean(axis=1)
    return {'mean': float(x.mean()), 'nominal_bootstrap_95': np.quantile(means, [.025, .975]).tolist(),
            'n_base_pairs': len(x)}


def analyze(records, reference, cpu32):
    ids = list(dict.fromkeys(r['case_id'] for r in records))
    effects, a_errors, b_errors = [], [], []
    vectors = []
    for key in ids:
        pair = [r for r in records if r['case_id'] == key]
        m = [r['margins'] for r in pair]
        effects.append([[v[a]-v['baseline'] for a in ('full', 'read_row', 'read_null')] for v in m])
        # Prediction vectors: A=(full effect, 0), B=(0, full effect).
        # Average ABSOLUTE errors before averaging directions; never cancel signs.
        a_errors.append(np.mean([abs(v['read_row']-v['full'])+abs(v['read_null']-v['baseline']) for v in m])/2)
        b_errors.append(np.mean([abs(v['read_null']-v['full'])+abs(v['read_row']-v['baseline']) for v in m])/2)
        vectors.append(np.mean([abs(v['full']-v['baseline']) for v in m]))
    e = np.asarray(effects)
    out = {'status': 'development_pilot_not_confirmation', 'independent_unit': 'base_prompt_pair',
           'n_base_pairs': len(ids), 'n_directed_pairs': len(records),
           'estimand': 'IO minus subject logit margin; positive effect increases IO preference',
           'effects_by_receiver_pattern': {pattern: {arm: estimate(e[:, i, j])
                for j, arm in enumerate(('full', 'read_row', 'read_null'))}
                for i, pattern in enumerate(('ABB', 'BAB'))},
           'A_visible_read_MAE': estimate(a_errors), 'B_null_read_MAE': estimate(b_errors),
           'paired_MAE_A_minus_B_positive_favors_B': estimate(np.array(a_errors)-b_errors),
           'conditional_rival_separation_mean_absolute_full_effect': estimate(vectors),
           'identity_controls_passed': all(r['margins']['identity'] == r['margins']['baseline'] for r in records),
           'fidelity_controls_passed': all(x['passed'] for r in records for x in r['fidelity'].values()),
           'baseline_IO_wins': sum(r['margins']['baseline'] > 0 for r in records),
           'baseline_ties': sum(r['margins']['baseline'] == 0 for r in records),
           'interpretation': 'Relative predictive fit of two patch-read-source explanations; '
                             'neither semantic identification nor a preregistered adequacy decision.'}
    def compare(left, right):
        index = {(r['case_id'], r['receiver_pattern']): r for r in left}
        diffs = {a: [] for a in ('baseline', 'full', 'read_row', 'read_null')}
        for r in right:
            old = index[(r['case_id'], r['receiver_pattern'])]['margins']
            for a in diffs:
                # Compare the ENDPOINT and the causal effect, not just raw activations.
                diffs[a].append([abs(old[a]-r['margins'][a]),
                    abs((old[a]-old['baseline'])-(r['margins'][a]-r['margins']['baseline']))])
        return {'n_directed_pairs': len(right), 'max_discrepancy': {
            a: {'margin': float(np.max(v, axis=0)[0]), 'effect': float(np.max(v, axis=0)[1])}
            for a, v in diffs.items()}}
    if reference:
        out['precision_MPS32_vs_CPU64'] = compare(records, reference)
        out['backend_MPS32_vs_CPU32'] = compare(records, cpu32)
        out['precision_CPU32_vs_CPU64'] = compare(cpu32, reference)
        out['precision_scope'] = ('First predeclared base pairs only; float64 uses the same '
            'processed float32 weights and directions. A reference, not exact truth or a universal error bound.')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT/'artifacts/makelov_source')
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--seed', type=int, default=20260920)
    p.add_argument('--reference-pairs', type=int, default=4)
    p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = p.parse_args()
    if args.n < 1 or not 0 <= args.reference_pairs <= args.n:
        p.error('invalid sample or reference count')
    if args.output.exists():
        p.error('output exists; choose a new directory to preserve past results')
    args.output.mkdir(parents=True)
    start = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from transformer_lens import HookedTransformer
    source = json.loads((args.source/'source_manifest.json').read_text())
    for file, digest in source['files'].items():
        if experiment.sha256(args.source/file) != digest:
            raise RuntimeError(f'source hash mismatch: {file}')
    snapshot = Path(snapshot_download('gpt2', local_files_only=True))
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    hf = GPT2LMHeadModel.from_pretrained(snapshot, local_files_only=True,
                                       attn_implementation='eager').eval()
    cases = experiment.make_cases(args.source, tok, args.n, args.seed)
    model = HookedTransformer.from_pretrained('gpt2-small', hf_model=hf, tokenizer=tok,
        device='cpu', dtype=torch.float32, center_unembed=True,
        center_writing_weights=True, fold_ln=True, refactor_factored_attn_matrices=True)
    del hf
    model.eval().requires_grad_(False)
    directions, decomposition = experiment.decompose_direction(
        experiment.load_published_vector(args.source/'das_mlp8.joblib'),
        model.W_out[8].detach().cpu().numpy())
    np.savez(args.output/'directions.npz', **directions)
    save(args.output/'cases.json', cases)
    manifest = {'status': 'frozen_before_model_evaluation', 'purpose': 'development pilot',
        'created_unix': time.time(), 'seed': args.seed, 'n_base_pairs': args.n,
        'source': source, 'layer_zero_based': 8, 'site': 'post-GELU / pre-W_out',
        'position': 'absolute final prompt token from cases.json',
        'arms': ['baseline', *experiment.ARMS], 'same_write_vector_all_arms': True,
        'separate_component_normalization': False, 'working_dtype': 'float32',
        'working_device': args.device, 'model_snapshot': str(snapshot),
        'model_transforms': {'center_unembed': True, 'center_writing_weights': True,
                            'fold_ln': True, 'refactor_factored_attn_matrices': True},
        'runtime': {x: importlib.metadata.version(x) for x in ('torch', 'transformers', 'transformer-lens', 'numpy')},
        'platform': platform.platform(), 'git_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'files': {str(f.relative_to(ROOT)): experiment.sha256(f) for f in
                  (ROOT/'src/makelov_read_source.py', Path(__file__).resolve())},
        'cases_sha256': experiment.sha256(args.output/'cases.json'),
        'directions_sha256': experiment.sha256(args.output/'directions.npz'),
        'decomposition': decomposition,
        'reference_case_ids': [c['case_id'] for c in cases[:args.reference_pairs]],
        'selection': 'unique paired prompts; single-token names; equal token lengths; no behavioral filtering',
        'freshness': 'new seed/draws; original training/test membership cannot be exhaustively audited',
        'intervals': 'nominal 95% paired bootstrap over base pairs, exploratory, not simultaneous',
        'hypotheses': {'A': 'row-read reproduces full; null-read reproduces baseline',
                       'B': 'null-read reproduces full; row-read reproduces baseline'},
        'adequacy': 'No confirmatory adequacy threshold, no forced winner; report both errors.'}
    weight_files = [f for f in snapshot.iterdir() if f.suffix in ('.safetensors', '.bin', '.json')]
    manifest['model_files_sha256'] = {f.name: experiment.sha256(f) for f in weight_files}
    save(args.output/'manifest.json', manifest)
    print(json.dumps({'stage': 'manifest_frozen', 'decomposition': decomposition}), flush=True)
    records = []
    model.to(args.device)
    with (args.output/'records.jsonl').open('w') as handle:
        for i, c in enumerate(cases):
            new = experiment.run_case(model, c, directions)
            records.extend(new)
            for r in new:
                handle.write(json.dumps(r, allow_nan=False)+'\n')
            handle.flush()
            if (i+1) % 8 == 0:
                print(json.dumps({'stage': 'pilot', 'done': i+1, 'elapsed_s': round(time.time()-start, 1)}), flush=True)
    reference, cpu32 = [], []
    if args.reference_pairs:
        model.to('cpu')
        for c in cases[:args.reference_pairs]:
            cpu32.extend(experiment.run_case(model, c, directions))
        # TL's to(dtype) updates config as well as parameters.
        model.to(torch.float64)
        for c in cases[:args.reference_pairs]:
            reference.extend(experiment.run_case(model, c, directions))
        save(args.output/'reference_cpu32.json', cpu32)
        save(args.output/'reference_cpu64.json', reference)
    summary = analyze(records, reference, cpu32)
    summary['elapsed_seconds'] = time.time()-start
    summary['model_forward_calls'] = 5*args.n+10*args.reference_pairs
    summary['input_hashes'] = {f.name: experiment.sha256(f) for f in args.output.iterdir() if f.is_file()}
    save(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
