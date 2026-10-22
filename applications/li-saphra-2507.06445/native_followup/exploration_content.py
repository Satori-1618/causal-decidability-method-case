"""Exploratory native value diagnostics on the two published input files only.

No new intervention is evaluated. Gradients linearize the already measured
routing displacement; regression characterizes predictable value content,
not causally necessary information or a native algorithm.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qualify import digest, dump, source_rows
from runtime import DyckRuntime, HERE


def downstream_gradient(rt, residual, preprojection):
    """Exact native local derivative of False−True with respect to head AV."""
    block = rt.model.transformer.h[rt.layer]
    pre = preprojection.clone().requires_grad_(True)
    x = residual.clone() + block.attn.c_proj(pre)
    x = x + block.mlpf(block.ln_2(x))
    logits = rt.model.lm_head(rt.model.transformer.ln_f(x))
    margin = logits[:, 0] - logits[:, 1]
    gradient, = torch.autograd.grad(margin.sum(), pre)
    start = rt.head * (block.attn.n_embd // block.attn.n_head)
    return margin.detach(), gradient[:, start:start + block.attn.n_embd // block.attn.n_head].detach()


def partition_change(a, z, groups):
    """Split uniform−native routing into allocation and conditional changes.

    Each term is invariant to adding one constant to every projected value.
    Empty partitions are omitted. This is a local-gradient decomposition,
    not an empirical nonlinear intervention decomposition.
    """
    mass = float(a.sum())
    native = a / mass
    between = within = 0.
    for group in np.unique(groups):
        selected = groups == group
        p = float(selected.mean())
        q = float(native[selected].sum())
        native_mean = float(np.dot(native[selected], z[selected]) / q)
        uniform_mean = float(z[selected].mean())
        between += mass * (p - q) * native_mean
        within += mass * p * (uniform_mean - native_mean)
    direct = float(np.dot(mass / len(a) - a, z))
    assert abs(between + within - direct) < 1e-9
    return between, within


def regression_cv(features, outcomes, case_ids, semantic):
    # All rows of a string stay together; folds are deterministic index modulo5.
    # Both target and features are centered within each case, removing arbitrary
    # case-dependent projected-value offsets. Centering uses covariates/targets
    # only for descriptive regression, never to define causal claim success.
    n_base = 9
    x = features if semantic else features[:, :n_base]
    prediction = np.empty_like(outcomes)
    for fold in range(5):
        test = case_ids % 5 == fold
        scale = np.std(x[~test], axis=0)
        scale[scale < 1e-12] = 1
        beta = np.linalg.lstsq(x[~test] / scale, outcomes[~test], rcond=None)[0]
        prediction[test] = (x[test] / scale) @ beta
    residual = outcomes - prediction
    return {"out_of_fold_r2": float(1 - np.dot(residual, residual) / np.dot(outcomes, outcomes)),
            "out_of_fold_rmse": float(np.sqrt(np.mean(residual ** 2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing overwrite')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rt = DyckRuntime.from_checkpoint(dtype=torch.float64)
    known = [json.loads(line) for line in (HERE / 'results/development_001/cases.jsonl').read_text().splitlines()]
    known = {(r['dataset'], r['index']): r for r in known}
    records, summary = [], {}
    for name in ['ood', 'indist']:
        rows = source_rows(name)
        strings = [r['string'] for r in rows]
        residuals = []
        def capture(_module, args):
            residuals.append(args[0].detach().clone())
        hook = rt.model.transformer.h[rt.layer].register_forward_pre_hook(capture)
        try:
            native = rt.run(strings)
        finally:
            hook.remove()
        index = torch.arange(len(strings))
        residual = torch.cat(residuals)[index, native.eos_positions].clone()
        native_pre = native.preprojection[index, native.eos_positions].clone()
        margin, gradient = downstream_gradient(rt, residual, native_pre)
        assert torch.allclose(margin, native.margins, atol=1e-12, rtol=0)
        features, targets, case_ids = [], [], []
        for i, text in enumerate(strings):
            length = len(text)
            token = np.array([1 if c == '(' else -1 for c in text])
            depth = np.cumsum(token)
            absolute = np.arange(1, length + 1)
            position = absolute / length
            a = native.attention[i, 1:length+1].numpy()
            z = (native.values[i, 1:length+1] @ gradient[i]).numpy()
            z = z - z.mean()
            linear = float(np.dot(a.sum() / length - a, z))
            observed = known[(name, i)]['float64_routing_only'] - known[(name, i)]['float64_native']
            record = {'dataset': name, 'index': i, 'valid': rows[i]['balanced'] == 'True',
                      'final_depth': int(depth[-1]), 'negative_positions': int((depth < 0).sum()),
                      'observed_routing_effect': observed, 'linearized_routing_effect': linear}
            for label, groups in [('prefix_sign', depth < 0), ('token', token), ('position_half', position > .5)]:
                between, within = partition_change(a, z, groups)
                record[label + '_between'] = between
                record[label + '_within'] = within
            # Nine lexical/position covariates, followed by depth covariates.
            feat = np.column_stack([token, position, position**2, position**3,
                                    token*position, token*position**2, absolute, absolute**2,
                                    token*absolute, depth, np.abs(depth), (depth < 0).astype(float),
                                    depth/length, token*depth, (depth == 0).astype(float)])
            feat -= feat.mean(axis=0)
            features.append(feat)
            targets.append(z)
            case_ids.extend([i] * length)
            records.append(record)
        features, targets, case_ids = np.vstack(features), np.concatenate(targets), np.array(case_ids)
        data = [r for r in records if r['dataset'] == name]
        observed = np.array([r['observed_routing_effect'] for r in data])
        linear = np.array([r['linearized_routing_effect'] for r in data])
        summary[name] = {
            'n': len(data), 'tokens': len(targets),
            'native_downstream_reconstruction_max_error': float((margin-native.margins).abs().max()),
            'linearization_correlation': float(np.corrcoef(observed, linear)[0, 1]),
            'linearization_mae': float(abs(observed-linear).mean()),
            'observed_routing_effect_mean': float(observed.mean()),
            'linearized_effect_mean': float(linear.mean()),
            'lexical_position_regression': regression_cv(features, targets, case_ids, False),
            'lexical_position_and_depth_regression': regression_cv(features, targets, case_ids, True),
            'partitions': {
                label: {term: {'mean': float(np.mean([r[label+'_'+term] for r in data])),
                              'mean_absolute': float(np.mean([abs(r[label+'_'+term]) for r in data]))}
                        for term in ['between', 'within']}
                for label in ['prefix_sign', 'token', 'position_half']
            },
        }
    summary['status'] = 'exploratory_native_content_diagnostic_no_new_intervention_or_confirmation'
    dump(args.output/'summary.json', summary)
    (args.output/'cases.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
    dump(args.output/'manifest.json', {'sources': {name: digest(HERE/name) for name in ['exploration_content.py', 'runtime.py', 'ASSET_LOCK.json', 'results/development_001/cases.jsonl']},
                                      'outputs': {name: digest(args.output/name) for name in ['summary.json', 'cases.jsonl']}})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
