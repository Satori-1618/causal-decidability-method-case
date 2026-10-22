"""Exploratory four-cell separation on the already published input sets."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from qualify import source_rows, dump, digest
from runtime import DyckRuntime, HERE

MODES = ['native', 'routing_only', 'gate_only', 'both']


def sign_match(text, row):
    depth = 0
    negative, nonnegative = [], []
    for i, char in enumerate(text, start=1):
        depth += 1 if char == '(' else -1
        (negative if depth < 0 else nonnegative).append(float(row[i]))
    if not negative or not nonnegative:
        return None
    return min(negative) > max(nonnegative) if depth < 0 else max(negative) < min(nonnegative)


def summarize(records):
    out = {}
    for name in ['ood', 'indist', 'indist_valid', 'indist_invalid']:
        selected = [r for r in records if r['dataset'] == name or
                    (name.startswith('indist_') and r['dataset'] == 'indist' and
                     r['valid'] == (name == 'indist_valid'))]
        data = {mode: np.array([r['float64_' + mode] for r in selected]) for mode in MODES}
        valid = np.array([r['valid'] for r in selected])
        full = data['both'] - data['native']
        routes = data['routing_only'] - data['native']
        gates = data['gate_only'] - data['native']
        route_error = np.maximum(abs(data['routing_only'] - data['both']), abs(data['gate_only'] - data['native']))
        gate_error = np.maximum(abs(data['routing_only'] - data['native']), abs(data['gate_only'] - data['both']))
        out[name] = {
            'n': len(selected), 'valid': int(valid.sum()),
            'correct': {mode: int(((m < 0) == valid).sum()) for mode, m in data.items()},
            'margin_effect_mean': {'full': float(full.mean()), 'routing': float(routes.mean()), 'gate': float(gates.mean())},
            'margin_effect_mean_absolute': {'full': float(abs(full).mean()), 'routing': float(abs(routes).mean()), 'gate': float(abs(gates).mean())},
            'pure_routing_error_mean': float(route_error.mean()), 'pure_gating_error_mean': float(gate_error.mean()),
            'pure_routing_error_quantiles': np.quantile(route_error, [0,.25,.5,.75,.9,1]).tolist(),
            'pure_gating_error_quantiles': np.quantile(gate_error, [0,.25,.5,.75,.9,1]).tolist(),
            'sign_pattern_preservation_mismatches_gate': sum(r['sign_native'] != r['sign_gate'] for r in selected),
            'fp32_fp64_max_margin_difference': max(abs(r['float32_' + mode] - r['float64_' + mode]) for r in selected for mode in MODES),
            'constant_shift_best_fit_residual_mae': float(abs(full - full.mean()).mean()),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite run directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    records = []
    tensor_store = {}
    for name in ['ood', 'indist']:
        rows = source_rows(name)
        strings = [r['string'] for r in rows]
        outputs = {}
        for dtype in [torch.float32, torch.float64]:
            rt = DyckRuntime.from_checkpoint(dtype=dtype)
            tag = str(dtype).split('.')[-1]
            for mode in MODES:
                outputs[(tag, mode)] = rt.run(strings, mode=mode)
                print(name, tag, mode, flush=True)
        native = outputs[('float64', 'native')]
        for mode in MODES:
            result = outputs[('float64', mode)]
            # A last-layer intervention cannot change upstream current values.
            assert torch.equal(result.values, native.values)
            tensor_store[name + '_' + mode + '_node'] = result.node.numpy()
            tensor_store[name + '_' + mode + '_attention'] = result.attention.numpy()
        # The effect is only at target head, target EOS. Check all other slices.
        for mode in ['routing_only', 'gate_only', 'both']:
            diff = outputs[('float64', mode)].preprojection - native.preprojection
            for i, eos in enumerate(native.eos_positions):
                diff[i, eos, 32:64] = 0
            assert torch.count_nonzero(diff) == 0
        for i, row in enumerate(rows):
            record = {'dataset': name, 'index': i, 'string': row['string'], 'valid': row['balanced'] == 'True',
                      'sign_native': sign_match(row['string'], native.attention[i]),
                      'sign_gate': sign_match(row['string'], outputs[('float64', 'gate_only')].attention[i])}
            for (dtype, mode), result in outputs.items():
                record[dtype + '_' + mode] = float(result.margins[i])
            records.append(record)
    summary = summarize(records)
    summary['status'] = 'exploratory_on_published_inputs_not_confirmation'
    summary['elapsed_seconds'] = time.monotonic() - started
    (args.output / 'cases.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    np.savez_compressed(args.output / 'nodes.npz', **tensor_store)
    dump(args.output / 'summary.json', summary)
    dump(args.output / 'manifest.json', {'sources': {name: digest(HERE / name) for name in ['runtime.py','develop.py','qualify.py','ASSET_LOCK.json','DEVELOPMENT.md']},
                                        'outputs': {name: digest(args.output / name) for name in ['summary.json','cases.jsonl','nodes.npz']}})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
