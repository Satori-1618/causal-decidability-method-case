"""Reproduce published endpoints and validate the runtime before hybrid effects."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from runtime import DyckRuntime, HERE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_rows(name):
    path = HERE / 'cache/data/model_preds' / (name + '_data_preds.csv')
    return list(csv.DictReader(path.open()))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite run directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    all_rows = {name: source_rows(name) for name in ['ood', 'indist']}
    strings = {name: [r['string'] for r in rows] for name, rows in all_rows.items()}
    outputs = {}
    summary = {}
    started = time.monotonic()
    for dtype in [torch.float32, torch.float64]:
        tag = str(dtype).split('.')[-1]
        rt = DyckRuntime.from_checkpoint(dtype=dtype)
        native_ood = rt.run(strings['ood'])
        mean_pattern = native_ood.full_attention.mean(0)
        for name in ['ood', 'indist']:
            native = native_ood if name == 'ood' else rt.run(strings[name])
            identity = rt.run(strings[name], mode='identity')
            uniform = rt.run(strings[name], mode='uniform_all_queries')
            eos_uniform = rt.run(strings[name], mode='both')
            # The paper's mean replacement is defined on OOD inputs; do not
            # invent a published ID mean endpoint.
            arms = {'native': native, 'uniform': uniform}
            if name == 'ood':
                mean_rows = mean_pattern[native.eos_positions]
                arms['mean'] = rt.run(strings[name], mode='custom_attention', attention_override=mean_rows)
            truth = torch.tensor([r['balanced'] == 'True' for r in all_rows[name]])
            # Upstream predictions are probabilities of the True class.
            original = torch.tensor([float(r['1aez5d6p']) > 0.5 for r in all_rows[name]])
            checks = {
                'correct': {arm: int((result.predicted_valid == truth).sum()) for arm, result in arms.items()},
                'native_public_prediction_mismatches': int((native.predicted_valid != original).sum()),
                'identity_max_margin_error': float((identity.margins - native.margins).abs().max()),
                'identity_max_node_error': float((identity.node - native.node).abs().max()),
                'uniform_scope_max_margin_error': float((eos_uniform.margins - uniform.margins).abs().max()),
                'minimum_bracket_mass': min(float(row[1:eos].sum()) for row, eos in zip(native.attention, native.eos_positions)),
                'forward_batches_per_arm': native.forward_calls,
            }
            assert checks['identity_max_margin_error'] <= (1e-5 if dtype == torch.float32 else 1e-12)
            assert checks['identity_max_node_error'] <= (1e-5 if dtype == torch.float32 else 1e-12)
            assert checks['uniform_scope_max_margin_error'] <= (1e-5 if dtype == torch.float32 else 1e-12)
            expected = {'native': 779, 'uniform': 823, 'mean': 827} if name == 'ood' else {'native': 998, 'uniform': 974}
            assert checks['correct'] == expected, (tag, name, checks['correct'], expected)
            assert checks['native_public_prediction_mismatches'] == 0
            summary[tag + '_' + name] = checks
            outputs[(tag, name)] = arms
            print(tag, name, checks, flush=True)
    records = []
    for name, rows in all_rows.items():
        for i, row in enumerate(rows):
            native = outputs[('float64', name)]['native']
            eos = int(native.eos_positions[i])
            record = {'dataset': name, 'index': i, 'string': row['string'], 'valid': row['balanced'] == 'True',
                      'bos_mass': float(native.attention[i, 0]), 'eos_mass': float(native.attention[i, eos]),
                      'bracket_mass': float(native.attention[i, 1:eos].sum())}
            for dtype in ['float32', 'float64']:
                for arm, result in outputs[(dtype, name)].items():
                    record[dtype + '_' + arm] = float(result.margins[i])
            records.append(record)
    (args.output / 'cases.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    for name in all_rows:
        summary[name + '_precision'] = {
            arm: float((outputs[('float32', name)][arm].margins.double() - result.margins).abs().max())
            for arm, result in outputs[('float64', name)].items()
        }
    summary['elapsed_seconds'] = time.monotonic() - started
    summary['environment'] = {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu', 'threads': 1}
    dump(args.output / 'summary.json', summary)
    dump(args.output / 'manifest.json', {'status': 'qualification_only', 'sources': {name: digest(HERE / name) for name in ['runtime.py', 'qualify.py', 'ASSET_LOCK.json', 'DEVELOPMENT.md']},
                                        'outputs': {name: digest(args.output / name) for name in ['summary.json', 'cases.jsonl']}})


if __name__ == '__main__':
    main()
