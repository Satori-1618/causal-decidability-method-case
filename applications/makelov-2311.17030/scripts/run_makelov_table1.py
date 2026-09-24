"""Re-run four rows of Makelov, Lange & Nanda Table 1 and export every per-example cell.

    python3 scripts/run_makelov_table1.py --output results/makelov_replication_001

The reproduction gate runs first: the four aggregated means must land on the
published ``patching_metrics_ioi.joblib`` values before anything else is written.
``--tolerance`` sets how close counts as reproduced (default 5e-4 on both columns,
which is half a unit in the last published digit of the accuracy column).

Two conventions of the upstream code are not recoverable from its source and are
therefore switches here rather than assumptions:

``--symbol-order``  upstream's ``list(set(pattern))`` has no defined order across
                    processes; it decides which of the two sampled names plays A.
``--row-basis``     ``qr`` is upstream's own reduced QR of ``W_out``, which keeps
                    all 768 columns; ``svd`` drops the numerically dead one.

``--no-normalise`` skips renormalising the sub-directions, which the patch operator
is invariant to; it exists so that invariance can be measured, not asserted.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import makelov_table1 as M  # noqa: E402

SOURCE_DIR = ROOT / 'artifacts' / 'makelov_source'


def load_model(device):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name='gpt2-small', center_unembed=True, center_writing_weights=True,
        fold_ln=True, refactor_factored_attn_matrices=True, device=device)
    model.requires_grad_(False)
    return model


def gate(rows, tolerance):
    """Compare against the published means before any new analysis is trusted."""
    report = []
    for row in rows:
        published = M.PUBLISHED_ROWS[row['condition']]
        placeholder = published.get('accuracy_is_placeholder', False)
        entry = {'condition': row['condition'], 'intervention': row['intervention'],
                 'accuracy_is_placeholder_upstream': placeholder}
        for column in ('accuracy', 'logit_diff'):
            entry[f'{column}_observed'] = row[column]
            entry[f'{column}_published'] = published[column]
            entry[f'{column}_delta'] = row[column] - published[column]
        checked = ('logit_diff',) if placeholder else ('accuracy', 'logit_diff')
        entry['checked_columns'] = list(checked)
        entry['passed'] = all(abs(entry[f'{c}_delta']) <= tolerance for c in checked)
        report.append(entry)
    inert = abs(next(r for r in rows if r['condition'] == 'nullspace')['logit_diff']
                - next(r for r in rows if r['condition'] == 'clean')['logit_diff'])
    return {'rows': report, 'tolerance': tolerance,
            'nullspace_minus_clean_logit_diff': inert,
            'passed': all(e['passed'] for e in report)}


def export(path, dataset, results):
    """One JSONL line per base/source pair per condition; nothing is pre-averaged."""
    with open(path, 'w') as handle:
        for condition in M.CONDITIONS:
            cell = results[condition]
            for i, record in enumerate(dataset):
                handle.write(json.dumps({
                    'pair_id': record['pair_id'], 'index': record['index'],
                    'half': record['half'], 'condition': condition,
                    'base_sentence': record['base_sentence'],
                    'source_sentence': record['source_sentence'],
                    'base_io_name': record['base_io_name'],
                    'base_s_name': record['base_s_name'],
                    'patched_correct_name': record['patched_answer_names'][0],
                    'patched_incorrect_name': record['patched_answer_names'][1],
                    'logit_patched_correct': float(cell['target'][i]),
                    'logit_patched_incorrect': float(cell['foil'][i]),
                    'logit_diff_patched_ordering':
                        float(cell['target'][i] - cell['foil'][i]),
                    'logit_base_io': float(cell['io'][i]),
                    'logit_base_s': float(cell['s'][i]),
                    'logit_diff_base_ordering': float(cell['io'][i] - cell['s'][i]),
                    'argmax_token_id': int(cell['argmax'][i]),
                    'interchange_correct': int(cell['interchange'][i]),
                }) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--symbol-order', default='AB', choices=['AB', 'BA'])
    ap.add_argument('--row-basis', default=M.ROW_BASIS_QR,
                    choices=[M.ROW_BASIS_QR, M.ROW_BASIS_SVD])
    ap.add_argument('--no-normalise', action='store_true')
    ap.add_argument('--batch-size', type=int, default=100)
    ap.add_argument('--samples-per-combination', type=int, default=1000)
    ap.add_argument('--tolerance', type=float, default=5e-4)
    ap.add_argument('--allow-gate-failure', action='store_true',
                    help='still write the export when the gate fails; for '
                         'convention sweeps only, never for the reported run')
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} already exists and is not empty')
    started = time.time()

    dist = M.load_distribution(SOURCE_DIR)
    dataset = M.build_patching_dataset(
        dist, list(args.symbol_order),
        samples_per_combination=args.samples_per_combination)
    model = load_model(args.device)
    vector = M.load_published_vector(
        SOURCE_DIR / 'das_mlp8.joblib').to(args.device)
    directions, audit = M.decompose(vector, model.W_out[M.SITE_LAYER].detach(),
                                    basis=args.row_basis)

    def progress(done, total):
        print(f'  {done}/{total}', flush=True)

    results = M.run_conditions(model, dataset, directions,
                               batch_size=args.batch_size,
                               normalise=not args.no_normalise, progress=progress)
    rows = M.aggregate(results)
    report = gate(rows, args.tolerance)

    for row in rows:
        published = M.PUBLISHED_ROWS[row['condition']]
        print(f"{row['intervention']:<32} acc {row['accuracy']:.4f} "
              f"(pub {published['accuracy']:.4f})  ld {row['logit_diff']:.6f} "
              f"(pub {published['logit_diff']:.6f})")
    print('gate passed:', report['passed'])

    if not report['passed'] and not args.allow_gate_failure:
        raise SystemExit('reproduction gate failed; refusing to write the export')

    out.mkdir(parents=True, exist_ok=True)
    export(out / 'per_example.jsonl', dataset, results)
    (out / 'aggregate.json').write_text(json.dumps({
        'rows': rows, 'gate': report, 'direction_audit': audit,
        'config': {'device': args.device, 'symbol_order': args.symbol_order,
                   'row_basis': args.row_basis,
                   'normalise_subdirections': not args.no_normalise,
                   'batch_size': args.batch_size,
                   'samples_per_combination': args.samples_per_combination,
                   'n_pairs': len(dataset)},
        'environment': {'python': platform.python_version(),
                        'torch': torch.__version__,
                        'platform': platform.platform(),
                        'makelov_table1_sha256': M.sha256(
                            ROOT / 'src' / 'makelov_table1.py'),
                        'upstream_manifest': json.loads(
                            (SOURCE_DIR / 'source_manifest.json').read_text())},
        'runtime_seconds': time.time() - started,
    }, indent=1))
    (out / 'dataset.jsonl').write_text(''.join(
        json.dumps({k: v for k, v in r.items() if k != 'template'}) + '\n'
        for r in dataset))
    print('wrote', out)


if __name__ == '__main__':
    main()
