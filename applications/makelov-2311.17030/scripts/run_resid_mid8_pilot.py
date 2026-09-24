"""The blinded pilot of ``PREREG_RESID_MID8.md`` (§7). Runs once.

    PYTHONHASHSEED=0 python3 scripts/run_resid_mid8_pilot.py
    PYTHONHASHSEED=0 python3 scripts/run_resid_mid8_pilot.py --check

Writes ``results/resid_mid8_pilot/pilot.json`` and prints the same record. The
record holds ``mean E(full)`` per readout, the SDs of ``d_inert`` and ``d_all`` per
component and readout, the decomposition audit and provenance, and nothing else;
per-pair values are neither written nor printed.

Before anything is loaded the script refuses to run unless ``PYTHONHASHSEED=0``
gives upstream's pinned symbol order, the preregistration, calculator and direction
match their pinned hashes, every file the pilot depends on is committed unchanged
at ``HEAD``, and the output directory does not exist yet. ``--check`` runs those
gates and builds the 200 pilot pairs, without loading the model.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import makelov_table1 as M  # noqa: E402
import resid_mid8_pilot as P  # noqa: E402

SOURCE_DIR = ROOT / 'artifacts' / 'makelov_source'
DIRECTION = SOURCE_DIR / 'das_resid_mid.joblib'
PREREG = ROOT / 'PREREG_RESID_MID8.md'
CALCULATOR = ROOT / 'src' / 'decidability.py'
OUTPUT = ROOT / 'results' / 'resid_mid8_pilot'
#: the pilot's own code and every pinned input it reads
COMMITTED = (
    'scripts/run_resid_mid8_pilot.py', 'src/resid_mid8_pilot.py', 'src/makelov_table1.py',
    'src/decidability.py', 'PREREG_RESID_MID8.md',
    'artifacts/makelov_source/das_resid_mid.joblib',
    'artifacts/makelov_source/data/names.json', 'artifacts/makelov_source/data/objects.json',
    'artifacts/makelov_source/data/places.json',
    'artifacts/makelov_source/data/prefixes.json',
    'artifacts/makelov_source/data/templates.json',
)


def _git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)


def gates():
    """Every refusal happens here, before the model or the data is touched."""
    if os.environ.get('PYTHONHASHSEED') != P.REQUIRED_HASHSEED:
        sys.exit('refused: run with PYTHONHASHSEED=0 (prereg §6)')
    if P.upstream_symbol_order() != P.SYMBOL_ORDER:
        sys.exit(f'refused: list(set(pattern)) gives {P.upstream_symbol_order()}, '
                 f'pinned {P.SYMBOL_ORDER}')
    if M.sha256(PREREG) != P.PREREG_SHA:
        sys.exit('refused: PREREG_RESID_MID8.md does not match the pinned hash')
    if M.sha256(CALCULATOR) != P.CALCULATOR_SHA:
        sys.exit('refused: src/decidability.py does not match the hash pinned in §8')
    if M.sha256(DIRECTION) != P.DIRECTION_SHA:
        sys.exit('refused: das_resid_mid.joblib does not match the pinned hash')
    tracked = _git('ls-files', '--error-unmatch', *COMMITTED)
    if tracked.returncode != 0:
        sys.exit(f'refused: not committed: {tracked.stderr.strip()}')
    if _git('diff', '--quiet', 'HEAD', '--', *COMMITTED).returncode != 0:
        sys.exit('refused: the pilot or one of its inputs differs from HEAD (§7: the '
                 'pilot script is committed and hashed before it runs)')
    if OUTPUT.exists():
        sys.exit(f'refused: {OUTPUT.relative_to(ROOT)} exists; the pilot runs once')
    return _git('rev-parse', 'HEAD').stdout.strip()


def load_model():
    """As in the replication: GPT-2-small, float32, CPU, same processing flags."""
    from transformer_lens import HookedTransformer, utils
    if utils.get_act_name('resid_mid', layer=P.SITE_LAYER) != P.SITE_NAME:
        sys.exit('refused: hook name does not resolve to the preregistered site')
    model = HookedTransformer.from_pretrained(
        model_name='gpt2-small', center_unembed=True, center_writing_weights=True,
        fold_ln=True, refactor_factored_attn_matrices=True, device='cpu')
    model.requires_grad_(False)
    if model.cfg.dtype != torch.float32:
        sys.exit('refused: model is not float32')
    return model


def dataset_fingerprint(dataset):
    joined = '|'.join(r['pair_id'] for r in dataset)
    return hashlib.sha256(joined.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='run the gates and build the pairs; do not load the model')
    args = parser.parse_args()

    head = gates()
    dist = M.load_distribution(SOURCE_DIR)
    dataset = P.build_pilot_dataset(dist)
    direction = P.load_direction(DIRECTION)
    if args.check:
        print(json.dumps({'gates': 'passed', 'head': head, 'pairs': len(dataset),
                          'dataset_sha256': dataset_fingerprint(dataset)}, indent=1))
        return

    started = time.time()
    model = load_model()
    w_qs = [model.W_Q[layer, head_index].detach().cpu()
            for layer, head_index in P.NAME_MOVERS]
    directions, audit = P.decompose_namemovers(direction, w_qs)
    summary = P.blinded_summary(P.effects(P.run_conditions(model, dataset, directions)))
    record = P.validate_record({
        'summary': summary,
        'decomposition': audit,
        'provenance': {
            'prereg': 'PREREG_RESID_MID8.md', 'git_head': head,
            'seed': P.PILOT_SEED, 'per_combination': P.PILOT_PER_COMBINATION,
            'pythonhashseed': os.environ['PYTHONHASHSEED'],
            'symbol_order': ''.join(P.SYMBOL_ORDER),
            'site': P.SITE_NAME, 'position': 'last',
            'name_movers': ['.'.join(map(str, h)) for h in P.NAME_MOVERS],
            'dtype': 'float32', 'device': 'cpu',
            'dataset_sha256': dataset_fingerprint(dataset),
            'name_mover_queries_sha256': P.weights_fingerprint(w_qs),
            'file_sha256': {path: M.sha256(ROOT / path) for path in COMMITTED[:6]},
            'python': platform.python_version(), 'torch': torch.__version__,
            'runtime_seconds': round(time.time() - started, 1),
        },
    })
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with open(OUTPUT / 'pilot.json', 'x') as handle:
        handle.write(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
