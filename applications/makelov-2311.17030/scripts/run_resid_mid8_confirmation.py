"""The confirmation run of ``PREREG_RESID_MID8.md`` (§9–§10). Runs once, after Freeze B.

    PYTHONHASHSEED=0 python3 scripts/run_resid_mid8_confirmation.py
    PYTHONHASHSEED=0 python3 scripts/run_resid_mid8_confirmation.py --check

Refuses unless the Freeze B predictions exist with the hash pinned here and every
input is committed unchanged at ``HEAD``. Generates the seed-7013 pairs only then,
evaluates them in the interleaved order (§13, item 6), and writes
``results/resid_mid8_confirmation/`` with the scored predictions, the secondary block
rates, the now-unblinded component effects, and every per-pair cell.
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
import resid_mid8_freeze as F  # noqa: E402
import resid_mid8_pilot as P  # noqa: E402

SOURCE_DIR = ROOT / 'artifacts' / 'makelov_source'
PREDICTIONS = ROOT / 'results' / 'resid_mid8_freeze_b' / 'predictions.json'
PILOT = ROOT / 'results' / 'resid_mid8_pilot' / 'pilot.json'
OUTPUT = ROOT / 'results' / 'resid_mid8_confirmation'
#: Freeze B, as committed
PREDICTIONS_SHA = '3cb542b9cfd990e5176df674285663015fa6df3e37f2f0964c8630c72abc40d2'
COMMITTED = (
    'scripts/run_resid_mid8_confirmation.py', 'src/resid_mid8_freeze.py',
    'src/resid_mid8_pilot.py', 'src/makelov_table1.py', 'src/decidability.py',
    'PREREG_RESID_MID8.md', 'results/resid_mid8_freeze_b/predictions.json',
    'results/resid_mid8_pilot/pilot.json',
    'artifacts/makelov_source/das_resid_mid.joblib',
    'artifacts/makelov_source/data/names.json', 'artifacts/makelov_source/data/objects.json',
    'artifacts/makelov_source/data/places.json',
    'artifacts/makelov_source/data/prefixes.json',
    'artifacts/makelov_source/data/templates.json',
)


def _git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)


def gates():
    if os.environ.get('PYTHONHASHSEED') != P.REQUIRED_HASHSEED:
        sys.exit('refused: run with PYTHONHASHSEED=0 (prereg §6)')
    if P.upstream_symbol_order() != P.SYMBOL_ORDER:
        sys.exit('refused: symbol order under this hash seed is not the pinned one')
    if not PREDICTIONS.exists() or M.sha256(PREDICTIONS) != PREDICTIONS_SHA:
        sys.exit('refused: Freeze B predictions missing or not the committed ones')
    for path, sha in ((ROOT / 'PREREG_RESID_MID8.md', P.PREREG_SHA),
                      (ROOT / 'src' / 'decidability.py', P.CALCULATOR_SHA),
                      (SOURCE_DIR / 'das_resid_mid.joblib', P.DIRECTION_SHA)):
        if M.sha256(path) != sha:
            sys.exit(f'refused: {path.name} does not match its pinned hash')
    tracked = _git('ls-files', '--error-unmatch', *COMMITTED)
    if tracked.returncode != 0:
        sys.exit(f'refused: not committed: {tracked.stderr.strip()}')
    if _git('diff', '--quiet', 'HEAD', '--', *COMMITTED).returncode != 0:
        sys.exit('refused: the confirmation or one of its inputs differs from HEAD')
    if OUTPUT.exists():
        sys.exit(f'refused: {OUTPUT.relative_to(ROOT)} exists; the confirmation runs once')
    return _git('rev-parse', 'HEAD').stdout.strip()


def load_model():
    """As in the pilot and the replication: GPT-2-small, float32, CPU."""
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


def build_confirmation(dist):
    dataset = M.build_patching_dataset(
        dist, list(P.SYMBOL_ORDER), seed=F.CONFIRMATION_SEED,
        samples_per_combination=F.CONFIRMATION_PER_COMBINATION)
    return F.interleave(dataset)


def fingerprint(dataset):
    return hashlib.sha256('|'.join(r['pair_id'] for r in dataset).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='run the gates only; generate no confirmation pair')
    args = parser.parse_args()
    head = gates()
    if args.check:
        print(json.dumps({'gates': 'passed', 'head': head}, indent=1))
        return

    predictions = json.loads(PREDICTIONS.read_text())
    pilot = json.loads(PILOT.read_text())
    dist = M.load_distribution(SOURCE_DIR)
    dataset = build_confirmation(dist)
    pilot_ids = {r['pair_id'] for r in P.build_pilot_dataset(dist)}

    started = time.time()
    model = load_model()
    w_qs = [model.W_Q[layer, h].detach().cpu() for layer, h in P.NAME_MOVERS]
    if P.weights_fingerprint(w_qs) != pilot['provenance']['name_mover_queries_sha256']:
        sys.exit('refused: name-mover queries differ from the pilot')
    directions, audit = P.decompose_namemovers(P.load_direction(
        SOURCE_DIR / 'das_resid_mid.joblib'), w_qs)
    if audit != pilot['decomposition']:
        sys.exit('refused: decomposition differs from the pilot')
    cells = P.run_conditions(model, dataset, directions)
    effect = P.effects(cells)
    d = P.contrasts(effect)

    realized = {(r, x, n): F.realized(d[r][x]['d_inert'][:n], d[r][x]['d_all'][:n])
                for r in F.READOUTS for x in F.COMPONENTS for n in F.LADDER}
    scored = F.score(predictions['primary'], realized)
    observed_crossover = {
        f'{r}/{x}': next((n for n in F.LADDER if realized[(r, x, n)]['decided']), None)
        for r in F.READOUTS for x in F.COMPONENTS}
    secondary = []
    for cell in predictions['secondary']:
        r, x, n = cell['readout'], cell['component'], cell['n']
        secondary.append(dict(cell, **F.block_rate(d[r][x]['d_inert'], d[r][x]['d_all'], n)))
    full_n = len(dataset)
    unblinded = {
        r: {'mean_E_full': float(effect[r]['full'].mean()),
            **{x: {'mean_E': float(effect[r][x].mean()),
                   'realized_at_n_max': realized[(r, x, full_n)]}
               for x in F.COMPONENTS}}
        for r in F.READOUTS}

    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = {
        'primary': scored, 'predicted_crossover': predictions['crossover'],
        'observed_crossover': observed_crossover, 'secondary': secondary,
        'unblinded': unblinded, 'z': F.Z,
        'provenance': {
            'git_head': head, 'predictions_sha256': PREDICTIONS_SHA,
            'seed': F.CONFIRMATION_SEED, 'pairs': full_n, 'order': 'interleaved',
            'dataset_sha256': fingerprint(dataset),
            'pairs_also_in_pilot': len(pilot_ids & {r['pair_id'] for r in dataset}),
            'pythonhashseed': os.environ['PYTHONHASHSEED'],
            'symbol_order': ''.join(P.SYMBOL_ORDER), 'site': P.SITE_NAME,
            'python': platform.python_version(), 'torch': torch.__version__,
            'runtime_seconds': round(time.time() - started, 1),
        },
    }
    with open(OUTPUT / 'result.json', 'x') as handle:
        handle.write(json.dumps(result, indent=1) + '\n')
    with open(OUTPUT / 'per_pair.jsonl', 'x') as handle:
        for i, record in enumerate(dataset):
            row = {'position': i, 'pair_id': record['pair_id'], 'half': record['half']}
            for condition in P.CONDITIONS:
                row[f'ld_{condition}'] = float(cells[condition]['ld'][i])
                row[f'interchange_{condition}'] = int(cells[condition]['interchange'][i])
            handle.write(json.dumps(row) + '\n')
    print(json.dumps({k: result[k] for k in ('predicted_crossover', 'observed_crossover')},
                     indent=1))
    print(f"primary: {scored['matches']}/{scored['total']} match, "
          f"{scored['outside_band_misses']} outside-band misses -> {scored['verdict']}")
    for s in secondary:
        print(f"secondary A {s['component']:>4} n={s['n']:>2} ratio={s['ratio']:.2f} "
              f"predicted={s['decidable']} realized_rate={s['rate']:.3f} ({s['blocks']} blocks)")
    for r in F.READOUTS:
        for x in F.COMPONENTS:
            u = unblinded[r][x]['realized_at_n_max']
            print(f"{r} {x:>4}: E(full)={unblinded[r]['mean_E_full']:.4f} "
                  f"E(X)={unblinded[r][x]['mean_E']:.4f}  inert compatible="
                  f"{u['inert_compatible']}  all compatible={u['all_compatible']}")


if __name__ == '__main__':
    main()
