"""Apply the category preflight to pinned public Goodfire MCQA input tables.

Standard library only. Downloads hash-checked source files if absent; never
imports upstream code, loads a model, fits a subspace, or generates completions.
"""
from collections import Counter
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import urllib.request

APP = Path(__file__).resolve().parent
ROOT = APP.parents[1]
CACHE = APP / 'artifacts' / 'upstream'
RESULTS = APP / 'results'
RIVALS = ('position_transfer', 'symbol_transfer', 'preserve_task_answer')


def load_inputs():
    manifest = json.loads((CACHE / 'source_manifest.json').read_text())
    files = {}
    for name, entry in manifest['files'].items():
        path = CACHE / 'raw' / entry['path']
        if not path.exists():
            with urllib.request.urlopen(entry['url'], timeout=30) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError(f'upstream hash mismatch: {name}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f'cached hash mismatch: {name}')
        files[name] = raw
    return manifest, files


def predictions(row):
    """Evaluate the high-level rules, NOT unmeasured neural behavior."""
    if len(row['counterfactual_inputs_variables']) != 1:
        raise ValueError('expected one donor per recipient')
    donor = row['counterfactual_inputs_variables'][0]
    base_pos = [row[f'choices[{i}]'] for i in range(2)].index(row['color'])
    donor_pos = [donor[f'choices[{i}]'] for i in range(2)].index(donor['color'])
    base_answer = row[f'symbols[{base_pos}]']
    donor_answer = donor[f'symbols[{donor_pos}]']
    if (base_pos != int(row['answer_position'])
            or donor_pos != int(donor['answer_position'])
            or base_answer != row['answer'] or donor_answer != donor['answer']
            or base_answer != row['base_answer'].strip()
            or donor_answer != row['cf_answer'].strip()):
        raise ValueError('task equations disagree with serialized variables')
    return {
        'position_transfer': row[f'symbols[{donor_pos}]'],
        'symbol_transfer': donor_answer,
        # This is a task-level rule, not an observed unpatched neural output.
        'preserve_task_answer': base_answer,
    }


def make_config(design, rows, revision):
    config = {
        'prediction_kind': 'category',
        'prediction_source': f'Goodfire causalab {revision}; MCQA equations. '
                             'Task-level predictions, not observed model answers. '
                             'A true no-effect rule needs measured recipient baselines.',
        'cells': [], 'predictions': {rival: [] for rival in RIVALS},
    }
    for index, row in enumerate(rows):
        values = predictions(row)
        target = 'position_transfer' if design == 'pointer' else 'symbol_transfer'
        if values[target] != row['label'].strip():
            raise ValueError(f'{design} row {index}: source target label mismatch')
        config['cells'].append({'id': f'{design}_{row["split"]}_{index:03d}'})
        for rival in RIVALS:
            config['predictions'][rival].append(values[rival])
    return config


def describe(config, check):
    result = check(config, structure_only=True)
    disagreements = {
        f'{a} vs {b}': sum(x != y for x, y in zip(config['predictions'][a], config['predictions'][b]))
        for a, b in combinations(RIVALS, 2)
    }
    return {'rows': len(config['cells']),
            'identical_prediction_groups': result['identical_category_groups'],
            'disagreeing_rows': disagreements,
            'measurement_resolution': 'not assessed; categorical predictions have no numeric gap'}


def main():
    manifest, files = load_inputs()
    spec = importlib.util.spec_from_file_location('causal_preflight', ROOT / 'examples/causal_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    RESULTS.mkdir(exist_ok=True)
    summary = {
        'source_revision': manifest['revision'],
        'analysis_sources_sha256': {
            'run.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'examples/causal_preflight.py': hashlib.sha256((ROOT / 'examples/causal_preflight.py').read_bytes()).hexdigest(),
            'src/causal_decidability/design.py': hashlib.sha256((ROOT / 'src/causal_decidability/design.py').read_bytes()).hexdigest(),
            'source_manifest.json': hashlib.sha256((CACHE / 'source_manifest.json').read_bytes()).hexdigest(),
        },
        'analysis': 'Retrospective design audit of public inputs; zero model forwards.',
        'scope': 'Finite categorical prediction patterns of declared task-level rules only.',
        'designs': {},
        'empirical_no_effect_status': 'UNKNOWN per case: task answer is not a measured baseline.',
        'confirmation_status': 'Not a fresh prediction, held-out method validation, or neural mechanism finding.',
    }
    configs = []
    for design in ('pointer', 'symbol'):
        rows = json.loads(files[f'{design}_data'])
        config = make_config(design, rows, manifest['revision'])
        configs.append(config)
        summary['designs'][design] = {
            **describe(config, module.check),
            'splits': dict(Counter(row['split'] for row in rows)),
            'test_design_only': describe(make_config(design, [r for r in rows if r['split'] == 'test'], manifest['revision']), module.check),
            'published_clean_summary_not_recomputed': json.loads(files[f'{design}_results'])['clean'],
            'first_source_row': {
                'recipient': rows[0]['input'],
                'donor': rows[0]['counterfactual_inputs'][0],
                'predictions': predictions(rows[0]),
            },
        }
        (RESULTS / f'{design}_predictions.json').write_text(json.dumps(config, indent=2) + '\n')
    combined = {
        'prediction_kind': 'category',
        'prediction_source': configs[0]['prediction_source'] +
                             ' Combined input menu requires ONE fixed intervention for an empirical comparison; '
                             'separately fitted upstream subspaces are not the same intervention.',
        'cells': configs[0]['cells'] + configs[1]['cells'],
        'predictions': {r: configs[0]['predictions'][r] + configs[1]['predictions'][r] for r in RIVALS},
    }
    summary['combined_input_menu_only'] = describe(combined, module.check)
    summary['next_step'] = (
        'Hold one layer, token position and learned patch operator fixed; evaluate both input families; '
        'record per-case native and patched output categories and scores. Keep development and confirmation separate.'
    )
    (RESULTS / 'combined_predictions.json').write_text(json.dumps(combined, indent=2) + '\n')
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Goodfire MCQA: pinned source tables -> existing category preflight (no model run)')
    for name, result in list(summary['designs'].items()) + [('combined input menu', summary['combined_input_menu_only'])]:
        print(f'\n{name}: {result["rows"]} rows; {len(result["identical_prediction_groups"])} prediction groups')
        for group in result['identical_prediction_groups']:
            print('  ' + ' = '.join(group))
    print('\nMeasurement resolution: UNKNOWN; no numeric gap/power forecast.')
    print('No-effect interpretation: requires measured neural baselines, not task labels.')
    print('Saved reproducible results to', RESULTS)


if __name__ == '__main__':
    main()
