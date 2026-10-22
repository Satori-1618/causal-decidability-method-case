"""Apply the category preflight to pinned public Goodfire CausaLab MCQA input tables.

Standard library only. Downloads hash-checked source files if absent; never
imports upstream code, loads a model, fits a subspace, or generates completions.

    python3 applications/goodfire-mcqa-preflight/run.py           # check, write nothing
    python3 applications/goodfire-mcqa-preflight/run.py --write   # regenerate results/

The default ``--check`` keeps downloads in memory and compares everything it derives
with the committed files in results/. ``--write`` caches the downloads under
artifacts/upstream/raw/ (git-ignored) and overwrites the files in results/.
"""
import argparse
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
# A fourth, proposed rule: transfer the donor's correct colour and look up its letter in
# the recipient's options. It is kept out of the three-rule exports above.
FOUR_RULES = RIVALS + ('color_transfer',)


def load_inputs(write_cache):
    manifest = json.loads((CACHE / 'source_manifest.json').read_text())
    files = {}
    for name, entry in manifest['files'].items():
        path = CACHE / 'raw' / entry['path']
        if path.exists():
            raw = path.read_bytes()
        else:
            with urllib.request.urlopen(entry['url'], timeout=30) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError(f'upstream hash mismatch: {name}')
            if write_cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
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


def color_transfer(recipient, donor):
    """Letter of the donor's correct colour among the recipient's options, or None."""
    choices = [recipient[f'choices[{i}]'] for i in range(2)]
    if donor['color'] not in choices:
        return None
    return recipient[f'symbols[{choices.index(donor["color"])}]']


def four_rule_row(row):
    values = predictions(row)
    values['color_transfer'] = color_transfer(row, row['counterfactual_inputs_variables'][0])
    return values


def proposed_color_row(row):
    """Proposed donor: the recipient's own options, but the other colour is correct."""
    choices = [row[f'choices[{i}]'] for i in range(2)]
    other = choices[1 - choices.index(row['color'])]
    donor_pos = choices.index(other)
    return {
        'position_transfer': row[f'symbols[{donor_pos}]'],
        'symbol_transfer': row[f'symbols[{donor_pos}]'],
        'preserve_task_answer': row[f'symbols[{choices.index(row["color"])}]'],
        'color_transfer': row[f'symbols[{donor_pos}]'],
    }


def four_rule_config(families):
    config = {'prediction_kind': 'category',
              'prediction_source': 'Task-level rules on published recipients; the colour family '
                                   'is a proposed design, not published data.',
              'cells': [], 'predictions': {rule: [] for rule in FOUR_RULES}}
    for family, rows in families:
        for index, values in enumerate(rows):
            config['cells'].append({'id': f'{family}_{index:03d}'})
            for rule in FOUR_RULES:
                config['predictions'][rule].append(values[rule])
    return config


def full_patch_by_layer(results):
    """The published fixed full-vector patch, scored on one input family."""
    return {str(entry['layer']): entry['test_iia'] for entry in results['full_patch']}


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


def build(files, manifest, check):
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
            **describe(config, check),
            'splits': dict(Counter(row['split'] for row in rows)),
            'test_design_only': describe(make_config(design, [r for r in rows if r['split'] == 'test'], manifest['revision']), check),
            'published_clean_summary_not_recomputed': json.loads(files[f'{design}_results'])['clean'],
            'first_source_row': {
                'recipient': rows[0]['input'],
                'donor': rows[0]['counterfactual_inputs'][0],
                'predictions': predictions(rows[0]),
            },
        }
    combined = {
        'prediction_kind': 'category',
        'prediction_source': configs[0]['prediction_source'] +
                             ' Combined input menu requires ONE fixed intervention for an empirical comparison; '
                             'separately fitted upstream subspaces are not the same intervention.',
        'cells': configs[0]['cells'] + configs[1]['cells'],
        'predictions': {r: configs[0]['predictions'][r] + configs[1]['predictions'][r] for r in RIVALS},
    }
    summary['combined_input_menu_only'] = describe(combined, check)
    summary['next_step'] = (
        'Hold one layer, token position and learned patch operator fixed; evaluate both input families; '
        'record per-case native and patched output categories and scores. Keep development and confirmation separate.'
    )
    pointer, symbol = (json.loads(files[f'{d}_data']) for d in ('pointer', 'symbol'))
    families = {'published_position_design': [four_rule_row(r) for r in pointer],
                'published_letter_design': [four_rule_row(r) for r in symbol],
                'proposed_color_design': [proposed_color_row(r) for r in pointer]}
    menus = {'position design only': ['published_position_design'],
             'letter design only': ['published_letter_design'],
             'proposed colour design only': ['proposed_color_design'],
             'both published designs': ['published_position_design', 'published_letter_design'],
             'all three designs': list(families)}
    four_rules = {
        'rules': list(FOUR_RULES),
        'status': 'design analysis of task-level rules; the colour design is proposed, not run',
        'menus': {name: {'rows': sum(len(families[f]) for f in members),
                         'identical_prediction_groups': check(four_rule_config(
                             [(f, families[f]) for f in members]), structure_only=True)[
                             'identical_category_groups']}
                  for name, members in menus.items()},
    }
    full_patch = {
        'source': 'iia_plotted.json of each published design (field full_patch, test split)',
        'reading': 'One fixed full-vector patch per layer, scored against each design\'s own '
                   'target label. This is the authors\' measurement, reported here unchanged.',
        'position_design': full_patch_by_layer(json.loads(files['pointer_results'])),
        'letter_design': full_patch_by_layer(json.loads(files['symbol_results'])),
    }
    outputs = {'pointer_predictions.json': configs[0], 'symbol_predictions.json': configs[1],
               'combined_predictions.json': combined, 'summary.json': summary,
               'four_rule_extension.json': {'four_rule_design_analysis': four_rules,
                                            'published_full_patch_iia': full_patch}}
    return {name: json.dumps(value, indent=2) + '\n' for name, value in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true',
                        help='cache downloads and overwrite the committed files in results/')
    args = parser.parse_args()
    manifest, files = load_inputs(write_cache=args.write)
    spec = importlib.util.spec_from_file_location('causal_preflight', ROOT / 'examples/causal_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outputs = build(files, manifest, module.check)
    if args.write:
        RESULTS.mkdir(exist_ok=True)
        for name, text in outputs.items():
            (RESULTS / name).write_text(text)
    else:
        report = {'mode': 'check; nothing written', 'files': {}}
        for name, text in outputs.items():
            path = RESULTS / name
            committed = path.read_text() if path.exists() else None
            if name == 'summary.json' and committed is not None:
                fresh, stored = json.loads(text), json.loads(committed)
                stale = sorted(k for k, v in fresh['analysis_sources_sha256'].items()
                               if stored['analysis_sources_sha256'].get(k) != v)
                fresh.pop('analysis_sources_sha256'), stored.pop('analysis_sources_sha256')
                report['files'][name] = {'matches_except_analysis_hashes': fresh == stored,
                                         'stale_analysis_hashes': stale}
            else:
                report['files'][name] = {'byte_identical': committed == text}
        report['four_rule_design_analysis'] = json.loads(outputs['four_rule_extension.json'])[
            'four_rule_design_analysis']
        report['published_full_patch_iia'] = json.loads(outputs['four_rule_extension.json'])[
            'published_full_patch_iia']
        report['summary'] = json.loads(outputs['summary.json'])
        print(json.dumps(report, indent=2))
        ok = all(v.get('byte_identical', v.get('matches_except_analysis_hashes'))
                 for v in report['files'].values())
        raise SystemExit(0 if ok else 1)
    summary = json.loads(outputs['summary.json'])
    print('CausaLab MCQA: pinned source tables -> existing category preflight (no model run)')
    for name, result in list(summary['designs'].items()) + [('combined input menu', summary['combined_input_menu_only'])]:
        print(f'\n{name}: {result["rows"]} rows; {len(result["identical_prediction_groups"])} prediction groups')
        for group in result['identical_prediction_groups']:
            print('  ' + ' = '.join(group))
    print('\nMeasurement resolution: UNKNOWN; no numeric gap/power forecast.')
    print('No-effect interpretation: requires measured neural baselines, not task labels.')
    print('Saved reproducible results to', RESULTS)


if __name__ == '__main__':
    main()
