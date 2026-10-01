"""Create-only CPU DEVELOPMENT benchmark; no confirmation/holdout execution path."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from causal_decidability.compatible_set import compatible_set


def load_local(name):
    spec = importlib.util.spec_from_file_location('design_comparison_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worlds = load_local('worlds')
selection = load_local('selection')
BANDS = {'small': (0.1, 0.3), 'medium': (0.3, 1.0), 'large': (1.0, 3.0)}
TRUTHS = ('channel_a', 'channel_b', 'balanced', 'joint', 'context_switch',
          'context_switch_inverse', worlds.OUTSIDE_CANDIDATE)
SOURCES = [HERE / name for name in ('run.py', 'selection.py', 'worlds.py', 'PROTOCOL.md')]
SOURCES += [ROOT / 'src/causal_decidability/compatible_set.py',
            ROOT / 'docs/MECHANISM_DISCRIMINATION_GOAL.md']
PUBLIC_KEYS = {'case_id', 'family', 'amplitude', 'menu', 'parameters', 'predictions',
               'equivalence_groups', 'numerical_bound', 'controls', 'amplitude_band',
               'noise_level', 'known_sigmas'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Existing artifact will not be overwritten: {path}')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def write_lines(path, values):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def git_info():
    def get(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return {'commit': get('rev-parse', 'HEAD'), 'branch': get('branch', '--show-current'),
            'dirty_status': get('status', '--short')}


def verify_sources(manifest):
    for name, expected in manifest['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Source changed since generation: {name}; use a NEW run directory')


def verify_artifacts(directory, bindings):
    for name, expected in bindings.items():
        if sha(directory / name) != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')


def generate(directory, *, cases=512, seed=2026092901, samples=8):
    if cases < 1 or samples < 1:
        raise ValueError('Positive independent case and sample counts required')
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = {'status': 'development_only', 'cases': cases, 'seed': seed, 'samples': samples,
              'alpha_per_case_policy': 0.005, 'primary_extra_cells': 3,
              'extra_cell_budgets': [1, 2, 3, 4], 'amplitude_bands': BANDS,
              'noise_levels': [0.03, 0.10, 0.30],
              'development_families': list(worlds.DEVELOPMENT_FAMILIES),
              'reserved_families_NOT_EXECUTED': list(worlds.HELDOUT_FAMILIES),
              'truth_sampling': list(TRUTHS),
              'measurement_model': 'known independent Gaussian errors added to deterministic fp32 graph outputs',
              'unit': 'one independently generated circuit/context instance; policies and cells paired'}
    manifest = {'config': config, 'git': git_info(),
                'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
                'environment': {'python': sys.version, 'numpy': np.__version__,
                                'torch': torch.__version__, 'platform': platform.platform(),
                                'device': 'cpu', 'graph_dtype': 'float32', 'reference_dtype': 'float64'},
                'created_unix': time.time()}
    write_json(directory / 'manifest.json', manifest)  # before the first case outcome
    rng = random.Random(seed)
    public, observations, labels = [], [], []
    started = time.perf_counter()
    qualification_seconds, truth_seconds = 0.0, 0.0
    for index in range(cases):
        case_seed = rng.randrange(2**63)
        family = rng.choice(worlds.DEVELOPMENT_FAMILIES)
        band = rng.choice(list(BANDS))
        amplitude = rng.uniform(*BANDS[band])
        noise = rng.choice(config['noise_levels'])
        truth = rng.choice(TRUTHS)
        phase_started = time.perf_counter()
        case = worlds.public_case(case_seed, family, amplitude)
        qualification_seconds += time.perf_counter() - phase_started
        assert family not in worlds.HELDOUT_FAMILIES
        nrng = np.random.default_rng(case_seed)
        # Known heterogeneous costs of measurement, available equally to every policy.
        sigmas = noise * nrng.uniform(0.6, 1.6, len(case['menu']))
        case.update(amplitude_band=band, noise_level=noise, known_sigmas=sigmas.tolist())
        phase_started = time.perf_counter()
        values = np.asarray(worlds.observe_truth(case, truth))
        truth_seconds += time.perf_counter() - phase_started
        # Potential measurements are generated once; policy budgets count consumed draws.
        # Full-only can spend up to twice the ordinary per-cell samples on its anchors.
        draws = values[:, None] + sigmas[:, None] * nrng.normal(size=(len(values), 2 * samples))
        public.append(case)
        observations.append({'case_id': case['case_id'], 'samples': draws.tolist()})
        truth_class = next((g for g in case['equivalence_groups'] if truth in g), [])
        labels.append({'case_id': case['case_id'], 'truth_name': truth,
                       'true_full_menu_class': sorted(truth_class), 'in_set': bool(truth_class)})
        if (index + 1) % 128 == 0:
            print(f'Generated {index + 1}/{cases} development instances', flush=True)
    write_lines(directory / 'public_cases.jsonl', public)
    write_lines(directory / 'measurements.jsonl', observations)
    write_lines(directory / 'private_labels.jsonl', labels)
    write_json(directory / 'generation.json', {
        'status': 'development_only', 'elapsed_seconds': time.perf_counter() - started,
        'artifacts': {name: sha(directory / name) for name in
                      ('manifest.json', 'public_cases.jsonl', 'measurements.jsonl', 'private_labels.jsonl')},
        'all_controls_passed': all(c['controls']['all_passed'] for c in public),
        'maximum_numerical_bound': max(c['numerical_bound'] for c in public),
        'reserved_structures_executed': 0,
        'common_qualification_seconds': qualification_seconds,
        'truth_execution_seconds': truth_seconds,
        'common_qualification_graph_forwards': sum(c['controls']['qualification_graph_forward_count'] for c in public),
        'common_analytic_formula_evaluations': sum(c['controls']['analytic_formula_evaluation_count'] for c in public),
        'actual_truth_graph_cell_evaluations': sum(len(c['menu']) for c in public),
        'potential_measurement_draws': sum(len(c['menu']) * 2 * samples for c in public),
        'cost_note': 'Qualification executes all candidate/reference graphs as common setup. '
                     'Budget comparisons count consumed potential measurements, not physical LLM savings.'})


def prediction_rows(directory):
    manifest = read_json(directory / 'manifest.json')
    verify_sources(manifest)
    generation = read_json(directory / 'generation.json')
    # Prediction does NOT open the private label file, even to hash it.
    verify_artifacts(directory, {name: value for name, value in generation['artifacts'].items()
                                 if name != 'private_labels.jsonl'})
    cases = lines(directory / 'public_cases.jsonl')
    measured = {row['case_id']: row for row in lines(directory / 'measurements.jsonl')}
    cfg = manifest['config']
    result = []
    for case in cases:
        if set(case) != PUBLIC_KEYS:
            raise ValueError('Unexpected public case fields (possible label leakage)')
        if case['family'] not in worlds.DEVELOPMENT_FAMILIES or not case['controls']['all_passed']:
            raise ValueError('Reserved or unqualified case')
        raw_record = measured[case['case_id']]
        if set(raw_record) != {'case_id', 'samples'}:
            raise ValueError('Unexpected measurement fields')
        raw = np.asarray(raw_record['samples'])
        if raw.shape != (len(case['menu']), 2 * cfg['samples']) or not np.isfinite(raw).all():
            raise ValueError('Invalid or incomplete potential measurements')
        for extras in cfg['extra_cell_budgets']:
            start = time.perf_counter()
            # Only public, pre-outcome quantities cross this boundary.
            picked = selection.select(case['predictions'], case['equivalence_groups'],
                case['menu'], case['known_sigmas'], case['numerical_bound'],
                mandatory_count=worlds.MANDATORY_COUNT, extras=extras,
                samples=cfg['samples'], alpha=cfg['alpha_per_case_policy'],
                seed=int(case['case_id'][:12], 16) + extras)
            elapsed = time.perf_counter() - start
            for policy, plan in picked['plans'].items():
                cells, counts = plan['cells'], plan['counts']
                all_counts = np.full(len(case['menu']), cfg['samples'], dtype=int)
                all_counts[cells] = counts
                all_radii = selection.radii(case['known_sigmas'], all_counts,
                    cfg['alpha_per_case_policy'], case['numerical_bound'])
                estimated = [float(raw[j, :n].mean()) for j, n in zip(cells, counts)]
                prediction = {name: [values[j] for j in cells]
                              for name, values in case['predictions'].items()}
                decision = compatible_set(prediction, estimated, all_radii[cells].tolist(),
                                          equivalence_groups=case['equivalence_groups'])
                result.append({'case_id': case['case_id'], 'policy': policy, 'extras': extras,
                    'cells': cells, 'cell_names': [case['menu'][j] for j in cells],
                    'counts': counts, 'cost': plan['cost'], 'estimate': estimated,
                    'radius': all_radii[cells].tolist(), 'retained': decision['retained'],
                    'retained_groups': decision['retained_groups'], 'outcome': decision['outcome'],
                    'planned_min_separation_ratio': picked['maximin_score'],
                    'shared_selection_seconds': elapsed,
                    'subsets_examined': picked['subsets_examined']})
    return result


def predict(directory):
    if (directory / 'predictions.jsonl').exists():
        raise FileExistsError('Predictions are already sealed')
    result = prediction_rows(directory)
    write_lines(directory / 'predictions.jsonl', result)
    write_json(directory / 'prediction_seal.json', {
        'status': 'development_only',
        'artifacts': {name: sha(directory / name) for name in
                      ('generation.json', 'predictions.jsonl')},
        'records': len(result), 'labels_opened_by_prediction_stage': False,
        'timestamp_unix': time.time()})


def binomial_upper(k, n, alpha):
    """Exact one-sided Clopper-Pearson bound, standard library, including k=0."""
    if not 0 <= k <= n or n < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid binomial inputs')
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(alpha) / n)
    constants = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                 for j in range(k + 1)]
    lo, hi = k / n, 1.0
    for _ in range(70):
        p = (lo + hi) / 2
        terms = [c + j * math.log(p) + (n-j) * math.log1p(-p)
                 for j, c in enumerate(constants)]
        largest = max(terms)
        cdf = math.exp(largest) * sum(math.exp(t - largest) for t in terms)
        if cdf > alpha:
            lo = p
        else:
            hi = p
    return hi


def metrics(rows):
    n = len(rows)
    return {'n': n, **{name: sum(r[name] for r in rows) for name in
        ('exact_class', 'truth_excluded', 'false_single_class', 'no_candidate_fits',
         'multiple_remain', 'all_remain')},
        'exact_class_rate': sum(r['exact_class'] for r in rows) / n if n else None,
        'mean_cost': sum(r['cost'] for r in rows) / n if n else None}


def summarize(directory):
    manifest = read_json(directory / 'manifest.json')
    verify_sources(manifest)
    verify_artifacts(directory, read_json(directory / 'generation.json')['artifacts'])
    verify_artifacts(directory, read_json(directory / 'prediction_seal.json')['artifacts'])
    public = {r['case_id']: r for r in lines(directory / 'public_cases.jsonl')}
    truth = {r['case_id']: r for r in lines(directory / 'private_labels.jsonl')}
    predictions = lines(directory / 'predictions.jsonl')
    expected_count = len(public) * 4 * (len(selection.POLICIES) + 1)
    keys = [(r['case_id'], r['extras'], r['policy']) for r in predictions]
    if len(keys) != expected_count or len(keys) != len(set(keys)):
        raise ValueError('Missing or duplicate predictions')
    scored = []
    for row in predictions:
        label = truth[row['case_id']]
        retained = set(row['retained'])
        expected = set(label['true_full_menu_class'])
        in_set = label['in_set']
        case = public[row['case_id']]
        ratio = row['planned_min_separation_ratio']
        scored.append({**row, **label, 'family': case['family'],
            'amplitude_band': case['amplitude_band'], 'noise_level': case['noise_level'],
            'planned_ratio_band': '<1' if ratio < 1 else '1–2' if ratio < 2 else '>=2',
            'exact_class': retained == expected,
            'truth_excluded': bool(in_set and not expected.issubset(retained)),
            'false_single_class': bool(len(row['retained_groups']) == 1 and retained != expected),
            'no_candidate_fits': not retained,
            'multiple_remain': len(row['retained_groups']) > 1,
            'all_remain': row['outcome'] == 'insufficient_evidence'})
    primary = [r for r in scored if r['extras'] == manifest['config']['primary_extra_cells']]
    table, strata = {}, {}
    for policy in (*selection.POLICIES, 'full_menu'):
        inside = [r for r in primary if r['policy'] == policy and r['in_set']]
        outside = [r for r in primary if r['policy'] == policy and not r['in_set']]
        value = {'in_set': metrics(inside), 'out_of_set': metrics(outside)}
        value['in_set']['false_exclusion_upper_simultaneous_95'] = (
            binomial_upper(sum(r['truth_excluded'] for r in inside), len(inside), 0.05/6)
            if inside else None)
        table[policy] = value
    by_key = {(r['case_id'], r['policy']): r for r in primary if r['in_set']}
    comparisons = {}
    for baseline in ('mean_pairwise', 'balanced_split', 'random'):
        differences = [int(r['exact_class']) - int(by_key[(r['case_id'], baseline)]['exact_class'])
                       for r in primary if r['in_set'] and r['policy'] == 'maximin']
        n = len(differences)
        mean = sum(differences) / n if n else None
        radius = math.sqrt(2 * math.log(2 * 3 / 0.05) / n) if n else None
        comparisons[baseline] = {'n': n, 'paired_advantage': mean,
            'simultaneous_95_hoeffding_interval':
                [max(-1, mean-radius), min(1, mean+radius)] if n else None,
            'wins': differences.count(1), 'losses': differences.count(-1),
            'ties': differences.count(0),
            'interpretation': 'Conservative development interval; not fresh confirmation'}
    for dimension in ('family', 'truth_name', 'amplitude_band', 'noise_level', 'planned_ratio_band'):
        grouped = defaultdict(list)
        for row in primary:
            if row['in_set']:
                grouped[(str(row[dimension]), row['policy'])].append(row)
        strata[dimension] = {f'{value}/{policy}': metrics(rows)
                             for (value, policy), rows in sorted(grouped.items())}
    budget_table = {}
    for budget in manifest['config']['extra_cell_budgets']:
        budget_table[str(budget)] = {policy: metrics([r for r in scored if
            r['extras'] == budget and r['policy'] == policy and r['in_set']])
            for policy in (*selection.POLICIES, 'full_menu')}
    return {'status': 'development_only', 'case_count': len(public),
        'config': manifest['config'],
        'controls': read_json(directory / 'generation.json'), 'primary': table,
        'paired_comparisons': comparisons, 'strata': strata, 'budget_curve': budget_table,
        'all_equivalence_groups_preserved': all(
            ('channel_a' in r['retained']) == ('channel_a_alias' in r['retained']) for r in predictions),
        'reserved_structures_executed': 0,
        'primary_comparator': 'mean_pairwise',
        'claim': 'Development comparison of explicit selection objectives on declared known circuits. '
                 'No confirmation, novelty, native LLM mechanism, or general superiority established.'}


def report_text(summary):
    text = ['# Intervention selection: development results', '',
            '**Development only. No reserved structural family was evaluated.**', '',
            f"Independent circuit instances: {summary['case_count']}. Primary: four anchors + three additional cells; {summary['config']['samples']} samples per cell.", '',
            '| Policy | Correct class (in set) | Truth excluded | False single class | OOS rejected | Cost |',
            '|---|---:|---:|---:|---:|---:|']
    for policy, result in summary['primary'].items():
        inside, outside = result['in_set'], result['out_of_set']
        text.append(f"| {policy} | {inside['exact_class']}/{inside['n']} | "
                    f"{inside['truth_excluded']}/{inside['n']} | {inside['false_single_class']} | "
                    f"{outside['no_candidate_fits']}/{outside['n']} | {inside['mean_cost']} |")
    text += ['', 'Full-menu is a higher-cost reference. Full-only is a negative control.', '',
             '| Maximin minus | Paired advantage | Simultaneous conservative 95% interval | Wins / losses / ties |',
             '|---|---:|---:|---:|']
    for policy, result in summary['paired_comparisons'].items():
        if not result['n']:
            continue
        lo, hi = result['simultaneous_95_hoeffding_interval']
        text.append(f"| {policy} | {100*result['paired_advantage']:.2f} pp | "
                    f"[{100*lo:.2f}, {100*hi:.2f}] pp | "
                    f"{result['wins']} / {result['losses']} / {result['ties']} |")
    text += ['', '## Interpretation', '',
             'The primary comparator is mean_pairwise, not full-only or random. An advantage over the latter '
             'does not establish an advantage over established discrimination design. All policies use the '
             'same production compatible-set classifier.', '',
             'Truth-exclusion upper bounds (simultaneous over six reported policies), per-family, '
             'amplitude/noise/ratio strata, and secondary budgets are in summary.json. Graph outputs are '
             'deterministic; Gaussian measurement noise is added deliberately and its variance is known. '
             'These are generic scores, not LLM nats. Parameterized two-unit circuits and their known '
             'prediction tables are a restricted testbed.', '',
             'The structural holdout remains unopened. A fresh evaluation requires a reviewed, powered '
             'confirmation contract. This development run cannot be promoted by relabeling fresh seeds.', '',
             summary['claim'], '']
    return '\n'.join(text)


def score(directory):
    summary = summarize(directory)
    write_json(directory / 'summary.json', summary)
    report = directory / 'REPORT.md'
    with report.open('x') as handle:
        handle.write(report_text(summary))
    write_json(directory / 'score_seal.json', {'artifacts': {
        name: sha(directory / name) for name in ('prediction_seal.json', 'summary.json', 'REPORT.md')}})
    return summary


def verify(directory):
    verify_artifacts(directory, read_json(directory / 'score_seal.json')['artifacts'])
    stored = read_json(directory / 'summary.json')
    if summarize(directory) != stored:
        raise ValueError('Stored summary differs from recomputation')
    # Also re-run production inference, independent of saved retained sets.
    recomputed = prediction_rows(directory)
    saved = lines(directory / 'predictions.jsonl')
    for actual, expected in zip(recomputed, saved):
        actual.pop('shared_selection_seconds')
        expected.pop('shared_selection_seconds')
        if actual != expected:
            raise ValueError('Decision does not reproduce from public predictions and measurements')
    return {'verified': True, 'cases': stored['case_count'],
            'reserved_structures_executed': 0, 'status': 'development_only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'predict', 'score', 'verify', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2026092901)
    parser.add_argument('--samples', type=int, default=8)
    args = parser.parse_args()
    if args.command in ('generate', 'run'):
        generate(args.out, cases=args.cases, seed=args.seed, samples=args.samples)
    if args.command in ('predict', 'run'):
        predict(args.out)
    if args.command in ('score', 'run'):
        summary = score(args.out)
        print(report_text(summary))
    if args.command in ('verify', 'run'):
        print(json.dumps(verify(args.out), indent=2))


if __name__ == '__main__':
    main()
