"""Verify Round-3A records without loading a tokenizer, model, torch or joblib.

The default checks use only the standard library. --full-bootstrap additionally
requires NumPy and reruns the frozen analyzer's two seeded whole-family bootstraps.
Available original inputs are byte-checked; absent originals are reported explicitly
and can be made fatal with --require-original-inputs. Serialized audits cannot prove
that model forwards occurred or reconstruct activation tensors from their hashes.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

APPLICATION = Path(__file__).resolve().parents[1]
REPOSITORY = APPLICATION.parents[1]
sys.path.insert(0, str(APPLICATION / 'src'))
from donor_factor_sampling import (DONOR_INDICES, HISTORICAL_CASE_FILES,
                                   OPTIONAL_HISTORICAL_FILES, validate_family)

PRECISIONS = ('float32', 'float64')
CELLS = ('00', '01', '10', '11')
PROFILES = ('position_only', 'identity_only')
CONTRASTS = ('I', 'P', 'J')
SITE = 'blocks.8.mlp.hook_post'
REQUIRED_FILES = {'manifest.json', 'cases.json', 'records.jsonl', 'summary.json',
                  'RUN_STARTED.json', 'measurements_float32.jsonl', 'measurements_float64.jsonl'}
CODE_PATHS = {'scripts/run_donor_factor.py', 'scripts/run_query_route.py',
              'src/makelov_read_source.py', 'src/query_route_sampling.py',
              'src/query_route_analysis.py', 'src/donor_factor.py',
              'src/donor_factor_sampling.py', 'src/donor_factor_analysis.py',
              'src/donor_factor_planning.py', 'DONOR_FACTOR_PROTOCOL.md'}
FLOAT_TOLERANCE = 1e-12


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def load(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def hexhash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def relative_path(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, f'unsafe artifact path: {name}')
    return Path(root) / path


def check_hash(path, digest, label):
    require(hexhash(digest), f'{label}: invalid sha256')
    require(Path(path).is_file(), f'missing required file: {label}')
    require(sha256(path) == digest, f'hash mismatch: {label}')


def finite(value, label, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0), f'{label}: invalid finite numeric value')
    return value


def vector(value, label, n=2, nonnegative=False):
    require(isinstance(value, list) and len(value) == n, f'{label}: wrong vector length')
    return [finite(v, label, nonnegative) for v in value]


def close(actual, expected, label):
    finite(actual, label)
    require(abs(actual - expected) <= FLOAT_TOLERANCE * max(1., abs(expected)),
            f'{label}: recomputed numeric value differs')


def independent_cp(k, n):
    """Direct binomial-polynomial inversion, independent of the producer's lgamma code."""
    tail = .025 / 4
    coefficients = [math.comb(n, j) for j in range(n + 1)]

    def probability(p, indices):
        return math.fsum(coefficients[j] * p**j * (1-p)**(n-j) for j in indices)

    def invert(indices, increasing):
        left, right = 0., 1.
        for _ in range(80):
            middle = (left + right) / 2
            if (probability(middle, indices) > tail) == increasing:
                right = middle
            else:
                left = middle
        return (left + right) / 2

    return [0. if k == 0 else invert(range(k, n + 1), True),
            1. if k == n else invert(range(k + 1), False)]


def audit_precision(raw, case, precision):
    label = f"{case['case_id']}.{precision}"
    baseline = raw['baseline_answer_logits']
    require(isinstance(baseline, list) and len(baseline) == 4, f'{label}: baseline row count')
    for row in baseline:
        vector(row, label)
    patched = raw['patched_answer_logits']
    require(set(patched) == set((*CELLS, 'zero')), f'{label}: missing patched logits')
    for cell, rows in patched.items():
        require(isinstance(rows, list) and len(rows) == 4, f'{label}.{cell}: expected four rows')
        for row in rows:
            vector(row, label)
    margins = [row[0] - row[1] for row in baseline]
    require(raw['baseline_margins'] == margins, f'{label}: baseline margin disagrees with logits')
    scale = max(1., *(abs(x) for row in baseline for x in row),
                *(abs(x) for rows in patched.values() for row in rows for x in row))
    tolerance = 64 * 2.**(-23 if precision == 'float32' else -52) * scale
    controls = raw['controls']
    require(controls['passed'] is True and controls['site'] == SITE
            and controls['absolute_position'] == case['position'], f'{label}: controls/site/position mismatch')
    require(controls['answer_logit_identity_tolerance'] == tolerance
            and controls['identity_tolerance'] == 2*tolerance, f'{label}: incorrect identity tolerance')
    errors = {'self_vs_baseline': max(abs(x-y) for a, b in zip(patched['00'], baseline)
                                      for x, y in zip(a, b)),
              'zero_vs_baseline': max(abs(x-y) for a, b in zip(patched['zero'], baseline)
                                      for x, y in zip(a, b)),
              'untouched_batch_rows': max(abs(x-y) for rows in patched.values()
                                           for a, b in zip(rows[2:], baseline[2:]) for x, y in zip(a, b))}
    require(controls['identity_errors'] == errors and all(v <= tolerance for v in errors.values()),
            f'{label}: identity error differs or exceeds tolerance')
    require(controls['identity_exact'] is all(v == 0 for v in errors.values()),
            f'{label}: incorrect exact identity flag')
    require(vector(controls['zero_alphas'], label) == [0., 0.], f'{label}: zero alpha is nonzero')
    expected_sources = {cell: [DONOR_INDICES[r][cell] for r in (0, 1)] for cell in CELLS}
    expected_sources['zero'] = [0, 1]
    require(canonical_hash(raw['source_indices']) == canonical_hash(expected_sources),
            f'{label}: wrong source donor mapping')
    require(hexhash(raw['source_activations_sha256']), f'{label}: missing source activation hash')
    per_prompt = raw['source_activation_sha256_per_prompt']
    require(isinstance(per_prompt, list) and len(per_prompt) == 4 and all(map(hexhash, per_prompt)),
            f'{label}: missing per-prompt activation hashes')
    fidelity = controls['fidelity']
    require(set(fidelity) == set((*CELLS, 'zero')), f'{label}: missing fidelity audits')
    for cell, audit in fidelity.items():
        require(type(audit['calls']) is int and audit['calls'] == 1 and audit['passed'] is True
                and audit['other_positions_unchanged'] is True, f'{label}.{cell}: failed insertion audit')
        error = vector(audit['insertion_error_max_per_item'], label, 4, True)
        budget = vector(audit['rounding_budget_per_item'], label, 4, True)
        norms = vector(audit['actual_delta_l2_per_item'], label, 4, True)
        require(all(e <= b for e, b in zip(error, budget)), f'{label}.{cell}: insertion exceeds budget')
        require(norms[2:] == [0., 0.], f'{label}.{cell}: nonrecipient rows changed')
        if cell in ('00', 'zero'):
            require(norms == [0.]*4 and error == [0.]*4, f'{label}.{cell}: self/zero delta changed tensor')
    panels = raw['panels']
    require(isinstance(panels, list) and len(panels) == 2, f'{label}: panel count differs')
    for r, panel in enumerate(panels):
        require(type(panel['recipient_index']) is int and panel['recipient_index'] == r
                and canonical_hash(panel['donor_indices']) == canonical_hash(DONOR_INDICES[r]),
                f'{label}: panel donor mapping differs')
        require(panel['baseline_margin'] == margins[r], f'{label}: panel baseline differs')
        require(panel['controls']['passed'] is True
                and panel['controls']['identity_tolerance'] == 2*tolerance,
                f'{label}: panel controls differ')
        for field in ('patched_margins', 'alphas', 'delta_l2'):
            require(set(panel[field]) == set(CELLS), f'{label}: missing panel {field}')
            for value in panel[field].values():
                finite(value, label, field == 'delta_l2')
        for cell in CELLS:
            require(panel['patched_margins'][cell] == patched[cell][r][0] - patched[cell][r][1],
                    f'{label}.{cell}: patched margin disagrees with logits')
            # GPT-2 Small's MLP width is 3072. Reverse triangle inequality bounds
            # the difference between intended and inserted vector norms from the
            # recorded maximum coordinate insertion error, without tensor recovery.
            audit = fidelity[cell]
            norm_error = abs(panel['delta_l2'][cell] - audit['actual_delta_l2_per_item'][r])
            norm_budget = math.sqrt(3072) * audit['insertion_error_max_per_item'][r]
            require(norm_error <= norm_budget + FLOAT_TOLERANCE * max(1., panel['delta_l2'][cell]),
                    f'{label}.{cell}: delta norm contradicts insertion audit')
        require(panel['alphas']['00'] == panel['delta_l2']['00'] == 0,
                f'{label}: self alpha or delta norm is nonzero')
    require(raw['model_forward_calls'] == 6 and raw['model_prompt_evaluations'] == 24,
            f'{label}: forward/prompt counts differ')


def independent_readouts(records):
    """Recompute cell contrasts and unit predicates without analyzer helper imports."""
    counts, resolved_count = dict.fromkeys(PROFILES, 0), 0
    unit_means = {p: {c: [] for c in CONTRASTS} for p in PRECISIONS}
    unresolved_ids = []
    for record in records:
        fits = dict.fromkeys(PROFILES, True)
        flat = {}
        for precision in PRECISIONS:
            values, panel_contrasts = [], []
            for panel in record['precisions'][precision]['panels']:
                d = {c: panel['patched_margins'][c] - panel['baseline_margin'] for c in CELLS}
                ip0, ip1 = d['10'] - d['00'], d['11'] - d['01']
                pi0, pi1 = d['01'] - d['00'], d['11'] - d['10']
                contrasts = {'I': ip0/2 + ip1/2, 'P': pi0/2 + pi1/2, 'J': ip1-ip0}
                fits['position_only'] &= max(abs(ip0), abs(ip1)) <= .25
                fits['identity_only'] &= max(abs(pi0), abs(pi1)) <= .25
                values.extend([*d.values(), ip0, ip1, pi0, pi1, *contrasts.values()])
                panel_contrasts.append(contrasts)
            for c in CONTRASTS:
                mean = panel_contrasts[0][c]/2 + panel_contrasts[1][c]/2
                unit_means[precision][c].append(mean)
                values.append(mean)
            flat[precision] = values
        resolved = max(abs(a-b) for a, b in zip(flat['float32'], flat['float64'])) <= .01
        resolved_count += resolved
        if not resolved:
            unresolved_ids.append(record['case_id'])
        for profile in PROFILES:
            counts[profile] += bool(resolved and fits[profile])
    return counts, resolved_count, unresolved_ids, unit_means


def interval_decision(bounds):
    low, high = bounds
    if low > .10:
        return 'positive_relevant'
    if high < -.10:
        return 'negative_relevant'
    if low > -.10 and high < .10:
        return 'equivalent'
    return 'unresolved'


def audit_summary(summary, records):
    n = len(records)
    require(summary['schema_version'] == 'donor-factor-analysis-v1', 'unsupported summary schema')
    expected_contract = {'mean_epsilon_nat': .10, 'invariance_epsilon_nat': .25,
                         'coverage': .8, 'profile_family_alpha': .025, 'mean_family_alpha': .025,
                         'numerical_budget_nat': .01, 'bootstrap_method': 'whole-unit percentile Bonferroni',
                         'bootstrap_draws_per_seed': 20000, 'primary_seed': 320260924,
                         'secondary_seed': 320260925, 'mc_endpoint_budget_nat': .01,
                         'mc_requires_identical_decisions': True}
    require(canonical_hash(summary['contract']) == canonical_hash(expected_contract),
            'summary analysis contract differs from frozen rules')
    counts, resolved, unresolved_ids, means = independent_readouts(records)
    require(summary['n_units'] == n and summary['n_resolved_units'] == resolved
            and summary['n_unresolved_units'] == n-resolved, 'independent unit/resolution count differs')
    require(set(summary['profiles']) == set(PROFILES), 'summary profile names differ')
    cp_error = 0.
    for profile, count in counts.items():
        stored = summary['profiles'][profile]
        require(stored['successes'] == count and stored['n'] == n and stored['fraction'] == count/n,
                f'independent count differs: {profile}')
        expected = independent_cp(count, n)
        interval = vector(stored['interval'], profile)
        error = max(abs(a-b) for a, b in zip(expected, interval))
        cp_error = max(cp_error, error)
        require(error <= FLOAT_TOLERANCE, f'independent CP interval differs: {profile}')
        status = 'adequate' if expected[0] > .8 else 'excluded' if expected[1] < .8 else 'unresolved'
        require(stored['status'] == status, f'independent profile status differs: {profile}')
    require(set(summary['mean_effects']) == set(CONTRASTS), 'summary mean contrasts differ')
    for c in CONTRASTS:
        result = summary['mean_effects'][c]
        precisions = result['precisions']
        require(set(precisions) == set(PRECISIONS), 'mean precision set differs')
        for precision, entry in precisions.items():
            close(entry['mean'], math.fsum(means[precision][c])/n, f'independent mean {precision}.{c}')
            first, second = (vector(entry[key], c) for key in ('interval', 'secondary_interval'))
            require(first[0] <= first[1] and second[0] <= second[1], 'bootstrap interval inverted')
            decision, repeat = interval_decision(first), interval_decision(second)
            require(entry['decision'] == decision and entry['secondary_decision'] == repeat,
                    f'mean interval decision differs: {c}')
            shift = max(abs(a-b) for a, b in zip(first, second))
            close(entry['mc_endpoint_shift'], shift, f'Monte Carlo shift {c}')
            require(entry['mc_stable'] is (shift <= .01 and decision == repeat),
                    f'Monte Carlo stability differs: {c}')
        interval_difference = max(abs(a-b) for key in ('interval', 'secondary_interval')
                                  for a, b in zip(precisions['float32'][key], precisions['float64'][key]))
        close(result['maximum_interval_precision_discrepancy'], interval_difference, c)
        reasons = []
        if n < 2:
            reasons.append('fewer_than_two_independent_units')
        if unresolved_ids or interval_difference > .01:
            reasons.append('cross_precision_disagreement')
        if not all(entry['mc_stable'] for entry in precisions.values()):
            reasons.append('bootstrap_monte_carlo_instability')
        decisions = {entry['decision'] for entry in precisions.values()}
        if len(decisions) != 1:
            reasons.append('precision_decisions_disagree')
        require(result['blocked_reasons'] == reasons and result['discrepant_case_ids'] == unresolved_ids,
                f'mean blocked reasons differ: {c}')
        expected_status = next(iter(decisions)) if not reasons else 'unresolved'
        require(result['status'] == expected_status, f'mean final decision differs: {c}')
    return counts, resolved, cp_error


def compare_tree(actual, expected, label='analyzer reproduction'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f'{label}: keys differ')
        for key in expected:
            compare_tree(actual[key], expected[key], f'{label}.{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'{label}: list length differs')
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_tree(left, right, f'{label}[{index}]')
    elif type(expected) is float:
        close(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, f'{label}: value differs')


def verify(results, *, repository_root=REPOSITORY, source_root=None, model_snapshot=None,
           full_bootstrap=False, require_original_inputs=False):
    results, repository_root = Path(results), Path(repository_root)
    application = repository_root / 'applications/makelov-2311.17030'
    require(not (results/'FAILED.json').exists(), 'run has FAILED.json')
    artifacts = load(results/'artifact_hashes.json')
    require(REQUIRED_FILES <= set(artifacts), 'artifact manifest omits required result file')
    verified, unavailable = [], []

    def optional(path, digest, label):
        require(hexhash(digest), f'{label}: invalid sha256')
        if Path(path).is_file():
            check_hash(path, digest, label)
            verified.append(label)
            return True
        unavailable.append(label)
        return False

    for name, digest in artifacts.items():
        require(Path(name).name == name, f'unsafe result artifact: {name}')
        if name == 'directions.npz':
            optional(results/name, digest, name)
        else:
            check_hash(results/name, digest, name)
    manifest, cases = load(results/'manifest.json'), load(results/'cases.json')
    summary, started = load(results/'summary.json'), load(results/'RUN_STARTED.json')
    frozen_hash = sha256(results/'manifest.json')
    require(started['manifest_sha256'] == summary['manifest_sha256'] == frozen_hash,
            'freeze hash differs between manifest, start and summary')
    check_hash(results/'cases.json', manifest['cases_sha256'], 'frozen cases')
    check_hash(results/'records.jsonl', summary['records_sha256'], 'summary records')
    if 'directions.npz' in artifacts:
        require(artifacts['directions.npz'] == manifest['directions_sha256'], 'direction hash chain differs')
    else:
        optional(results/'directions.npz', manifest['directions_sha256'], 'directions.npz')
    require(manifest['schema_version'] == 1 and manifest['experiment'] == 'donor_factor_3a'
            and manifest['stage'] in ('development', 'confirmation'), 'unsupported manifest schema/stage')
    require(manifest['site'] == SITE and manifest['precisions'] == list(PRECISIONS), 'wrong instrument')
    contract = manifest['contract']
    expected_contract = {'profile_residual_tolerance': .25, 'profile_coverage': .8,
                         'profile_family_alpha': .025, 'profiles': list(PROFILES),
                         'mean_practical_boundary': .10, 'mean_family_alpha': .025,
                         'mean_contrasts': list(CONTRASTS), 'mean_bootstrap_draws': 20000,
                         'mean_bootstrap_seed': 320260924, 'secondary_bootstrap_seed': 320260925,
                         'bootstrap_endpoint_stability_budget': .01, 'numerical_discrepancy_budget': .01}
    for key, expected in expected_contract.items():
        require(contract[key] == expected, f'unexpected scientific contract: {key}')
    prefix = 'applications/makelov-2311.17030/'
    require(set(manifest['code_files_sha256']) == {prefix+p for p in CODE_PATHS}, 'frozen code set differs')
    for name, digest in manifest['code_files_sha256'].items():
        optional(relative_path(repository_root, name), digest, 'code:'+name)
    qualification = application/'results/donor_factor_environment/qualification.json'
    if optional(qualification, manifest['qualification_sha256'], 'runtime qualification'):
        require(load(qualification)['passed'] is True, 'frozen runtime qualification failed')
    for kind in ('planning_input', 'development_input'):
        item = manifest.get(kind)
        if item is not None:
            optional(relative_path(repository_root, item['path']), item['sha256'], kind)
    n = manifest['n_families']
    require(type(n) is int and n > 0 and len(cases) == n, 'case count differs from frozen n')
    if manifest['stage'] == 'confirmation':
        require(n in (128, 192, 256, 384) and manifest['planning_input']['selected_n'] == n,
                'confirmation sample size differs from declared planning grid/decision')
        planning_path = relative_path(repository_root, manifest['planning_input']['path'])
        if planning_path.is_file():
            require(load(planning_path)['selected_n'] == n, 'planning artifact selected n differs')
    sampling = manifest['sampling']
    require(sampling['n_family_draws'] == n and sampling['draw_cases_sha256'] == canonical_hash(cases),
            'sampling draw count/hash differs')
    definition = sampling['sampling_definition']
    require(sampling['sampling_version'] == definition['version'] == 'donor-factor-iid-v1',
            'unsupported sampling definition version')
    population = canonical_hash(definition)
    require(sampling['population_definition_sha256'] == population, 'sampling population hash differs')
    require(sampling['stage'] == manifest['stage'] and sampling['seed'] == manifest['seed'],
            'sampling stage/seed differs')
    excluded_list = definition['excluded_prompts']
    require(isinstance(excluded_list, list) and all(isinstance(p, str) and p for p in excluded_list),
            'invalid frozen prompt exclusions')
    excluded = set(excluded_list)
    require(excluded_list == sorted(excluded), 'exclusion list must be sorted unique prompt strings')
    provenance = sampling['exclusion_provenance']
    require(provenance['excluded_prompts_sha256'] == canonical_hash(excluded_list)
            and provenance['unique_excluded_prompts'] == len(excluded), 'exclusion provenance hash/count differs')
    sources = provenance['sources']
    source_names = [entry['path'] for entry in sources]
    require(len(set(source_names)) == len(source_names) and set(HISTORICAL_CASE_FILES) <= set(source_names),
            'historical exclusion provenance omits required rounds or duplicates sources')
    for name in OPTIONAL_HISTORICAL_FILES:
        require((name in source_names) != (name in provenance['optional_sources_absent']),
                'optional historical source neither checked nor declared absent')
    for entry in sources:
        path = relative_path(application, entry['path'])
        if optional(path, entry['sha256'], 'history:'+entry['path']):
            prior = jsonl(path) if path.suffix == '.jsonl' else load(path)
            prompts = {p for record in prior for p in record.get('prompts',
                       [record.get('base_sentence'), record.get('source_sentence')])}
            require(None not in prompts and prompts <= excluded, 'historical prompts missing from exclusions')
            require(entry['records'] == len(prior) and entry['unique_prompts'] == len(prompts)
                    and entry['prompts_sha256'] == canonical_hash(sorted(prompts)),
                    'historical source prompt hash/count differs')
    development = manifest.get('development_input')
    if development and relative_path(repository_root, development['path']).is_file():
        prior = load(relative_path(repository_root, development['path']))
        prompts = {p for case in prior for p in case['prompts']}
        require(all(len(case['prompts']) == 4 for case in prior) and prompts <= excluded,
                'new development donor prompts missing from exclusions')
        require(provenance['additional_development_cases_sha256'] == canonical_hash(prior)
                and provenance['additional_development_draws'] == len(prior)
                and provenance['additional_development_prompts_sha256'] == canonical_hash(sorted(prompts)),
                'new development exclusion provenance differs')
    elif manifest['stage'] == 'confirmation':
        unavailable.append('new development prompt contents (hash only)')
    source_root = Path(source_root) if source_root else application/'artifacts/makelov_source'
    source_manifest = source_root/'source_manifest.json'
    if source_manifest.exists():
        require(load(source_manifest) == manifest['source'], 'original source manifest differs')
    for name, digest in manifest['source']['files'].items():
        optional(relative_path(source_root, name), digest, 'source:'+name)
    for name, digest in definition['source_files'].items():
        require(manifest['source']['files'].get(name) == digest, 'sampling/source input hashes differ')
    if model_snapshot is None:
        hub = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ.get('HF_HOME', Path.home()/'.cache/huggingface'))/'hub'))
        model_snapshot = hub/'models--gpt2/snapshots'/manifest['model_snapshot_revision']
    for name, digest in manifest['model_files_sha256'].items():
        optional(relative_path(model_snapshot, name), digest, 'model:'+name)
    ids = [case['case_id'] for case in cases]
    require(len(set(ids)) == n and all(isinstance(x, str) and x for x in ids), 'draw IDs must be unique')
    for index, case in enumerate(cases):
        validate_family(case)
        unique = canonical_hash([sampling['sampling_version'], population, manifest['seed'],
                                 manifest['stage'], index])[:24]
        require(case['case_id'] == case['unique_draw_id'] == unique
                and case['draw_index'] == case['draw'] == index, 'IID draw identity/order differs')
        require(not (set(case['prompts']) & excluded), 'sample contains historical/excluded donor prompt')
        require(all(row[0] == definition['bos_token_id'] for row in case['token_ids']), 'BOS token differs')
    proposals = [case['proposal_index'] for case in cases]
    require(all(type(i) is int and i >= 0 for i in proposals)
            and proposals == sorted(set(proposals))
            and sampling['proposal_count'] == proposals[-1]+1
            and sampling['proposal_count'] == n+sum(sampling['rejection_counts'].values())
            and sampling['proposal_count'] <= sampling['max_proposals'], 'proposal/rejection counts differ')
    multiplicities = Counter(c['content_family_id'] for c in cases)
    require(sampling['unique_content_families'] == len(multiplicities)
            and sampling['repeated_content_draws'] == n-len(multiplicities)
            and sampling['content_multiplicities_above_one'] == {k: v for k, v in multiplicities.items() if v > 1},
            'sampling content multiplicities differ')
    records = jsonl(results/'records.jsonl')
    require([record['case_id'] for record in records] == ids, 'records omit/reorder/duplicate frozen draws')
    for case, record in zip(cases, records):
        require(record['content_family_id'] == case['content_family_id']
                and set(record['precisions']) == set(PRECISIONS), 'record content/precision differs')
    for precision in PRECISIONS:
        measured = jsonl(results/f'measurements_{precision}.jsonl')
        require([record['case_id'] for record in measured] == ids, f'{precision}: measurement IDs differ')
        for case, record, measurement in zip(cases, records, measured):
            expected = dict(measurement)
            del expected['case_id']
            require(record['precisions'][precision] == expected, f'{precision}: measurement content differs')
            audit_precision(expected, case, precision)
    counts, resolved, cp_error = audit_summary(summary, records)
    require(summary['model_forward_calls'] == 12*n and summary['model_prompt_evaluations'] == 48*n
            and summary['stage'] == manifest['stage']
            and summary['confirmation'] is (manifest['stage'] == 'confirmation'), 'summary stage/count differs')
    if full_bootstrap:
        from donor_factor_analysis import analyze_records
        reproduced = analyze_records(records, bootstrap_draws=contract['mean_bootstrap_draws'],
                                     seed=contract['mean_bootstrap_seed'],
                                     secondary_seed=contract['secondary_bootstrap_seed'])
        compare_tree({key: summary[key] for key in reproduced}, reproduced)
    if require_original_inputs:
        require(not unavailable, 'original inputs unavailable: ' + ', '.join(unavailable))
    return {'verified': True, 'mode': 'records_only', 'n_families': n,
            'independent_profile_counts': counts, 'n_resolved_units': resolved,
            'profiles': summary['profiles'],
            'mean_statuses': {c: summary['mean_effects'][c]['status'] for c in CONTRASTS},
            'full_seeded_bootstrap_reproduced': full_bootstrap,
            'cp_interval_max_abs_error': cp_error, 'numeric_reproduction_tolerance': FLOAT_TOLERANCE,
            'optional_original_inputs_verified': verified, 'optional_original_inputs_unavailable': unavailable,
            'scope': ['No tokenizer or model was loaded or run. Raw A-minus-B margins, factorial maps, '
                      'controls, complete IID draw records, and saved exclusions were checked.',
                      'Profile counts, numeric resolution, CP bounds, family I/P/J sample means and '
                      'decision rules were independently recomputed from raw cells.',
                      'Full seeded bootstrap and enriched records reproduced with NumPy analyzer.' if full_bootstrap else
                      'Bootstrap endpoints and enriched summary records were not regenerated; use --full-bootstrap with NumPy.',
                      'Available originals were byte-checked. Missing originals are listed, not treated as verified.',
                      'Saved token alignments and audits are consistency checks; hashes do not recover '
                      'activations, prove forwards happened, or establish an independent preregistration timestamp.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--full-bootstrap', action='store_true')
    parser.add_argument('--require-original-inputs', action='store_true')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--model-snapshot', type=Path)
    args = parser.parse_args()
    try:
        report = verify(args.results, full_bootstrap=args.full_bootstrap,
                        require_original_inputs=args.require_original_inputs,
                        source_root=args.source_root, model_snapshot=args.model_snapshot)
    except (ValueError, KeyError, TypeError, OSError, OverflowError, ImportError) as exc:
        parser.exit(1, f'Verification failed: {exc}\n')
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
