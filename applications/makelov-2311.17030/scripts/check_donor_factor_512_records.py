"""Verify the authorized Round-3A 512-family amendment without model inference.

Standard-library mode reuses the preserved earlier checker's raw-record/statistical
checks and adds strict amendment/planning bindings. --full-bootstrap needs NumPy;
it reproduces inference bootstrap summaries, not the sample-size simulations.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_donor_factor_records import (
    REPOSITORY, REQUIRED_FILES, PRECISIONS, PROFILES, CONTRASTS, SITE,
    HISTORICAL_CASE_FILES, OPTIONAL_HISTORICAL_FILES, FLOAT_TOLERANCE,
    VerificationError, require, load, jsonl, sha256, canonical_hash, hexhash,
    relative_path, check_hash, finite, vector, close, validate_family,
    audit_precision, audit_summary, compare_tree, CODE_PATHS as ORIGINAL_CODE_PATHS)

CODE_PATHS = ORIGINAL_CODE_PATHS | {'scripts/run_donor_factor_512.py',
                                   'src/donor_factor_planning_512.py',
                                   'DONOR_FACTOR_512_AMENDMENT.md'}
AMENDMENT_ID = 'round3a-512-v1'
APPLICATION_PREFIX = 'applications/makelov-2311.17030/'
STOP_PATH = APPLICATION_PREFIX + 'results/donor_factor_planning/planning.json'
STOP_SHA256 = 'fa30e74a514e91a9eeac28930384addfa1ca9ae73b871d9423defd2312a40a28'
PROTOCOL_PATH = APPLICATION_PREFIX + 'DONOR_FACTOR_512_AMENDMENT.md'
PLANNER_PATH = APPLICATION_PREFIX + 'src/donor_factor_planning_512.py'


def binomial_probability(k, n, p, upper):
    """Sum a binomial tail outward from its mode, avoiding huge coefficients."""
    low, high = (k, n) if upper else (0, k)
    if p == 0:
        return float(low == 0)
    if p == 1:
        return float(high == n)
    mode = min(high, max(low, int((n+1)*p)))
    mass = math.exp(math.lgamma(n+1)-math.lgamma(mode+1)-math.lgamma(n-mode+1)
                    + mode*math.log(p)+(n-mode)*math.log1p(-p))
    terms = [mass]
    value = mass
    for j in range(mode, high):
        value *= (n-j)/(j+1)*p/(1-p)
        terms.append(value)
        if value < mass*1e-17:
            break
    value = mass
    for j in range(mode, low, -1):
        value *= j/(n-j+1)*(1-p)/p
        terms.append(value)
        if value < mass*1e-17:
            break
    return min(1., math.fsum(terms))


def mc_interval(k, n):
    """Independent recurrence/bisection CP interval for stored MC detection counts."""
    bounds = []
    for upper_tail, boundary in ((True, 0.), (False, 1.)):
        if (upper_tail and k == 0) or (not upper_tail and k == n):
            bounds.append(boundary)
            continue
        low, high = 0., 1.
        for _ in range(70):
            middle = (low+high)/2
            if (binomial_probability(k, n, middle, upper_tail) > .025) == upper_tail:
                high = middle
            else:
                low = middle
        bounds.append((low+high)/2)
    return bounds


def audit_power_report(report, simulations, bootstrap):
    require(set(report['directions']) == {'positive', 'negative'}, 'planning direction set differs')
    decisions, intervals = [], []
    for sign, contrasts in report['directions'].items():
        require(set(contrasts) == set(CONTRASTS), f'planning contrasts differ: {sign}')
        for name, entry in contrasts.items():
            count = entry['detections']
            require(type(count) is int and 0 <= count <= simulations
                    and entry['simulations'] == simulations, 'planning MC count/denominator differs')
            require(entry['probability'] == count/simulations, 'planning MC probability differs')
            expected = mc_interval(count, simulations)
            supplied = vector(entry['monte_carlo_95_percent_interval'], 'planning MC interval')
            require(max(abs(a-b) for a, b in zip(expected, supplied)) <= FLOAT_TOLERANCE,
                    f'planning MC CP interval differs: {sign}.{name}')
            passed = expected[0] > .8 if bootstrap else count/simulations >= .8
            require(entry['passes'] is passed, f'planning MC gate differs: {sign}.{name}')
            decisions.append(passed)
            intervals.append(expected[0])
    require(report['passes'] is all(decisions), 'planning aggregate pass flag differs')
    return {'passes': all(decisions), 'smallest_mc_lower_bound': min(intervals)}


def audit_amendment(manifest, repository_root):
    """Require the preserved STOP and actual new planning artifacts, not just claims."""
    amendment = manifest['amendment']
    require(set(amendment) == {'id', 'protocol', 'original_stop'}
            and amendment['id'] == AMENDMENT_ID, '512 amendment identity differs')
    require(amendment['original_stop'] == {'path': STOP_PATH, 'sha256': STOP_SHA256},
            'original STOP binding differs from the preserved artifact')
    require(amendment['protocol']['path'] == PROTOCOL_PATH
            and amendment['protocol']['sha256'] == manifest['code_files_sha256'][PROTOCOL_PATH],
            'amendment protocol/code binding differs')
    for label in ('original_stop', 'protocol'):
        binding = amendment[label]
        check_hash(relative_path(repository_root, binding['path']), binding['sha256'], 'amendment:'+label)
    old = load(relative_path(repository_root, STOP_PATH))
    require(old['status'] == 'STOP' and old['selected_n'] is None
            and old['settings']['candidate_sizes'] == [128, 192, 256, 384], 'preserved planner was not STOP')
    binding = manifest['planning_input']
    path = relative_path(repository_root, binding['path'])
    check_hash(path, binding['sha256'], '512 planning artifact')
    plan = load(path)
    require(plan['schema_version'] == 'donor-factor-planning-512-v1'
            and plan['status'] == 'SELECTED' and plan['selected_n'] == binding['selected_n'] == 512,
            'amended planning did not select exactly 512')
    require(plan['amendment'] == amendment, 'planning/manifest amendment bindings differ')
    require(plan['planner_code_sha256'] == manifest['code_files_sha256'][PLANNER_PATH],
            'planning/manifest planner code binding differs')
    require(plan['protocol_matches_frozen_simulation_counts'] is True
            and plan['protocol_matches_development_size'] is True and plan['n_development_units'] == 32
            and plan['confirmation_authorized'] is False, 'planning qualification flags differ')
    settings = dict(old['settings'], candidate_sizes=[128, 192, 256, 384, 512])
    require(plan['settings'] == settings, 'amendment changes a planning rule beyond the new 512 cap')
    require(plan['development_binding'] == old['development_binding'], 'planning development binding changed')
    require(manifest['development_input']['sha256'] == plan['development_binding']['cases.json'],
            'confirmation/planning development cases differ')
    development = relative_path(repository_root, manifest['development_input']['path']).parent
    for name, digest in plan['development_binding'].items():
        check_hash(relative_path(development, name), digest, 'planning development:'+name)
    original_manifest = load(development/'manifest.json')
    original_contract = original_manifest['contract']
    expected_contract = dict(original_contract, planning=dict(original_contract['planning'],
                             n_grid=[128, 192, 256, 384, 512]))
    require(manifest['contract'] == expected_contract, 'amendment changed the original scientific contract')
    require(all(manifest['code_files_sha256'].get(path) == digest
                for path, digest in original_manifest['code_files_sha256'].items()),
            'amendment changed a frozen original producer file')
    candidates = plan['candidates']
    require(len(candidates) == 5 and candidates[:4] == old['candidates']
            and all(c['passes'] is False for c in candidates[:4]), 'original four planning candidates changed')
    candidate = candidates[-1]
    require(candidate['n'] == 512, 'new planning candidate is not 512')
    threshold = next(k for k in range(513) if binomial_probability(k, 512, .8, True) < .025/4)
    power = binomial_probability(threshold, 512, .9, True)
    coverage = candidate['coverage']
    require(coverage['n'] == 512 and coverage['required_successes'] == threshold
            and coverage['passes'] is (power >= .8), 'planning coverage gate differs')
    close(coverage['power_at_90_percent_success'], power, 'planning coverage power')
    normal = audit_power_report(candidate['normal_reference'], 10000, False)
    checks = candidate['bootstrap_checks']
    require(isinstance(checks, dict) and set(checks) == {'gaussian', 'empirical'}, 'missing bootstrap planning gates')
    bootstrap = {}
    for offset, distribution in ((1, 'gaussian'), (2, 'empirical')):
        report = checks[distribution]
        require(report['distribution'] == distribution and report['bootstrap_draws'] == 2000
                and report['fixed_bootstrap_seed'] == 320260924
                and report['simulation_seed'] == 320260926+400+offset,
                'bootstrap planning method/seed differs')
        bootstrap[distribution] = audit_power_report(report, 1000, True)
    passes = coverage['passes'] and normal['passes'] and all(v['passes'] for v in bootstrap.values())
    require(candidate['passes'] is True and passes, '512 candidate fails a saved planning gate')
    return {'id': AMENDMENT_ID, 'selected_n': 512, 'original_stop_sha256': STOP_SHA256,
            'planning_sha256': binding['sha256'], 'development_binding_unchanged': True,
            'original_four_candidates_unchanged': True, 'coverage_required_successes': threshold,
            'normal_reference': normal, 'bootstrap_checks': bootstrap,
            'scope': 'Saved planning counts, CP intervals, rules and bindings checked; no planning simulations rerun.'}


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
            and manifest['stage'] == 'confirmation' and manifest['seed'] == 24092432, 'unsupported manifest schema/stage')
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
    require(contract['planning']['n_grid'] == [128, 192, 256, 384, 512],
            '512 amendment planning grid differs')
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
        require(n == 512 and manifest['planning_input']['selected_n'] == n,
                'confirmation sample size differs from declared planning grid/decision')
        planning_path = relative_path(repository_root, manifest['planning_input']['path'])
        if planning_path.is_file():
            require(load(planning_path)['selected_n'] == n, 'planning artifact selected n differs')
    amendment_audit = audit_amendment(manifest, repository_root)
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
            'profiles': summary['profiles'], 'amendment_audit': amendment_audit,
            'mean_statuses': {c: summary['mean_effects'][c]['status'] for c in CONTRASTS},
            'full_seeded_bootstrap_reproduced': full_bootstrap,
            'cp_interval_max_abs_error': cp_error, 'numeric_reproduction_tolerance': FLOAT_TOLERANCE,
            'optional_original_inputs_verified': verified, 'optional_original_inputs_unavailable': unavailable,
            'scope': ['The separately versioned 512-family amendment and saved planning gates were checked; no planning simulations were rerun.',
                      'No tokenizer or model was loaded or run. Raw A-minus-B margins, factorial maps, '
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
