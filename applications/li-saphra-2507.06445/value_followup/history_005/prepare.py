"""Prepare matched history-edit bundles from strings alone; no neural imports.

Every quartet shares a stem, first 21 symbols and future suffix. Two disjoint
adjacent swaps cross last-minimum recency with the final two bracket symbols.
They are controlled history-edit bundles, not isolated scalar interventions.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile

HERE = Path(__file__).resolve().parent
VALUE = HERE.parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('history_input_design', VALUE / 'design.py')
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)

CORES = {6: '(())((', 10: '(()()('}
ENDINGS = {'alt': '()()', 'close': '(())'}
LAST_TWO = {'alt': '()', 'close': '))'}
SUFFIXES = tuple(sorted({')(((', '()((', '(()(', '((()'}))
ROLE_NAMESPACE = 'matched-history-005-role:'
ROLE_RULE = 'int(SHA256(UTF8("matched-history-005-role:"+first20)),16)%2;0=calibration,1=target'
RECIPE = {
    'round': 'history_005', 'model': 'a9g0io1r', 'layer_one_based': 2, 'head_one_based': 1,
    'scientific_stage': 'prospective_development', 'input_hash_partition': 'confirmation',
    'native_candidate_count': 2048, 'family_count': 256,
    'recipient_seed': 26100451, 'donor_seed': 26100452,
    'length': 32, 'open_count': 16, 'position': 28, 'balance': -2, 'minimum': -4,
    'read_symbol': ')', 'stem_length': 18, 'stem_balance': -4,
    'recencies': [6, 10], 'cores': {str(k): v for k, v in CORES.items()},
    'endings': ENDINGS, 'last_two': LAST_TWO, 'suffixes': list(SUFFIXES),
    'role_rule': ROLE_RULE, 'phase_rule': "int(sha256('value-prefix-v1:' + text),16)%5 != 0",
    'roles': ['calibration', 'target'], 'replicas_per_role': 2,
    'stem_sampler': 'random.Random(donor_seed).sample(sorted_eligible_role_pool, 2) for each role in each family',
    'suffix_sampler': 'rng.choice(sorted_suffixes) once per role/replica quartet, in calibration then target order',
    'sampling': 'With replacement between families and recipient draws; four distinct stems within a family',
    'historical_donor_ban': 'Both first20 and first28 of every donor, across all historical roles',
    'quartet_matching': 'Common first21 and common future suffix; recency swaps positions22/23; ending swaps26/27',
    'assignment': 'Template index i goes to the i-th accepted recipient; no outcome-based donor selection',
    'candidate_attempt_limit_per_draw': 100000,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dumps(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def jsonl(values):
    return ''.join(json.dumps(value, sort_keys=True, allow_nan=False) + '\n' for value in values)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def role_of(text):
    if len(text) < 20:
        raise ValueError('Role assignment requires twenty symbols')
    return ('calibration', 'target')[int(design.sha(ROLE_NAMESPACE + text[:20]), 16) % 2]


def history_properties(prefix):
    depth = low = 0
    last_minimum = 0
    for index, symbol in enumerate(prefix, 1):
        if symbol not in '()':
            raise ValueError('Non-bracket symbol')
        depth += 1 if symbol == '(' else -1
        if depth < low:
            low = depth
            last_minimum = index
        elif depth == low:
            last_minimum = index
    return {'balance': depth, 'minimum': low, 'last_minimum_position': last_minimum,
            'recency': len(prefix) - last_minimum, 'last_two': prefix[-2:]}


def historical_exclusions():
    prior = VALUE / 'averaged_anchors_004/inputs'
    preparation = read(prior / 'preparation.json')
    sources = {str((prior / 'preparation.json').relative_to(ROOT)): sha(prior / 'preparation.json')}
    for name, digest in preparation['files'].items():
        if sha(prior / name) != digest:
            raise ValueError('Historical prepared input changed: ' + name)
        sources[str((prior / name).relative_to(ROOT))] = digest
    texts = set(read(prior / 'exclusions.json')['recipient_strings'])
    texts.update(row['recipient'] for row in read_rows(prior / 'candidates.jsonl'))
    texts.update(donor['string'] for family in read_rows(prior / 'donor_families.jsonl') for donor in family['donors'])
    prefixes = {str(p): sorted({s[:p] for s in texts}) for p in (20, 28)}
    if (len(texts), len(prefixes['20']), len(prefixes['28'])) != (24617, 22548, 24587):
        raise ValueError('Historical population changed; amend rather than silently regenerating')
    return {'recipient_strings': sorted(texts), 'prefixes': prefixes, 'source_hashes': sources,
            'rule': 'All previously prepared/opened strings, including every 004 candidate and donor, not only measured cells'}


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(None)
def stem_completion_count(remaining, depth):
    if depth < -4 or abs(-4 - depth) > remaining:
        return 0
    if remaining == 0:
        return int(depth == -4)
    return stem_completion_count(remaining - 1, depth + 1) + stem_completion_count(remaining - 1, depth - 1)


def all_stems(remaining=18, depth=0, prefix=''):
    if not stem_completion_count(remaining, depth):
        return
    if remaining == 0:
        yield prefix
        return
    for symbol, step in (('(', 1), (')', -1)):
        yield from all_stems(remaining - 1, depth + step, prefix + symbol)


def eligible_stems(exclusions):
    banned = {p: set(exclusions['prefixes'][str(p)]) for p in (20, 28)}
    counts = Counter()
    pools = {'calibration': [], 'target': []}
    for stem in all_stems():
        counts['raw_stems'] += 1
        if stem + '((' in banned[20]:
            continue
        counts['fresh_first20'] += 1
        prefixes = [stem + core + ending for core in CORES.values() for ending in ENDINGS.values()]
        if any(prefix in banned[28] for prefix in prefixes):
            continue
        counts['all_four_fresh_first28'] += 1
        if not all(design.phase_of(prefix) == 'confirmation' for prefix in prefixes):
            continue
        counts['all_four_confirmation'] += 1
        pools[role_of(prefixes[0])].append(stem)
    pools = {role: sorted(values) for role, values in pools.items()}
    if counts['raw_stems'] != 13260 or counts['all_four_confirmation'] != 5158:
        raise ValueError('Finite matched-prefix support changed')
    if any(len(values) < 2 for values in pools.values()):
        raise ValueError('Insufficient distinct matched stems; do not relax roles')
    return {'pools': pools, 'counts': dict(sorted(counts.items())),
            'role_pool_counts': {role: len(values) for role, values in pools.items()},
            'scope': 'Uniform matched-stem population after full historical bans and all-four-cell hash eligibility; not uniform over every prefix with the factor labels'}


def draw_candidates(exclusions, recipe=RECIPE):
    rng = random.Random(recipe['recipient_seed'])
    banned = set(exclusions['recipient_strings'])
    result, attempts = [], 0
    for index in range(recipe['native_candidate_count']):
        for _ in range(recipe['candidate_attempt_limit_per_draw']):
            attempts += 1
            chars = list('(' * 16 + ')' * 16)
            rng.shuffle(chars)
            text = ''.join(chars)
            if (text not in banned and design.phase_of(text) == recipe['input_hash_partition']
                    and design.prefix_state(text)[1] < 0):
                break
        else:
            raise RuntimeError('Recipient sampling exhausted; no silent relaxation')
        result.append({'candidate_index': index, 'recipient': text,
                       'candidate_id': design.sha('history-005-recipient:{}:{}'.format(recipe['recipient_seed'], index))[:20]})
    return result, attempts


def draw_families(support, recipe=RECIPE):
    rng = random.Random(recipe['donor_seed'])
    families = []
    for index in range(recipe['family_count']):
        donors = []
        for role in ('calibration', 'target'):
            stems = rng.sample(support['pools'][role], 2)
            for replica, stem in enumerate(stems):
                suffix = rng.choice(SUFFIXES)
                for recency, core in CORES.items():
                    for ending, final in ENDINGS.items():
                        prefix = stem + core + final
                        cell = '{}_t{}_{}_{}'.format(role, recency, ending, replica)
                        donors.append({'role': role, 'replica': replica, 'recency': recency,
                                       'ending': ending, 'last_two': LAST_TWO[ending], 'cell': cell,
                                       'balance': -2, 'minimum': -4, 'position': 28,
                                       'last_minimum_position': 28 - recency, 'string': prefix + suffix,
                                       'prefix_sha256': design.sha(prefix), 'stem_sha256': design.sha(stem)})
        families.append({'family_index': index,
                         'family_id': design.sha('history-005-family:{}:{}'.format(recipe['donor_seed'], index))[:20],
                         'input_hash_partition': recipe['input_hash_partition'], 'donors': donors})
    return families


def validate_records(candidates, families, exclusions, support, recipe=RECIPE):
    if len(candidates) != recipe['native_candidate_count'] or len(families) != recipe['family_count']:
        raise ValueError('Unexpected number of candidates or families')
    banned = {p: set(exclusions['prefixes'][str(p)]) for p in (20, 28)}
    old_strings = set(exclusions['recipient_strings'])
    pools = {role: set(stems) for role, stems in support['pools'].items()}
    for i, row in enumerate(candidates):
        text = row['recipient']
        if (set(row) != {'candidate_index', 'candidate_id', 'recipient'} or row['candidate_index'] != i
                or row['candidate_id'] != design.sha('history-005-recipient:{}:{}'.format(recipe['recipient_seed'], i))[:20]
                or len(text) != 32 or text.count('(') != 16 or set(text) != {'(', ')'}
                or text in old_strings or design.prefix_state(text)[1] >= 0
                or design.phase_of(text) != recipe['input_hash_partition']):
            raise ValueError('Invalid candidate input')
    expected = {(role, r, ending, replica) for role in ('calibration', 'target')
                for r in (6, 10) for ending in ('alt', 'close') for replica in (0, 1)}
    role_prefixes = {role: {p: set() for p in (20, 28)} for role in ('calibration', 'target')}
    donor_fields = {'role', 'replica', 'recency', 'ending', 'last_two', 'cell', 'balance', 'minimum',
                    'position', 'last_minimum_position', 'string', 'prefix_sha256', 'stem_sha256'}
    for index, family in enumerate(families):
        if (set(family) != {'family_index', 'family_id', 'input_hash_partition', 'donors'}
                or family['family_index'] != index or family['input_hash_partition'] != recipe['input_hash_partition']
                or family['family_id'] != design.sha('history-005-family:{}:{}'.format(recipe['donor_seed'], index))[:20]):
            raise ValueError('Invalid family identity')
        seen, quartets = set(), {}
        for donor in family['donors']:
            role, recency, ending, replica = (donor[k] for k in ('role', 'recency', 'ending', 'replica'))
            key = role, recency, ending, replica
            if key not in expected or key in seen or set(donor) != donor_fields:
                raise ValueError('Invalid or duplicate donor cell')
            seen.add(key)
            text = donor['string']; stem = text[:18]; prefix = text[:28]
            properties = history_properties(prefix)
            if (len(text) != 32 or text.count('(') != 16 or set(text) != {'(', ')'}
                    or stem not in pools[role] or prefix != stem + CORES[recency] + ENDINGS[ending]
                    or text[28:] not in SUFFIXES or donor['cell'] != '{}_t{}_{}_{}'.format(role, recency, ending, replica)
                    or donor['position'] != 28 or donor['balance'] != -2 or donor['minimum'] != -4
                    or donor['last_minimum_position'] != 28 - recency or donor['last_two'] != LAST_TWO[ending]
                    or properties != {'balance': -2, 'minimum': -4, 'last_minimum_position': 28 - recency,
                                      'recency': recency, 'last_two': LAST_TWO[ending]}
                    or donor['prefix_sha256'] != design.sha(prefix) or donor['stem_sha256'] != design.sha(stem)
                    or design.phase_of(prefix) != 'confirmation' or role_of(text) != role
                    or any(text[:p] in banned[p] for p in (20, 28))):
                raise ValueError('Invalid factor labels, prefix history, or donor freshness')
            quartets.setdefault((role, replica), {})[recency, ending] = text
            for p in (20, 28):
                role_prefixes[role][p].add(text[:p])
        if seen != expected:
            raise ValueError('Incomplete factorial donor grid')
        stems = set()
        for quartet in quartets.values():
            if len({text[:21] for text in quartet.values()}) != 1 or len({text[28:] for text in quartet.values()}) != 1:
                raise ValueError('Quartet stem/future matching violated')
            stems.add(next(iter(quartet.values()))[:18])
            for ending in ('alt', 'close'):
                changed = [j + 1 for j, (a, b) in enumerate(zip(quartet[6, ending], quartet[10, ending])) if a != b]
                if changed != [22, 23]:
                    raise ValueError('Recency edit changed undeclared positions')
            for recency in (6, 10):
                changed = [j + 1 for j, (a, b) in enumerate(zip(quartet[recency, 'alt'], quartet[recency, 'close'])) if a != b]
                if changed != [26, 27]:
                    raise ValueError('Ending edit changed undeclared positions')
        if len(stems) != 4:
            raise ValueError('Each family needs four distinct matched stems')
    if any(role_prefixes['calibration'][p] & role_prefixes['target'][p] for p in (20, 28)):
        raise ValueError('Global calibration/target prefix leakage')


def statistics(candidates, families, exclusions, support):
    recipients = Counter(row['recipient'] for row in candidates)
    donors = [d for f in families for d in f['donors']]
    strings = Counter(d['string'] for d in donors)
    prefixes = Counter(d['prefix_sha256'] for d in donors)
    stems = Counter(d['stem_sha256'] for d in donors)
    cells = Counter(d['cell'] for d in donors)
    return {'candidate_draws': len(candidates), 'unique_candidates': len(recipients),
            'maximum_candidate_reuse': max(recipients.values()), 'family_templates': len(families),
            'donor_draws': len(donors), 'unique_donor_strings': len(strings),
            'unique_measured_prefixes': len(prefixes), 'maximum_measured_prefix_reuse': max(prefixes.values()),
            'matched_quartets': len(families) * 4, 'unique_matched_stems': len(stems),
            'maximum_stem_quartet_reuse': max(stems.values()) // 4,
            'factorial_cell_draws': dict(sorted(cells.items())),
            'global_cross_role_prefix_overlap': {str(p): len({d['string'][:p] for d in donors if d['role'] == 'calibration'} &
                                                                {d['string'][:p] for d in donors if d['role'] == 'target'}) for p in (20, 28)},
            'new_recipient_donor_full_string_overlap': len(set(recipients) & set(strings)),
            'new_recipient_donor_prefix_overlap': {str(p): len({s[:p] for s in recipients} & {s[:p] for s in strings}) for p in (20, 28)},
            'historical_full_strings': len(exclusions['recipient_strings']),
            'historical_prefixes': {p: len(values) for p, values in exclusions['prefixes'].items()},
            'finite_stem_support': support['counts'], 'eligible_role_stems': support['role_pool_counts']}


def prepare(output=HERE / 'inputs'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite prepared inputs')
    exclusions = historical_exclusions(); support = eligible_stems(exclusions)
    candidates, attempts = draw_candidates(exclusions); families = draw_families(support)
    validate_records(candidates, families, exclusions, support)
    payload = {'exclusions.json': dumps(exclusions), 'eligible_stems.json': dumps(support),
               'candidates.jsonl': jsonl(candidates), 'donor_families.jsonl': jsonl(families)}
    manifest = {'schema_version': 1, 'status': 'inputs_only_no_measurements', 'recipe': RECIPE,
                'source_hashes': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                                  str((VALUE / 'design.py').relative_to(ROOT)): sha(VALUE / 'design.py')},
                'files': {name: hashlib.sha256(content.encode()).hexdigest() for name, content in payload.items()},
                'recipient_generation_attempts': attempts,
                'statistics': statistics(candidates, families, exclusions, support)}
    output.mkdir(parents=True)
    for name, content in payload.items():
        (output / name).write_text(content)
    (output / 'preparation.json').write_text(dumps(manifest))
    return manifest


def validate(directory=HERE / 'inputs', regenerate=True):
    directory = Path(directory)
    expected = {'exclusions.json', 'eligible_stems.json', 'candidates.jsonl', 'donor_families.jsonl', 'preparation.json'}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError('Unexpected prepared file inventory')
    manifest = read(directory / 'preparation.json')
    if manifest['recipe'] != RECIPE or manifest['status'] != 'inputs_only_no_measurements':
        raise ValueError('Prepared recipe/status differs')
    for name, digest in manifest['source_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Preparation source changed: ' + name)
    for name, digest in manifest['files'].items():
        if sha(directory / name) != digest:
            raise ValueError('Prepared data changed: ' + name)
    exclusions = historical_exclusions(); support = eligible_stems(exclusions)
    if read(directory / 'exclusions.json') != exclusions or read(directory / 'eligible_stems.json') != support:
        raise ValueError('Historical bans or finite support differ')
    candidates, families = read_rows(directory / 'candidates.jsonl'), read_rows(directory / 'donor_families.jsonl')
    validate_records(candidates, families, exclusions, support)
    if manifest['statistics'] != statistics(candidates, families, exclusions, support):
        raise ValueError('Prepared statistics differ')
    if regenerate:
        with tempfile.TemporaryDirectory() as temp:
            fresh = Path(temp) / 'inputs'
            prepare(fresh)
            for name in expected:
                if (fresh / name).read_bytes() != (directory / name).read_bytes():
                    raise ValueError('Prepared bytes are not deterministic: ' + name)
    return manifest


def validate_inputs(plan, directory=HERE / 'inputs', regenerate=True):
    sampling, model = plan['sampling'], plan['model']
    checks = (plan['round'] == RECIPE['round'], model['model'] == RECIPE['model'],
              model['layer_one_based'] == 2, model['head_one_based'] == 1,
              sampling['phase'] == RECIPE['input_hash_partition'],
              sampling['recipient_seed'] == RECIPE['recipient_seed'], sampling['donor_seed'] == RECIPE['donor_seed'],
              sampling['native_candidates'] == RECIPE['native_candidate_count'], sampling['families'] == RECIPE['family_count'],
              sampling['length'] == 32, sampling['opens'] == 16, sampling['closes'] == 16,
              sampling['recipient_requires_negative_prefix'] is True,
              sampling['role_rule'] == ROLE_RULE, sampling['role_assignment_uses_measurements'] is False,
              sampling['no_global_deduplication'] is True)
    if not all(checks):
        raise ValueError('Plan sampling differs from the frozen recipe')
    directory = Path(directory); manifest = validate(directory, regenerate=regenerate)
    if (sampling['historical_full_strings'], sampling['historical_prefix20'], sampling['historical_prefix28']) != (24617, 22548, 24587):
        raise ValueError('Plan history population differs')
    return manifest, read_rows(directory / 'candidates.jsonl'), read_rows(directory / 'donor_families.jsonl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'inputs')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    result = validate(args.output) if args.validate_only else prepare(args.output)
    print(dumps({'status': result['status'], 'statistics': result['statistics']}), end='')


if __name__ == '__main__':
    main()
