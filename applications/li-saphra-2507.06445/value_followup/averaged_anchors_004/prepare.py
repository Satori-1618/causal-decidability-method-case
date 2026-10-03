"""Input-only preparation for the averaged-anchor round; no model imports.

The historical ``confirmation`` hash partition supplies fresh strings. This is
a new prospective development round, not confirmation of a unique mechanism.
Sampling is with replacement between families. The role hash separates all
calibration and target prefixes globally, including across read positions.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile

HERE = Path(__file__).resolve().parent
VALUE = HERE.parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('averaged_input_design', VALUE / 'design.py')
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)

RECIPE = {
    'round': 'averaged_anchors_004',
    'model': 'a9g0io1r', 'layer_one_based': 2, 'head_one_based': 1,
    'input_hash_partition': 'confirmation',
    'scientific_stage': 'prospective_development',
    'native_candidate_count': 2048, 'family_count': 256,
    'recipient_seed': 26100341, 'donor_seed': 26100342,
    'length': 32, 'open_count': 16, 'positions': [20, 28],
    'balances': [-2, 2], 'minimum': -4, 'read_symbol': ')',
    'roles': ['calibration', 'target'], 'replicas_per_role_cell': 2,
    'role_hash': "int(sha256('averaged-anchor-004-role:' + first20).hexdigest(), 16) % 2",
    'role_parity': {'0': 'calibration', '1': 'target'},
    'phase_hash': "int(sha256('value-prefix-v1:' + measured_prefix).hexdigest(), 16) % 5 != 0",
    'sampling': 'with replacement between families and between recipient draws',
    'within_cell_rule': 'four distinct measured prefixes across the two roles',
    'historical_donor_ban': 'both first20 and first28 of every full donor, regardless of read position',
    'assignment': 'template index i is assigned to the i-th accepted recipient; do not replace templates',
    'attempt_limit_per_draw': 100000,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dumps(obj):
    return json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n'


def jsonl(rows):
    return ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def role_of(text):
    if len(text) < 20:
        raise ValueError('Role assignment requires the first 20 tokens')
    parity = int(design.sha('averaged-anchor-004-role:' + text[:20]), 16) % 2
    return ('calibration', 'target')[parity]


def historical_exclusions():
    prior = VALUE / 'screen_transfer_003/inputs'
    manifest = json.loads((prior / 'preparation.json').read_text())
    sources = {str((prior / 'preparation.json').relative_to(ROOT)): sha(prior / 'preparation.json')}
    for relative, digest in manifest['files'].items():
        if sha(prior / relative) != digest:
            raise ValueError('Historical prepared inputs changed: ' + relative)
        sources[str((prior / relative).relative_to(ROOT))] = digest
    texts = set(json.loads((prior / 'exclusions.json').read_text())['recipient_strings'])
    for relative in sorted(manifest['files']):
        if not relative.endswith('.jsonl'):
            continue
        for row in read_rows(prior / relative):
            texts.add(row['recipient'])
            texts.update(donor['string'] for donor in row.get('donors', []))
    result = {'recipient_strings': sorted(texts),
              'prefixes': {str(p): sorted({text[:p] for text in texts}) for p in (20, 28)},
              'source_hashes': sources,
              'rule': 'All historical prepared/opened strings, including unused templates and all six 003 pools'}
    counts = len(texts), len(result['prefixes']['20']), len(result['prefixes']['28'])
    if counts != (18475, 16753, 18447):
        raise ValueError('Historical population changed; amend before preparing new inputs')
    return result


def draw_candidates(exclusions, recipe=RECIPE):
    rng = random.Random(recipe['recipient_seed'])
    banned = set(exclusions['recipient_strings'])
    rows, attempts = [], 0
    for index in range(recipe['native_candidate_count']):
        for _ in range(recipe['attempt_limit_per_draw']):
            attempts += 1
            chars = list('(' * 16 + ')' * 16)
            rng.shuffle(chars)
            text = ''.join(chars)
            if (text not in banned and design.phase_of(text) == recipe['input_hash_partition']
                    and design.prefix_state(text)[1] < 0):
                break
        else:
            raise RuntimeError('Recipient sampling exhausted; no silent relaxation')
        rows.append({'candidate_index': index,
                     'candidate_id': design.sha('averaged-004-recipient:{}:{}'.format(recipe['recipient_seed'], index))[:20],
                     'recipient': text})
    return rows, attempts


def draw_families(exclusions, recipe=RECIPE):
    rng = random.Random(recipe['donor_seed'])
    banned = {p: set(exclusions['prefixes'][str(p)]) for p in (20, 28)}
    rows, attempts = [], Counter()
    for index in range(recipe['family_count']):
        donors = []
        for balance in (-2, 2):
            for position in (20, 28):
                selected = set()
                for role in ('calibration', 'target'):
                    key = '{}_{}_{}'.format(role, balance, position)
                    for replica in range(2):
                        for _ in range(recipe['attempt_limit_per_draw']):
                            attempts[key] += 1
                            prefix = design.draw_prefix(rng, position, balance)
                            if (prefix in selected or design.phase_of(prefix) != recipe['input_hash_partition']
                                    or role_of(prefix) != role or prefix[:20] in banned[20]
                                    or (position == 28 and prefix in banned[28])):
                                continue
                            suffix = list('(' * (16 - prefix.count('(')) + ')' * (16 - prefix.count(')')))
                            rng.shuffle(suffix)
                            text = prefix + ''.join(suffix)
                            # This includes the unmeasured p28 prefix of p20 donors.
                            if text[:28] not in banned[28]:
                                break
                        else:
                            raise RuntimeError('Donor sampling exhausted for {}; no silent relaxation'.format(key))
                        selected.add(prefix)
                        cell = role + '_' + ('neg' if balance < 0 else 'pos') + '_{}_{}'.format(position, replica)
                        donors.append({'role': role, 'cell': cell, 'replica': replica,
                                       'balance': balance, 'position': position, 'string': text,
                                       'prefix_sha256': design.sha(prefix)})
        rows.append({'family_index': index,
                     'family_id': design.sha('averaged-004-family:{}:{}'.format(recipe['donor_seed'], index))[:20],
                     'input_hash_partition': recipe['input_hash_partition'], 'donors': donors})
    return rows, dict(sorted(attempts.items()))


def validate_records(candidates, families, exclusions, recipe=RECIPE):
    if len(candidates) != recipe['native_candidate_count'] or len(families) != recipe['family_count']:
        raise ValueError('Unexpected pool or family size')
    banned_texts = set(exclusions['recipient_strings'])
    banned = {p: set(exclusions['prefixes'][str(p)]) for p in (20, 28)}
    for i, row in enumerate(candidates):
        text = row['recipient']
        if (set(row) != {'candidate_index', 'candidate_id', 'recipient'} or row['candidate_index'] != i
                or len(text) != 32 or text.count('(') != 16 or set(text) != {'(', ')'}
                or text in banned_texts or design.prefix_state(text)[1] >= 0
                or design.phase_of(text) != recipe['input_hash_partition']
                or row['candidate_id'] != design.sha('averaged-004-recipient:{}:{}'.format(recipe['recipient_seed'], i))[:20]):
            raise ValueError('Invalid recipient input')
    all_prefixes = {role: {p: set() for p in (20, 28)} for role in ('calibration', 'target')}
    expected = {(role, b, p, r) for role in ('calibration', 'target')
                for b in (-2, 2) for p in (20, 28) for r in range(2)}
    for i, family in enumerate(families):
        if (set(family) != {'family_index', 'family_id', 'input_hash_partition', 'donors'}
                or family['family_index'] != i or family['input_hash_partition'] != recipe['input_hash_partition']
                or family['family_id'] != design.sha('averaged-004-family:{}:{}'.format(recipe['donor_seed'], i))[:20]):
            raise ValueError('Invalid family identity')
        seen, measured_by_cell = set(), {}
        for donor in family['donors']:
            role, b, p, r = (donor[k] for k in ('role', 'balance', 'position', 'replica'))
            key = role, b, p, r
            if key not in expected or key in seen:
                raise ValueError('Invalid or duplicate donor cell')
            seen.add(key)
            text, cell = donor['string'], role + '_' + ('neg' if b < 0 else 'pos') + '_{}_{}'.format(p, r)
            prefix = text[:p]
            if (set(donor) != {'role', 'cell', 'replica', 'balance', 'position', 'string', 'prefix_sha256'}
                    or len(text) != 32 or text.count('(') != 16 or set(text) != {'(', ')'}
                    or text[p - 1] != ')' or design.prefix_state(prefix) != (b, -4)
                    or role_of(text) != role or design.phase_of(prefix) != recipe['input_hash_partition']
                    or donor['prefix_sha256'] != design.sha(prefix) or donor['cell'] != cell
                    or any(text[:q] in banned[q] for q in (20, 28))):
                raise ValueError('Invalid donor or historical/role prefix leakage')
            if prefix in measured_by_cell.setdefault((b, p), set()):
                raise ValueError('Repeated measured prefix within family cell')
            measured_by_cell[(b, p)].add(prefix)
            for q in (20, 28):
                all_prefixes[role][q].add(text[:q])
        if seen != expected:
            raise ValueError('Incomplete donor family')
    if any(all_prefixes['calibration'][p] & all_prefixes['target'][p] for p in (20, 28)):
        raise ValueError('Global calibration/target prefix leakage')


def finite_p20_support(exclusions):
    """Exact input-only support, before extending prefixes with a suffix."""
    banned = set(exclusions['prefixes']['20'])

    def walk(remaining, depth, touched, target, prefix=''):
        if design.completion_count(remaining, depth, touched, target) == 0:
            return
        if remaining == 0:
            yield prefix + ')'
            return
        for step, symbol in ((1, '('), (-1, ')')):
            yield from walk(remaining - 1, depth + step, touched or depth + step == -4,
                            target, prefix + symbol)

    result = {}
    for balance in (-2, 2):
        counts = Counter()
        for prefix in walk(19, 0, False, balance + 1):
            if prefix not in banned:
                counts[design.phase_of(prefix) + '_' + role_of(prefix)] += 1
        result[str(balance)] = dict(sorted(counts.items()))
    return result


def statistics(candidates, families, exclusions):
    recipients = Counter(row['recipient'] for row in candidates)
    donors = [donor for family in families for donor in family['donors']]
    strings = Counter(donor['string'] for donor in donors)
    prefixes = Counter(donor['prefix_sha256'] for donor in donors)
    by_role = {role: [d for d in donors if d['role'] == role] for role in ('calibration', 'target')}
    return {
        'candidate_draws': len(candidates), 'unique_candidates': len(recipients),
        'maximum_candidate_reuse': max(recipients.values()), 'family_templates': len(families),
        'donor_draws': len(donors), 'unique_donor_strings': len(strings),
        'unique_measured_prefixes': len(prefixes), 'maximum_measured_prefix_reuse': max(prefixes.values()),
        'new_recipient_donor_full_string_overlap': len(set(recipients) & set(strings)),
        'new_recipient_donor_prefix_overlap': {str(p): len({s[:p] for s in recipients} & {s[:p] for s in strings}) for p in (20, 28)},
        'global_cross_role_prefix_overlap': {str(p): len({d['string'][:p] for d in by_role['calibration']} & {d['string'][:p] for d in by_role['target']}) for p in (20, 28)},
        'role_cell_measured_prefixes': {
            '{}_{}_{}'.format(role, b, p): len({d['prefix_sha256'] for d in donors if (d['role'], d['balance'], d['position']) == (role, b, p)})
            for role in ('calibration', 'target') for b in (-2, 2) for p in (20, 28)},
        'historical_full_strings': len(exclusions['recipient_strings']),
        'historical_prefixes': {p: len(rows) for p, rows in exclusions['prefixes'].items()},
        'exact_fresh_p20_support_before_suffix_check': finite_p20_support(exclusions),
    }


def prepare(output=HERE / 'inputs', recipe=RECIPE):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite prepared inputs')
    exclusions = historical_exclusions()
    candidates, recipient_attempts = draw_candidates(exclusions, recipe)
    families, donor_attempts = draw_families(exclusions, recipe)
    validate_records(candidates, families, exclusions, recipe)
    payload = {'exclusions.json': dumps(exclusions), 'candidates.jsonl': jsonl(candidates),
               'donor_families.jsonl': jsonl(families)}
    manifest = {'schema_version': 1, 'status': 'inputs_only_no_measurements', 'recipe': recipe,
                'source_hashes': {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__),
                                  str((VALUE / 'design.py').relative_to(ROOT)): sha(VALUE / 'design.py')},
                'files': {name: hashlib.sha256(content.encode()).hexdigest() for name, content in payload.items()},
                'generation_attempts': {'recipients': recipient_attempts, 'donors_by_role_cell': donor_attempts},
                'statistics': statistics(candidates, families, exclusions)}
    output.mkdir(parents=True)
    for name, content in payload.items():
        (output / name).write_text(content)
    (output / 'preparation.json').write_text(dumps(manifest))
    return manifest


def validate(directory=HERE / 'inputs', regenerate=True):
    directory = Path(directory)
    expected_files = {'preparation.json', 'exclusions.json', 'candidates.jsonl', 'donor_families.jsonl'}
    if {p.name for p in directory.iterdir()} != expected_files:
        raise ValueError('Unexpected input inventory')
    manifest = json.loads((directory / 'preparation.json').read_text())
    if manifest['recipe'] != RECIPE or manifest['status'] != 'inputs_only_no_measurements':
        raise ValueError('Prepared recipe/status differs')
    for relative, digest in manifest['source_hashes'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Preparation source changed: ' + relative)
    for relative, digest in manifest['files'].items():
        if sha(directory / relative) != digest:
            raise ValueError('Prepared file changed: ' + relative)
    exclusions = historical_exclusions()
    if (directory / 'exclusions.json').read_text() != dumps(exclusions):
        raise ValueError('Historical exclusion inventory differs')
    candidates, families = read_rows(directory / 'candidates.jsonl'), read_rows(directory / 'donor_families.jsonl')
    validate_records(candidates, families, exclusions)
    if manifest['statistics'] != statistics(candidates, families, exclusions):
        raise ValueError('Prepared summary differs')
    if regenerate:
        with tempfile.TemporaryDirectory() as temp:
            fresh = Path(temp) / 'inputs'
            prepare(fresh)
            for name in expected_files:
                if (fresh / name).read_bytes() != (directory / name).read_bytes():
                    raise ValueError('Prepared bytes are not deterministic: ' + name)
    return manifest


def validate_inputs(plan, directory=HERE / 'inputs', regenerate=True):
    """Runtime-facing adapter; scientific sampling values must match the freeze."""
    sampling, intervention, model = plan['sampling'], plan['intervention'], plan['model']
    checks = (
        plan['round'] == RECIPE['round'],
        model['model'] == RECIPE['model'],
        model['layer_one_based'] == RECIPE['layer_one_based'],
        model['head_one_based'] == RECIPE['head_one_based'],
        sampling['phase'] == RECIPE['input_hash_partition'],
        sampling['recipient_seed'] == RECIPE['recipient_seed'],
        sampling['donor_seed'] == RECIPE['donor_seed'],
        sampling['native_candidates'] == RECIPE['native_candidate_count'],
        sampling['families'] == RECIPE['family_count'],
        sampling['length'] == RECIPE['length'], sampling['opens'] == RECIPE['open_count'],
        sampling['closes'] == RECIPE['open_count'], sampling['recipient_requires_negative_prefix'] is True,
        sampling['role_rule'] == 'int(SHA256(UTF8("averaged-anchor-004-role:"+first20)),16)%2;0=calibration,1=target',
        sampling['role_assignment_uses_measurements'] is False, sampling['no_global_deduplication'] is True,
        intervention['positions'] == RECIPE['positions'], intervention['balances'] == RECIPE['balances'],
        intervention['minimum_prefix_balance'] == RECIPE['minimum'], intervention['target_symbol'] == RECIPE['read_symbol'],
        intervention['calibration_prefixes_per_cell'] == 2, intervention['target_prefixes_per_cell'] == 2,
    )
    if not all(checks):
        raise ValueError('Plan sampling differs from the prepared input recipe')
    directory = Path(directory)
    manifest = validate(directory, regenerate=regenerate)
    if (sampling['historical_full_strings'] != manifest['statistics']['historical_full_strings']
            or sampling['historical_prefix20'] != manifest['statistics']['historical_prefixes']['20']
            or sampling['historical_prefix28'] != manifest['statistics']['historical_prefixes']['28']):
        raise ValueError('Plan historical support differs')
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
