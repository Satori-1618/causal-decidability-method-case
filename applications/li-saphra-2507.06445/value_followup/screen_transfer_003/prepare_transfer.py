"""Prepare all six frozen input pools without loading a model or any outcomes."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
VALUE = HERE.parent
NATIVE = VALUE.parent / 'native_followup'
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('transfer_input_design', VALUE / 'design.py')
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dumps(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def jsonl(rows):
    return ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def head_key(task):
    return '{}_head{}'.format(task['model'], task['head_one_based'])


def input_recipe(plan):
    """Ignore release bookkeeping, which changes after preparation and review."""
    return {'round': plan['round'], 'phase': plan['screen']['phase'],
            'pool_size': plan['screen']['native_candidates_per_head'],
            'per_stratum': plan['screen']['families_per_stratum'],
            'heads': [{key: task[key] for key in ('model', 'head_one_based', 'recipient_seed', 'donor_seed')}
                      for task in plan['cohort']]}


def historical_exclusions():
    # The committed prior exclusion surface already contains public/native
    # inputs. A clean export therefore needs neither cached CSVs nor a download.
    base = VALUE / 'screen_002/inputs/exclusions.json'
    old_prep = json.loads((base.parent / 'preparation.json').read_text())
    if sha(base) != old_prep['files']['exclusions.json']:
        raise ValueError('Committed historical exclusion surface changed')
    exclusions = json.loads(base.read_text())
    texts = set(exclusions['recipient_strings'])
    sources = {str(base.relative_to(ROOT)): sha(base),
               str((base.parent / 'preparation.json').relative_to(ROOT)): sha(base.parent / 'preparation.json')}
    for relative in ('inputs/development_001/cases.jsonl',
                     'screen_002/inputs/candidates.jsonl',
                     'screen_002/inputs/donor_families.jsonl'):
        path = VALUE / relative
        for row in read_rows(path):
            texts.add(row['recipient'])
            texts.update(donor['string'] for donor in row.get('donors', []))
        sources[str(path.relative_to(ROOT))] = sha(path)
    result = {'recipient_strings': sorted(texts),
              'prefixes': {str(p): sorted({text[:p] for text in texts if len(text) >= p})
                           for p in design.POSITIONS},
              'source_hashes': sources,
              'rule': 'All prior prepared/opened full strings; both p20 and p28 prefixes from every role'}
    if (len(texts), len(result['prefixes']['20']), len(result['prefixes']['28'])) != (5424, 5054, 3671):
        raise ValueError('Historical exclusion population changed; amend before generating inputs')
    return result


def draw_candidates(exclusions, seed, n):
    rng = random.Random(seed)
    banned = set(exclusions['recipient_strings'])
    rows = []
    for index in range(n):
        for _ in range(100000):
            chars = list('(' * 16 + ')' * 16)
            rng.shuffle(chars)
            text = ''.join(chars)
            if (text not in banned and design.phase_of(text) == 'development'
                    and design.prefix_state(text)[1] < 0):
                break
        else:
            raise RuntimeError('Candidate generation exhausted; do not relax constraints')
        rows.append({'candidate_id': design.sha('screen-transfer-003:{}:{}'.format(seed, index))[:20],
                     'candidate_index': index, 'recipient': text})
    return rows


def head_payload(task, recipe, exclusions):
    candidates = draw_candidates(exclusions, task['recipient_seed'], recipe['pool_size'])
    donors = design.generate(task['donor_seed'], 2 * recipe['per_stratum'], 'development', exclusions)
    prefixes = Counter(d['prefix_sha256'] for family in donors for d in family['donors'])
    recipients = Counter(row['recipient'] for row in candidates)
    statistics = {'unique_candidate_strings': len(recipients), 'maximum_candidate_reuse': max(recipients.values()),
                  'donor_prefix_draws': sum(prefixes.values()), 'unique_donor_prefixes': len(prefixes),
                  'maximum_donor_prefix_reuse': max(prefixes.values()),
                  'sampling': 'With replacement across families; donor templates independent of native screening'}
    return candidates, donors, statistics


def overlap_statistics(payloads):
    """Describe reuse; never remove a draw or change the declared sampling."""
    recipients = {key: {r['recipient'] for r in value[0]} for key, value in payloads.items()}
    prefixes = {key: {d['prefix_sha256'] for f in value[1] for d in f['donors']}
                for key, value in payloads.items()}
    keys = sorted(payloads)
    pairs = [{'head_a': a, 'head_b': b,
              'shared_recipient_strings': len(recipients[a] & recipients[b]),
              'shared_donor_prefix_hashes': len(prefixes[a] & prefixes[b])}
             for i, a in enumerate(keys) for b in keys[i + 1:]]
    return {'scope': 'All prepared donors, including unmeasured cells and unused templates',
            'unique_candidate_strings_across_heads': len(set().union(*recipients.values())),
            'unique_donor_prefixes_across_heads': len(set().union(*prefixes.values())),
            'pairs': pairs}


def prepare(plan, output):
    if output.exists():
        raise ValueError('Refusing to overwrite prepared inputs')
    recipe = input_recipe(plan)
    if (recipe['pool_size'], recipe['per_stratum'], len(recipe['heads']), recipe['phase']) != (1024, 64, 6, 'development'):
        raise ValueError('This preparation implements only the frozen six-head 1024/64 design')
    exclusions = historical_exclusions()
    # Generate everything in memory first: no half-written set after a generation failure.
    payloads = {head_key(task): head_payload(task, recipe, exclusions) for task in plan['cohort']}
    output.mkdir(parents=True)
    (output / 'exclusions.json').write_text(dumps(exclusions))
    metadata = {'round': plan['round'], 'recipe': recipe,
                'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                            str((VALUE / 'design.py').relative_to(ROOT)): sha(VALUE / 'design.py')},
                'historical_unique_full_strings': len(exclusions['recipient_strings']),
                'historical_prefix_counts': {p: len(v) for p, v in exclusions['prefixes'].items()},
                'files': {'exclusions.json': sha(output / 'exclusions.json')}, 'heads': {},
                'between_head_overlaps': overlap_statistics(payloads)}
    for task in plan['cohort']:
        key = head_key(task)
        directory = output / key
        directory.mkdir()
        candidates, donors, stats = payloads[key]
        (directory / 'candidates.jsonl').write_text(jsonl(candidates))
        (directory / 'donor_families.jsonl').write_text(jsonl(donors))
        info = {'head_key': key, 'recipient_seed': task['recipient_seed'], 'donor_seed': task['donor_seed'],
                'candidate_pool_size': len(candidates), 'donor_templates': len(donors), **stats}
        (directory / 'preparation.json').write_text(dumps(info))
        metadata['heads'][key] = info
        for name in ('candidates.jsonl', 'donor_families.jsonl', 'preparation.json'):
            metadata['files'][key + '/' + name] = sha(directory / name)
    (output / 'preparation.json').write_text(dumps(metadata))
    return metadata


def validate_inputs(plan, directory, regenerate=True):
    """Byte-compare the deterministic recipe, including all six donors and bans."""
    prep = json.loads((directory / 'preparation.json').read_text())
    if prep['recipe'] != input_recipe(plan):
        raise ValueError('Prepared recipe differs from plan')
    expected_files = {'exclusions.json'} | {head_key(task) + '/' + name for task in plan['cohort']
                       for name in ('candidates.jsonl', 'donor_families.jsonl', 'preparation.json')}
    if set(prep['files']) != expected_files or set(prep['heads']) != {head_key(t) for t in plan['cohort']}:
        raise ValueError('Prepared file/head inventory changed')
    if prep['sources'] != {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                           str((VALUE / 'design.py').relative_to(ROOT)): sha(VALUE / 'design.py')}:
        raise ValueError('Prepared generator inventory changed')
    for relative, expected in prep['files'].items():
        path = (directory / relative).resolve()
        if directory.resolve() not in path.parents or sha(path) != expected:
            raise ValueError('Prepared input hash mismatch: ' + relative)
    for relative, expected in prep['sources'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Generator changed: ' + relative)
    exclusions = historical_exclusions()
    if (directory / 'exclusions.json').read_text() != dumps(exclusions):
        raise ValueError('Historical exclusions differ')
    if (prep['historical_unique_full_strings'] != len(exclusions['recipient_strings']) or
            prep['historical_prefix_counts'] != {p: len(v) for p, v in exclusions['prefixes'].items()}):
        raise ValueError('Historical counts differ')
    inputs = {}
    for task in plan['cohort']:
        key = head_key(task)
        candidates = read_rows(directory / key / 'candidates.jsonl')
        donors = read_rows(directory / key / 'donor_families.jsonl')
        if regenerate:
            expected_candidates, expected_donors, stats = head_payload(task, prep['recipe'], exclusions)
            if (jsonl(candidates) != jsonl(expected_candidates) or jsonl(donors) != jsonl(expected_donors)):
                raise ValueError('Deterministic regeneration differs: ' + key)
            info = {'head_key': key, 'recipient_seed': task['recipient_seed'], 'donor_seed': task['donor_seed'],
                    'candidate_pool_size': len(candidates), 'donor_templates': len(donors), **stats}
            if prep['heads'][key] != info or (directory / key / 'preparation.json').read_text() != dumps(info):
                raise ValueError('Prepared head metadata differs: ' + key)
        inputs[key] = (candidates, donors)
    if prep['between_head_overlaps'] != overlap_statistics(inputs):
        raise ValueError('Between-head overlap metadata differs')
    return prep, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'inputs')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads((HERE / 'plan.json').read_text())
    result = validate_inputs(plan, args.output)[0] if args.validate_only else prepare(plan, args.output)
    print(dumps(result))


if __name__ == '__main__':
    main()
