"""IID base-pair sampling for the query-route experiment; no model dependencies.

Draw with replacement from a fixed Cartesian proposal, rejecting only predeclared
text/token eligibility failures and excluded historical prompts. Every accepted draw
is a new statistical unit, even when its content appeared in another draw. This is
uniform conditional sampling over eligible proposal tuples, not forced uniqueness.

The large eligible population is defined factorwise, not materialized. Its definition
hash binds the source files, axes, tokenizer, eligibility rules and exclusions; it is
explicitly NOT a hash of an enumerated eligible-pair list. Confirmation is iid conditional
on its frozen exclusions, which include the complete development prompt set.
"""
import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

VERSION = 'query-route-iid-v1'
PILOT_N, PILOT_SEED = 32, 24092401
CONFIRMATION_N, CONFIRMATION_SEED = 192, 24092402
HISTORICAL_CASE_FILES = (
    'results/makelov_read_source_001/cases.json',
    'results/makelov_read_source_q1/cases.json',
)
OPTIONAL_HISTORICAL_FILES = ('results/makelov_replication_001/dataset.jsonl',)
SOURCE_NAMES = ('names', 'objects', 'places', 'prefixes', 'templates')
CASE_FIELDS = ('prompts', 'patterns', 'token_ids', 'position', 'io', 'subject',
               'answer_token_ids')
ELIGIBILITY = {
    'answer_names': 'distinct names, each exactly one token with a leading space; distinct token ids',
    'prompt_pair': 'ABB and BAB; same IO and subject; equal token lengths including BOS',
    'position': 'last absolute prompt token',
    'exclusion': 'reject entire pair if either exact prompt is in the frozen exclusion set',
    'outcome_filter': 'none: no model predictions, activations or outcome values are used',
    'duplicate_policy': 'retain repeated content as independent with-replacement draws',
}


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _mapping(case):
    return asdict(case) if is_dataclass(case) else dict(case)


def _prompts(records):
    found = set()
    for record in records:
        record = _mapping(record)
        if 'prompts' in record:
            prompts = record['prompts']
        elif 'base_sentence' in record and 'source_sentence' in record:
            prompts = [record['base_sentence'], record['source_sentence']]
        else:
            raise ValueError('prior record contains no complete prompt pair')
        if not isinstance(prompts, (list, tuple)) or len(prompts) != 2:
            raise ValueError('prior record needs exactly two prompts')
        if not all(isinstance(p, str) and p for p in prompts):
            raise ValueError('prior prompts must be nonempty strings')
        found.update(prompts)
    return found


def historical_exclusions(application_root, pilot_cases=(), extra_case_files=()):
    """Required Q1/pilot prompts, optional archived Table-1 prompts, and all new pilot draws.

    Missing required files fail closed. Optional files are explicitly listed as absent
    rather than silently represented as checked. Additional supplied files must exist.
    Only prompt fields and byte hashes are read; no old outcome values are inspected.
    """
    root = Path(application_root)
    paths = [root / p for p in HISTORICAL_CASE_FILES]
    absent = []
    for name in OPTIONAL_HISTORICAL_FILES:
        if (root / name).is_file():
            paths.append(root / name)
        else:
            absent.append(name)
    paths.extend(Path(p) if Path(p).is_absolute() else root / p for p in extra_case_files)
    excluded, sources = set(), []
    for path in dict.fromkeys(paths):
        content = path.read_bytes()
        records = ([json.loads(line) for line in content.splitlines() if line.strip()]
                   if path.suffix == '.jsonl' else json.loads(content))
        if not isinstance(records, list) or not records:
            raise ValueError(f'prior case file must contain a nonempty record list: {path.name}')
        prompts = _prompts(records)
        excluded.update(prompts)
        label = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
        sources.append({'path': label, 'sha256': hashlib.sha256(content).hexdigest(),
                        'records': len(records), 'unique_prompts': len(prompts)})
    pilot = [_mapping(c) for c in pilot_cases]
    excluded.update(_prompts(pilot))
    return excluded, {
        'sources': sources, 'optional_sources_absent': absent,
        'additional_pilot_draws': len(pilot), 'additional_pilot_cases_sha256': _hash(pilot),
        'unique_excluded_prompts': len(excluded), 'excluded_prompts_sha256': _hash(sorted(excluded)),
        'scope': 'Exact prompts in the listed artifacts and supplied pilot draws; no claim '
                 'of exhaustive historical or original model-training membership auditing.',
    }


def _fingerprint(tokenizer, supplied):
    if supplied is not None:
        if not isinstance(supplied, str) or not supplied:
            raise ValueError('tokenizer_fingerprint must be a nonempty declared string')
        return {'kind': 'caller_supplied', 'value': supplied}
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is None or not hasattr(backend, 'to_str'):
        raise ValueError('supply a tokenizer_fingerprint or use a serializable fast tokenizer')
    return {'kind': 'fast_tokenizer_backend',
            'backend_sha256': hashlib.sha256(backend.to_str().encode()).hexdigest(),
            'special_token_ids': {name: getattr(tokenizer, name, None) for name in
                                  ('bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id')}}


def _draw_declarations(n, seed, stage):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if not isinstance(stage, str) or not stage:
        raise ValueError('stage must be a nonempty string')


def _validated_case(case):
    source = _mapping(case)
    case = {key: copy.deepcopy(source[key]) for key in CASE_FIELDS}
    _prompts([case])
    if case['patterns'] != ['ABB', 'BAB'] or case['prompts'][0] == case['prompts'][1]:
        raise ValueError('case must contain distinct reciprocal ABB/BAB prompts')
    ids = case['token_ids']
    if (len(ids) != 2 or not ids[0] or len(ids[0]) != len(ids[1])
            or case['position'] != len(ids[0]) - 1):
        raise ValueError('case token lengths or absolute position are inconsistent')
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for row in ids for x in row):
        raise ValueError('token ids must be nonnegative integers')
    answers = case['answer_token_ids']
    if len(answers) != 2 or answers[0] == answers[1] or case['io'] == case['subject']:
        raise ValueError('two distinct answer names and token ids are required')
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in answers):
        raise ValueError('answer token ids must be nonnegative integers')
    # Preserve the historical content ID algorithm, but never use it as the draw/unit ID.
    case['content_pair_id'] = hashlib.sha256(json.dumps(case['prompts']).encode()).hexdigest()[:16]
    return case


def _append_draw(draws, case, population_hash, seed, stage, proposal_index):
    index = len(draws)
    unique = _hash([VERSION, population_hash, seed, stage, index])[:24]
    record = copy.deepcopy(case)
    record.update(case_id=unique, unique_draw_id=unique, draw=index, draw_index=index,
                  proposal_index=proposal_index)
    draws.append(record)


def _manifest(draws, definition, seed, stage, proposals, rejection_counts,
              exclusion_provenance):
    counts = Counter(c['content_pair_id'] for c in draws)
    return {
        'sampling_version': VERSION, 'stage': stage, 'seed': seed,
        'n_base_pair_draws': len(draws), 'sampling_definition': definition,
        'population_definition_sha256': _hash(definition),
        'definition_hash_scope': 'Definition of the eligible sampling population; '
                                 'not a materialized eligible-population hash.',
        'proposal_count': proposals, 'rejection_counts': dict(rejection_counts),
        'unique_content_pairs': len(counts), 'repeated_content_draws': len(draws) - len(counts),
        'content_multiplicities_above_one': {key: value for key, value in sorted(counts.items())
                                           if value > 1},
        'draw_cases_sha256': _hash(draws), 'exclusion_provenance': exclusion_provenance,
        'independent_unit': 'unique_draw_id / case_id; both reciprocal directions belong '
                            'to one accepted iid base-pair draw; do not merge repeated content',
        'population_scope': 'Conditional on the fixed source, tokenizer and exclusions. '
                            'Freshness excludes declared prompts, not every possible past use.',
    }


def sample_iid_cases(source_dir, tokenizer, *, n, seed, stage, excluded_prompts=(),
                     exclusion_provenance=None, tokenizer_fingerprint=None, max_proposals=None):
    """IID rejection draws; never deduplicate accepted content or use model outcomes."""
    _draw_declarations(n, seed, stage)
    if max_proposals is None:
        max_proposals = max(1000, n * 10_000)
    if isinstance(max_proposals, bool) or not isinstance(max_proposals, int) or max_proposals < n:
        raise ValueError('max_proposals must be an integer at least n')
    root = Path(source_dir)
    data, hashes = {}, {}
    for name in SOURCE_NAMES:
        content = (root / 'data' / f'{name}.json').read_bytes()
        data[name] = json.loads(content)
        hashes[f'data/{name}.json'] = hashlib.sha256(content).hexdigest()
        if (not isinstance(data[name], list) or not data[name]
                or not all(isinstance(s, str) and s for s in data[name])
                or len(set(data[name])) != len(data[name])):
            raise ValueError(f'{name} must be a nonempty list of unique strings')
    if len(data['prefixes']) < 3 or len(data['templates']) < 3:
        raise ValueError('the declared held-out prefix/template split is unavailable')
    bos = tokenizer.bos_token_id
    if isinstance(bos, bool) or not isinstance(bos, int) or bos < 0:
        raise ValueError('tokenizer must declare a nonnegative integer BOS token')
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    heldout_names = data['names'][len(data['names']) // 2:]
    name_ids = {name: encode(' ' + name) for name in heldout_names}
    names = [name for name in heldout_names if len(name_ids[name]) == 1]
    if len(names) < 2:
        raise ValueError('fewer than two token-eligible held-out names')
    excluded = set(excluded_prompts)
    if not all(isinstance(p, str) and p for p in excluded):
        raise ValueError('excluded prompts must be nonempty strings')
    axes = {'names': names, 'objects': data['objects'][len(data['objects']) // 2:],
            'places': data['places'][len(data['places']) // 2:],
            'templates': data['templates'][2:], 'prefix': data['prefixes'][2]}
    definition = {'version': VERSION, 'source_files': hashes, 'axes': axes,
                  'tokenizer': _fingerprint(tokenizer, tokenizer_fingerprint), 'bos_token_id': bos,
                  'eligibility': ELIGIBILITY,
                  'proposal': 'uniform distinct ordered name pair × uniform object × '
                              'uniform place × uniform template, independently with replacement',
                  'proposal_tuple_count': len(names) * (len(names) - 1) * len(axes['objects'])
                                          * len(axes['places']) * len(axes['templates']),
                  'excluded_prompts': sorted(excluded),
                  'eligible_population_count': None,
                  'eligible_population_count_reason': 'Defined by deterministic eligibility; not enumerated.'}
    population_hash = _hash(definition)
    rng, draws, rejected = random.Random(seed), [], Counter()
    for proposal_index in range(max_proposals):
        io, subject = rng.sample(names, 2)
        obj, place, template = (rng.choice(axes[key]) for key in ('objects', 'places', 'templates'))
        prompts = [axes['prefix'] + template.format(name_A=a, name_B=b, name_C=subject,
                                                    object=obj, place=place)
                   for a, b in ((io, subject), (subject, io))]
        if any(p in excluded for p in prompts):
            rejected['excluded_prompt'] += 1
            continue
        ids = [[bos] + encode(p) for p in prompts]
        if len(ids[0]) != len(ids[1]):
            rejected['unequal_token_lengths'] += 1
            continue
        if name_ids[io] == name_ids[subject]:
            rejected['indistinct_answer_tokens'] += 1
            continue
        case = _validated_case({'prompts': prompts, 'patterns': ['ABB', 'BAB'],
                                'token_ids': ids, 'position': len(ids[0]) - 1,
                                'io': io, 'subject': subject,
                                'answer_token_ids': [name_ids[io][0], name_ids[subject][0]]})
        _append_draw(draws, case, population_hash, seed, stage, proposal_index)
        if len(draws) == n:
            manifest = _manifest(draws, definition, seed, stage, proposal_index + 1,
                                 rejected, exclusion_provenance)
            manifest['max_proposals'] = max_proposals
            return draws, manifest
    raise ValueError('fixed proposal budget exhausted; no complete sample produced; '
                     'do not conditionally extend or replace failed draws based on outcomes')


def sample_from_eligible_pool(eligible_cases, *, n, seed, stage, excluded_prompts=(),
                              exclusion_provenance=None):
    """Small enumerated-pool variant: uniform iid draws WITH replacement.

    The caller constructs token eligibility; this function checks the supplied case
    structure and frozen prompt exclusions. Duplicate pool entries are refused, while
    duplicate sampled content is retained. Useful for small pools and CPU-only tests.
    """
    _draw_declarations(n, seed, stage)
    excluded = set(excluded_prompts)
    pool = [_validated_case(c) for c in eligible_cases]
    pool = [c for c in pool if not any(p in excluded for p in c['prompts'])]
    if not pool:
        raise ValueError('eligible population is empty after exclusions')
    by_id = {c['content_pair_id']: c for c in pool}
    if len(by_id) != len(pool):
        raise ValueError('duplicate content in the declared eligible pool; do not change sampling weights')
    pool = [by_id[key] for key in sorted(by_id)]
    definition = {'version': VERSION, 'proposal': 'uniform enumerated eligible pool with replacement',
                  'eligibility': ELIGIBILITY, 'eligible_population_count': len(pool),
                  'eligible_pool_sha256': _hash(pool), 'excluded_prompts': sorted(excluded)}
    population_hash = _hash(definition)
    rng, draws = random.Random(seed), []
    for index in range(n):
        _append_draw(draws, rng.choice(pool), population_hash, seed, stage, index)
    return draws, _manifest(draws, definition, seed, stage, n, {}, exclusion_provenance)
