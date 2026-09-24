"""Outcome-free, IID four-prompt family sampling for Plan 3A.

The fixed ordered names are A=``io`` and B=``subject``. A family contains ABB,
BAB, BAA, ABA, with ABB and BAB as the two recipients. All eight donor cells,
both precisions and all controls belong to this ONE with-replacement draw.
"""
import copy
import hashlib
import json
import random
import string
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

VERSION = 'donor-factor-iid-v1'
DEVELOPMENT_N, DEVELOPMENT_SEED = 32, 24092431
CONFIRMATION_N, CONFIRMATION_SEED = 192, 24092432
PATTERNS = ['ABB', 'BAB', 'BAA', 'ABA']
RECIPIENT_INDICES = [0, 1]
DONOR_INDICES = [{'00': 0, '01': 1, '10': 2, '11': 3},
                 {'00': 1, '01': 0, '10': 3, '11': 2}]
HISTORICAL_CASE_FILES = (
    'results/makelov_read_source_001/cases.json',
    'results/makelov_read_source_q1/cases.json',
    'results/query_route_development/cases.json',
    'results/query_route_confirmation/cases.json',
)
OPTIONAL_HISTORICAL_FILES = ('results/makelov_replication_001/dataset.jsonl',)
SOURCE_NAMES = ('names', 'objects', 'places', 'prefixes', 'templates')
ELIGIBILITY = {
    'answer_names': 'distinct held-out names; one leading-space token each; distinct token ids',
    'prompt_family': 'ABB, BAB, BAA, ABA in this fixed order; four distinct prompts',
    'token_lengths': 'all four prompts have equal token lengths including exactly one explicit BOS',
    'name_slots': 'exactly three varying token positions; name_A, name_B, name_C each occur once '
                  'in that order; prefix-through-name tokenization matches the full prompt and '
                  'the leading-space answer token at every slot',
    'position': 'last absolute prompt token',
    'exclusion': 'reject the whole family if ANY of its four exact prompt strings is excluded',
    'outcome_filter': 'none: no model predictions, activations or outcomes are used',
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
    """Read complete historical pairs or complete new families; fail closed."""
    found = set()
    for item in records:
        record = _mapping(item)
        if 'prompts' in record:
            prompts = record['prompts']
        elif 'base_sentence' in record and 'source_sentence' in record:
            prompts = [record['base_sentence'], record['source_sentence']]
        else:
            raise ValueError('prior record contains no complete prompt pair or family')
        if (not isinstance(prompts, (list, tuple)) or len(prompts) not in (2, 4)
                or not all(isinstance(p, str) and p for p in prompts)):
            raise ValueError('prior record requires two or four nonempty prompt strings')
        found.update(prompts)
    return found


def historical_exclusions(application_root, development_cases=(), extra_case_files=()):
    """Exclude old pilot/Q1/archive, both Round-2 stages and supplied 3A development.

    For confirmation the caller must supply the complete new development case list
    or its file via ``extra_case_files``. Every actual file and every exact excluded
    string is hashed. Missing required files fail closed; an absent optional archive
    is explicitly declared. Only prompt fields are used, never historical outcomes.
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
            raise ValueError(f'prior case file must contain a nonempty record list: {path}')
        prompts = _prompts(records)
        excluded.update(prompts)
        label = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        sources.append({'path': label, 'sha256': hashlib.sha256(content).hexdigest(),
                        'records': len(records), 'unique_prompts': len(prompts),
                        'prompts_sha256': _hash(sorted(prompts))})
    development = [_mapping(c) for c in development_cases]
    if any(not isinstance(c.get('prompts'), (list, tuple)) or len(c['prompts']) != 4
           for c in development):
        raise ValueError('new development exclusions require all four family prompts')
    development_prompts = _prompts(development)
    excluded.update(development_prompts)
    return excluded, {
        'sources': sources, 'optional_sources_absent': absent,
        'additional_development_draws': len(development),
        'additional_development_cases_sha256': _hash(development),
        'additional_development_prompts_sha256': _hash(sorted(development_prompts)),
        'unique_excluded_prompts': len(excluded),
        'excluded_prompts_sha256': _hash(sorted(excluded)),
        'scope': 'Exact prompt strings from the listed actual artifacts and supplied development '
                 'draws; no claim of exhaustive historical or model-training membership auditing.',
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


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _roles(io, subject, answer_ids):
    roles = []
    for index, pattern in enumerate(PATTERNS):
        answer_letter = 'A' if index < 2 else 'B'
        roles.append({
            'prompt_index': index, 'pattern': pattern,
            'correct_answer': io if answer_letter == 'A' else subject,
            'correct_answer_token_id': answer_ids[0 if answer_letter == 'A' else 1],
            'correct_answer_identity': answer_letter,
            'correct_answer_mention_position': pattern[:2].index(answer_letter),
            'first_mentioned_name': io if pattern[0] == 'A' else subject,
            'second_mentioned_name': io if pattern[1] == 'A' else subject,
            'repeated_subject': subject if answer_letter == 'A' else io,
        })
    return roles


def validate_family(record):
    """Validate stored family structure without inferring/repairing missing fields.

    Tokenizer-to-text correspondence is checked during generation; this record-only
    check validates its saved slots, factorial maps, roles and constant scaffold.
    """
    case = copy.deepcopy(_mapping(record))
    required = ('prompts', 'patterns', 'token_ids', 'position', 'io', 'subject',
                'answer_token_ids', 'name_token_positions', 'recipient_indices',
                'donor_indices', 'prompt_roles', 'scaffold')
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError('family missing required fields: ' + ', '.join(missing))
    prompts = case['prompts']
    if (not isinstance(prompts, list) or len(prompts) != 4
            or not all(isinstance(p, str) and p for p in prompts) or len(set(prompts)) != 4
            or case['patterns'] != PATTERNS):
        raise ValueError('family requires four distinct ABB/BAB/BAA/ABA prompts')
    names, answers = [case['io'], case['subject']], case['answer_token_ids']
    if (not all(isinstance(name, str) and name for name in names) or names[0] == names[1]
            or not isinstance(answers, list) or len(answers) != 2
            or not all(_integer(t) for t in answers) or answers[0] == answers[1]):
        raise ValueError('family requires two distinct answer names and token ids')
    scaffold = case['scaffold']
    if (not isinstance(scaffold, dict)
            or any(not isinstance(scaffold.get(key), str)
                   for key in ('prefix', 'template', 'object', 'place'))):
        raise ValueError('family requires its complete source scaffold')
    named = dict(zip(('A', 'B'), names))
    expected_prompts = [_render(scaffold['prefix'], scaffold['template'],
                                dict(zip(('name_A', 'name_B', 'name_C'),
                                         (named[letter] for letter in pattern)),
                                     object=scaffold['object'], place=scaffold['place']))[0]
                        for pattern in PATTERNS]
    if prompts != expected_prompts:
        raise ValueError('prompt strings do not match the declared name assignment and scaffold')
    rows = case['token_ids']
    if (not isinstance(rows, list) or len(rows) != 4
            or any(not isinstance(row, list) or not row for row in rows)
            or len({len(row) for row in rows}) != 1
            or any(not _integer(t) for row in rows for t in row)
            or not _integer(case['position']) or case['position'] != len(rows[0]) - 1):
        raise ValueError('family token lengths or absolute position are inconsistent')
    slots = case['name_token_positions']
    if (not isinstance(slots, list) or len(slots) != 3
            or not all(_integer(p) and 0 < p < len(rows[0]) for p in slots)
            or sorted(set(slots)) != slots):
        raise ValueError('family requires three ordered absolute name token positions')
    varying = [p for p, values in enumerate(zip(*rows)) if len(set(values)) > 1]
    if varying != slots:
        raise ValueError('only the three declared name token positions may vary')
    for pattern, row in zip(PATTERNS, rows):
        if [row[p] for p in slots] != [answers[0 if name == 'A' else 1] for name in pattern]:
            raise ValueError('name token positions do not match factorial answer identities')
    if (_bytes(case['recipient_indices']) != _bytes(RECIPIENT_INDICES)
            or _bytes(case['donor_indices']) != _bytes(DONOR_INDICES)):
        raise ValueError('recipient and donor maps must align both mirrored panels')
    if _bytes(case['prompt_roles']) != _bytes(_roles(*names, answers)):
        raise ValueError('prompt role metadata does not match the fixed factorial semantics')
    content_id = _hash(prompts)[:16]
    if 'content_family_id' in case and case['content_family_id'] != content_id:
        raise ValueError('content_family_id does not match the complete ordered prompt family')
    case['content_family_id'] = content_id
    return case


def _render(prefix, template, replacements):
    """Render text and track the actual three template-field character spans."""
    text, spans, fields = prefix, [], []
    for literal, field, spec, conversion in string.Formatter().parse(template):
        text += literal
        if field is None:
            continue
        if field not in replacements or spec or conversion:
            raise ValueError('unsupported template field')
        value = replacements[field]
        if field.startswith('name_'):
            fields.append(field)
            spans.append((len(text), len(text) + len(value), value))
        text += value
    if fields != ['name_A', 'name_B', 'name_C']:
        raise ValueError('template must contain name_A, name_B, name_C exactly once in order')
    return text, spans


def _name_positions(prompt, spans, tokens, encode, name_ids):
    positions = []
    for start, end, name in spans:
        through = encode(prompt[:end])
        before = prompt[:start]
        # A GPT-2 leading-space name token consumes the immediate preceding space.
        if before.endswith(' '):
            before = before[:-1]
        preceding = encode(before)
        if (not through or through[-1] != name_ids[name][0]
                or through != tokens[:len(through)] or preceding != through[:-1]):
            return None
        positions.append(len(through))  # One explicit BOS shifts the zero-based index.
    return positions


def sample_iid_families(source_dir, tokenizer, *, n, seed, stage, excluded_prompts=(),
                        exclusion_provenance=None, tokenizer_fingerprint=None, max_proposals=None):
    """Draw complete eligible families before inference, without effect filtering."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if not isinstance(stage, str) or not stage:
        raise ValueError('stage must be a nonempty string')
    if max_proposals is None:
        max_proposals = max(1000, n * 10_000)
    if not _integer(max_proposals) or max_proposals < n:
        raise ValueError('max_proposals must be an integer at least n')
    root, data, hashes = Path(source_dir), {}, {}
    for name in SOURCE_NAMES:
        content = (root / 'data' / f'{name}.json').read_bytes()
        data[name] = json.loads(content)
        hashes[f'data/{name}.json'] = hashlib.sha256(content).hexdigest()
        if (not isinstance(data[name], list) or not data[name]
                or not all(isinstance(s, str) and (name == 'prefixes' or bool(s))
                           for s in data[name])):
            raise ValueError(f'{name} must be a nonempty string list; only prefixes may be empty')
    if len(data['prefixes']) < 3 or len(data['templates']) < 3:
        raise ValueError('the declared held-out prefix/template split is unavailable')
    bos = tokenizer.bos_token_id
    if not _integer(bos):
        raise ValueError('tokenizer must declare a nonnegative integer BOS token')
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    heldout_names = data['names'][len(data['names']) // 2:]
    name_ids = {name: encode(' ' + name) for name in heldout_names}
    names = [name for name in heldout_names if len(name_ids[name]) == 1
             and _integer(name_ids[name][0])]
    if len(set(names)) < 2:
        raise ValueError('fewer than two token-eligible held-out names')
    excluded = set(excluded_prompts)
    if not all(isinstance(p, str) and p for p in excluded):
        raise ValueError('excluded prompts must be nonempty strings')
    # Lists retain the source's repeated proposal slots and their sampling weights.
    axes = {'names': names, 'objects': data['objects'][len(data['objects']) // 2:],
            'places': data['places'][len(data['places']) // 2:],
            'templates': data['templates'][2:], 'prefix': data['prefixes'][2]}
    definition = {
        'version': VERSION, 'source_files': hashes, 'axes': axes,
        'tokenizer': _fingerprint(tokenizer, tokenizer_fingerprint), 'bos_token_id': bos,
        'eligibility': copy.deepcopy(ELIGIBILITY),
        'proposal': 'uniform distinct ordered name slots × uniform object × uniform place × '
                    'uniform template, independently with replacement; reject equal names/tokens',
        'proposal_tuple_count': len(names) * (len(names) - 1) * len(axes['objects'])
                                * len(axes['places']) * len(axes['templates']),
        'patterns': PATTERNS.copy(), 'recipient_indices': RECIPIENT_INDICES.copy(),
        'donor_indices': copy.deepcopy(DONOR_INDICES),
        'excluded_prompts': sorted(excluded), 'eligible_population_count': None,
        'eligible_population_count_reason': 'Defined by deterministic eligibility; not enumerated.',
    }
    population_hash = _hash(definition)
    rng, draws, rejected = random.Random(seed), [], Counter()
    for proposal_index in range(max_proposals):
        io, subject = rng.sample(names, 2)
        obj, place, template = (rng.choice(axes[key]) for key in ('objects', 'places', 'templates'))
        if io == subject or name_ids[io] == name_ids[subject]:
            rejected['indistinct_answer_tokens'] += 1
            continue
        named = {'A': io, 'B': subject}
        try:
            rendered = [_render(axes['prefix'], template,
                                dict(zip(('name_A', 'name_B', 'name_C'),
                                         (named[letter] for letter in pattern)),
                                     object=obj, place=place)) for pattern in PATTERNS]
        except ValueError:
            rejected['invalid_name_template'] += 1
            continue
        prompts = [p for p, _ in rendered]
        if any(p in excluded for p in prompts):
            rejected['excluded_prompt'] += 1
            continue
        bare_tokens = [encode(p) for p in prompts]
        if len({len(row) for row in bare_tokens}) != 1:
            rejected['unequal_token_lengths'] += 1
            continue
        positions = [_name_positions(p, spans, row, encode, name_ids)
                     for (p, spans), row in zip(rendered, bare_tokens)]
        if positions[0] is None or any(p != positions[0] for p in positions):
            rejected['name_token_alignment'] += 1
            continue
        ids = [[bos] + row for row in bare_tokens]
        varying = [p for p, values in enumerate(zip(*ids)) if len(set(values)) > 1]
        if varying != positions[0]:
            rejected['non_name_token_variation'] += 1
            continue
        answers = [name_ids[io][0], name_ids[subject][0]]
        case = validate_family({
            'prompts': prompts, 'patterns': PATTERNS.copy(), 'token_ids': ids,
            'position': len(ids[0]) - 1, 'io': io, 'subject': subject,
            'answer_token_ids': answers, 'name_token_positions': positions[0],
            'recipient_indices': RECIPIENT_INDICES.copy(),
            'donor_indices': copy.deepcopy(DONOR_INDICES), 'prompt_roles': _roles(io, subject, answers),
            'scaffold': {'prefix': axes['prefix'], 'template': template, 'object': obj, 'place': place},
        })
        index = len(draws)
        unique = _hash([VERSION, population_hash, seed, stage, index])[:24]
        case.update(case_id=unique, unique_draw_id=unique, draw=index, draw_index=index,
                    proposal_index=proposal_index)
        draws.append(case)
        if len(draws) == n:
            counts = Counter(c['content_family_id'] for c in draws)
            return draws, {
                'sampling_version': VERSION, 'stage': stage, 'seed': seed,
                'n_family_draws': len(draws), 'sampling_definition': definition,
                'population_definition_sha256': population_hash,
                'definition_hash_scope': 'Definition of the eligible sampling population; '
                                         'not a materialized eligible-population hash.',
                'proposal_count': proposal_index + 1, 'max_proposals': max_proposals,
                'rejection_counts': dict(rejected), 'unique_content_families': len(counts),
                'repeated_content_draws': len(draws) - len(counts),
                'content_multiplicities_above_one': {key: value for key, value in sorted(counts.items())
                                                   if value > 1},
                'draw_cases_sha256': _hash(draws), 'exclusion_provenance': exclusion_provenance,
                'inference_order': 'All complete families, eligibility decisions, and exclusions '
                                   'are determined before any model inference; no outcome filtering.',
                'independent_unit': 'unique_draw_id / case_id: one four-prompt family including '
                                    'both mirrored recipient panels, all controls and precisions; '
                                    'retain repeated content as independent draws.',
                'population_scope': 'Conditional on the fixed source, tokenizer and exclusions. '
                                    'Freshness excludes declared prompts, not every possible past use.',
            }
    raise ValueError('fixed proposal budget exhausted; no complete sample produced; '
                     'do not extend or replace failed draws based on outcomes')
