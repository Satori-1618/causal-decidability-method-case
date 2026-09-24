"""IID native-only Stage A families, frozen before any model outputs.

Every family contains all 48 wording/context/query rows, including duplicate
prompt strings under distinct context IDs. Eligibility uses source membership,
exact strings and tokenizer behavior only. Accepted content is never deduplicated.
The count-noun proposal whitelist is fixed before sampling; it is not an outcome
filter. This module neither loads nor calls a model.
"""
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import role_design
from donor_factor_sampling import (_fingerprint, _hash, HISTORICAL_CASE_FILES as EARLIER_CASE_FILES,
                                  OPTIONAL_HISTORICAL_FILES, historical_exclusions as _earlier_exclusions)

VERSION = 'role-baseline-iid-v1'
DEVELOPMENT_N, DEVELOPMENT_SEED = 32, 24092441
STAGE = 'baseline_development'
PREFIX = 'Then, '
CONTEXTS = {**role_design.DONORS, **role_design.RECIPIENTS}
ROUND3A_CASE_FILES = ('results/donor_factor_development/cases.json',
                      'results/donor_factor_confirmation_512/cases.json')
HISTORICAL_CASE_FILES = (*EARLIER_CASE_FILES, *ROUND3A_CASE_FILES)
OBJECT_WHITELIST = (
    'camera', 'guitar', 'necklace', 'mirror', 'cup', 'flag', 'shovel', 'cooler',
    'hammer', 'wrench', 'towel', 'glove', 'speaker', 'remote', 'leash', 'magazine',
    'notebook', 'candle', 'feather', 'laptop', 'pamphlet', 'knife', 'kettle', 'scarf',
    'tie', 'joystick', 'bookmark', 'microphone', 'hat', 'harness', 'roller', 'blanket',
    'folder', 'bag', 'crate', 'pot', 'watch', 'mug', 'sandwich', 'ring', 'backpack',
    'pencil', 'broom', 'baseball', 'basket', 'loaf', 'helmet', 'bible', 'jacket',
)
ELIGIBILITY = {
    'names': 'three distinct ordered held-out names; each one leading-space token; distinct token ids',
    'objects': 'prespecified count-noun whitelist intersected with the held-out source slots',
    'prompts': 'exact role_design.render output, including its common Then, prefix; 48 rows retained',
    'mentions': 'every name occurs exactly once per prompt; each actual mention is the corresponding '
                'single leading-space name token, checked against prefix tokenization',
    'query_lengths': 'all three query prompts have equal token length within each context and wording',
    'donor_collision': 'd0/giver and d1/receiver have equal length, all three first-mention token '
                       'positions, and the same correct name/token/mention position within each wording',
    'positions': 'one explicit BOS; absolute final positions recorded; different forms need not have equal length',
    'exclusions': 'reject the whole family if any exact prompt matches a frozen historical exclusion',
    'outcome_filter': 'none; no predictions, native competence, activations, patch effects or other outputs are used',
    'duplicate_policy': 'retain with-replacement content repeats as distinct independent draws',
}


def historical_exclusions(application_root, extra_case_files=()):
    """Require old pilot/Q1, both Round-2 stages, both executed 3A stages; read archive if present."""
    return _earlier_exclusions(application_root,
                               extra_case_files=(*ROUND3A_CASE_FILES, *extra_case_files))


def expected_row_ids():
    return [f'{wording}/{context}/{query}' for wording in role_design.WORDINGS
            for context in CONTEXTS for query in role_design.ROLES]


def _integer(value):
    return type(value) is int and value >= 0


def _name_spans(prompt, names):
    matches = []
    for index, name in enumerate(names):
        found = list(re.finditer(r'(?<!\w)'+re.escape(name)+r'(?!\w)', prompt))
        if len(found) != 1:
            return None
        matches.append((found[0].start(), found[0].end(), index))
    return sorted(matches)


def _mention_tokens(prompt, spans, bare_tokens, encode, answer_ids):
    positions = []
    for start, end, name_index in spans:
        if start == 0 or prompt[start-1] != ' ':
            return None
        through = encode(prompt[:end])
        preceding = encode(prompt[:start-1])
        if (not through or through[-1] != answer_ids[name_index]
                or through != bare_tokens[:len(through)] or preceding != through[:-1]):
            return None
        positions.append(len(through))  # Explicit BOS moves the token index right by one.
    return positions


def validate_family(case):
    """Validate saved row coverage, semantics and token structure without a tokenizer."""
    required = ('names', 'answer_token_ids', 'item', 'bos_token_id', 'rows', 'prompts')
    if not isinstance(case, dict) or any(key not in case for key in required):
        raise ValueError('family is missing required structural fields')
    names, answers, rows = case['names'], case['answer_token_ids'], case['rows']
    if (not isinstance(names, list) or len(names) != 3
            or not all(isinstance(n, str) and n for n in names) or len(set(names)) != 3
            or not isinstance(answers, list) or len(answers) != 3
            or not all(_integer(a) for a in answers) or len(set(answers)) != 3
            or not _integer(case['bos_token_id'])):
        raise ValueError('three distinct names/answer token ids and a valid BOS are required')
    if case['item'] not in OBJECT_WHITELIST:
        raise ValueError('item is outside the frozen object whitelist')
    if (not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows)
            or len(rows) != 48 or [r.get('row_id') for r in rows] != expected_row_ids()
            or case['prompts'] != [r.get('prompt') for r in rows]):
        raise ValueError('family must contain all 48 rows in the frozen order')
    by_id = {}
    for row in rows:
        needed = ('row_id', 'context_id', 'context', 'form', 'wording', 'query', 'correct_name_index',
                  'correct_answer_token_id', 'prompt', 'token_ids', 'position',
                  'name_token_positions', 'name_char_spans', 'first_mention_name_indices')
        if any(key not in row for key in needed):
            raise ValueError('row is missing required structural fields')
        wording, context_id, query = row['row_id'].split('/')
        context = CONTEXTS[context_id]
        target = role_design.answer(context, query)
        if (row['context_id'] != context_id or row['context'] != context_id
                or row['form'] != context['form'] or row['wording'] != wording or row['query'] != query
                or type(row['correct_name_index']) is not int or row['correct_name_index'] != target
                or row['correct_answer_token_id'] != answers[target]):
            raise ValueError('row semantics differ from the frozen role design')
        prompt = role_design.render(context, query, names=names, item=case['item'], wording=wording)
        if row['prompt'] != prompt or not prompt.startswith(PREFIX):
            raise ValueError('prompt differs from the exact prefixed role_design rendering')
        spans = _name_spans(prompt, names)
        order = list(role_design.mentions(context))
        if (spans is None or [i for _, _, i in spans] != order
                or row['name_char_spans'] != [[a, b] for a, b, _ in spans]
                or row['first_mention_name_indices'] != order):
            raise ValueError('names must occur once each in the declared mention order')
        ids, positions = row['token_ids'], row['name_token_positions']
        if (not isinstance(ids, list) or not ids or not all(_integer(t) for t in ids)
                or ids[0] != case['bos_token_id'] or not _integer(row['position'])
                or row['position'] != len(ids)-1 or not isinstance(positions, list) or len(positions) != 3
                or not all(_integer(p) and 0 < p < len(ids) for p in positions)
                or positions != sorted(set(positions))
                or [ids[p] for p in positions] != [answers[i] for i in order]):
            raise ValueError('row token ids, absolute position or single-name slots are inconsistent')
        by_id[row['row_id']] = row
    for wording in role_design.WORDINGS:
        for context_id in CONTEXTS:
            group = [by_id[f'{wording}/{context_id}/{q}'] for q in role_design.ROLES]
            if len({len(row['token_ids']) for row in group}) != 1:
                raise ValueError('query lengths differ within context/wording')
            if any(row['name_token_positions'] != group[0]['name_token_positions'] for row in group):
                raise ValueError('first-mention positions differ across queries of one story')
        left, right = (by_id[f'{wording}/{context}/{query}']
                       for context, query in (('d0', 'giver'), ('d1', 'receiver')))
        if (len(left['token_ids']) != len(right['token_ids'])
                or left['name_token_positions'] != right['name_token_positions']
                or left['first_mention_name_indices'] != right['first_mention_name_indices']
                or left['correct_name_index'] != right['correct_name_index']):
            raise ValueError('matched donor collision does not preserve name/token/mention positions')
    return copy.deepcopy(case)


def sample_families(source_dir, tokenizer, n=DEVELOPMENT_N, seed=DEVELOPMENT_SEED, *,
                    stage=STAGE, excluded_prompts=(), exclusion_provenance=None,
                    tokenizer_fingerprint=None, max_proposals=None):
    """Generate the complete Stage A sample before inference, using IID rejection draws."""
    if not _integer(n) or n < 1 or type(seed) is not int or not isinstance(stage, str) or not stage:
        raise ValueError('n must be positive, seed an integer and stage a nonempty string')
    if len(expected_row_ids()) != 48:
        raise ValueError('the frozen baseline design must contain exactly 48 rows')
    if max_proposals is None:
        max_proposals = max(1000, n*10000)
    if not _integer(max_proposals) or max_proposals < n:
        raise ValueError('max_proposals must be an integer at least n')
    root, data, hashes = Path(source_dir), {}, {}
    for name in ('names', 'objects'):
        path = root/'data'/f'{name}.json'
        content = path.read_bytes()
        values = json.loads(content)
        if not isinstance(values, list) or not values or not all(isinstance(s, str) and s for s in values):
            raise ValueError(f'{name} must contain a nonempty source string list')
        data[name] = values
        hashes[f'data/{name}.json'] = hashlib.sha256(content).hexdigest()
    bos = tokenizer.bos_token_id
    if not _integer(bos):
        raise ValueError('tokenizer must declare a nonnegative integer BOS')
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    heldout_names = data['names'][len(data['names'])//2:]
    name_ids = {name: encode(' '+name) for name in heldout_names}
    names = [name for name in heldout_names if len(name_ids[name]) == 1 and _integer(name_ids[name][0])]
    objects = [item for item in data['objects'][len(data['objects'])//2:] if item in OBJECT_WHITELIST]
    if len(set(names)) < 3 or len({name_ids[name][0] for name in names}) < 3:
        raise ValueError('fewer than three distinct token-eligible held-out names')
    if not objects:
        raise ValueError('held-out source has no object slots in the frozen count-noun whitelist')
    excluded = set(excluded_prompts)
    if not all(isinstance(p, str) and p for p in excluded):
        raise ValueError('excluded prompts must be nonempty strings')
    prefix_ids = encode(PREFIX.rstrip())
    if not prefix_ids or not all(_integer(t) for t in prefix_ids):
        raise ValueError('common prefix must tokenize to finite integer token ids')
    module_dir = Path(__file__).resolve().parent
    definition = {
        'version': VERSION, 'mode': 'native_baseline_only', 'source_files': hashes,
        'code_files_sha256': {name: hashlib.sha256((module_dir/name).read_bytes()).hexdigest()
                              for name in ('role_baseline_sampling.py', 'role_design.py', 'donor_factor_sampling.py')},
        'axes': {'names': names, 'objects': objects}, 'object_whitelist': list(OBJECT_WHITELIST),
        'heldout_split': 'each source list from floor(len/2) onward; retain repeated source slots',
        'tokenizer': _fingerprint(tokenizer, tokenizer_fingerprint), 'bos_token_id': bos,
        'common_prefix': PREFIX, 'common_prefix_stable_token_ids_with_bos': [bos]+prefix_ids,
        'contexts': copy.deepcopy(CONTEXTS), 'queries': list(role_design.ROLES),
        'wordings': list(role_design.WORDINGS), 'row_ids': expected_row_ids(),
        'eligibility': copy.deepcopy(ELIGIBILITY), 'excluded_prompts': sorted(excluded),
        'proposal': 'uniform three ordered distinct source name slots and one uniform object slot, '
                    'with replacement across proposals; reject equal names/tokens and deterministic token failures',
        'proposal_tuple_count': len(names)*(len(names)-1)*(len(names)-2)*len(objects),
        'eligible_population_count': None,
        'eligible_population_count_reason': 'Factorwise eligibility definition, not an enumerated population.',
    }
    population_hash = _hash(definition)
    rng, draws, rejected = random.Random(seed), [], Counter()
    for proposal_index in range(max_proposals):
        selected, item = rng.sample(names, 3), rng.choice(objects)
        answers = [name_ids[name][0] for name in selected]
        if len(set(selected)) != 3 or len(set(answers)) != 3:
            rejected['indistinct_names_or_answer_tokens'] += 1
            continue
        rows, failure = [], None
        for wording in role_design.WORDINGS:
            for context_id, context in CONTEXTS.items():
                for query in role_design.ROLES:
                    prompt = role_design.render(context, query, names=selected, item=item, wording=wording)
                    if prompt in excluded:
                        failure = 'excluded_prompt'
                        break
                    if not prompt.startswith(PREFIX):
                        raise ValueError('role_design.render no longer supplies the declared common prefix')
                    spans = _name_spans(prompt, selected)
                    if spans is None or [i for _, _, i in spans] != list(role_design.mentions(context)):
                        failure = 'name_mentions'
                        break
                    bare = encode(prompt)
                    if not bare or not all(_integer(t) for t in bare) or bare[:len(prefix_ids)] != prefix_ids:
                        failure = 'invalid_tokens_or_prefix'
                        break
                    positions = _mention_tokens(prompt, spans, bare, encode, answers)
                    if positions is None:
                        failure = 'name_token_alignment'
                        break
                    target = role_design.answer(context, query)
                    rows.append({'row_id': f'{wording}/{context_id}/{query}', 'context_id': context_id,
                                 'context': context_id, 'form': context['form'], 'wording': wording, 'query': query,
                                 'correct_name_index': target, 'correct_answer_token_id': answers[target],
                                 'prompt': prompt, 'token_ids': [bos]+bare, 'position': len(bare),
                                 'name_token_positions': positions,
                                 'name_char_spans': [[a, b] for a, b, _ in spans],
                                 'first_mention_name_indices': [i for _, _, i in spans]})
                if failure:
                    break
            if failure:
                break
        if failure:
            rejected[failure] += 1
            continue
        case = {'names': selected, 'answer_token_ids': answers, 'item': item, 'bos_token_id': bos,
                'rows': rows, 'prompts': [row['prompt'] for row in rows]}
        try:
            validate_family(case)
        except ValueError as exc:
            if 'query lengths' in str(exc):
                rejected['unequal_query_lengths'] += 1
            elif 'collision' in str(exc):
                rejected['donor_collision_alignment'] += 1
            elif 'first-mention positions differ across queries' in str(exc):
                rejected['query_name_position_alignment'] += 1
            else:
                raise
            continue
        index = len(draws)
        unique = _hash([VERSION, population_hash, seed, stage, index])[:24]
        case.update(case_id=unique, unique_draw_id=unique, content_family_id=_hash(case['prompts'])[:16],
                    draw=index, draw_index=index, proposal_index=proposal_index)
        draws.append(case)
        if len(draws) == n:
            counts = Counter(c['content_family_id'] for c in draws)
            return draws, {
                'sampling_version': VERSION, 'stage': stage, 'seed': seed, 'mode': 'native_baseline_only',
                'n_family_draws': n, 'rows_per_family': 48, 'n_prompt_rows': 48*n,
                'sampling_definition': definition, 'population_definition_sha256': population_hash,
                'definition_hash_scope': 'Eligible population definition; not an enumerated population hash.',
                'draw_cases_sha256': _hash(draws), 'exclusion_provenance': exclusion_provenance,
                'proposal_count': proposal_index+1, 'max_proposals': max_proposals,
                'rejection_counts': dict(rejected), 'unique_content_families': len(counts),
                'repeated_content_draws': n-len(counts),
                'content_multiplicities_above_one': {k: v for k, v in sorted(counts.items()) if v > 1},
                'independent_unit': 'unique_draw_id / case_id; all 48 native rows are components of one family',
                'inference_order': 'All accepted families, eligibility decisions and exclusions precede inference.',
                'outcome_filter': 'none; no model output is consulted or used to retain/replace a family',
            }
    raise ValueError('fixed proposal budget exhausted; no complete sample produced; '
                     'do not extend sampling based on later native accuracy or patch outcomes')
