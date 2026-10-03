"""Standard-library audit and analysis of saved screen-transfer records.

No model, runner or generator is imported. Tensor arithmetic is reconstructed
from the archived snapshots; statistical arithmetic uses the declared planning
calculator. This is not an independent neural rerun.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import plan_analysis as stats

_spec = importlib.util.spec_from_file_location('transfer_snapshot_primitives', HERE.parent/'verify.py')
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
require, number, close, digest = old.require, old.number, old.close, old.digest
ANCHORS = ('neg_20_0', 'pos_28_0')
DTYPES = ('float32', 'float64')
ARRAYS = ('v_r', 'v_d', 'h_r', 'h_patch_intended', 'h_patch_delivered',
          'requested_node_delta', 'delivered_node_delta')
GAP, NUMERICAL, N, B = .202, .001, 64, 1024


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: '+key)
        result[key] = value
    return result


def load_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_object)


def read_jsonl(path):
    rows = [json.loads(line, object_pairs_hook=_object) for line in Path(path).read_text().splitlines()]
    require(all(isinstance(row, dict) for row in rows), 'Expected JSON object records')
    return rows


def head_key(head):
    return head['model']+'_head'+str(head['head_one_based'])


def safe_path(base, name):
    base = Path(base).resolve()
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Artifact path must be relative')
    path = (base/name).resolve()
    require(base in path.parents, 'Artifact path escapes its bound directory')
    return path


def check_hashes(base, hashes, required=()):
    require(isinstance(hashes, dict) and set(required) <= set(hashes), 'Missing required artifact hash')
    for name, expected in hashes.items():
        require(isinstance(expected, str) and len(expected) == 64,
                'Invalid recorded SHA256: '+name)
        require(digest(safe_path(base, name)) == expected, 'Artifact changed: '+name)


def check_normalization(record, reference_cutoff):
    """Check saved weight algebra; checkpoint provenance is verified separately."""
    vectors = {}
    for key in ('w_false', 'w_true', 'ln_gamma', 'ln_beta'):
        require(isinstance(record[key], list) and len(record[key]) == 64,
                'Normalization needs four saved 64-dimensional vectors')
        vectors[key] = [number(v) for v in record[key]]
    w = [a-b for a, b in zip(vectors['w_false'], vectors['w_true'])]
    center = math.fsum(a*b for a, b in zip(w, vectors['ln_beta']))
    weighted = [a*b for a, b in zip(w, vectors['ln_gamma'])]
    mean = math.fsum(weighted)/64
    radius = 8*math.sqrt(math.fsum((v-mean)**2 for v in weighted))
    require(radius > 0 and math.isfinite(radius), 'Normalization radius must be finite and positive')
    close(record['center'], center, 'Normalization center mismatch')
    close(record['radius'], radius, 'Normalization radius mismatch')
    close(record['normalized_cutoff'], reference_cutoff, 'Normalized cutoff changed', atol=1e-15, rtol=0)
    cutoff = center+reference_cutoff*radius
    close(record['raw_cutoff'], cutoff, 'Derived raw cutoff mismatch')
    return {'center': center, 'radius': radius, 'normalized_cutoff': reference_cutoff,
            'raw_cutoff': cutoff}


def check_donors(template, exclusions):
    require(template['phase'] == 'development', 'Unexpected donor phase')
    require(isinstance(template['donors'], list) and len(template['donors']) == 8,
            'Expected eight prepared donor cells')
    donors, within_cell = {}, {}
    for donor in template['donors']:
        cell = donor['cell']
        require(cell in old.CELLS and cell not in donors, 'Duplicate or extra donor cell')
        sign, pos, rep = cell.split('_')
        pos, rep = int(pos), int(rep)
        depth = -2 if sign == 'neg' else 2
        require((donor['balance'], donor['position'], donor['replica']) == (depth, pos, rep),
                'Donor metadata differs from cell name')
        text = donor['string']
        old.shape(text)
        prefix = text[:pos]
        require(text[pos-1] == ')' and old.walk(prefix) == (depth, -4), 'Donor semantic constraints failed')
        require(old.phase(prefix) == 'development' and old.text_hash(prefix) == donor['prefix_sha256'],
                'Donor prefix phase/hash mismatch')
        require(text not in exclusions['recipient_strings']
                and prefix not in exclusions['prefixes'][str(pos)], 'Previously exposed donor value')
        key = depth, pos
        require(prefix not in within_cell.setdefault(key, set()), 'Within-cell donor prefixes must differ')
        within_cell[key].add(prefix)
        donors[cell] = donor
    require(set(donors) == set(old.CELLS), 'Incomplete prepared donor grid')
    return donors


def check_screen(candidates, screening, exclusions, normalization, *, allow_shortfall=False):
    require(len(candidates) == len(screening) == B, 'Every head must preserve all 1024 native candidates')
    require(len({c['candidate_id'] for c in candidates}) == B, 'Candidate identifiers must be unique')
    indices = {'accepted': [], 'rejected': []}
    maximum = 0.
    c, radius, zcut = (normalization[k] for k in ('center', 'radius', 'normalized_cutoff'))
    require(number(radius) > 0, 'Invalid screen radius')
    for i, (candidate, row) in enumerate(zip(candidates, screening)):
        require(candidate['candidate_index'] == i, 'Prepared candidate ordering changed')
        for key in ('candidate_id', 'candidate_index', 'recipient'):
            require(row[key] == candidate[key], 'Native output/input binding differs')
        text = candidate['recipient']
        old.shape(text)
        require(text not in exclusions['recipient_strings'] and old.phase(text) == 'development'
                and old.walk(text)[1] < 0, 'Recipient outside the fixed fresh population')
        margins = {tag: number(row['margin_'+tag]) for tag in DTYPES}
        for tag in DTYPES:
            if 'normalized_margin_'+tag in row:
                close(row['normalized_margin_'+tag], (margins[tag]-c)/radius,
                      'Recorded normalized native margin differs')
        labels = {tag: (v-c)/radius < zcut for tag, v in margins.items()}
        maximum = max(maximum, abs(margins['float32']-margins['float64']))
        require(abs(margins['float32']-margins['float64']) <= NUMERICAL
                and labels['float32'] == labels['float64'], 'Native precision/classification gate failed')
        for tag in DTYPES:
            require(row['screen_accept_'+tag] is labels[tag], 'Incorrect saved normalized screen label')
        stratum = 'accepted' if labels['float64'] else 'rejected'
        require(row['stratum'] == stratum, 'Screen stratum changed')
        indices[stratum].append(i)
    counts = {s: len(v) for s, v in indices.items()}
    shortfall = any(v < N for v in counts.values())
    require(allow_shortfall or not shortfall, 'Insufficient native screening yield')
    chosen = {s: v[:N] for s, v in indices.items()}
    selected_set = set(chosen['accepted']+chosen['rejected'])
    for i, row in enumerate(screening):
        require(row['selected'] is (i in selected_set), 'Selection is not first64 per frozen stratum')
    return chosen, counts, maximum, shortfall


def check_selection(selection):
    attention = selection['native_eos_attention']
    require(isinstance(attention, list) and len(attention) == 42, 'Expected complete 42-position attention row')
    attention = [number(v) for v in attention]
    require(min(attention) >= 0 and all(v == 0 for v in attention[34:]), 'Invalid attention support')
    close(math.fsum(attention), 1., 'Attention row is not normalized', atol=1e-12, rtol=0)
    closing = [i+1 for i, token in enumerate(selection['recipient']) if token == ')']
    position = max(closing, key=lambda i: attention[i])
    require(type(selection['recipient_position']) is int and selection['recipient_position'] == position,
            'Recipient site is not first maximum-attention closing bracket')
    return position, attention


def check_snapshot(snapshot, dtype, recipient, position, donor, head_width):
    require(head_width in (16, 32), 'Head width must derive from the audited model head count')
    require(snapshot['dtype'] == 'torch.'+dtype, 'Snapshot dtype mismatch')
    require(snapshot['recipient_string'] == recipient and snapshot['donor_string'] == donor['string'],
            'Snapshot strings differ from prepared sources')
    require(snapshot['recipient_position'] == position and snapshot['donor_position'] == donor['position'],
            'Snapshot positions differ from sources')
    require(recipient[position-1] == ')', 'Recipient target token changed')
    arrays = {}
    for key in ARRAYS:
        require(isinstance(snapshot[key], list) and len(snapshot[key]) == head_width,
                'Snapshot dimension differs from actual head width: '+key)
        arrays[key] = [number(x) for x in snapshot[key]]
    cast = old.float32 if dtype == 'float32' else float
    a = number(snapshot['a_r'])
    require(0 <= a <= 1, 'Attention coefficient is not a probability')
    for key in ('native_margin', 'patched_margin', 'donor_native_margin', 'margin_change'):
        number(snapshot[key])
    if dtype == 'float32':
        require(cast(a) == a and all(cast(x) == x for values in arrays.values() for x in values),
                'Float32 snapshot includes nonrepresentable tensor values')
        require(all(cast(snapshot[key]) == snapshot[key] for key in
                    ('native_margin', 'patched_margin', 'donor_native_margin', 'margin_change')),
                'Float32 score snapshot contains non-float32 values')
    diff = [cast(d-r) for d, r in zip(arrays['v_d'], arrays['v_r'])]
    requested = [cast(a*v) for v in diff]
    intended = [cast(h+delta) for h, delta in zip(arrays['h_r'], requested)]
    delivered = [cast(h-before) for h, before in zip(intended, arrays['h_r'])]
    require(arrays['requested_node_delta'] == requested, 'Requested contribution formula mismatch')
    require(arrays['h_patch_intended'] == intended, 'Intended post-dtype formula mismatch')
    require(arrays['h_patch_delivered'] == intended, 'Delivered node differs from intended node')
    require(arrays['delivered_node_delta'] == delivered, 'Delivered node delta mismatch')
    reference = [h+a*(d-r) for h, d, r in zip(arrays['h_r'], arrays['v_d'], arrays['v_r'])]
    roundoff = max(abs(x-y) for x, y in zip(intended, reference))
    close(snapshot['construction_roundoff_linf'], roundoff, 'Recorded construction roundoff differs')
    tolerance = 2e-6 if dtype == 'float32' else 1e-11
    close(snapshot['value_difference_l2'], math.sqrt(math.fsum(v*v for v in diff)),
          'Value-difference norm differs', atol=tolerance, rtol=tolerance)
    close(snapshot['node_delta_l2'], math.sqrt(math.fsum(v*v for v in delivered)),
          'Delivered-delta norm differs', atol=tolerance, rtol=tolerance)
    require(snapshot['margin_change'] == cast(snapshot['patched_margin']-snapshot['native_margin']),
            'Saved margin change disagrees with execution dtype')
    return roundoff


def check_completed_records(head, candidates, templates, screening, families, selections, rows,
                            exclusions, normalization):
    require(stats.N == N, 'Frozen calculator is not the64-per-stratum amendment')
    require(head['model_heads'] in (2, 4) and 64 % head['model_heads'] == 0, 'Unsupported model head count')
    width = 64//head['model_heads']
    require(len(templates) == len(families) == len(selections) == len(rows) == 2*N,
            'Completed head requires128 donor templates and output families')
    chosen, counts, native_error, _ = check_screen(candidates, screening, exclusions, normalization)
    order = [(s, i) for s in ('accepted', 'rejected') for i in chosen[s]]
    hits = {'accepted': 0, 'rejected': 0}
    family_results, prefix_values = [], {}
    maximum_gap_error = 0.
    roundoff = {tag: 0. for tag in DTYPES}
    for k, ((stratum, index), template, family, selection, row) in enumerate(zip(order, templates, families, selections, rows)):
        candidate, observed = candidates[index], screening[index]
        require(template['family_id'] == old.text_hash(f"development:{head['donor_seed']}:{k}")[:20],
                'Donor template identity/order changed')
        donors = check_donors(template, exclusions)
        for item in (family, row):
            require(item['family_id'] == item['candidate_id'] == candidate['candidate_id']
                    and item['candidate_index'] == index and item['stratum'] == stratum
                    and item['recipient'] == candidate['recipient']
                    and item['donor_template_id'] == template['family_id'], 'Selected family/template binding differs')
            for tag in DTYPES:
                require(item['margin_'+tag] == observed['margin_'+tag], 'Selected native margin changed')
        require(family['donors'] == template['donors'], 'Assigned donor grid changed')
        require(selection['family_id'] == candidate['candidate_id'] and selection['recipient'] == candidate['recipient'],
                'Recipient selection belongs to another family')
        position, attention = check_selection(selection)
        require(all(item['recipient_position'] == position for item in (family, row, observed)),
                'Recipient intervention position changed')
        close(selection['native_margin'], observed['margin_float64'], 'Selection baseline differs')
        require(set(row['cells']) == set(ANCHORS), 'Only the two frozen anchor cells may be measured')
        baseline = {}
        for name in ANCHORS:
            cell, donor = row['cells'][name], donors[name]
            require(set(cell) == {'donor_metadata', *DTYPES}, 'Missing or extra cell/dtype fields')
            require(cell['donor_metadata'] == donor, 'Measured donor metadata differs from prepared grid')
            for tag in DTYPES:
                snapshot = cell[tag]
                roundoff[tag] = max(roundoff[tag], check_snapshot(snapshot, tag, family['recipient'], position, donor, width))
                state = {key: snapshot[key] for key in ('a_r', 'v_r', 'h_r', 'native_margin')}
                require(baseline.setdefault(tag, state) == state, 'Recipient state changed within family')
                close(snapshot['native_margin'], observed['margin_'+tag], 'Transfer baseline differs from screen',
                      atol=1e-5 if tag == 'float32' else 1e-10, rtol=0)
                if tag == 'float64':
                    require(snapshot['a_r'] == attention[position], 'Transferred attention weight changed')
                key = tag, donor['position'], donor['prefix_sha256']
                require(prefix_values.setdefault(key, snapshot['v_d']) == snapshot['v_d'],
                        'Repeated donor prefix has inconsistent value vector')
        contrasts = {tag: number(row['cells'][ANCHORS[1]][tag]['patched_margin'])-
                           number(row['cells'][ANCHORS[0]][tag]['patched_margin']) for tag in DTYPES}
        discrepancy = abs(contrasts['float32']-contrasts['float64'])
        maximum_gap_error = max(maximum_gap_error, discrepancy)
        require(discrepancy <= NUMERICAL and (abs(contrasts['float32']) > GAP) == (abs(contrasts['float64']) > GAP),
                'Anchor precision/separation-boundary gate failed')
        separates = abs(contrasts['float64']) > GAP
        for tag in DTYPES:
            require(row['anchor_gap_'+tag] == abs(contrasts[tag]), 'Saved anchor gap differs')
        require(row['anchor_separating'] is separates, 'Saved separation label differs')
        hits[stratum] += separates
        family_results.append({'family_id': row['family_id'], 'stratum': stratum,
                               'anchor_gap': abs(contrasts['float64']), 'separating': separates})
    return {'head_key': head_key(head), 'status': 'completed', 'head_width': width,
            'pool_counts': counts, 'statistics': stats.head_result(hits['accepted'], hits['rejected']),
            'cost_estimates': stats.cost_estimates(counts['accepted'], hits['accepted'], hits['rejected']),
            'family_results': family_results,
            'verification': {'native_candidates': B, 'families': 2*N, 'snapshots': 8*N,
                             'max_native_dtype_difference': native_error,
                             'max_signed_anchor_dtype_difference': maximum_gap_error,
                             'maximum_construction_roundoff': roundoff}}


def check_controls(controls, result, rows):
    old.compare(controls['native_screen'], {
        'max_dtype_margin_difference': result['verification']['max_native_dtype_difference'],
        'dtype_classification_mismatches': 0,
        'selected_counts': {'accepted': N, 'rejected': N}, 'pool_counts': result['pool_counts']})
    old.compare(controls['anchor_precision'], {
        'maximum_signed_contrast_dtype_difference': result['verification']['max_signed_anchor_dtype_difference'],
        'separation_boundary_straddles': 0})
    self_cases = sum(row['recipient'] == cell['donor_metadata']['string'] and
                     row['recipient_position'] == cell['donor_metadata']['position']
                     for row in rows for cell in row['cells'].values())
    for tag in DTYPES:
        control = controls['anchors'][tag]
        tolerance = 1e-5 if tag == 'float32' else 1e-10
        require(control['identity_tolerance'] == tolerance, 'Identity tolerance changed')
        for key in ('identity_max_margin_error', 'identity_max_node_error', 'self_same_position_max_margin_error'):
            require(0 <= number(control[key]) <= tolerance, 'Recorded identity control failed')
        require(control['self_same_position_cases'] == self_cases, 'Self-patch control count differs')
        for key in ('inserted_node_exactly_intended_after_dtype', 'all_target_layer_attention_weights_unchanged',
                    'all_target_layer_value_projections_unchanged', 'nontarget_head_and_query_preprojection_exactly_unchanged'):
            require(control[key] is True, 'Recorded intervention control failed: '+key)


def cohort_report(plan, head_results):
    expected = [head_key(h) for h in plan['cohort']]
    require(len(expected) == 6 and len(set(expected)) == 6, 'Plan must retain six distinct heads')
    require(len(head_results) == 6 and [h['head_key'] for h in head_results] == expected,
            'All six fixed heads must appear exactly once in frozen order')
    counts = []
    for result in head_results:
        require(result['status'] in ('completed', 'insufficient_yield', 'technical_failure'), 'Unknown head status')
        if result['status'] == 'completed':
            decision = result['statistics']
            recomputed = stats.head_result(decision['accepted_hits'], decision['rejected_hits'])
            old.compare(decision, recomputed)
            counts.append((decision['accepted_hits'], decision['rejected_hits']))
        else:
            require('statistics' not in result and 'cost_estimates' not in result,
                    'Unavailable head must not receive imputed scientific statistics or policy ratios')
            counts.append(None)
    decision = stats.cohort_result(counts)
    return {'status': 'saved_record_audit_completed', 'heads': head_results,
            'cohort_decision': decision,
            'scope': 'Fixed selected six-head cohort; anchor separability, not mechanism identification.',
            'limits': ['Unavailable heads remain in the six; no replacement or zero-effect imputation.',
                       'Statistical bounds and numerical gates are different quantities.',
                       'Cost ratios and random-selection yields are descriptive plug-in estimates.',
                       'No model forward is repeated by this audit.',
                       'Full-layer unchangedness is producer-reported; only target snapshots are archived.',
                       'Local hashes/timestamps do not establish external preregistration.',
                       'Saved readout vectors can verify normalization algebra, not independently extract checkpoint weights.']}


def check_exclusions(exclusions, root):
    """Rebuild the frozen union from committed input records, without raw caches."""
    app = Path(root)/'applications/li-saphra-2507.06445'
    base_path = app/'value_followup/screen_002/inputs/exclusions.json'
    check_hashes(root, exclusions['source_hashes'])
    base = load_json(base_path)
    texts = set(base['recipient_strings'])
    paths = [app/'value_followup/inputs/development_001/cases.jsonl',
             app/'value_followup/screen_002/inputs/candidates.jsonl',
             app/'value_followup/screen_002/inputs/donor_families.jsonl']
    for path in paths:
        for row in read_jsonl(path):
            texts.add(row['recipient'])
            texts.update(d['string'] for d in row.get('donors', []))
    require(set(exclusions['recipient_strings']) == texts, 'Historical full-string exclusions changed')
    require(len(exclusions['recipient_strings']) == len(texts), 'Duplicate full-string exclusions')
    for position in (20, 28):
        required = {text[:position] for text in texts if len(text) >= position}
        require(set(exclusions['prefixes'][str(position)]) == required,
                'Historical cross-position/role prefix exclusions changed')
    # The union sources must be explicit; originals outside this committed
    # snapshot need not be available to this saved-record analysis.
    for path in [base_path]+paths:
        relative = str(path.relative_to(root))
        require(exclusions['source_hashes'].get(relative) == digest(path),
                'Exclusion union source is not bound: '+relative)


def check_input_inventory(inputs, preparation, plan, root):
    recipe = {'round': plan['round'], 'phase': 'development', 'pool_size': B, 'per_stratum': N,
              'heads': [{k: head[k] for k in ('model', 'head_one_based', 'recipient_seed', 'donor_seed')}
                        for head in plan['cohort']]}
    require(preparation['recipe'] == recipe, 'Prepared sampling recipe changed')
    required = {'exclusions.json'} | {head_key(head)+'/'+name for head in plan['cohort']
                                    for name in ('candidates.jsonl', 'donor_families.jsonl', 'preparation.json')}
    require(set(preparation['files']) == required, 'Missing or extra prepared input file binding')
    check_hashes(inputs, preparation['files'])
    check_hashes(root, preparation['sources'])
    require(set(preparation['heads']) == {head_key(h) for h in plan['cohort']}, 'Prepared head inventory changed')
    exclusions = load_json(Path(inputs)/'exclusions.json')
    check_exclusions(exclusions, root)
    records = {}
    for head in plan['cohort']:
        key = head_key(head)
        directory = Path(inputs)/key
        info = load_json(directory/'preparation.json')
        require(info == preparation['heads'][key], 'Per-head preparation differs from global manifest')
        require((info['head_key'], info['recipient_seed'], info['donor_seed'],
                 info['candidate_pool_size'], info['donor_templates']) ==
                (key, head['recipient_seed'], head['donor_seed'], B, 2*N), 'Prepared head settings changed')
        candidates, templates = read_jsonl(directory/'candidates.jsonl'), read_jsonl(directory/'donor_families.jsonl')
        require(len(candidates) == B and len(templates) == 2*N, 'Incorrect prepared population size')
        for i, candidate in enumerate(candidates):
            require(candidate['candidate_id'] == old.text_hash(f"screen-transfer-003:{head['recipient_seed']}:{i}")[:20]
                    and candidate['candidate_index'] == i, 'Prepared candidate identifier/order changed')
        prefixes = Counter()
        for i, template in enumerate(templates):
            require(template['family_id'] == old.text_hash(f"development:{head['donor_seed']}:{i}")[:20],
                    'Prepared donor identifier/order changed')
            donors = check_donors(template, exclusions)
            prefixes.update(d['prefix_sha256'] for d in donors.values())
        recipients = Counter(c['recipient'] for c in candidates)
        old.compare(info, {'unique_candidate_strings': len(recipients), 'maximum_candidate_reuse': max(recipients.values()),
                          'donor_prefix_draws': sum(prefixes.values()), 'unique_donor_prefixes': len(prefixes),
                          'maximum_donor_prefix_reuse': max(prefixes.values())})
        records[key] = candidates, templates
    return exclusions, records


def _time(text):
    result = datetime.fromisoformat(text)
    require(result.tzinfo is not None, 'Local provenance timestamps need time zones')
    return result


def check_cost_ledger(status, events):
    ledger = status['cost']
    count = {name: 0 for name in ('sequence_forwards_attempted', 'sequence_forwards_completed',
                                'forward_batches_attempted', 'forward_batches_completed')}
    by_stage = {stage: {tag: 0 for tag in DTYPES} for stage in ('native', 'anchors')}
    pending = {}
    previous = _time(status['started_at'])
    for event in events:
        tag, stage, kind, size = (event[k] for k in ('dtype', 'stage', 'event', 'sequences'))
        require(tag in DTYPES and stage in by_stage and kind in ('started', 'completed'), 'Unknown forward ledger event')
        require(type(size) is int and 0 < size <= 64, 'Unexpected forward batch size')
        timestamp = _time(event['at'])
        require(previous <= timestamp <= _time(status['finished_at']), 'Forward events are not locally chronological')
        previous = timestamp
        if kind == 'started':
            require(tag not in pending, 'A dtype starts another forward before its previous call returns')
            pending[tag] = stage, size
            count['sequence_forwards_attempted'] += size
            count['forward_batches_attempted'] += 1
        else:
            require(pending.pop(tag, None) == (stage, size), 'Completed forward has no matching started call')
            count['sequence_forwards_completed'] += size
            count['forward_batches_completed'] += 1
            by_stage[stage][tag] += size
    old.compare(ledger, count)
    require(ledger['failed_forward_internal_progress_unknown'] is bool(pending), 'Incorrect unfinished-forward cost marker')
    require(count['sequence_forwards_attempted'] <= B*2+2*N*16, 'Forward budget exceeded')
    for field, maximum in (('baseline_candidates_completed_by_dtype', B), ('anchor_jobs_completed_by_dtype', 4*N)):
        require(set(ledger[field]) == set(DTYPES), 'Incomplete per-dtype completion ledger')
        for tag, completed in ledger[field].items():
            require(type(completed) is int and completed in (0, maximum), 'Unexpected returned-stage job count')
    if status['status'] in ('completed', 'insufficient_yield'):
        require(not pending, 'Completed scientific stage has a pending model forward')
        require(by_stage['native'] == {tag: B for tag in DTYPES}, 'Native forward cost differs from complete pool')
        require(ledger['baseline_candidates_completed_by_dtype'] == {tag: B for tag in DTYPES},
                'Not all native candidates returned')
        anchor_jobs = 4*N if status['status'] == 'completed' else 0
        require(ledger['anchor_jobs_completed_by_dtype'] == {tag: anchor_jobs for tag in DTYPES},
                'Incorrect completed anchor-job count')
        require(by_stage['anchors'] == {tag: 4*anchor_jobs for tag in DTYPES}, 'Anchor forward cost differs from operator')
    return {**count, 'failed_forward_internal_progress_unknown': bool(pending),
            'completed_sequence_forwards_by_stage_dtype': by_stage}


def audit(run, inputs, *, root=ROOT, plan_path=None, release_path=None):
    run, inputs, root = Path(run), Path(inputs), Path(root).resolve()
    relative = Path('applications/li-saphra-2507.06445/value_followup/screen_transfer_003')
    plan_path = Path(plan_path) if plan_path else root/relative/'plan.json'
    release_path = Path(release_path) if release_path else root/relative/'EXECUTION_RELEASE.json'
    plan, release = load_json(plan_path), load_json(release_path)
    require(stats.N == N and stats.HEADS == 6 and stats.TAIL_ALPHA == .05/24,
            'Analysis calculator differs from the64-family six-head contract')
    require(plan['round'] == 'screen_transfer_003' and plan['screen']['native_candidates_per_head'] == B
            and plan['screen']['families_per_stratum'] == N and plan['inference']['families_per_stratum'] == N,
            'Wrong experiment or family count')
    require(plan['intervention']['anchors'] == list(ANCHORS)
            and plan['intervention']['strict_gap_nat'] == GAP
            and plan['intervention']['numerical_allowance_nat'] == NUMERICAL
            and plan['intervention']['secondary_target_transfers'] is False,
            'Frozen intervention or numerical/scientific limits changed')
    require(plan['execution_authorized'] is True and release['execution_authorized'] is True
            and release['final_review']['status'] == 'approved', 'Execution release was not approved')
    manifest = load_json(run/'manifest.json')
    require(manifest['round'] == plan['round'] and manifest['status'] in
            ('completed', 'completed_with_unavailable_heads'), 'Incomplete/shared-failure cohort has no scientific report')
    require(manifest['plan_sha256'] == digest(plan_path)
            and manifest['execution_release_sha256'] == digest(release_path), 'Run plan or release binding differs')
    required_sources = {str(relative/name) for name in ('plan.json', 'PROTOCOL.md', 'EXECUTION_RELEASE.json',
                        'run_transfer.py', 'prepare_transfer.py', 'analyze_transfer.py', 'plan_analysis.py')}
    required_sources |= {str(relative.parent/name) for name in ('verify.py', 'design.py', 'value_runtime.py')}
    check_hashes(root, manifest['source_hashes'], required_sources)
    require(manifest['reviewed_commit'] == release['final_review']['reviewed_commit'], 'Review provenance changed')
    require(manifest['environment']['device'] == 'cpu' and manifest['environment']['threads'] == 1,
            'Frozen execution environment changed')
    preparation = load_json(inputs/'preparation.json')
    require(manifest['preparation'] == preparation, 'Run input manifest differs from supplied inputs')
    exclusions, payloads = check_input_inventory(inputs, preparation, plan, root)
    for name, expected_hash in preparation['sources'].items():
        require(manifest['source_hashes'].get(name) == expected_hash, 'Preparation source absent from run bindings')
    for name in ['preparation.json']+list(preparation['files']):
        path = safe_path(inputs, name)
        require(root in path.parents and manifest['source_hashes'].get(str(path.relative_to(root))) == digest(path),
                'Prepared input absent from run source binding: '+name)
    require(manifest['normalization_sha256'] == digest(run/'normalization.json'), 'Normalization artifact changed')
    norms = load_json(run/'normalization.json')
    head_order = [head_key(h) for h in plan['cohort']]
    require(len(head_order) == len(set(head_order)) == 6 and set(norms['heads']) == set(head_order),
            'Normalization has a missing or extra head')
    require(len(manifest['heads']) == 6 and [h['head_key'] for h in manifest['heads']] == head_order,
            'Manifest must retain exactly the fixed six heads in order')
    zcut = number(plan['reference']['normalized_cutoff'])
    reference = check_normalization(norms['reference'], zcut)
    for key in ('center', 'radius'):
        close(reference[key], plan['reference'][key], 'Reference normalization differs from frozen plan')
    require(norms['reference']['model_id'] == plan['reference']['model']
            and norms['reference']['checkpoint_sha256'] == plan['reference']['checkpoint_sha256'],
            'Reference checkpoint identity differs')
    start, normalized_at, all_screened, finished = map(_time, [manifest['started_at'],
        norms['written_before_native_at'], manifest['all_screening_completed_at'], manifest['finished_at']])
    require(start <= normalized_at <= all_screened <= finished, 'Global local chronology is invalid')
    results = []
    for head, entry in zip(plan['cohort'], manifest['heads']):
        key = head_key(head)
        directory = run/'heads'/key
        require(entry['status_sha256'] == digest(directory/'status.json'), 'Head status changed: '+key)
        status = load_json(directory/'status.json')
        require(status['head_key'] == key and status['status'] == entry['status']
                and status['status'] in ('completed', 'insufficient_yield', 'technical_failure'), 'Unknown/inconsistent head status')
        require(status['task'] == {'model_id': head['model'], 'head': head['head_one_based'],
                                   'model_heads': head['model_heads'], 'n_layer': head['layer_one_based']},
                'Measured task differs from frozen head')
        require(status['source_hashes'] == manifest['source_hashes'], 'Head source bindings differ')
        require(set(status['output_hashes']) == {p.name for p in directory.iterdir() if p.is_file() and p.name != 'status.json'},
                'Unbound or missing head output file')
        check_hashes(directory, status['output_hashes'])
        require(normalized_at <= _time(status['started_at']) <= _time(status['finished_at']) <= finished,
                'Head timing precedes normalization or exceeds run')
        require(status['normalization'] == norms['heads'][key], 'Head normalization changed after screening')
        norm_record = norms['heads'][key]
        require(norm_record['model_id'] == head['model'] and norm_record['checkpoint_sha256'] == head['sha256'],
                'Normalization checkpoint binding differs')
        normalization = check_normalization(norm_record, zcut)
        events = read_jsonl(directory/'forward_events.jsonl') if (directory/'forward_events.jsonl').exists() else []
        cost = check_cost_ledger(status, events)
        if status['status'] == 'technical_failure':
            require(isinstance(status.get('error'), str) and bool(status['error'])
                    and isinstance(status.get('error_type'), str) and bool(status['error_type'])
                    and 'failure.txt' in status['output_hashes'], 'Technical failure lacks a preserved reason')
            results.append({'head_key': key, 'status': 'technical_failure', 'error': status['error'],
                            'recorded_cost': cost, 'science_status': 'not_estimable'})
            continue
        required_files = {'screening.jsonl', 'controls.json', 'forward_events.jsonl'}
        if status['status'] == 'completed':
            required_files |= {'selected_families.jsonl', 'recipient_selection.json', 'screening_receipt.json', 'cases.jsonl'}
        require(set(status['output_hashes']) == required_files, 'Missing or extra scientific output artifact')
        candidates, templates = payloads[key]
        screening = read_jsonl(directory/'screening.jsonl')
        chosen, counts, native_error, shortfall = check_screen(candidates, screening, exclusions, normalization,
                                                             allow_shortfall=status['status']=='insufficient_yield')
        controls = load_json(directory/'controls.json')
        old.compare(controls['native_screen'], {'max_dtype_margin_difference': native_error,
                    'dtype_classification_mismatches': 0, 'pool_counts': counts,
                    'selected_counts': {s: len(v) for s, v in chosen.items()}})
        if status['status'] == 'insufficient_yield':
            require(shortfall and 'anchors_measurement_started_at' not in status,
                    'Insufficient-yield status is not supported by the frozen screen')
            results.append({'head_key': key, 'status': 'insufficient_yield', 'pool_counts': counts,
                            'acceptance_rate': counts['accepted']/B, 'recorded_cost': cost,
                            'science_status': 'not_estimable'})
            continue
        receipt = load_json(directory/'screening_receipt.json')
        for name in ('screening.jsonl', 'selected_families.jsonl', 'recipient_selection.json'):
            field = name.split('.')[0]+'_sha256'
            require(receipt[field] == digest(directory/name), 'Screen receipt artifact changed')
        require(receipt['source_hashes'] == manifest['source_hashes'] and receipt['normalization'] == norm_record,
                'Screen receipt source/normalization changed')
        committed_at = _time(receipt['written_before_transfers_at'])
        anchor_at = _time(status['anchors_measurement_started_at'])
        require(_time(status['started_at']) <= committed_at <= all_screened < anchor_at <= _time(status['finished_at']),
                'All native screen predictions must precede every anchor measurement')
        for event in events:
            require((_time(event['at']) <= all_screened if event['stage'] == 'native'
                     else _time(event['at']) >= anchor_at), 'Forward event violates prospective measurement order')
        rows = read_jsonl(directory/'cases.jsonl')
        result = check_completed_records(head, candidates, templates, screening,
            read_jsonl(directory/'selected_families.jsonl'), load_json(directory/'recipient_selection.json'), rows,
            exclusions, normalization)
        check_controls(controls, result, rows)
        result['recorded_cost'] = cost
        results.append(result)
    require((manifest['status'] == 'completed') == all(r['status']=='completed' for r in results),
            'Global availability status misstates its six heads')
    report = cohort_report(plan, results)
    report['provenance'] = {'git_head': manifest['git_head'], 'manifest_sha256': digest(run/'manifest.json'),
                            'plan_sha256': digest(plan_path), 'analysis_sha256': digest(__file__),
                            'verified_source_files': len(manifest['source_hashes'])}
    report['recorded_cohort_cost'] = {
        'sequence_forwards_attempted': sum(r['recorded_cost']['sequence_forwards_attempted'] for r in results),
        'sequence_forwards_completed': sum(r['recorded_cost']['sequence_forwards_completed'] for r in results),
        'failed_forward_internal_progress_unknown': any(r['recorded_cost']['failed_forward_internal_progress_unknown'] for r in results)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-report', type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.inputs)
    if args.check_report:
        old.compare(load_json(args.check_report), result)
    encoded = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as handle:
            handle.write(encoded)
    print(encoded)


if __name__ == '__main__':
    main()
