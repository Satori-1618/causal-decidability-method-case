"""Synthetic records only: no checkpoint, model forward or measured outcome."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import analyze_transfer as audit
import design  # Input-only generator, used solely to make valid synthetic fixtures.


def snapshot(recipient, pos, donor, tag, width, margin, gap):
    cast = audit.old.float32 if tag == 'float32' else float
    vd = [float(i == 0) for i in range(width)]
    native = cast(margin)
    patched = cast(native+gap)
    return {'recipient_string': recipient, 'donor_string': donor['string'],
            'recipient_position': pos, 'donor_position': donor['position'],
            'native_margin': native, 'patched_margin': patched, 'donor_native_margin': 0.,
            'margin_change': cast(patched-native), 'dtype': 'torch.'+tag, 'a_r': 1.,
            'v_r': [0.]*width, 'v_d': vd, 'h_r': [0.]*width,
            'requested_node_delta': vd[:], 'h_patch_intended': vd[:],
            'h_patch_delivered': vd[:], 'delivered_node_delta': vd[:],
            'value_difference_l2': 1., 'node_delta_l2': 1., 'construction_roundoff_linf': 0.}


def fixture(model_heads=2):
    exclusions = {'recipient_strings': [], 'prefixes': {'20': [], '28': []}}
    head = {'model': 'synthetic', 'model_heads': model_heads, 'head_one_based': 1,
            'layer_one_based': 2, 'donor_seed': 5678, 'recipient_seed': 1234}
    seed_family = design.generate(head['donor_seed'], 1, 'development', exclusions)[0]
    text = seed_family['recipient']
    pos = next(i+1 for i, c in enumerate(text) if c == ')')
    attention = [0.]*42
    attention[pos] = 1.
    candidates = [{'candidate_id': f'candidate-{i}', 'candidate_index': i,
                   'recipient': text} for i in range(1024)]
    normalization = {'center': 0., 'radius': 10., 'normalized_cutoff': .8, 'raw_cutoff': 8.}
    screening = []
    for i, candidate in enumerate(candidates):
        accepted = i < 512
        screening.append({**candidate, 'margin_float32': 0. if accepted else 9.,
                          'margin_float64': 0. if accepted else 9.,
                          'screen_accept_float32': accepted, 'screen_accept_float64': accepted,
                          'stratum': 'accepted' if accepted else 'rejected',
                          'selected': i < 64 or 512 <= i < 576, 'recipient_position': pos})
    templates, families, selections, rows = [], [], [], []
    for k, index in enumerate(list(range(64))+list(range(512, 576))):
        template = copy.deepcopy(seed_family)
        template['family_id'] = audit.old.text_hash(f'development:{head["donor_seed"]}:{k}')[:20]
        templates.append(template)
        observed = screening[index]
        family = {**copy.deepcopy(template), **{key: observed[key] for key in
                  ('candidate_id', 'candidate_index', 'recipient', 'recipient_position',
                   'stratum', 'margin_float32', 'margin_float64')},
                  'family_id': observed['candidate_id'], 'donor_template_id': template['family_id']}
        families.append(family)
        selections.append({'family_id': family['family_id'], 'recipient': text,
                           'recipient_position': pos, 'native_margin': observed['margin_float64'],
                           'native_eos_attention': attention[:]})
        row = {key: family[key] for key in ('family_id', 'candidate_id', 'candidate_index', 'recipient',
               'recipient_position', 'stratum', 'margin_float32', 'margin_float64', 'donor_template_id')}
        row['cells'] = {}
        gap = .5 if family['stratum'] == 'accepted' else .0625
        for cell in audit.ANCHORS:
            donor = next(d for d in family['donors'] if d['cell'] == cell)
            row['cells'][cell] = {'donor_metadata': copy.deepcopy(donor), **{
                tag: snapshot(text, pos, donor, tag, 64//model_heads, observed['margin_'+tag],
                              gap if cell == audit.ANCHORS[1] else 0.) for tag in audit.DTYPES}}
        row.update({'anchor_gap_float32': gap, 'anchor_gap_float64': gap,
                    'anchor_separating': gap > .202})
        rows.append(row)
    return dict(head=head, candidates=candidates, templates=templates, screening=screening,
                families=families, selections=selections, rows=rows, exclusions=exclusions,
                normalization=normalization)


def directory_fixture(root, model_heads=2):
    """One complete synthetic head and five honest quota shortfalls."""
    root = Path(root)
    relative = Path('applications/li-saphra-2507.06445/value_followup/screen_transfer_003')
    source = root/relative
    source.mkdir(parents=True)
    inputs, run = source/'inputs', root/'synthetic_run'
    inputs.mkdir()
    run.mkdir()
    base = fixture(model_heads)
    stamp = lambda second: f'2026-10-03T00:00:{second:02d}+00:00'
    def js(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False))
    def jl(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in value))
    history = source.parent/'screen_002/inputs'
    js(history/'exclusions.json', base['exclusions'])
    prior = [source.parent/'inputs/development_001/cases.jsonl', history/'candidates.jsonl', history/'donor_families.jsonl']
    for path in prior:
        jl(path, [])
    exclusions = {**base['exclusions'], 'source_hashes': {
        str(path.relative_to(root)): audit.digest(path) for path in [history/'exclusions.json']+prior}}
    js(inputs/'exclusions.json', exclusions)
    norm = {'w_false': [10/math.sqrt(63)]+[0.]*63, 'w_true': [0.]*64,
            'ln_gamma': [1.]*64, 'ln_beta': [0.]*64,
            'center': 0., 'radius': 10., 'normalized_cutoff': .8, 'raw_cutoff': 8., 'checkpoint_sha256': 'a'*64}
    heads = [{**base['head'], 'model': f'synthetic{i}', 'recipient_seed': 1234+i,
              'sha256': 'a'*64} for i in range(6)]
    plan = {'round': 'screen_transfer_003', 'execution_authorized': True, 'cohort': heads,
            'screen': {'native_candidates_per_head': 1024, 'families_per_stratum': 64},
            'inference': {'families_per_stratum': 64},
            'intervention': {'anchors': list(audit.ANCHORS), 'strict_gap_nat': .202,
                             'numerical_allowance_nat': .001, 'secondary_target_transfers': False},
            'reference': {'model': 'reference', 'checkpoint_sha256': 'a'*64, 'center': 0.,
                          'radius': 10., 'normalized_cutoff': .8}}
    release = {'execution_authorized': True, 'final_review': {'status': 'approved', 'reviewed_commit': 'synthetic-review'}}
    js(source/'plan.json', plan)
    js(source/'EXECUTION_RELEASE.json', release)
    for name in ('PROTOCOL.md', 'run_transfer.py', 'prepare_transfer.py', 'analyze_transfer.py', 'plan_analysis.py'):
        (source/name).write_text('synthetic bound source '+name)
    for name in ('verify.py', 'design.py', 'value_runtime.py'):
        (source.parent/name).write_text('synthetic bound dependency '+name)
    prep = {'recipe': {'round': plan['round'], 'phase': 'development', 'pool_size': 1024, 'per_stratum': 64,
             'heads': [{k: h[k] for k in ('model', 'head_one_based', 'recipient_seed', 'donor_seed')} for h in heads]},
            'sources': {str(path.relative_to(root)): audit.digest(path) for path in
                        (source/'prepare_transfer.py', source.parent/'design.py')},
            'files': {'exclusions.json': audit.digest(inputs/'exclusions.json')}, 'heads': {}}
    payloads = []
    for i, head in enumerate(heads):
        f = copy.deepcopy(base)
        f['head'] = head
        for index, candidate in enumerate(f['candidates']):
            candidate['candidate_id'] = audit.old.text_hash(f'screen-transfer-003:{head["recipient_seed"]}:{index}')[:20]
            f['screening'][index]['candidate_id'] = candidate['candidate_id']
        for family, selection, row in zip(f['families'], f['selections'], f['rows']):
            identifier = f['candidates'][family['candidate_index']]['candidate_id']
            family['family_id'] = family['candidate_id'] = identifier
            row['family_id'] = row['candidate_id'] = identifier
            selection['family_id'] = identifier
        key = audit.head_key(head)
        directory = inputs/key
        jl(directory/'candidates.jsonl', f['candidates'])
        jl(directory/'donor_families.jsonl', f['templates'])
        info = {'head_key': key, 'recipient_seed': head['recipient_seed'], 'donor_seed': head['donor_seed'],
                'candidate_pool_size': 1024, 'donor_templates': 128, 'unique_candidate_strings': 1,
                'maximum_candidate_reuse': 1024, 'donor_prefix_draws': 1024, 'unique_donor_prefixes': 8,
                'maximum_donor_prefix_reuse': 128}
        js(directory/'preparation.json', info)
        prep['heads'][key] = info
        for name in ('candidates.jsonl', 'donor_families.jsonl', 'preparation.json'):
            prep['files'][key+'/'+name] = audit.digest(directory/name)
        payloads.append(f)
    js(inputs/'preparation.json', prep)
    source_hashes = {str(path.relative_to(root)): audit.digest(path) for path in source.parent.rglob('*') if path.is_file()}
    normalization = {'written_before_native_at': stamp(1), 'reference': {**norm, 'model_id': 'reference'},
                     'heads': {audit.head_key(h): {**norm, 'model_id': h['model']} for h in heads}}
    js(run/'normalization.json', normalization)
    manifest = {'round': plan['round'], 'status': 'completed_with_unavailable_heads', 'started_at': stamp(0),
                'finished_at': stamp(9), 'all_screening_completed_at': stamp(4), 'git_head': 'synthetic',
                'reviewed_commit': 'synthetic-review', 'plan_sha256': audit.digest(source/'plan.json'),
                'execution_release_sha256': audit.digest(source/'EXECUTION_RELEASE.json'),
                'normalization_sha256': audit.digest(run/'normalization.json'), 'source_hashes': source_hashes,
                'preparation': prep, 'environment': {'device': 'cpu', 'threads': 1}, 'heads': []}
    for i, f in enumerate(payloads):
        key = audit.head_key(f['head'])
        directory = run/'heads'/key
        directory.mkdir(parents=True)
        completed = i == 0
        if not completed:
            for index, row in enumerate(f['screening']):
                for tag in audit.DTYPES:
                    row['margin_'+tag] = 9.
                    row['screen_accept_'+tag] = False
                row['stratum'] = 'rejected'
                row['selected'] = index < 64
        jl(directory/'screening.jsonl', f['screening'])
        controls = {'native_screen': {'max_dtype_margin_difference': 0., 'dtype_classification_mismatches': 0,
                    'selected_counts': {'accepted': 64 if completed else 0, 'rejected': 64},
                    'pool_counts': {'accepted': 512 if completed else 0, 'rejected': 512 if completed else 1024}}}
        events = []
        for stage in (('native', 'anchors') if completed else ('native',)):
            for tag in audit.DTYPES:
                for _ in range(16):
                    for kind in ('started', 'completed'):
                        events.append({'event': kind, 'dtype': tag, 'stage': stage, 'sequences': 64,
                                       'at': stamp(2 if stage == 'native' else 6)})
        jl(directory/'forward_events.jsonl', events)
        status = {'head_key': key, 'status': 'completed' if completed else 'insufficient_yield',
                  'started_at': stamp(2), 'finished_at': stamp(8 if completed else 3),
                  'task': {'model_id': f['head']['model'], 'head': 1, 'model_heads': model_heads, 'n_layer': 2},
                  'normalization': normalization['heads'][key], 'source_hashes': source_hashes,
                  'cost': {'sequence_forwards_attempted': 4096 if completed else 2048,
                           'sequence_forwards_completed': 4096 if completed else 2048,
                           'forward_batches_attempted': 64 if completed else 32,
                           'forward_batches_completed': 64 if completed else 32,
                           'failed_forward_internal_progress_unknown': False,
                           'baseline_candidates_completed_by_dtype': {t: 1024 for t in audit.DTYPES},
                           'anchor_jobs_completed_by_dtype': {t: 256 if completed else 0 for t in audit.DTYPES}}}
        if completed:
            status['anchors_measurement_started_at'] = stamp(5)
            jl(directory/'selected_families.jsonl', f['families'])
            js(directory/'recipient_selection.json', f['selections'])
            jl(directory/'cases.jsonl', f['rows'])
            receipt = {'written_before_transfers_at': stamp(3), 'source_hashes': source_hashes,
                       'normalization': normalization['heads'][key]}
            for name in ('screening.jsonl', 'selected_families.jsonl', 'recipient_selection.json'):
                receipt[name.split('.')[0]+'_sha256'] = audit.digest(directory/name)
            js(directory/'screening_receipt.json', receipt)
            controls['anchor_precision'] = {'maximum_signed_contrast_dtype_difference': 0., 'separation_boundary_straddles': 0}
            controls['anchors'] = {t: {'identity_tolerance': 1e-5 if t == 'float32' else 1e-10,
                'identity_max_margin_error': 0., 'identity_max_node_error': 0.,
                'self_same_position_cases': 0, 'self_same_position_max_margin_error': 0.,
                'inserted_node_exactly_intended_after_dtype': True,
                'all_target_layer_attention_weights_unchanged': True,
                'all_target_layer_value_projections_unchanged': True,
                'nontarget_head_and_query_preprojection_exactly_unchanged': True} for t in audit.DTYPES}
        js(directory/'controls.json', controls)
        status['output_hashes'] = {p.name: audit.digest(p) for p in directory.iterdir()}
        js(directory/'status.json', status)
        manifest['heads'].append({'head_key': key, 'status': status['status'], 'status_sha256': audit.digest(directory/'status.json')})
    js(run/'manifest.json', manifest)
    return run, inputs, plan


class SyntheticAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = fixture()

    def fresh(self):
        return copy.deepcopy(self.base)

    def test_complete_two_head_model_uses_width32(self):
        result = audit.check_completed_records(**self.base)
        self.assertEqual(result['head_width'], 32)
        self.assertEqual(result['statistics']['accepted_hits'], 64)
        self.assertEqual(result['statistics']['rejected_hits'], 0)
        self.assertEqual(result['verification']['snapshots'], 512)
        self.assertEqual(result['cost_estimates']['validation_total_cost'], 4096)

    def test_complete_four_head_model_uses_width16(self):
        result = audit.check_completed_records(**fixture(4))
        self.assertEqual(result['head_width'], 16)

    def test_wrong_head_width_is_rejected(self):
        f = self.fresh()
        f['head']['model_heads'] = 4
        with self.assertRaisesRegex(ValueError, 'dimension'):
            audit.check_completed_records(**f)

    def test_wrong_dtype_and_extra_cell_are_rejected(self):
        for change in ('dtype', 'extra', 'missing_dtype'):
            f = self.fresh()
            cell = f['rows'][0]['cells']['neg_20_0']
            if change == 'dtype':
                cell['float32']['dtype'] = 'torch.float64'
            elif change == 'extra':
                f['rows'][0]['cells']['pos_20_0'] = copy.deepcopy(cell)
            else:
                del cell['float32']
            with self.assertRaises(ValueError):
                audit.check_completed_records(**f)

    def test_delivered_tensor_change_is_rejected(self):
        f = self.fresh()
        f['rows'][0]['cells']['neg_20_0']['float64']['h_patch_delivered'][0] = 2.
        with self.assertRaisesRegex(ValueError, 'Delivered node'):
            audit.check_completed_records(**f)

    def test_changed_donor_metadata_is_rejected(self):
        f = self.fresh()
        f['rows'][0]['cells']['neg_20_0']['donor_metadata']['position'] = 28
        with self.assertRaisesRegex(ValueError, 'metadata'):
            audit.check_completed_records(**f)

    def test_family_swap_is_rejected(self):
        f = self.fresh()
        f['rows'][0], f['rows'][1] = f['rows'][1], f['rows'][0]
        with self.assertRaisesRegex(ValueError, 'binding'):
            audit.check_completed_records(**f)

    def test_selection_must_use_first64(self):
        f = self.fresh()
        f['screening'][0]['selected'] = False
        f['screening'][64]['selected'] = True
        with self.assertRaisesRegex(ValueError, 'first64'):
            audit.check_completed_records(**f)

    def test_missing_or_extra_candidate_fails(self):
        for delta in (-1, 1):
            f = self.fresh()
            if delta < 0:
                f['screening'].pop()
            else:
                f['screening'].append(copy.deepcopy(f['screening'][-1]))
            with self.assertRaisesRegex(ValueError, '1024'):
                audit.check_completed_records(**f)

    def test_native_precision_failure_not_dropped(self):
        f = self.fresh()
        f['screening'][1000]['margin_float32'] += .002
        with self.assertRaisesRegex(ValueError, 'precision'):
            audit.check_completed_records(**f)

    def test_native_threshold_is_signed_and_model_relative(self):
        f = self.fresh()
        for tag in audit.DTYPES:
            f['screening'][100]['margin_'+tag] = -100.
        selected, counts, _, _ = audit.check_screen(f['candidates'], f['screening'], f['exclusions'], f['normalization'])
        self.assertEqual(counts['accepted'], 512)
        self.assertEqual(selected['accepted'], list(range(64)))

    def test_recipient_site_must_maximize_native_attention(self):
        f = self.fresh()
        selection = f['selections'][0]
        alternatives = [i+1 for i,c in enumerate(selection['recipient']) if c == ')'
                        and i+1 != selection['recipient_position']]
        selection['native_eos_attention'] = [0.]*42
        selection['native_eos_attention'][alternatives[0]] = 1.
        with self.assertRaisesRegex(ValueError, 'first maximum'):
            audit.check_completed_records(**f)

    def test_anchor_boundary_straddle_fails_even_below_error_tolerance(self):
        f = self.fresh()
        cell = f['rows'][0]['cells']['pos_28_0']
        for tag, target in [('float32', .2019), ('float64', .2021)]:
            cell[tag]['patched_margin'] = audit.old.float32(target) if tag == 'float32' else target
            cell[tag]['margin_change'] = cell[tag]['patched_margin']
        with self.assertRaisesRegex(ValueError, 'boundary'):
            audit.check_completed_records(**f)

    def test_shortfall_retains_all_native_rows_without_imputation(self):
        f = self.fresh()
        for i, row in enumerate(f['screening']):
            accepted = i < 10
            for tag in audit.DTYPES:
                row['margin_'+tag] = 0. if accepted else 9.
                row['screen_accept_'+tag] = accepted
            row['stratum'] = 'accepted' if accepted else 'rejected'
            row['selected'] = i < 74
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            audit.check_screen(f['candidates'], f['screening'], f['exclusions'], f['normalization'])
        _, counts, _, shortfall = audit.check_screen(f['candidates'], f['screening'], f['exclusions'],
                                                     f['normalization'], allow_shortfall=True)
        self.assertTrue(shortfall)
        self.assertEqual(counts, {'accepted': 10, 'rejected': 1014})

    def test_normalization_recomputed_without_model(self):
        record = {'w_false': [1.]+[0.]*63, 'w_true': [0.]*64,
                  'ln_gamma': [1.]*64, 'ln_beta': [0.]*64,
                  'center': 0., 'radius': math.sqrt(63), 'normalized_cutoff': .8,
                  'raw_cutoff': .8*math.sqrt(63)}
        result = audit.check_normalization(record, .8)
        self.assertAlmostEqual(result['radius'], math.sqrt(63))
        record['center'] = 1
        with self.assertRaisesRegex(ValueError, 'center'):
            audit.check_normalization(record, .8)

    def test_hash_binding_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bound.txt'
            path.write_text('before')
            hashes = {'bound.txt': audit.digest(path)}
            audit.check_hashes(temp, hashes, ['bound.txt'])
            path.write_text('after')
            with self.assertRaisesRegex(ValueError, 'changed'):
                audit.check_hashes(temp, hashes)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                audit.safe_path(temp, '../escape')

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.json'
            path.write_text('{"status":"failed","status":"completed"}')
            with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
                audit.load_json(path)

    def test_full_synthetic_artifact_audit_retains_all_heads(self):
        with tempfile.TemporaryDirectory() as temp:
            run, inputs, _ = directory_fixture(temp)
            result = audit.audit(run, inputs, root=temp)
            self.assertEqual(len(result['heads']), 6)
            self.assertEqual(result['cohort_decision']['unavailable_heads'], 5)
            self.assertEqual(result['cohort_decision']['substantial_heads'], 1)
            self.assertFalse(result['cohort_decision']['primary_success'])
            self.assertEqual(result['recorded_cohort_cost']['sequence_forwards_completed'], 14336)

    def test_full_directory_audit_also_supports_width16(self):
        with tempfile.TemporaryDirectory() as temp:
            run, inputs, _ = directory_fixture(temp, model_heads=4)
            result = audit.audit(run, inputs, root=temp)
            self.assertEqual(result['heads'][0]['head_width'], 16)

    def test_full_directory_retains_technical_failure_without_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            run, inputs, _ = directory_fixture(temp)
            manifest = audit.load_json(run/'manifest.json')
            entry = manifest['heads'][-1]
            directory = run/'heads'/entry['head_key']
            status = audit.load_json(directory/'status.json')
            status.update(status='technical_failure', error='Synthetic failure after native return', error_type='RuntimeError')
            (directory/'failure.txt').write_text('Synthetic preserved exception record')
            status['output_hashes']['failure.txt'] = audit.digest(directory/'failure.txt')
            (directory/'status.json').write_text(json.dumps(status))
            entry.update(status='technical_failure', status_sha256=audit.digest(directory/'status.json'))
            (run/'manifest.json').write_text(json.dumps(manifest))
            result = audit.audit(run, inputs, root=temp)
            failed = result['heads'][-1]
            self.assertEqual(failed['status'], 'technical_failure')
            self.assertNotIn('statistics', failed)
            self.assertNotIn('cost_estimates', failed)
            self.assertEqual(failed['recorded_cost']['sequence_forwards_completed'], 2048)

    def test_full_directory_detects_raw_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            run, inputs, _ = directory_fixture(temp)
            directory = run/'heads'/'synthetic0_head1'
            with (directory/'cases.jsonl').open('a') as handle:
                handle.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'Artifact changed'):
                audit.audit(run, inputs, root=temp)

    def test_missing_or_duplicate_head_cannot_change_denominator(self):
        plan = {'cohort': [{'model': f'm{i}', 'head_one_based': 1} for i in range(6)]}
        rows = [{'head_key': f'm{i}_head1', 'status': 'technical_failure'} for i in range(6)]
        self.assertEqual(audit.cohort_report(plan, rows)['cohort_decision']['unavailable_heads'], 6)
        with self.assertRaisesRegex(ValueError, 'six'):
            audit.cohort_report(plan, rows[:-1])
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError, 'six'):
            audit.cohort_report(plan, rows)

    def test_unavailable_head_cannot_receive_imputed_statistics(self):
        plan = {'cohort': [{'model': f'm{i}', 'head_one_based': 1} for i in range(6)]}
        rows = [{'head_key': f'm{i}_head1', 'status': 'insufficient_yield'} for i in range(6)]
        rows[0]['statistics'] = audit.stats.head_result(0, 0)
        with self.assertRaisesRegex(ValueError, 'imputed'):
            audit.cohort_report(plan, rows)

    def test_unmatched_forward_retains_cost_uncertainty(self):
        status = {'status': 'technical_failure', 'started_at': '2026-10-03T00:00:00+00:00',
                  'finished_at': '2026-10-03T00:00:02+00:00', 'cost': {
                  'sequence_forwards_attempted': 64, 'sequence_forwards_completed': 0,
                  'forward_batches_attempted': 1, 'forward_batches_completed': 0,
                  'failed_forward_internal_progress_unknown': True,
                  'baseline_candidates_completed_by_dtype': {t: 0 for t in audit.DTYPES},
                  'anchor_jobs_completed_by_dtype': {t: 0 for t in audit.DTYPES}}}
        events = [{'dtype': 'float32', 'stage': 'native', 'event': 'started', 'sequences': 64,
                   'at': '2026-10-03T00:00:01+00:00'}]
        result = audit.check_cost_ledger(status, events)
        self.assertTrue(result['failed_forward_internal_progress_unknown'])
        status['cost']['failed_forward_internal_progress_unknown'] = False
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            audit.check_cost_ledger(status, events)


if __name__ == '__main__':
    unittest.main()
