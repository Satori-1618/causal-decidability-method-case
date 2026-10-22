#!/usr/bin/env python3
"""Small, read-only derivations from stored records that public claims cite.

Loads no model and prints JSON. Nothing is written to the checkout; the Tracr and Li/Saphra
subcommands write only into temporary copies. Each subcommand
recomputes values from per-example or per-family records that the frozen pipelines
stored; it reaches no new decision.

    python3 scripts/describe_claim_inputs.py makelov-table1
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep imported application modules from writing caches

ROOT = Path(__file__).resolve().parents[1]
MAKELOV = ROOT / 'applications/makelov-2311.17030'
MIXING = ROOT / 'applications/gur-arieh-2510.06182'
TRACR = ROOT / 'applications/tracr'
SAPHRA = ROOT / 'applications/li-saphra-2507.06445'


def _jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def makelov_table1():
    """Per-condition interchange accuracy and logit difference in the Table 1 replication."""
    rows = _jsonl(MAKELOV / 'results/makelov_replication_001/per_example.jsonl')
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row['condition'], []).append(row)
    return {
        'source': 'applications/makelov-2311.17030/results/makelov_replication_001/per_example.jsonl',
        'conditions': {
            name: {
                'n_pairs': len(group),
                'interchange_accuracy': statistics.fmean(r['interchange_correct'] for r in group),
                'mean_logit_diff_base_ordering': statistics.fmean(
                    r['logit_diff_base_ordering'] for r in group),
            } for name, group in sorted(by_condition.items())},
        'reading': 'interchange accuracy = share of pairs whose top answer under the patch is '
                   "the source's answer; logit difference = base IO minus base subject.",
    }


def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mixing_concentration():
    """Case-level concentration T in the Mixing Mechanisms confirmation, from raw records.

    Uses the frozen checker's own distribution() and measure() on the qualifying records,
    so the per-case values are the ones its decisions are built from.
    """
    results = MIXING / 'results/confirmation'
    checker = _load_module('mixing_checker', MIXING / 'scripts/check_mixing_round1_records.py')
    manifest = json.loads((results / 'manifest.json').read_text())
    records = _jsonl(results / 'records.jsonl')
    rule, anchors, cell = manifest['rule'], manifest['anchors'], manifest['cell']
    qualifying = [r for r in records if r.get('qualifies') is True]
    cases = [checker.measure(checker.distribution(r['answer_logits']), r['answer_mass_full_vocab'],
                             cell, rule['w'], rule['s_min']) for r in qualifying]
    resolved = [c for c in cases if c['resolved']]
    t_w, band = anchors['T_W'], rule['kappa'] * anchors['d']
    values = [c['T'] for c in resolved]
    mean_q = [statistics.fmean(c['q'][i] for c in resolved) for i in range(3)]
    return {
        'source': 'applications/gur-arieh-2510.06182/results/confirmation/records.jsonl',
        'setup': {'model': manifest['model']['id'], 'revision': manifest['model']['revision'],
                  'task': manifest['task'], 'n_groups': manifest['n_groups'],
                  't_entity': manifest['t_entity'], 'layer': manifest['layer'],
                  'patch_positions': manifest['patch_positions'], 'cell_key': manifest['cell_key'],
                  'cell': cell, 'readout': rule['readout'], 'N': manifest['N'],
                  'generated': len(records)},
        'T_W': t_w, 'T_A': anchors['T_A'], 'band_half_width': band,
        'resolved': len(resolved), 'unresolved': len(cases) - len(resolved),
        'answer_mass_below_floor': sum(r['answer_mass_full_vocab'] < rule['answer_mass_floor']
                                       for r in qualifying),
        'mean_q_resolved': mean_q, 'q_bar_B': anchors['q_bar_B'],
        'mean_max_abs_difference': max(abs(a - b) for a, b in zip(anchors['q_bar_B'], mean_q)),
        'mean_gate_delta': manifest['mean_gate']['delta'],
        'above_T_W': sum(t > t_w for t in values), 'below_T_W': sum(t < t_w for t in values),
        'deviating_above_band': sum(t - t_w > band for t in values),
        'deviating_below_band': sum(t_w - t > band for t in values),
        'median_T_resolved': statistics.median(values),
        'argmax_label_counts_resolved': {
            label: sum(label in c['labels'] for c in resolved) for label in checker.LABELS},
        'argmax_ties_resolved': sum(len(c['labels']) > 1 for c in resolved),
        'reading': 'Descriptive counts over resolved cases. The frozen decisions are those of '
                   'check_mixing_round1_records.py; these counts add no test. A case is resolved '
                   'when it passes both the support rule and the answer-mass rule, so with no '
                   'case below the mass floor every unresolved case failed the support rule.',
    }


def tracr_confirmation():
    """Run the portable Tracr verifier on a temporary copy, then report the verified summary.

    The frozen verifier writes verification.json into the run directory it checks, so it
    only ever sees a copy in a temporary directory; the checkout is not written to. The
    development run whose summary hash the freeze records is verified the same way.
    """
    import hashlib
    import shutil
    import tempfile
    run = TRACR / 'results/confirmation_001'
    freeze_path = TRACR / 'CONFIRMATION_FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    bound = [d for d in sorted((TRACR / 'results').glob('development_*'))
             if (d / 'summary.json').exists() and hashlib.sha256(
                 (d / 'summary.json').read_bytes()).hexdigest() == freeze['development_summary_sha256']]
    if len(bound) != 1:
        raise ValueError('expected exactly one development run bound by the freeze')
    portable = _load_module('tracr_portable', ROOT / 'scripts/check_tracr_records_portable.py')
    frozen_verifier = _load_module('tracr_frozen_verify', TRACR / 'scripts/verify.py')
    with tempfile.TemporaryDirectory(prefix='claims-tracr-') as temp:
        copy = Path(temp) / run.name
        shutil.copytree(run, copy)
        verification = portable.verify(copy, freeze_path)
        development_copy = Path(temp) / bound[0].name
        shutil.copytree(bound[0], development_copy)
        development_verification = frozen_verifier.verify(development_copy, None)
    development = json.loads((bound[0] / 'summary.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    decision = summary['population_decision']
    return {'verification': verification,
            'required_successes': freeze['precision_plan']['required_successes'],
            'sample_size': freeze['sample_size'],
            'numeric_allowance': freeze['numeric_allowance'],
            'scientific_tolerance': freeze['scientific_tolerance'],
            'total_radius': freeze['scientific_tolerance'] + freeze['numeric_allowance'],
            'development_run_bound_by_freeze': bound[0].name,
            'development_verification_status': development_verification['status'],
            'development_families': development['family_count'],
            'development_copy_controls_selected': development['site_B_positive_controls'],
            'families': summary['family_count'], 'successes': decision['successes'],
            'lower_rate': decision['bounds']['lower_rate'],
            'minimum_success_rate': decision['minimum_success_rate'],
            'status': decision['status'], 'record_count': summary['record_count'],
            'paired_precision_max_error': summary['paired_precision_max_error']}


def saphra_native():
    """Unpack the archived native follow-up into a temporary copy and run its verifier there.

    The continued run is stored as a hash-indexed archive; the verifier also needs the
    original failed run beside it. Both go into a temporary directory only.
    """
    import shutil
    import tempfile
    native = SAPHRA / 'native_followup'
    bundle = _load_module('saphra_bundle', native / 'result_bundle.py')
    verifier = _load_module('saphra_verify', native / 'verify_confirmation.py')
    with tempfile.TemporaryDirectory(prefix='claims-saphra-') as temp:
        results = Path(temp) / 'results'
        shutil.copytree(native / 'results/confirmation_001', results / 'confirmation_001')
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()) as restore_log:
            bundle.unpack(native / 'artifacts/confirmation_001.tar.gz',
                          native / 'artifacts/confirmation_001.index.json', results)
        audit = verifier.audit((native / 'frozen/confirmation_001/freeze.json').resolve(),
                               results / 'confirmation_001_continued',
                               native / 'results/confirmation_report.json')
    development = [t for t in audit['task_audits']
                   if t['role'] == 'development_checkpoint_fresh_inputs']
    both_stages = {f"{t['model_id']}/{t['head']}": t['stages'] for t in audit['task_audits']
                   if t.get('stages')
                   and t['stages']['stage1']['candidates']['H_R']['status'] == 'adequate'
                   and t['stages']['stage2']['candidates']['H_W']['status'] == 'adequate'}
    # The audit above compared the stored report with its own recomputation, so the
    # report's per-stage eligibility fields can be read from here.
    report = json.loads((native / 'results/confirmation_report.json').read_text())
    eligibility = {}
    with_valid = {}
    for record in report['tasks']:
        if not record['stages']:
            continue
        key = f"{record['task']['model_id']}/{record['task']['head']}"
        for stage, result in record['stages'].items():
            members = result.get('eligible_members_by_condition') or {}
            if key in both_stages:
                eligibility.setdefault(key, {})[stage] = members
            if any(name.startswith('valid') and count > 0 for name, count in members.items()):
                with_valid[stage] = with_valid.get(stage, 0) + 1
    freeze = json.loads((native / 'frozen/confirmation_001/freeze.json').read_text())
    transfer_layers = sorted({task['n_layer'] for task in freeze['tasks']
                              if task['role'] == 'transfer_cohort'})
    return {'archive': restore_log.getvalue().strip(),
            'transfer_task_n_layer_values': transfer_layers,
            'eligible_members_by_condition_for_two_stage_tasks': eligibility,
            'tasks_with_a_valid_separating_member': with_valid,
            'verified': audit['verified'], 'report_compared': audit['report_compared'],
            'tasks': audit['tasks'], 'completed': audit['completed'],
            'technical_invalidity': audit['technical_invalidity'], 'summary': audit['summary'],
            'development_task': development,
            'tasks_adequate_in_both_stages': both_stages,
            'tasks_adequate_in_both_stages_ids': sorted(both_stages),
            'limitations': audit['limitations']}


def saphra_averaged004():
    """Extract the averaged-anchor run into a temporary directory and recount it there.

    Checks every archived file against raw_index.json, then runs the independent SciPy
    recount (verify_statistics.py), which imports none of the producer's analysis.
    """
    import contextlib
    import hashlib
    import io
    import sys
    import tarfile
    import tempfile
    round004 = SAPHRA / 'value_followup/averaged_anchors_004'
    index = json.loads((round004 / 'results/raw_index.json').read_text())
    archive = round004 / 'results' / index['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != index['archive_sha256']:
        raise ValueError('archive hash differs from raw_index.json')
    recount = _load_module('saphra_avg004_stats', round004 / 'reporting/verify_statistics.py')
    with tempfile.TemporaryDirectory(prefix='claims-avg004-') as temp:
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.name.startswith('/') or '..' in member.name:
                    raise ValueError(f'unexpected archive member: {member.name}')
                (Path(temp) / member.name).parent.mkdir(parents=True, exist_ok=True)
                (Path(temp) / member.name).write_bytes(tar.extractfile(member).read())
        run = Path(temp) / 'run_001'
        for name, entry in index['files'].items():
            data = (run / name).read_bytes()
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError(f'archived file differs from raw_index.json: {name}')
        argv, sys.argv = sys.argv, ['verify_statistics.py', str(run)]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                recount.main()
        finally:
            sys.argv = argv
    report = json.loads(out.getvalue())
    report.pop('families_recomputed')
    report['archive_files_verified'] = len(index['files'])
    return report


def saphra_retro():
    """Recompute the retrospective outputs from the vendored upstream tables, then read rows.

    Calls the retrospective analyzer's own encoded_results(), which rebuilds every stored
    output from the released CSVs; nothing is read from results/.
    """
    import csv
    import io
    sys.path.insert(0, str(SAPHRA))
    analyze = _load_module('saphra_analyze', SAPHRA / 'analyze.py')
    outputs = analyze.encoded_results()
    heads = list(csv.DictReader(io.StringIO(outputs['heads.csv'])))
    def head(model, layer, number):
        row = next(r for r in heads if r['model_id'] == model and r['layer'] == str(layer)
                   and r['head'] == str(number))
        return {k: row[k] for k in ('head_type', 'native_correct', 'uniform_correct', 'mean_correct')}
    countermodels = json.loads(outputs['countermodels.json'])
    return {'illustrated_head_a9g0io1r_L2H1': head('a9g0io1r', 2, 1),
            'countermodel_correct_counts': {name: w['correct_counts']
                                            for name, w in countermodels['witnesses'].items()}}


COMMANDS = {'makelov-table1': makelov_table1, 'mixing-concentration': mixing_concentration,
            'tracr-confirmation': tracr_confirmation, 'saphra-native': saphra_native,
            'saphra-averaged004': saphra_averaged004, 'saphra-retro': saphra_retro}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=sorted(COMMANDS))
    args = parser.parse_args()
    print(json.dumps(COMMANDS[args.command](), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
