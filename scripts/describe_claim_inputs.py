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
from pathlib import Path

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
        'above_T_W': sum(t > t_w for t in values), 'below_T_W': sum(t < t_w for t in values),
        'deviating_above_band': sum(t - t_w > band for t in values),
        'deviating_below_band': sum(t_w - t > band for t in values),
        'median_T_resolved': statistics.median(values),
        'argmax_label_counts_resolved': {
            label: sum(label in c['labels'] for c in resolved) for label in checker.LABELS},
        'reading': 'Descriptive counts over resolved cases. The frozen decisions are those of '
                   'check_mixing_round1_records.py; these counts add no test.',
    }


def tracr_confirmation():
    """Run the portable Tracr verifier on a temporary copy, then report the verified summary.

    The frozen verifier writes verification.json into the run directory it checks, so it
    only ever sees a copy in a temporary directory; the checkout is not written to.
    """
    import shutil
    import tempfile
    run = TRACR / 'results/confirmation_001'
    portable = _load_module('tracr_portable', ROOT / 'scripts/check_tracr_records_portable.py')
    with tempfile.TemporaryDirectory(prefix='claims-tracr-') as temp:
        copy = Path(temp) / run.name
        shutil.copytree(run, copy)
        verification = portable.verify(copy, TRACR / 'CONFIRMATION_FREEZE.json')
    summary = json.loads((run / 'summary.json').read_text())
    decision = summary['population_decision']
    return {'verification': verification,
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
                   if t.get('stages') and all(
                       any(c['status'] == 'adequate' for c in s['candidates'].values())
                       for s in t['stages'].values())}
    return {'archive': restore_log.getvalue().strip(),
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


COMMANDS = {'makelov-table1': makelov_table1, 'mixing-concentration': mixing_concentration,
            'tracr-confirmation': tracr_confirmation, 'saphra-native': saphra_native,
            'saphra-averaged004': saphra_averaged004}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=sorted(COMMANDS))
    args = parser.parse_args()
    print(json.dumps(COMMANDS[args.command](), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
