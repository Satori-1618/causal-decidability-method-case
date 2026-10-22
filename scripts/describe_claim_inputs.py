#!/usr/bin/env python3
"""Small, read-only derivations from stored records that public claims cite.

Standard library only. Loads no model, writes nothing, and prints JSON. Each subcommand
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


COMMANDS = {'makelov-table1': makelov_table1, 'mixing-concentration': mixing_concentration}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=sorted(COMMANDS))
    args = parser.parse_args()
    print(json.dumps(COMMANDS[args.command](), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
