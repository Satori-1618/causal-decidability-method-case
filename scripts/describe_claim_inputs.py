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


COMMANDS = {'makelov-table1': makelov_table1}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=sorted(COMMANDS))
    args = parser.parse_args()
    print(json.dumps(COMMANDS[args.command](), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
