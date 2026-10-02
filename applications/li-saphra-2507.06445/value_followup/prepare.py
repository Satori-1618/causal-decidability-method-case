"""Prepare inputs before any value-transfer outcome is measured."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from design import generate, exclusion_inputs, completion_count, POSITIONS, BALANCES

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['development', 'confirmation'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--families', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing overwrite')
    if args.phase == 'development' and (args.seed != 25100201 or args.families != 32):
        raise SystemExit('Development is fixed at 32 families and seed 25100201')
    if args.phase == 'confirmation' and not (HERE/'CONFIRMATION.md').exists():
        raise SystemExit('No confirmation contract exists; development cannot silently authorize it')
    exclusions = exclusion_inputs(HERE.parent/'native_followup')
    families = generate(args.seed, args.families, args.phase, exclusions)
    args.output.mkdir(parents=True)
    path = args.output/'cases.jsonl'
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in families))
    banned = args.output/'exclusions.json'
    banned.write_text(json.dumps(exclusions, indent=2) + '\n')
    prefixes = Counter(d['prefix_sha256'] for f in families for d in f['donors'])
    recipients = Counter(f['recipient'] for f in families)
    record = {
        'phase': args.phase, 'seed': args.seed, 'families': args.families,
        'sampling': 'iid family draws with replacement; distinct prefixes within each cell',
        'cases_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'exclusions_sha256': hashlib.sha256(banned.read_bytes()).hexdigest(),
        'generator_sha256': hashlib.sha256((HERE/'design.py').read_bytes()).hexdigest(),
        'prefix_support_before_exclusions_and_hash_split': {
            f'{d}_{p}': completion_count(p-1, 0, False, d+1) for d in BALANCES for p in POSITIONS},
        'unique_donor_prefixes': len(prefixes), 'donor_prefix_draws': sum(prefixes.values()),
        'max_prefix_reuse_across_families': max(prefixes.values()),
        'unique_recipients': len(recipients), 'max_recipient_reuse': max(recipients.values()),
    }
    (args.output/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
