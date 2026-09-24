"""Rebuild makelov_read_source.csv from the pilot's raw records (provenance, not needed to run).

    python3 examples/data/make_makelov_read_source_csv.py <path to records.jsonl>

The records live on the branch applications/makelov-2311.17030 under
applications/makelov-2311.17030/results/makelov_read_source_001/records.jsonl
(sha256 5e63b3b9...). Only measured margins are exported: no prompts, no model weights,
no upstream vectors.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

RECORDS_SHA = '5e63b3b95cfd50324530ee62b703d37fcfda629460641926c07c803a39a93829'
CONDITIONS = ('baseline', 'full', 'read_row', 'read_null')


def main():
    source = Path(sys.argv[1])
    if hashlib.sha256(source.read_bytes()).hexdigest() != RECORDS_SHA:
        sys.exit('records.jsonl is not the pilot file this example was built from')
    out = Path(__file__).with_name('makelov_read_source.csv')
    with open(out, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['unit', 'repeat', 'condition', 'value'])
        for line in source.read_text().splitlines():
            record = json.loads(line)
            for condition in CONDITIONS:
                writer.writerow([record['case_id'], 'receiver_' + record['receiver_pattern'],
                                 condition, repr(record['margins'][condition])])
    print('wrote', out)


if __name__ == '__main__':
    main()
