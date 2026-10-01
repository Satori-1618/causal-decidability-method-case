"""Download and verify the already locked checkpoint cohort; no new selection."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent


def fetch(item):
    name,record=item
    path=ROOT/'cache'/name
    data=path.read_bytes() if path.exists() else urllib.request.urlopen(record['url'],timeout=60).read()
    if hashlib.sha256(data).hexdigest()!=record['sha256'] or len(data)!=record['bytes']:
        raise ValueError('Locked checkpoint mismatch: '+name)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_bytes(data)
    return name


if __name__=='__main__':
    lock=json.loads((ROOT/'frozen/confirmation_001/COHORT_ASSET_LOCK.json').read_text())
    with ThreadPoolExecutor(max_workers=6) as pool:
        names=list(pool.map(fetch,lock['files'].items()))
    print('Verified',len(names),'pinned checkpoints')
