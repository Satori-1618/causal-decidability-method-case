"""Prepare untouched cyclic families and a fully enumerated checkpoint cohort.

No model execution. The generator is intentionally frozen before its outputs
are inspected by the confirmation runner.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor
import urllib.request

from fetch_assets import REVISION, ROOT

SEED = 250710012026
N_FAMILIES = 512
LENGTH = 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotations(text):
    return sorted({text[i:] + text[:i] for i in range(len(text))})


def valid(text):
    depth = 0
    for ch in text:
        depth += 1 if ch == '(' else -1
        if depth < 0:
            return False
    return depth == 0


def choose(items, canonical, tag):
    index = int(hashlib.sha256((str(SEED)+tag+canonical).encode()).hexdigest(),16) % len(items)
    return items[index]


def generate():
    old = set()
    for name in ['ood','indist']:
        with (ROOT/'cache/data/model_preds'/f'{name}_data_preds.csv').open() as f:
            for row in csv.DictReader(f):
                text = row['string']
                if len(text) == LENGTH and text.count('(') == LENGTH//2:
                    old.add(rotations(text)[0])
    rng = random.Random(SEED)
    seen, records, attempts = set(), [], 0
    while len(seen) < N_FAMILIES:
        attempts += 1
        chars = list('('*(LENGTH//2) + ')'*(LENGTH//2))
        rng.shuffle(chars)
        orbit = rotations(''.join(chars))
        # A uniformly sampled balanced string gives an orbit mass proportional
        # to its number of distinct rotations. Correct that size bias exactly.
        if rng.randrange(len(orbit)) != 0:
            continue
        canonical = orbit[0]
        if canonical in old or canonical in seen:
            continue
        positive = [s for s in orbit if valid(s)]
        open_negative = [s for s in orbit if s[0] == '(' and not valid(s)]
        close_negative = [s for s in orbit if s[0] == ')' and not valid(s)]
        if not positive or not open_negative or not close_negative:
            continue
        family = hashlib.sha256(canonical.encode()).hexdigest()[:20]
        seen.add(canonical)
        for condition, pool, target in [('valid',positive,True),('invalid_open',open_negative,False),('invalid_close',close_negative,False)]:
            records.append({'family_id':family,'condition':condition,'string':choose(pool,canonical,condition),'valid':target})
    return records, {'seed':SEED,'families':len(seen),'length':LENGTH,'attempts':attempts,'excluded_public_orbits':len(old),'sampling':'uniform_without_replacement_over_eligible_rotation_orbits'}


def tasks():
    with (ROOT.parent/'results/heads.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    answer=[]
    for r in rows:
        if r['head_type'] != 'sign-matching' or int(r['n_layer']) < 2 or r['layer'] != r['n_layer']:
            continue
        seed_pair=(int(r['initialization_seed']),int(r['shuffle_seed']))
        focus=r['model_id']=='1aez5d6p' and int(r['head'])==2
        if seed_pair == (365,220) and not focus:
            continue
        answer.append({'model_id':r['model_id'],'n_layer':int(r['n_layer']),'n_head':int(r['n_head']),'head':int(r['head']),
                       'init_seed':seed_pair[0],'shuffle_seed':seed_pair[1], 'role':'development_checkpoint_fresh_inputs' if focus else 'transfer_cohort',
                       'sign_score_published':float(r['sign_score'])})
    return sorted(answer,key=lambda r:(r['role']!='development_checkpoint_fresh_inputs',r['model_id'],r['head']))


def fetch_model(model_id):
    name=f'data/model_weights/run_{model_id}/run_{model_id}_checkpoint_5.pt'
    url=f'https://raw.githubusercontent.com/vli31/id-predict-ood/{REVISION}/{name}'
    path=ROOT/'cache'/name
    data=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=60).read()
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return name,{'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite confirmation preparation')
    args.output.mkdir(parents=True)
    selected=tasks()
    model_ids=sorted({t['model_id'] for t in selected})
    with ThreadPoolExecutor(max_workers=6) as pool:
        assets=dict(pool.map(fetch_model,model_ids))
    lock_path=args.output/'COHORT_ASSET_LOCK.json'
    lock_path.write_text(json.dumps({'revision':REVISION,'files':assets},indent=2)+'\n')
    cases,meta=generate()
    (args.output/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases))
    freeze={'tasks':selected,'cases_file':'cases.jsonl','cases_sha256':sha(args.output/'cases.jsonl'),
            'asset_lock_file':'COHORT_ASSET_LOCK.json','asset_lock_sha256':sha(lock_path),
            'numerical_tolerance':0.001,'prediction_tolerance':0.25,'adequacy':0.8,'minimum_eligible_families':128,
            'gap_threshold':0.502,'alpha':0.05,'one_sided_bound_count':len(selected)*4*2,
            'confidence_method':'binary_kl_chernoff_without_replacement',
            'protocol_sha256':sha(ROOT/'CONFIRMATION.md'),'analyzer_sha256':sha(ROOT/'analyze_confirmation.py'),
            'generator':meta,'generator_sha256':sha(__file__),
            'status':'prepared_before_any_confirmation_model_execution'}
    (args.output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps({'tasks':len(selected),'unique_models':len(model_ids),'roles':{role:sum(t['role']==role for t in selected) for role in ['development_checkpoint_fresh_inputs','transfer_cohort']},'generator':meta},indent=2))


if __name__=='__main__':
    main()
