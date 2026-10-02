"""Prepare a frozen native-only screening pool and independent donor templates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
VALUE=HERE.parent
sys.path.insert(0,str(VALUE))
from design import POSITIONS,exclusion_inputs,generate,phase_of,prefix_state

SCREEN_SEED=25100202
DONOR_SEED=25100203
POOL_SIZE=1024
PER_STRATUM=32


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extended_exclusions():
    native=VALUE.parent/'native_followup'
    assets=json.loads((native/'ASSET_LOCK.json').read_text())
    for name in ['indist','ood']:
        relative='data/model_preds/'+name+'_data_preds.csv'
        if digest(native/'cache'/relative)!=assets['files'][relative]['sha256']:
            raise ValueError('Public exclusion source hash mismatch: '+relative)
    exclusions=exclusion_inputs(native)
    old_path=VALUE/'inputs/development_001/cases.jsonl'
    old=[json.loads(line) for line in old_path.read_text().splitlines()]
    all_old_strings={f['recipient'] for f in old}
    all_old_strings.update(d['string'] for f in old for d in f['donors'])
    exclusions['recipient_strings']=sorted(set(exclusions['recipient_strings'])|all_old_strings)
    # Every prior full input exposed both causal prefixes, irrespective of the
    # position targeted in that experiment or the role (donor/recipient).
    for position in POSITIONS:
        exclusions['prefixes'][str(position)]=sorted(set(exclusions['prefixes'][str(position)])|
                                                     {text[:position] for text in all_old_strings if len(text)>=position})
    exclusions['source_hashes']['value_followup/inputs/development_001/cases.jsonl']=digest(old_path)
    exclusions['extension']={'old_value_full_strings':len(all_old_strings),
                             'rule':'Both positions 20 and 28 from every previous donor and recipient full string'}
    return exclusions


def draw_candidates(exclusions,*,seed=SCREEN_SEED,n=POOL_SIZE):
    rng=random.Random(seed)
    banned=set(exclusions['recipient_strings'])
    candidates=[]
    for index in range(n):
        while True:
            chars=list('('*16+')'*16)
            rng.shuffle(chars)
            text=''.join(chars)
            if text not in banned and phase_of(text)=='development' and prefix_state(text)[1]<0:
                break
        candidates.append({'candidate_id':hashlib.sha256(f'screen-002:{seed}:{index}'.encode()).hexdigest()[:20],
                           'candidate_index':index,'recipient':text})
    return candidates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing overwrite')
    exclusions=extended_exclusions()
    candidates=draw_candidates(exclusions)
    donors=generate(DONOR_SEED,2*PER_STRATUM,'development',exclusions)
    args.output.mkdir(parents=True)
    for filename,rows in [('candidates.jsonl',candidates),('donor_families.jsonl',donors)]:
        (args.output/filename).write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
    (args.output/'exclusions.json').write_text(json.dumps(exclusions,indent=2)+'\n')
    prefixes=Counter(d['prefix_sha256'] for family in donors for d in family['donors'])
    recipients=Counter(c['recipient'] for c in candidates)
    metadata={'round':'screen_002','screen_seed':SCREEN_SEED,'donor_seed':DONOR_SEED,
              'candidate_pool_size':POOL_SIZE,'per_stratum':PER_STRATUM,'phase':'development',
              'sampling':'Recipient and donor pools drawn independently with replacement; within-cell donor prefixes distinct',
              'files':{name:digest(args.output/name) for name in ['candidates.jsonl','donor_families.jsonl','exclusions.json']},
              'sources':{'prepare_screen.py':digest(__file__),'design.py':digest(VALUE/'design.py')},
              'unique_candidate_strings':len(recipients),'max_candidate_reuse':max(recipients.values()),
              'unique_donor_prefixes':len(prefixes),'donor_prefix_draws':sum(prefixes.values()),
              'maximum_donor_prefix_reuse':max(prefixes.values())}
    (args.output/'preparation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
