"""Apply frozen family-level adequacy rules; no model execution or fitting."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import hashlib


def kl(q, p):
    if p <= 0:
        return 0.0 if q == 0 else math.inf
    if p >= 1:
        return 0.0 if q == 1 else math.inf
    return (q*math.log(q/p) if q else 0.0) + ((1-q)*math.log((1-q)/(1-p)) if q < 1 else 0.0)


def kl_interval(hits, n, alpha_one_sided):
    if n < 1 or not 0 <= hits <= n or not 0 < alpha_one_sided < 1:
        raise ValueError('Invalid binomial-bound input')
    q=hits/n
    target=math.log(1/alpha_one_sided)/n
    if hits == 0:
        lower=0.0
    else:
        lo,hi=0.0,q
        for _ in range(80):
            mid=(lo+hi)/2
            if kl(q,mid)>target: lo=mid
            else: hi=mid
        lower=lo
    if hits == n:
        upper=1.0
    else:
        lo,hi=q,1.0
        for _ in range(80):
            mid=(lo+hi)/2
            if kl(q,mid)>target: hi=mid
            else: lo=mid
        upper=hi
    return lower,upper


STAGES={
    'stage1':('both','routing_only','gate_only','H_R','H_G'),
    'stage2':('routing_only','within_token','token_mass','H_W','H_T'),
}


def evaluate_stage(rows, stage, freeze):
    end,first,second,c1,c2=STAGES[stage]
    families=defaultdict(list)
    for r in rows: families[r['family_id']].append(r)
    numerical=freeze['numerical_tolerance']
    tau=freeze['prediction_tolerance']
    gaps={}
    errors={c1:{},c2:{}}
    eligible_by_condition=Counter()
    error_precision=0.0
    for family, members in families.items():
        if len(members)!=3 or {r['condition'] for r in members}!={'valid','invalid_open','invalid_close'}:
            raise ValueError('Incomplete or duplicate family')
        gaps[family]=max(abs(r['float64_'+end]-r['float64_native']) for r in members)
        for r in members:
            if abs(r['float64_'+end]-r['float64_native'])>freeze['gap_threshold']:
                eligible_by_condition[r['condition']]+=1
        for candidate in [c1,c2]:
            per_dtype={}
            for dtype in ['float64','float32']:
                per_dtype[dtype]=max(max(abs(r[dtype+'_'+first]-r[dtype+'_'+(end if candidate==c1 else 'native')]),
                                               abs(r[dtype+'_'+second]-r[dtype+'_'+('native' if candidate==c1 else end)])) for r in members)
            errors[candidate][family]=per_dtype['float64']
            error_precision=max(error_precision,abs(per_dtype['float64']-per_dtype['float32']))
    eligible=[family for family in families if gaps[family]>freeze['gap_threshold']]
    decision={}
    for candidate in [c1,c2]:
        err=[errors[candidate][f] for f in eligible]
        definite=sum(x<=tau-numerical for x in err)
        possible=sum(x<=tau+numerical for x in err)
        n=len(err)
        lower=kl_interval(definite,n,freeze['alpha']/freeze['one_sided_bound_count'])[0] if n else 0.
        upper=kl_interval(possible,n,freeze['alpha']/freeze['one_sided_bound_count'])[1] if n else 1.
        if error_precision>numerical:
            status='technical_invalidity'
        elif n<freeze['minimum_eligible_families']:
            status='insufficient_eligible_families'
        elif lower>=freeze['adequacy']:
            status='adequate'
        elif upper<freeze['adequacy']:
            status='excluded'
        else:
            status='unresolved'
        decision[candidate]={'definite_hits':definite,'possible_hits':possible,'n':n,'lower':lower,'upper':upper,'status':status,
                             'mean_family_max_error_eligible':statistics.mean(err) if err else None,
                             'mean_family_max_error_all':statistics.mean(errors[candidate].values())}
    return {'eligible_families':len(eligible),'eligible_members_by_condition':dict(eligible_by_condition),
            'maximum_prediction_error_precision_difference':error_precision,'candidates':decision,
            'not_excluded':[name for name,r in decision.items() if r['status']!='excluded'],
            'adequate':[name for name,r in decision.items() if r['status']=='adequate']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise SystemExit('Refusing to overwrite report')
    freeze=json.loads(args.freeze.read_text())
    manifest=json.loads((args.run/'manifest.json').read_text())
    if manifest['status']!='completed' or len(manifest['tasks'])!=len(freeze['tasks']):
        raise ValueError('Run is incomplete; scientific analysis blocked')
    if manifest['freeze_sha256']!=hashlib.sha256(args.freeze.read_bytes()).hexdigest():
        raise ValueError('Run used a different freeze')
    for entry in manifest['tasks']:
        folder=args.run/'tasks'/(entry['task']['model_id']+'_head'+str(entry['task']['head']))
        for name,expected in entry['files'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Altered run artifact: '+str(folder/name))
    results=[]
    for task in freeze['tasks']:
        path=args.run/'tasks'/(task['model_id']+'_head'+str(task['head']))/'cases.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=freeze['generator']['families']*3: raise ValueError('Incomplete rows')
        record={'task':task,'stages':{stage:evaluate_stage(rows,stage,freeze) for stage in STAGES}}
        record['by_condition']={}
        for condition in ['valid','invalid_open','invalid_close']:
            selected=[r for r in rows if r['condition']==condition]
            record['by_condition'][condition]={}
            for arm in ['native','both','routing_only','gate_only','within_token','token_mass']:
                record['by_condition'][condition][arm]={
                    'correct':sum((r['float64_'+arm]<0)==r['valid'] for r in selected),
                    'n':len(selected),
                    'mean_margin_effect':statistics.mean(r['float64_'+arm]-r['float64_native'] for r in selected)}
        results.append(record)
    summary={}
    for role in ['development_checkpoint_fresh_inputs','transfer_cohort']:
        subset=[r for r in results if r['task']['role']==role]
        summary[role]={'tasks':len(subset),'unique_models':len({r['task']['model_id'] for r in subset})}
        for stage,(_,_,_,c1,c2) in STAGES.items():
            summary[role][stage]={candidate:dict(Counter(r['stages'][stage]['candidates'][candidate]['status'] for r in subset)) for candidate in [c1,c2]}
    out={'summary':summary,'tasks':results,'interpretation':'Restricted intervention forecasts, conditional on resolvable endpoint effects. Finite checkpoint cohort; not unique native mechanisms or method superiority.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
