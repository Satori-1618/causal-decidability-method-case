"""Report every frozen task, retaining invalid tasks without scientific decisions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from analyze_confirmation import evaluate_stage, STAGES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Refusing to overwrite')
    freeze=json.loads(args.freeze.read_text())
    manifest=json.loads((args.run/'manifest.json').read_text())
    if manifest['status'] not in {'completed','completed_with_invalid_tasks'} or len(manifest['tasks'])!=len(freeze['tasks']):
        raise ValueError('Cohort accounting incomplete')
    if manifest['freeze_sha256']!=digest(args.freeze):raise ValueError('Different freeze')
    entries={(e['task']['model_id'],int(e['task']['head'])):e for e in manifest['tasks']}
    if len(entries)!=len(freeze['tasks']):raise ValueError('Duplicate task')
    results=[]
    for task in freeze['tasks']:
        key=(task['model_id'],task['head'])
        entry=entries[key]
        folder=args.run/'tasks'/(key[0]+'_head'+str(key[1]))
        for name,h in entry.get('files',{}).items():
            if digest(folder/name)!=h:raise ValueError('Modified artifact')
        if entry['status']!='completed':
            results.append({'task':task,'execution_status':'technical_invalidity','error':entry.get('error'), 'stages':None})
            continue
        controls=json.loads((folder/'controls.json').read_text())
        if not controls['precision_gate_passed']:raise ValueError('Failed precision gate disguised as success')
        rows=[json.loads(s) for s in (folder/'cases.jsonl').read_text().splitlines()]
        if len(rows)!=freeze['generator']['families']*3:raise ValueError('Missing cases')
        record={'task':task,'execution_status':'completed','stages':{s:evaluate_stage(rows,s,freeze) for s in STAGES},'by_condition':{}}
        for condition in ['valid','invalid_open','invalid_close']:
            selected=[r for r in rows if r['condition']==condition]
            record['by_condition'][condition]={}
            for arm in ['native','both','routing_only','gate_only','within_token','token_mass']:
                record['by_condition'][condition][arm]={'correct':sum((r['float64_'+arm]<0)==r['valid'] for r in selected), 'n':len(selected),
                    'mean_margin_effect':statistics.mean(r['float64_'+arm]-r['float64_native'] for r in selected)}
        results.append(record)
    summary={}
    for role in ['development_checkpoint_fresh_inputs','transfer_cohort']:
        subset=[r for r in results if r['task']['role']==role]
        summary[role]={'tasks':len(subset),'unique_models':len({r['task']['model_id'] for r in subset}),'execution':dict(Counter(r['execution_status'] for r in subset))}
        for stage,(_,_,_,c1,c2) in STAGES.items():
            summary[role][stage]={candidate:dict(Counter(r['stages'][stage]['candidates'][candidate]['status'] if r['stages'] else 'technical_invalidity' for r in subset)) for candidate in [c1,c2]}
    out={'summary':summary,'tasks':results,'freeze_sha256':digest(args.freeze),'run_manifest_sha256':digest(args.run/'manifest.json'),
         'interpretation':'Frozen restricted intervention forecasts. All tasks retained, including technical failures; K unchanged. No unique algorithm or method-superiority claim.'}
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
