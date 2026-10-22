"""Diagnose the recorded rank-preservation failure; never rescue its verdict."""
import argparse
import json
from pathlib import Path

import torch

from confirm import HERE,load_cases,load_runtime,native_sign_pattern
from qualify import digest,dump


def rank_gap(text,row):
    depth=0
    negative=[]
    nonnegative=[]
    for index,symbol in enumerate(text,1):
        depth+=1 if symbol=='(' else -1
        (negative if depth<0 else nonnegative).append(float(row[index]))
    if not negative or not nonnegative:
        return None
    return min(nonnegative)-max(negative) if depth>=0 else min(negative)-max(nonnegative)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing overwrite')
    freeze=json.loads(args.freeze.read_text())
    _,cases=load_cases(freeze,args.freeze)
    lock=json.loads((args.freeze.parent/freeze['asset_lock_file']).read_text())
    task=next(t for t in freeze['tasks'] if t['model_id']=='0nbysgqs' and t['head']==2)
    strings=[r['string'] for r in cases]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report={'status':'technical_diagnostic_only_original_invalidity_unchanged',
            'freeze_sha256':digest(args.freeze),'code_sha256':digest(__file__),'dtype':{}}
    for dtype in [torch.float32,torch.float64]:
        runtime=load_runtime(task,lock,dtype)
        native=runtime.run(strings)
        gate=runtime.run(strings,mode='gate_only')
        differences=[]
        for i,text in enumerate(strings):
            before=native_sign_pattern(text,native.attention[i])
            after=native_sign_pattern(text,gate.attention[i])
            if before!=after:
                b=rank_gap(text,native.attention[i]);a=rank_gap(text,gate.attention[i])
                mass=float(native.attention[i,1:len(text)+1].sum())
                scale=(len(text)/(len(text)+2))/mass
                differences.append({'family_id':cases[i]['family_id'],'condition':cases[i]['condition'],
                                    'native_pattern':before,'gate_pattern':after,'native_gap':b,'gate_gap':a,
                                    'ideal_gap_under_positive_scaling':b*scale,'scale':scale,
                                    'native_tie':b==0,'gate_tie':a==0})
        report['dtype'][str(dtype)]={'n':len(cases),'mismatches':len(differences),'details':differences}
    dump(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
