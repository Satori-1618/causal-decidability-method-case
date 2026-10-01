"""Continue the declared finite cohort after a preserved per-task technical failure."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import torch

from confirm import ROOT,HERE,assert_committed,load_cases,resolve_path,run_task,utcnow
from qualify import digest,dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['freeze','source-run','amendment','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    source=args.source_run.resolve()
    prior=json.loads((source/'manifest.json').read_text())
    freeze_path=args.freeze.resolve()
    freeze=json.loads(freeze_path.read_text())
    _,cases=load_cases(freeze,freeze_path)
    lock_path=resolve_path(freeze_path.parent,freeze['asset_lock_file'])
    lock=json.loads(lock_path.read_text())
    if digest(freeze_path)!=prior['freeze_sha256'] or digest(lock_path)!=prior['asset_lock_sha256']:
        raise ValueError('Original freeze or asset lock changed')
    for relative,expected in prior['source_hashes'].items():
        if digest(ROOT/relative)!=expected:
            raise ValueError('Frozen source changed: '+relative)
        assert_committed(ROOT/relative)
    for path in [Path(__file__).resolve(),args.amendment.resolve()]:
        assert_committed(path)
    for relative,entry in json.loads((HERE.parent/'SOURCE_LOCK.json').read_text())['files'].items():
        if digest(HERE.parent/'upstream'/relative)!=entry['sha256']:
            raise ValueError('Upstream file changed: '+relative)
    dependency='utils/minGPT/utils.py'
    if digest(HERE/'cache'/dependency)!=json.loads((HERE/'ASSET_LOCK.json').read_text())['files'][dependency]['sha256']:
        raise ValueError('Upstream model dependency changed')
    if prior['status']!='failed' or prior['error']!="AssertionError('Gate-only intervention altered native within-bracket sign pattern')":
        raise ValueError('This amendment covers the documented gate failure only')
    if args.output.exists():
        raise ValueError('Refusing to overwrite output')
    args.output.mkdir(parents=True)
    completed={(r['task']['model_id'],r['task']['head']):r for r in prior['tasks']}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    manifest={**{k:prior[k] for k in ['freeze_sha256','cases_sha256','asset_lock_sha256','source_hashes','environment']},
              'status':'running','started_at':utcnow(),'tasks':[],
              'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'original_run':str(source),'original_manifest_sha256':digest(source/'manifest.json'),
              'amendment_sha256':digest(args.amendment),'continuation_code_sha256':digest(__file__),
              'fixed_error_family_K':len(freeze['tasks'])*4*2}
    dump(args.output/'manifest.json',manifest)
    try:
        for task in freeze['tasks']:
            key=(task['model_id'],task['head'])
            name=task['model_id']+'_head'+str(task['head'])
            out=args.output/'tasks'/name
            print('Processing',name,flush=True)
            if key in completed:
                entry=completed[key]
                for file,expected in entry['files'].items():
                    if digest(source/'tasks'/name/file)!=expected:
                        raise ValueError('Completed artifact changed: '+name+'/'+file)
                out.parent.mkdir(exist_ok=True,parents=True)
                shutil.copytree(source/'tasks'/name,out)
                result={**entry,'status':'completed','carried_from':str(source/'tasks'/name)}
            elif key==('0nbysgqs',2):
                if not (source/'tasks'/name/'stage1_receipt.json').exists():
                    raise ValueError('Documented failed task artifact missing')
                out.parent.mkdir(exist_ok=True,parents=True)
                shutil.copytree(source/'tasks'/name,out)
                result={'task':task,'status':'technical_invalidity','error':prior['error'],
                        'carried_from':str(source/'tasks'/name)}
            else:
                try:
                    result={**run_task(task,cases,lock,out,float(freeze['numerical_tolerance'])),'status':'completed'}
                except AssertionError as error:
                    out.mkdir(exist_ok=True,parents=True)
                    failure={'status':'technical_invalidity','error':repr(error),'traceback':traceback.format_exc()}
                    dump(out/'failure.json',failure)
                    result={'task':task,**failure}
            result['files']={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}
            manifest['tasks'].append(result)
            manifest['elapsed_seconds']=time.monotonic()-started
            dump(args.output/'manifest.json',manifest)
            print(result['status'],name,flush=True)
        assert len(manifest['tasks'])==len(freeze['tasks'])
        manifest['status']='completed_with_invalid_tasks'
    except Exception as error:
        manifest['status']='failed'
        manifest['error']=repr(error)
        raise
    finally:
        manifest['elapsed_seconds']=time.monotonic()-started
        manifest['finished_at']=utcnow()
        dump(args.output/'manifest.json',manifest)


if __name__=='__main__':
    main()
