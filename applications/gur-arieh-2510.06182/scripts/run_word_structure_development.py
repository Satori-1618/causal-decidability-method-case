"""Run/check the fixed 32-family word-versus-target-site development study offline.

No confirmation is run. Existing Round 1 code and artifacts remain hash-bound.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[1]
sys.path.insert(0, str(APP/'src'))
from mixing_word_structure import paired_measures, summarize


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def index_files(out):
    data = {p.name: sha(p) for p in sorted(out.iterdir())
            if p.is_file() and p.name != 'artifact_hashes.json'}
    save(out/'artifact_hashes.json', data)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_logits(out, records):
    """Independently reconstruct the saved readout from the full-vocabulary audit."""
    import math
    import numpy as np
    prior = json.loads((APP/'FREEZE.json').read_text())
    ids = prior['entity_pools']['answer_form_ids']
    expected = {}
    for record in records[:2]:
        if record['qualifies']:
            for i, arm in enumerate(record['arms']):
                indices = [ids[row[1]] for row in arm['matrix']]
                for kind, readout in [('conflict', arm['patch']),
                                      ('native', arm['native']['recipient']['readout'])]:
                    expected[f"{record['case_id']}_{i}_{kind}"] = (indices, readout)
    errors = {'logits':0.0, 'mass':0.0}
    with np.load(out/'audit_full_logits.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected):
            raise ValueError('full-logit audit coverage mismatch')
        for key,(indices,readout) in expected.items():
            raw=arrays[key].astype(np.float64)
            logsum=raw.max()+math.log(float(np.exp(raw-raw.max()).sum()))
            selected=raw[indices]
            errors['logits']=max(errors['logits'],float(np.max(np.abs(selected-readout['answer_logits']))))
            errors['mass']=max(errors['mass'],abs(float(np.exp(selected-logsum).sum())-readout['answer_mass_full_vocab']))
    if errors['logits'] > 1e-6 or errors['mass'] > 1e-5:
        raise ValueError('full-logit audit differs from readout')
    return {'passed':True, 'arrays':len(expected), 'max_errors':errors}


def decision(records, references, manifest):
    """Development gates; no profile-adquacy/exclusion claims from these data."""
    rule = manifest['rule']
    n = manifest['families']
    if len(records) != n or [r['seed'] for r in records] != list(range(manifest['seed_base'], manifest['seed_base']+n)):
        raise ValueError('incomplete or unexpected seed sequence')
    result = summarize(records, rule)
    qualified = [r for r in records if r['qualifies']]
    positive = [a['agreement'] for r in qualified for a in r['arms']]
    transfer = sum(a['transfer'] for a in positive)/len(positive) if positive else 0
    pos_mass = sum(a['readout']['answer_mass_full_vocab'] >= rule['answer_mass_floor']
                   for a in positive)/len(positive) if positive else 0
    expected = records[:manifest['cpu_reference_families']]
    cpu_complete = [r['case_id'] for r in references] == [r['case_id'] for r in expected]
    diffs, classifications = [], []
    for main, cpu in zip(expected, references):
        classifications.append(main['qualifies'] == cpu['qualifies'])
        if main['qualifies'] and cpu['qualifies']:
            a,b = paired_measures(main,rule), paired_measures(cpu,rule)
            diffs.extend(abs(x['delta']-y['delta']) for x,y in zip(a['arms'],b['arms']))
            classifications.append(a['label'] == b['label'])
    maxdiff = max(diffs, default=None)
    gates = {
        'technical': all(r['technical'] and all(r['technical'].values()) for r in records+references),
        'native_yield': result['yield'] >= manifest['gates']['native_yield_floor'],
        'agreement_transfer': transfer >= manifest['gates']['agreement_transfer_floor'],
        'agreement_mass': pos_mass >= manifest['gates']['agreement_transfer_floor'],
        'cpu_fp32': cpu_complete and bool(diffs) and all(classifications)
                    and maxdiff <= manifest['gates']['cpu_delta_tolerance_nats'],
    }
    result.update(gates=gates, technical_status='VALID' if all(gates.values()) else 'STOP',
                  agreement_transfer_rate=transfer, agreement_mass_rate=pos_mass,
                  max_cpu_delta_difference_nats=maxdiff,
                  planning_note='Development only. Freeze a fresh confirmation only after reviewing resolution and rival separation.')
    return result


def check(out):
    """Replay stored evidence; no model imports. New code does not alter Round 1."""
    index = json.loads((out/'artifact_hashes.json').read_text())
    actual = {p.name for p in out.iterdir() if p.is_file()}-{'artifact_hashes.json'}
    if actual != set(index) or any(sha(out/name) != digest for name,digest in index.items()):
        raise ValueError('artifact coverage/hash mismatch')
    manifest = json.loads((out/'manifest.json').read_text())
    if sha(APP/'FREEZE.json') != manifest['round1_freeze_sha256']:
        raise ValueError('Round 1 freeze changed')
    for name,digest in manifest['code_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('changed code: '+name)
    start = json.loads((out/'RUN_STARTED.json').read_text())
    if start['manifest_sha256'] != sha(out/'manifest.json'):
        raise ValueError('manifest changed')
    committed=subprocess.check_output(['git','show',start['git_head']+':'+
                         'applications/gur-arieh-2510.06182/ROUND2_DEVELOPMENT.json'],cwd=ROOT)
    if committed != (out/'manifest.json').read_bytes():
        raise ValueError('start commit does not contain this manifest')
    records, refs = read_jsonl(out/'records.jsonl'), read_jsonl(out/'cpu_reference.jsonl')
    # A second arithmetic expression checks baseline subtraction and swap sign.
    for r in records:
        from mixing_word_structure import swap_pair
        expected=swap_pair(r['arms'][0]['matrix'],r['cell'])
        for arm,case in zip(r['arms'],expected):
            if arm['matrix'] != [list(row) for row in case['recipient']] or arm['donor_matrix'] != [list(row) for row in case['donor']]:
                raise ValueError('stored matrix does not implement the declared swap')
        qualifying=all(arm['native'][role]['correct'] and arm['native'][role]['readout_matches_generation']
                       and arm['native'][role]['readout']['answer_mass_full_vocab'] >= 0.5
                       for arm in r['arms'] for role in ('recipient','donor'))
        if qualifying != r['qualifies']:
            raise ValueError('qualification differs from native records')
        if not r['qualifies']:
            continue
        measured = paired_measures(r,manifest['rule'])
        p,l = r['cell']['i_P'],r['cell']['i_L']
        for arm,got in zip(r['arms'],measured['arms']):
            patched = arm['patch']['answer_logits']
            native = arm['native']['recipient']['readout']['answer_logits']
            independent = (patched[p]-native[p])-(patched[l]-native[l])
            if abs(independent-got['delta']) > 1e-10:
                raise ValueError('margin identity mismatch')
    computed = decision(records,refs,manifest)
    computed['full_logit_audit']=audit_logits(out,records)
    if computed != json.loads((out/'summary.json').read_text()):
        raise ValueError('summary differs from records')
    return {key:computed[key] for key in ('stage','technical_status','generated','qualified','resolved','counts')}


def run(args):
    import numpy as np
    import torch
    import run_mixing_confirmation as frozen
    from mixing_prompts import load_adapter
    from mixing_round1_design import random_matrix
    from mixing_runner import Runner, load_model, round1_pools, pools_record
    from mixing_word_structure_runner import run_pair
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest['stage'] != 'DEVELOPMENT_ONLY' or git('status','--porcelain'):
        raise ValueError('requires development manifest and clean committed tree')
    committed = subprocess.check_output(['git','show','HEAD:'+str(manifest_path.relative_to(ROOT))],cwd=ROOT)
    if committed != manifest_path.read_bytes():
        raise ValueError('manifest is not committed')
    if args.output.exists():
        raise ValueError('output exists; never overwrite a development run')
    for path,digest in manifest['code_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('source hash mismatch: '+path)
    for name,version in manifest['runtime'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('runtime mismatch: '+name)
    prior_path = APP/'FREEZE.json'
    if sha(prior_path) != manifest['round1_freeze_sha256']:
        raise ValueError('Round 1 freeze changed')
    prior = json.loads(prior_path.read_text())
    hashes = frozen.verify_frozen_hashes(prior, cache_root=args.cache)
    adapter = load_adapter()
    if adapter.check(args.upstream, hashes['source_lock']) != prior['upstream']:
        raise ValueError('upstream changed')
    if not torch.backends.mps.is_available():
        raise ValueError('declared MPS backend is unavailable')
    args.output.mkdir(parents=True)
    out = args.output
    save(out/'manifest.json',manifest)
    save(out/'RUN_STARTED.json',{'started':now(),'git_head':git('rev-parse','HEAD'),
                                 'git_dirty':False,'manifest_sha256':sha(manifest_path),
                                 'model_and_round1_hashes_verified':hashes['counts']})
    start = time.perf_counter()
    try:
        model,tokenizer = load_model(hashes['snapshot'],torch.float32,'mps','eager')
        spec = prior['task_spec']
        pools,dropped,contexts,answers = round1_pools(tokenizer,spec)
        if pools_record(pools,dropped,contexts,answers,spec) != prior['entity_pools']:
            raise ValueError('token pools changed')
        runner = Runner(model,tokenizer,spec,pools,contexts,answers,prior['layer'])
        records, arrays = [], {}
        with (out/'records.jsonl').open('x') as handle:
            for i in range(manifest['families']):
                seed=manifest['seed_base']+i
                matrix = random_matrix(prior['n_groups'],[pools[c] for c in spec['categories']],random.Random(seed))
                target=prior['cell']['i_P' if i%2==0 else 'i_L']
                record,full = run_pair(runner,matrix,prior['cell'],case_id=f'word-structure-{i:03d}',
                                      seed=seed,control_target=target,audit=i<2)
                handle.write(json.dumps(record,allow_nan=False)+'\n'); handle.flush()
                records.append(record); arrays.update(full)
                print(f"{i+1}/{manifest['families']} native-qualified={record['qualifies']} {record['runtime_seconds']:.1f}s",flush=True)
        np.savez_compressed(out/'audit_full_logits.npz',**arrays)
        del runner,model
        torch.mps.empty_cache()
        cpu_model,_ = load_model(hashes['snapshot'],torch.float32,'cpu','eager')
        cpu_runner=Runner(cpu_model,tokenizer,spec,pools,contexts,answers,prior['layer'])
        references=[]
        with (out/'cpu_reference.jsonl').open('x') as handle:
            for record in records[:manifest['cpu_reference_families']]:
                reference,_=run_pair(cpu_runner,record['arms'][0]['matrix'],prior['cell'],
                                     case_id=record['case_id'],seed=record['seed'],
                                     control_target=record['control_target'])
                references.append(reference)
                handle.write(json.dumps(reference,allow_nan=False)+'\n');handle.flush()
        result=decision(records,references,manifest)
        result['full_logit_audit']=audit_logits(out,records)
        save(out/'summary.json',result)
        save(out/'timings.json',{'seconds':time.perf_counter()-start,'finished':now()})
        index_files(out)
        return check(out)
    except Exception as error:
        save(out/'FAILED.json',{'type':type(error).__name__,'error':str(error),'time':now()})
        if not (out/'artifact_hashes.json').exists():
            index_files(out)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','check'])
    parser.add_argument('--manifest',default=APP/'ROUND2_DEVELOPMENT.json')
    parser.add_argument('--upstream',type=Path,default=Path('/Users/felixb/causal-decidability-deps/mixing-mechs'))
    parser.add_argument('--cache',type=Path,default=Path.home()/'.cache/huggingface/hub')
    parser.add_argument('--output',type=Path,default=APP/'results/round2_word_structure_development')
    args=parser.parse_args()
    print(json.dumps(run(args) if args.action=='run' else check(args.output),indent=2))


if __name__=='__main__':
    main()
