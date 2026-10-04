"""Independent, stdlib-only audit of one completed averaged-anchor run; no inference."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
import subprocess

REVIEW = '80085bbd156fd2a3cfe56662b748a7d10ccf7227'
RELEASE = '5cb6f770494a2e42bfdfeaaa3206bc9beab3f75a'
REL = Path('applications/li-saphra-2507.06445/value_followup/averaged_anchors_004')
DTYPES = ('float32', 'float64')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def rows(path): return [json.loads(x) for x in path.read_text().splitlines()]
def git(root, *args): return subprocess.check_output(['git', *args], cwd=root)
def cast(value, dtype): return struct.unpack('f', struct.pack('f', value))[0] if dtype == 'float32' else value

def audit(root, run):
    h=root/REL
    plan=read(h/'plan.json'); manifest=read(run/'manifest.json'); release=read(h/'EXECUTION_RELEASE.json')
    assert manifest['status']=='completed' and manifest['git_head']==RELEASE and manifest['reviewed_commit']==REVIEW
    assert release['final_review']['reviewed_commit']==REVIEW and release['status']=='approved'
    changed=git(root,'diff','--name-only',REVIEW,RELEASE).decode().splitlines()
    assert set(changed)=={str(REL/name) for name in ('README.md','REVIEW.md','EXECUTION_RELEASE.json','SOURCE_LOCK.json','plan.json')}
    reviewed_plan=json.loads(git(root,'show',REVIEW+':'+str(REL/'plan.json')))
    assert {k:v for k,v in plan.items() if k not in ('status','execution_authorized','provenance')}=={k:v for k,v in reviewed_plan.items() if k not in ('status','execution_authorized','provenance')}
    for name, expected in manifest['source_hashes'].items():
        assert digest(root/name)==expected
        assert hashlib.sha256(git(root,'show',RELEASE+':'+name)).hexdigest()==expected
    assert manifest['plan_sha256']==digest(h/'plan.json') and manifest['execution_release_sha256']==digest(h/'EXECUTION_RELEASE.json')
    for name, expected in manifest['output_hashes'].items(): assert digest(run/name)==expected
    assert manifest['preparation']==read(h/'inputs/preparation.json')
    for name, expected in manifest['preparation']['files'].items(): assert digest(h/'inputs'/name)==expected
    for name in ('preparation.json','candidates.jsonl','donor_families.jsonl','exclusions.json'):
        assert (h/'inputs'/name).read_bytes()==git(root,'show',REVIEW+':'+str(REL/'inputs'/name))

    screen=rows(run/'screening.jsonl'); candidates=rows(h/'inputs/candidates.jsonl')
    assert len(screen)==len(candidates)==2048
    accepted=[]
    for i,(s,c) in enumerate(zip(screen,candidates)):
        assert all(s[k]==v for k,v in c.items())
        for dtype in DTYPES: assert s['screen_accept_'+dtype] == (s['margin_'+dtype]<8)
        assert s['screen_accept_float32']==s['screen_accept_float64']
        assert abs(s['margin_float32']-s['margin_float64']) <= .001
        if s['screen_accept_float64']: accepted.append(i)
    first=accepted[:256]
    assert len(first)==256 and [i for i,s in enumerate(screen) if s['selected']]==first
    selected=rows(run/'selected_families.jsonl'); templates=rows(h/'inputs/donor_families.jsonl')
    attention=read(run/'recipient_selection.json')
    assert len(selected)==len(templates)==len(attention)==256
    for s, i, donor, att in zip(selected,first,templates,attention):
        assert s['family_id']==candidates[i]['candidate_id'] and s['candidate_index']==i
        assert s['recipient']==candidates[i]['recipient'] and s['donor_template_id']==donor['family_id'] and s['donors']==donor['donors']
        possible=[j+1 for j,char in enumerate(s['recipient']) if char==')']
        pos=max(possible,key=lambda j:att['native_eos_attention'][j])
        assert pos==s['recipient_position']==att['recipient_position'] and att['family_id']==s['family_id']

    sr=read(run/'screening_receipt.json'); fr=read(run/'forecast_receipt.json')
    for receipt in (sr,fr): assert receipt['source_hashes']==manifest['source_hashes']
    for key,name in [('screening_sha256','screening.jsonl'),('selected_families_sha256','selected_families.jsonl'),('recipient_selection_sha256','recipient_selection.json')]: assert sr[key]==digest(run/name)
    for key,name in [('forecasts_sha256','forecasts.jsonl'),('descriptive_forecasts_sha256','descriptive_forecasts.jsonl'),('calibration_cases_sha256','calibration_cases.jsonl'),('screening_receipt_sha256','screening_receipt.json')]: assert fr[key]==digest(run/name)
    assert datetime.fromisoformat(release['released_at']) < datetime.fromisoformat(manifest['started_at'])
    assert sr['written_before_calibration_at'] <= manifest['calibration_measurement_started_at'] < fr['written_before_targets_at'] <= manifest['target_measurement_started_at'] < manifest['finished_at']

    cases=rows(run/'cases.jsonl'); calibration=rows(run/'calibration_cases.jsonl'); forecasts=rows(run/'forecasts.jsonl'); desc=rows(run/'descriptive_forecasts.jsonl')
    assert len(cases)==len(calibration)==len(forecasts)==len(desc)==256
    numeric_cells=0; separation=0; max_formula_error=0.; precision_gap=0.
    for case,cal,fore,ds,sel in zip(cases,calibration,forecasts,desc,selected):
        assert case['family_id']==cal['family_id']==fore['family_id']==ds['family_id']==sel['family_id']
        assert len(cal['cells'])==8 and all(k.startswith('calibration_') for k in cal['cells'])
        assert len(case['cells'])==16
        for key,value in cal['cells'].items(): assert case['cells'][key]==value
        donor_lookup={d['cell']:d for d in sel['donors']}
        assert set(case['cells'])==set(donor_lookup)
        baseline={}
        for cell,entry in case['cells'].items():
            assert entry['donor_metadata']==donor_lookup[cell]
            for dtype in DTYPES:
                r=entry[dtype]; numeric_cells+=1
                assert r['recipient_string']==sel['recipient'] and r['donor_string']==donor_lookup[cell]['string']
                assert r['recipient_position']==sel['recipient_position'] and r['donor_position']==donor_lookup[cell]['position']
                assert r['native_margin']==sel['margin_'+dtype]
                native=(r['a_r'],r['v_r'],r['h_r'],r['native_margin'])
                if dtype in baseline: assert baseline[dtype]==native
                else: baseline[dtype]=native
                assert all(len(r[key])==16 for key in ('v_r','v_d','h_r','h_patch_intended','h_patch_delivered','requested_node_delta','delivered_node_delta'))
                delta=[cast(r['a_r']*cast(d-v,dtype),dtype) for d,v in zip(r['v_d'],r['v_r'])]
                intended=[cast(v+d,dtype) for v,d in zip(r['h_r'],delta)]
                assert delta==r['requested_node_delta']
                assert intended==r['h_patch_intended']==r['h_patch_delivered']
                assert [cast(a-b,dtype) for a,b in zip(intended,r['h_r'])]==r['delivered_node_delta']
                assert cast(r['patched_margin']-r['native_margin'],dtype)==r['margin_change']
        computed={}; gaps={}; dcomputed={}
        for dtype in DTYPES:
            c={k:cell[dtype]['patched_margin'] for k,cell in cal['cells'].items()}
            means={(s,p):(c[f'calibration_{s}_{p}_0']+c[f'calibration_{s}_{p}_1'])/2 for s in ('neg','pos') for p in (20,28)}
            pr={name:{} for name in ('B_avg','P_avg','B_single4','B_legacy2')};dp={}
            for s in ('neg','pos'):
                for p in (20,28):
                    for rep in (0,1):
                        key=f'target_{s}_{p}_{rep}'
                        pr['B_avg'][key]=(means[s,20]+means[s,28])/2
                        pr['P_avg'][key]=(means['neg',p]+means['pos',p])/2
                        pr['B_single4'][key]=(c[f'calibration_{s}_20_0']+c[f'calibration_{s}_28_0'])/2
                        pr['B_legacy2'][key]=c['calibration_neg_20_0' if s=='neg' else 'calibration_pos_28_0']
                        dp[key]=means[s,p]
            computed[dtype]=pr;dcomputed[dtype]=dp
            gaps[dtype]=max(abs(pr['B_avg'][k]-pr['P_avg'][k]) for k in pr['B_avg'])
        assert computed==fore['forecasts']==case['forecasts'] and dcomputed==ds['forecasts']
        assert (gaps['float32']>.202)==(gaps['float64']>.202)==case['separating']==fore['separating']
        assert all(gaps[tag]==case['separation_gap_'+tag] for tag in DTYPES)
        assert max(abs((computed['float32']['B_avg'][k]-computed['float32']['P_avg'][k])-(computed['float64']['B_avg'][k]-computed['float64']['P_avg'][k])) for k in computed['float64']['B_avg'])<=.001
        separation+=case['separating']
    assert separation==151==manifest['separating_families'] and manifest['target_start_passed']
    assert fr['start_rule']['separating']==separation and fr['start_rule']['start_targets'] and separation>=128

    events=rows(run/'forward_events.jsonl'); pending={}; totals=Counter(); batch=Counter(); times={};prev=''
    for event in events:
        assert event['at']>=prev;prev=event['at']
        assert event['stage'] in ('native','calibration','target') and event['dtype'] in DTYPES
        key=(event['stage'],event['dtype']);ek=(event['event'],*key)
        totals[ek]+=event['sequences'];batch[ek]+=1
        if event['event']=='started':
            assert event['dtype'] not in pending;pending[event['dtype']]=(event['stage'],event['sequences'])
        else:
            assert event['event']=='completed' and pending.pop(event['dtype'])==(event['stage'],event['sequences'])
        times.setdefault(event['stage'],[]).append(event['at'])
    assert not pending
    for stage, expected in [('native',2048),('calibration',8192),('target',8192)]:
        for dtype in DTYPES:
            assert totals['started',stage,dtype]==totals['completed',stage,dtype]==expected
    assert max(times['native'])<=sr['written_before_calibration_at']<=min(times['calibration'])
    assert max(times['calibration'])<=fr['written_before_targets_at']<=min(times['target'])
    cost=manifest['cost']
    assert sum(v for k,v in totals.items() if k[0]=='started')==cost['sequence_forwards_attempted']==36864
    assert sum(v for k,v in totals.items() if k[0]=='completed')==cost['sequence_forwards_completed']==36864
    assert sum(v for k,v in batch.items() if k[0]=='started')==cost['forward_batches_attempted']==576
    assert sum(v for k,v in batch.items() if k[0]=='completed')==cost['forward_batches_completed']==576
    assert cost['failed_forward_internal_progress_unknown'] is False
    controls=read(run/'controls.json')
    assert controls['native_screen']['accepted_pool_count']==len(accepted)==886
    for stage in ('calibration','target'):
        for dtype in DTYPES:
            c=controls[stage][dtype]
            assert c['identity_max_margin_error']==c['identity_max_node_error']==0.
            assert all(c[k] is True for k in ('inserted_node_exactly_intended_after_dtype','all_target_layer_attention_weights_unchanged','all_target_layer_value_projections_unchanged','nontarget_head_and_query_preprojection_exactly_unchanged'))
            assert c['recipient_and_donor_native_forward_batches']==64 and c['identity_forward_batches']==c['transfer_forward_batches']==32
    return {'status':'PASS','reviewed_commit':REVIEW,'released_run_commit':RELEASE,
       'release_scientific_changes':0,'source_hashes_verified':len(manifest['source_hashes']),
       'output_hashes_verified':len(manifest['output_hashes']),'input_files_unchanged_from_review':4,
       'screened_candidates':len(screen),'accepted_candidates':len(accepted),'selected_first_accepted_families':len(selected),
       'calibration_families':len(calibration),'separating_families':separation,'target_families':len(cases),
       'numeric_transfer_records_with_exact_reconstructed_delivered_nodes':numeric_cells,
       'calibration_only_original_and_C_forecasts_reproduced':True,
       'screening_receipt_precedes_calibration':True,'forecast_receipt_precedes_target':True,
       'sequence_forwards_attempted':36864,'sequence_forwards_completed':36864,
       'forward_batches_attempted':576,'forward_batches_completed':576,
       'ledger_has_unfinished_forward':False,'reported_identity_error':0.,
       'limitations':['Raw snapshots independently support local node construction/delivery and fixed native inputs. Full attention/value/off-target preservation and native-node identity are producer controls; complete tensors for those checks are not serialized.',
        'This audit is arithmetic/provenance review by another agent, not an independently executed model replication or independently timestamped data collection.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.repo.resolve(),args.run.resolve()),indent=2,sort_keys=True))
