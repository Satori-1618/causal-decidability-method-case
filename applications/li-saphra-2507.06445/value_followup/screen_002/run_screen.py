"""Prospective native-margin screen; no transfer outcome selects a family."""
import argparse
import copy
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

HERE=Path(__file__).resolve().parent
VALUE=HERE.parent
sys.path.insert(0,str(VALUE))
from design import ANCHORS,predictions,validate_family,phase_of,prefix_state
from run import (TASK,NATIVE,ROOT,NUMERICAL_ALLOWANCE,choose_recipient_positions,forecast_precision,
                 now,sha,source_bindings,structural_receipt,write_json,write_jsonl)
from confirm import assert_committed,load_runtime
from value_runtime import run_value_transfers
from prepare_screen import SCREEN_SEED,DONOR_SEED,POOL_SIZE,PER_STRATUM

MARGIN_THRESHOLD=8.
ANCHOR_GAP_THRESHOLD=.202


def select_indices(margins,per_stratum=PER_STRATUM):
    return {name:[i for i,value in enumerate(margins) if (value<MARGIN_THRESHOLD)==wanted][:per_stratum]
            for name,wanted in [('accepted',True),('rejected',False)]}


def read_inputs(directory):
    prep=json.loads((directory/'preparation.json').read_text())
    if (prep['round']!='screen_002' or prep['screen_seed']!=SCREEN_SEED or prep['donor_seed']!=DONOR_SEED
            or prep['candidate_pool_size']!=POOL_SIZE or prep['per_stratum']!=PER_STRATUM):
        raise ValueError('Input preparation does not match the fixed screen')
    for name,expected in prep['files'].items():
        if sha(directory/name)!=expected:
            raise ValueError('Prepared input hash mismatch: '+name)
    if prep['sources']!={'prepare_screen.py':sha(HERE/'prepare_screen.py'),'design.py':sha(VALUE/'design.py')}:
        raise ValueError('Generator differs from prepared source')
    candidates=[json.loads(line) for line in (directory/'candidates.jsonl').read_text().splitlines()]
    donors=[json.loads(line) for line in (directory/'donor_families.jsonl').read_text().splitlines()]
    exclusions=json.loads((directory/'exclusions.json').read_text())
    banned=set(exclusions['recipient_strings'])
    old_path=VALUE/'inputs/development_001/cases.jsonl'
    if exclusions['source_hashes']['value_followup/inputs/development_001/cases.jsonl']!=sha(old_path):
        raise ValueError('Prior development input changed after exclusion preparation')
    old=[json.loads(line) for line in old_path.read_text().splitlines()]
    for family in old:
        for text in [family['recipient']]+[donor['string'] for donor in family['donors']]:
            if text not in banned or any(text[:p] not in exclusions['prefixes'][str(p)] for p in [20,28]):
                raise ValueError('An old donor or recipient prefix was not excluded at both positions')
    if len(candidates)!=POOL_SIZE or len(donors)!=2*PER_STRATUM:
        raise ValueError('Unexpected pool or donor-template size')
    for index,row in enumerate(candidates):
        text=row['recipient']
        if (row['candidate_index']!=index or len(text)!=32 or text.count('(')!=16 or set(text)!={'(',')'}
                or text in banned or phase_of(text)!='development' or prefix_state(text)[1]>=0):
            raise ValueError('Candidate is outside the declared population/order')
    if len({r['candidate_id'] for r in candidates})!=len(candidates):
        raise ValueError('Candidate identifiers must be unique')
    for family in donors:
        validate_family(family)
        if family['phase']!='development':
            raise ValueError('Donor template phase must remain development')
        for donor in family['donors']:
            if donor['string'][:donor['position']] in exclusions['prefixes'][str(donor['position'])]:
                raise ValueError('Previously exposed donor prefix')
    return prep,candidates,donors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing overwrite')
    prep,candidates,donor_templates=read_inputs(args.inputs)
    sources=source_bindings()
    extra=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'REVIEW_PRE_RUN.md',VALUE/'verify.py',
          VALUE/'inputs/development_001/exclusions.json']+list(args.inputs.iterdir())
    for path in extra:
        if path.is_file():
            assert_committed(path)
            sources[str(path.resolve().relative_to(ROOT))]=sha(path)
    # The old input manifest is itself bound, not merely the derived ban list.
    old_input=VALUE/'inputs/development_001/cases.jsonl'
    assert_committed(old_input)
    sources[str(old_input.relative_to(ROOT))]=sha(old_input)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock=json.loads((NATIVE/'frozen/confirmation_001/COHORT_ASSET_LOCK.json').read_text())
    started=time.monotonic()
    manifest={'status':'running','round':'screen_002','task':TASK,'started_at':now(),
              'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'source_hashes':sources,'preparation':prep,'native_margin_threshold':MARGIN_THRESHOLD,
              'anchor_gap_threshold':ANCHOR_GAP_THRESHOLD,'prediction_error_allowance':NUMERICAL_ALLOWANCE,
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':1}}
    write_json(args.output/'manifest.json',manifest)
    controls={}
    try:
        runtimes={tag:load_runtime(TASK,lock,dtype) for tag,dtype in [('float32',torch.float32),('float64',torch.float64)]}
        strings=[c['recipient'] for c in candidates]
        native={tag:runtime.run(strings) for tag,runtime in runtimes.items()}
        margins={tag:result.margins.tolist() for tag,result in native.items()}
        selected=select_indices(margins['float64'])
        chosen=choose_recipient_positions(strings,native['float64'])
        selected_set=set(selected['accepted']+selected['rejected'])
        screening=[]
        for i,candidate in enumerate(candidates):
            screening.append({**candidate,'margin_float32':margins['float32'][i],'margin_float64':margins['float64'][i],
                              'screen_accept_float32':margins['float32'][i]<MARGIN_THRESHOLD,
                              'screen_accept_float64':margins['float64'][i]<MARGIN_THRESHOLD,
                              'stratum':'accepted' if margins['float64'][i]<MARGIN_THRESHOLD else 'rejected',
                              'selected':i in selected_set,'recipient_position':chosen[i]})
        write_jsonl(args.output/'screening.jsonl',screening)
        margin_error=max(abs(a-b) for a,b in zip(margins['float32'],margins['float64']))
        mismatches=sum(row['screen_accept_float32']!=row['screen_accept_float64'] for row in screening)
        controls['native_screen']={'max_dtype_margin_difference':margin_error,'dtype_classification_mismatches':mismatches,
                                  'forward_batches':{tag:r.forward_calls for tag,r in native.items()},
                                  'cases_per_dtype':len(candidates),'selected_counts':{s:len(v) for s,v in selected.items()},
                                  'pool_counts':{s:sum(row['stratum']==s for row in screening) for s in selected}}
        write_json(args.output/'controls.json',controls)
        if margin_error>NUMERICAL_ALLOWANCE or mismatches:
            raise AssertionError('Native screen precision/classification gate failed; do not discard cases')
        if any(len(v)!=PER_STRATUM for v in selected.values()):
            manifest['status']='insufficient_screening_stratum_no_transfers'
            return
        families=[]
        for template,(stratum,index) in zip(donor_templates,[(s,i) for s in ['accepted','rejected'] for i in selected[s]]):
            family=copy.deepcopy(template)
            family.update({'donor_template_id':template['family_id'],'family_id':candidates[index]['candidate_id'],
                           'candidate_id':candidates[index]['candidate_id'],'candidate_index':index,'stratum':stratum,
                           'recipient':strings[index],'recipient_position':chosen[index],
                           'margin_float32':margins['float32'][index],'margin_float64':margins['float64'][index]})
            validate_family(family)
            families.append(family)
        write_jsonl(args.output/'selected_families.jsonl',families)
        write_json(args.output/'recipient_selection.json',[
            {'family_id':family['family_id'],'recipient':family['recipient'],
             'recipient_position':family['recipient_position'],
             'native_margin':margins['float64'][family['candidate_index']],
             'native_eos_attention':native['float64'].attention[family['candidate_index']].tolist()}
            for family in families])
        write_json(args.output/'screening_receipt.json',{'written_before_transfers_at':now(),
                   'screening_sha256':sha(args.output/'screening.jsonl'),
                   'selected_families_sha256':sha(args.output/'selected_families.jsonl'),
                   'recipient_selection_sha256':sha(args.output/'recipient_selection.json'),
                   'source_hashes':sources,'selection_rule':'First32 per frozen stratum in original candidate order'})
        del native
        records=[{key:family[key] for key in ['family_id','candidate_id','candidate_index','stratum','recipient','recipient_position','donor_template_id','margin_float32','margin_float64']}
                 for family in families]
        for record in records:
            record['cells']={}
        for stage in ['anchors','accepted_targets']:
            manifest[('anchors' if stage=='anchors' else 'target')+'_measurement_started_at']=now()
            write_json(args.output/'manifest.json',manifest)
            jobs=[(i,donor) for i,family in enumerate(families) for donor in family['donors']
                  if (stage=='anchors' and donor['cell'] in ANCHORS) or
                     (stage=='accepted_targets' and family['stratum']=='accepted' and donor['cell'] not in ANCHORS)]
            controls[stage]={}
            for tag,runtime in runtimes.items():
                result=run_value_transfers(runtime,[families[i]['recipient'] for i,_ in jobs],
                                          [d['string'] for _,d in jobs],[families[i]['recipient_position'] for i,_ in jobs],
                                          [d['position'] for _,d in jobs])
                controls[stage][tag]=result.controls
                for (i,donor),snapshot in zip(jobs,result.snapshots()):
                    records[i]['cells'].setdefault(donor['cell'],{'donor_metadata':donor})[tag]=snapshot
                print(stage,tag,'complete',flush=True)
            if stage=='anchors':
                gap_error=0.;gap_mismatches=0
                anchor_forecasts=[]
                for record in records:
                    values={tag:[record['cells'][name][tag]['patched_margin'] for name in ANCHORS] for tag in runtimes}
                    contrasts={tag:v[1]-v[0] for tag,v in values.items()}
                    gap_error=max(gap_error,abs(contrasts['float32']-contrasts['float64']))
                    gap_mismatches+=((abs(contrasts['float32'])>ANCHOR_GAP_THRESHOLD)!=(abs(contrasts['float64'])>ANCHOR_GAP_THRESHOLD))
                    record['anchor_gap_float32']=abs(contrasts['float32'])
                    record['anchor_gap_float64']=abs(contrasts['float64'])
                    record['anchor_separating']=abs(contrasts['float64'])>ANCHOR_GAP_THRESHOLD
                    if record['stratum']=='accepted':
                        predicted=predictions(dict(zip(ANCHORS,values['float64'])))
                        record['predictions']=predicted
                        anchor_forecasts.append({'family_id':record['family_id'],'anchors':{name:record['cells'][name] for name in ANCHORS},'predictions':predicted})
                controls['anchor_precision']={'maximum_signed_contrast_dtype_difference':gap_error,
                                              'separation_boundary_straddles':gap_mismatches}
                write_jsonl(args.output/'anchor_cases.jsonl',records)
                write_json(args.output/'controls.json',controls)
                if gap_error>NUMERICAL_ALLOWANCE or gap_mismatches:
                    raise AssertionError('Anchor precision or separation-boundary gate failed; no replacements')
                write_jsonl(args.output/'anchor_forecasts.jsonl',anchor_forecasts)
                write_json(args.output/'anchor_receipt.json',{'written_before_target_execution_at':now(),
                           'anchor_forecasts_sha256':sha(args.output/'anchor_forecasts.jsonl'),
                           'anchor_cases_sha256':sha(args.output/'anchor_cases.jsonl'),
                           'structural_check':structural_receipt(anchor_forecasts)})
        maximum=0.
        for record in records:
            if record['stratum']=='accepted':
                record['prediction_error_dtype_discrepancies']=forecast_precision(record['cells'])
                maximum=max(maximum,max(record['prediction_error_dtype_discrepancies'].values()))
        controls['target_precision']={'maximum_prediction_error_dtype_difference':maximum,'gate_passed':maximum<=NUMERICAL_ALLOWANCE}
        write_jsonl(args.output/'cases.jsonl',records)
        write_json(args.output/'controls.json',controls)
        if maximum>NUMERICAL_ALLOWANCE:
            raise AssertionError('Secondary forecast precision gate failed')
        manifest['status']='completed'
    except Exception as error:
        manifest['status']='technical_failure'
        manifest['error']=repr(error)
        raise
    finally:
        manifest['elapsed_seconds']=time.monotonic()-started
        manifest['finished_at']=now()
        manifest['output_hashes']={p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='manifest.json'}
        write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':
    main()
