"""Execute the bounded value-transfer development with recorded anchor forecasts."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / 'native_followup'
ROOT = HERE.parents[2]
sys.path.insert(0, str(NATIVE))
from confirm import assert_committed, load_runtime
from design import ANCHORS, predictions, validate_family
from value_runtime import run_value_transfers

TASK = {'model_id': 'a9g0io1r', 'n_layer': 2, 'n_head': 4, 'head': 1}
NUMERICAL_ALLOWANCE = .001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        handle.flush()


def now():
    return datetime.now(timezone.utc).isoformat()


def choose_recipient_positions(strings, native):
    """Highest native fp64 EOS attention on ')'; argmax gives the first tie."""
    if native.attention.dtype != torch.float64:
        raise ValueError('Recipient selection requires native float64 attention')
    positions = []
    for text, row in zip(strings, native.attention):
        closing = [j+1 for j,c in enumerate(text) if c == ')']
        if not closing:
            raise ValueError('No eligible closing bracket')
        positions.append(closing[int(torch.argmax(row[closing]))])
    return positions


def structural_receipt(anchor_rows):
    spec = importlib.util.spec_from_file_location('value_causal_preflight', ROOT/'examples/causal_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {'cells': [], 'predictions': {'H_state': [], 'H_position': []},
              'prediction_source': 'Fixed design formulas applied to previously measured anchor margins; no target-cell fit'}
    selected = []
    for row in anchor_rows:
        all_cells = {**{name: row['anchors'][name]['float64']['patched_margin'] for name in ANCHORS},
                     **row['predictions']['H_state']}
        for name in all_cells:
            cell_id = row['family_id'] + ':' + name
            config['cells'].append({'id': cell_id})
            if name in ANCHORS:
                selected.append(cell_id)
            for candidate in config['predictions']:
                value = row['anchors'][name]['float64']['patched_margin'] if name in ANCHORS else row['predictions'][candidate][name]
                config['predictions'][candidate].append(value)
    results = {}
    for name, choice in [('anchors_only', selected), ('with_declared_target_cells', None)]:
        result = module.check(config, selected=choice, structure_only=True)
        results[name] = {key: result[key] for key in ['scope', 'mode', 'identical_mean_groups', 'pairs', 'calibration']}
    return results


def forecast_precision(cells):
    """Check signed prediction-error differences, rather than only arm errors."""
    errors = {}
    for candidate in ['H_state', 'H_position']:
        for cell in predictions({a: 0. for a in ANCHORS})[candidate]:
            # Distinct marker values locate the fixed anchor used by each rule.
            marker = predictions({ANCHORS[0]: 0., ANCHORS[1]: 1.})[candidate][cell]
            anchor = ANCHORS[int(marker)]
            error32 = cells[cell]['float32']['patched_margin'] - cells[anchor]['float32']['patched_margin']
            error64 = cells[cell]['float64']['patched_margin'] - cells[anchor]['float64']['patched_margin']
            errors[candidate + ':' + cell] = abs(error32-error64)
    return errors


def source_bindings():
    source_paths = [HERE/'DEVELOPMENT.md', HERE/'design.py', HERE/'prepare.py', HERE/'analyze.py', HERE/'value_runtime.py', HERE/'run.py',
                    NATIVE/'runtime.py', NATIVE/'confirm.py', NATIVE/'qualify.py',
                    NATIVE/'frozen/confirmation_001/COHORT_ASSET_LOCK.json', NATIVE/'ASSET_LOCK.json',
                    HERE.parent/'SOURCE_LOCK.json', ROOT/'examples/causal_preflight.py', ROOT/'src/causal_decidability/design.py']
    for path in source_paths:
        assert_committed(path)
    for relative, entry in json.loads((HERE.parent/'SOURCE_LOCK.json').read_text())['files'].items():
        if sha(HERE.parent/'upstream'/relative) != entry['sha256']:
            raise ValueError('Upstream source mismatch: '+relative)
    dependency = 'utils/minGPT/utils.py'
    if sha(NATIVE/'cache'/dependency) != json.loads((NATIVE/'ASSET_LOCK.json').read_text())['files'][dependency]['sha256']:
        raise ValueError('Upstream dependency mismatch')
    return {str(path.relative_to(ROOT)): sha(path) for path in source_paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a run directory')
    families = [json.loads(line) for line in args.cases.read_text().splitlines()]
    if len(families) != 32 or any(f['phase'] != 'development' for f in families):
        raise ValueError('This runner is scoped to the declared 32-family development')
    if len({f['family_id'] for f in families}) != len(families):
        raise ValueError('Family identifiers must be unique')
    for family in families:
        validate_family(family)
    preparation_path = args.cases.parent/'preparation.json'
    exclusions_path = args.cases.parent/'exclusions.json'
    for path in (args.cases, preparation_path, exclusions_path):
        assert_committed(path)
    preparation = json.loads(preparation_path.read_text())
    if (preparation['phase'] != 'development' or preparation['seed'] != 25100201
            or preparation['families'] != 32 or preparation['cases_sha256'] != sha(args.cases)
            or preparation['generator_sha256'] != sha(HERE/'design.py')
            or preparation['exclusions_sha256'] != sha(exclusions_path)):
        raise ValueError('Input preparation differs from the declared development design')
    sources = source_bindings()
    sources.update({str(path.resolve().relative_to(ROOT)):sha(path)
                    for path in (args.cases, preparation_path, exclusions_path)})
    lock_path = NATIVE/'frozen/confirmation_001/COHORT_ASSET_LOCK.json'
    lock = json.loads(lock_path.read_text())
    checkpoint_relative = 'data/model_weights/run_a9g0io1r/run_a9g0io1r_checkpoint_5.pt'
    if sha(NATIVE/'cache'/checkpoint_relative) != lock['files'][checkpoint_relative]['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    manifest = {'status':'running', 'phase':'development', 'task':TASK, 'started_at':now(),
                'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'input_cases_sha256':sha(args.cases), 'source_hashes':sources,
                'checkpoint_sha256':lock['files'][checkpoint_relative]['sha256'],
                'numerical_allowance_on_prediction_error':NUMERICAL_ALLOWANCE,
                'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':1}}
    write_json(args.output/'manifest.json',manifest)
    controls, records = {}, []
    try:
        runtimes = {tag:load_runtime(TASK,lock,dtype) for tag,dtype in [('float32',torch.float32),('float64',torch.float64)]}
        recipients = [f['recipient'] for f in families]
        selection = runtimes['float64'].run(recipients)
        chosen = choose_recipient_positions(recipients,selection)
        write_json(args.output/'recipient_selection.json', [
            {'family_id':f['family_id'], 'recipient':f['recipient'], 'recipient_position':j,
             'native_margin':float(selection.margins[i]),
             'native_eos_attention':selection.attention[i].tolist()}
            for i,(f,j) in enumerate(zip(families,chosen))])
        manifest['selection_forward_batches'] = selection.forward_calls
        for family, position in zip(families,chosen):
            records.append({'family_id':family['family_id'],'phase':family['phase'],'recipient':family['recipient'],
                            'recipient_position':position,'cells':{}})
        for stage, anchor_only in [('anchors',True),('remaining_targets',False)]:
            if not anchor_only:
                manifest['target_measurement_started_at'] = now()
                write_json(args.output/'manifest.json',manifest)
            jobs = [(i,donor) for i,family in enumerate(families) for donor in family['donors']
                    if (donor['cell'] in ANCHORS) == anchor_only]
            recipient_strings = [families[i]['recipient'] for i,_ in jobs]
            donor_strings = [donor['string'] for _,donor in jobs]
            recipient_positions = [chosen[i] for i,_ in jobs]
            donor_positions = [donor['position'] for _,donor in jobs]
            controls[stage] = {}
            for tag,runtime in runtimes.items():
                result = run_value_transfers(runtime,recipient_strings,donor_strings,recipient_positions,donor_positions)
                controls[stage][tag] = result.controls
                for (i,donor),snapshot in zip(jobs,result.snapshots()):
                    cell = records[i]['cells'].setdefault(donor['cell'],{'donor_metadata':donor})
                    cell[tag] = snapshot
                print(stage,tag,'complete',flush=True)
            if anchor_only:
                anchor_rows=[]
                for record in records:
                    anchors = {name:record['cells'][name] for name in ANCHORS}
                    predicted = predictions({name:anchors[name]['float64']['patched_margin'] for name in ANCHORS})
                    record['predictions'] = predicted
                    anchor_rows.append({'family_id':record['family_id'],'recipient':record['recipient'],
                                        'recipient_position':record['recipient_position'],'anchors':anchors,'predictions':predicted})
                write_jsonl(args.output/'anchor_forecasts.jsonl',anchor_rows)
                receipt = {'written_before_target_execution_at':now(),'anchor_forecasts_sha256':sha(args.output/'anchor_forecasts.jsonl'),
                           'source_hashes':sources,'structural_check':structural_receipt(anchor_rows)}
                write_json(args.output/'anchor_receipt.json',receipt)
                manifest['anchor_forecasts_sha256'] = receipt['anchor_forecasts_sha256']
                manifest['phase_completed'] = 'anchors_and_forecasts_before_targets'
                write_json(args.output/'manifest.json',manifest)
        maximum_precision_error = 0.
        for record in records:
            record['prediction_error_dtype_discrepancies'] = forecast_precision(record['cells'])
            maximum_precision_error = max(maximum_precision_error,max(record['prediction_error_dtype_discrepancies'].values()))
        controls['maximum_prediction_error_dtype_discrepancy'] = maximum_precision_error
        controls['prediction_error_dtype_allowance'] = NUMERICAL_ALLOWANCE
        controls['precision_gate_passed'] = maximum_precision_error <= NUMERICAL_ALLOWANCE
        write_jsonl(args.output/'cases.jsonl',records)
        write_json(args.output/'controls.json',controls)
        if not controls['precision_gate_passed']:
            raise AssertionError('Paired precision discrepancy exceeds frozen prediction-error allowance')
        manifest['status'] = 'completed'
    except Exception as error:
        manifest['status'] = 'technical_failure'
        manifest['error'] = repr(error)
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic()-started
        manifest['finished_at'] = now()
        manifest['output_hashes'] = {path.name:sha(path) for path in sorted(args.output.iterdir()) if path.is_file() and path.name!='manifest.json'}
        write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':
    main()
