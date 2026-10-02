"""Evaluate the declared value-transfer forecasts without fitting any candidate."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from design import ANCHORS, predictions

TAU = 0.10
NUMERICAL = 0.001
GAP = 0.202


def analyze_family(row):
    cells = row['cells']
    values = {name: float(cell['float64']['patched_margin']) for name,cell in cells.items()}
    expected = predictions({name:values[name] for name in ANCHORS})
    if set(values) != set(ANCHORS) | set(expected['H_state']):
        raise ValueError('Incomplete or extra measured cell')
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite margin')
    anchor_gap = abs(values[ANCHORS[1]] - values[ANCHORS[0]])
    errors = {name:max(abs(values[cell]-v) for cell,v in forecast.items())
              for name,forecast in expected.items()}
    precision = 0.
    for name,forecast in expected.items():
        for cell,v in forecast.items():
            anchor = ANCHORS[0] if ((name=='H_state' and cell.startswith('neg'))
                                    or (name=='H_position' and '_20_' in cell)) else ANCHORS[1]
            e32=cells[cell]['float32']['patched_margin']-cells[anchor]['float32']['patched_margin']
            precision=max(precision,abs(e32-(values[cell]-v)))
    if precision > NUMERICAL:
        raise ValueError('Prediction-error precision exceeds the declared floor')
    same_label = {f'{s}_{p}':abs(values[f'{s}_{p}_1']-values[f'{s}_{p}_0'])
                  for s in ('neg','pos') for p in (20,28)}
    mean_cell={f'{s}_{p}':statistics.mean(values[f'{s}_{p}_{r}'] for r in (0,1))
               for s in ('neg','pos') for p in (20,28)}
    return {'family_id':row['family_id'], 'eligible':anchor_gap>GAP,
            'anchor_gap':anchor_gap, 'max_prediction_error':errors,
            'max_error_precision_difference':precision,
            'same_label_prefix_differences':same_label,
            'max_same_label_prefix_difference':max(same_label.values()),
            'balance_contrast':statistics.mean(mean_cell[f'pos_{p}']-mean_cell[f'neg_{p}'] for p in (20,28)),
            'position_contrast':statistics.mean(mean_cell[f'{s}_28']-mean_cell[f'{s}_20'] for s in ('neg','pos')),
            'interaction':mean_cell['pos_28']-mean_cell['pos_20']-mean_cell['neg_28']+mean_cell['neg_20'],
            'max_native_margin_change':max(abs(c['float64']['margin_change']) for c in cells.values()),
            'recipient_attention':cells[ANCHORS[0]]['float64']['a_r']}


def summary(rows):
    families=[analyze_family(row) for row in rows]
    eligible=[r for r in families if r['eligible']]
    candidates={}
    for name in ('H_state','H_position'):
        errors=[f['max_prediction_error'][name] for f in eligible]
        candidates[name]={'eligible_families':len(errors),
                          'definite_hits':sum(e<=TAU-NUMERICAL for e in errors),
                          'possible_hits':sum(e<=TAU+NUMERICAL for e in errors),
                          'mean_max_error':statistics.mean(errors) if errors else None,
                          'max_error':max(errors) if errors else None}
    gate=len(eligible)>=16 and any(c['definite_hits']/len(eligible)>=.90 for c in candidates.values())
    describe=lambda xs:{'min':min(xs),'median':statistics.median(xs),
                        'mean':statistics.mean(xs),'max':max(xs)}
    out={'status':'development_only', 'families':len(families),'eligible_families':len(eligible),
         'thresholds':{'prediction_tolerance':TAU,'numerical_allowance':NUMERICAL,'strict_anchor_gap':GAP},
         'candidates':candidates, 'confirmation_start_gate':gate,
         'gate_rule':'at least16/32eligible and >=90%definite max-six-cell hits for one candidate',
         'same_label_control_exceeds_tolerance':sum(f['max_same_label_prefix_difference']>TAU for f in families),
         'same_label_control_exceeds_tolerance_eligible':sum(f['max_same_label_prefix_difference']>TAU for f in eligible),
         'descriptive':{key:describe([r[key] for r in families]) for key in
            ('anchor_gap','max_same_label_prefix_difference','balance_contrast','position_contrast',
             'interaction','max_native_margin_change','recipient_attention')},
         'max_error_precision_difference':max(f['max_error_precision_difference'] for f in families),
         'family_results':families,
         'limits':['Development estimates; no confirmatory exclusions or semantic identification.',
                   'Current balance class, exact depth, sign and normalized-depth aliases remain.',
                   'Whole-value donor transfer carries correlated contextual information.']}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing overwrite')
    manifest=json.loads((a.run/'manifest.json').read_text())
    if manifest['status']!='completed':raise SystemExit('Technical failure/incomplete run cannot support analysis')
    for name, digest in manifest['output_hashes'].items():
        if hashlib.sha256((a.run/name).read_bytes()).hexdigest()!=digest:
            raise SystemExit('Altered run artifact: '+name)
    rows=[json.loads(line) for line in (a.run/'cases.jsonl').read_text().splitlines()]
    if len(rows)!=32:raise SystemExit('This development analyzer expects exactly32families')
    result=summary(rows)
    result['provenance']={name:hashlib.sha256((a.run/name).read_bytes()).hexdigest()
                          for name in ('manifest.json','cases.jsonl','anchor_forecasts.jsonl')}
    result['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='family_results'},indent=2))


if __name__=='__main__':main()
