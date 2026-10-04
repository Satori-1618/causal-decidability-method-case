"""Independent raw recount of averaged-anchor round004; imports no repo code.

Requires scipy solely for an independent implementation of CP/binomial tails.
Reads only the explicitly supplied completed run directory. Does not load models.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import beta, binom

DTYPES = ('float32', 'float64')
GROUPS = tuple((d, p) for d in ('neg', 'pos') for p in (20, 28))
BASES = tuple(f'{d}_{p}_{r}' for d, p in GROUPS for r in (0, 1))
NAMES = ('B_avg', 'P_avg', 'B_single4', 'B_legacy2', 'C')


def close(x, y, context, atol=1e-11):
    if not math.isclose(float(x), float(y), rel_tol=1e-11, abs_tol=atol):
        raise AssertionError((context, x, y))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def predictions(cal):
    # Independent sums over named raw calibration cells.
    cell = {(d, p): sum(cal[f'{d}_{p}_{r}'] for r in (0, 1))/2 for d, p in GROUPS}
    b = {d: sum(cal[f'{d}_{p}_{r}'] for p in (20, 28) for r in (0, 1))/4 for d in ('neg', 'pos')}
    p = {pos: sum(cal[f'{d}_{pos}_{r}'] for d in ('neg', 'pos') for r in (0, 1))/4 for pos in (20, 28)}
    result = {name: {} for name in NAMES}
    for d, pos in GROUPS:
        for r in (0, 1):
            key = f'target_{d}_{pos}_{r}'
            result['B_avg'][key] = b[d]
            result['P_avg'][key] = p[pos]
            result['B_single4'][key] = (cal[f'{d}_20_0']+cal[f'{d}_28_0'])/2
            result['B_legacy2'][key] = cal['neg_20_0' if d == 'neg' else 'pos_28_0']
            result['C'][key] = cell[(d, pos)]
    return result


def summarize(values):
    v = sorted(values)
    if not v:
        return {'n': 0}
    def q(p):
        x = (len(v)-1)*p
        a = int(x)
        return v[a]+(v[min(a+1, len(v)-1)]-v[a])*(x-a)
    return {'n': len(v), 'mean': statistics.mean(v), 'median': q(.5),
            'q1': q(.25), 'q3': q(.75), 'iqr': q(.75)-q(.25),
            'min': v[0], 'max': v[-1], 'range': v[-1]-v[0]}


def family_record(row, frozen, frozen_c):
    roles = {role: {dt: {base: float(row['cells'][f'{role}_{base}'][dt]['patched_margin'])
                        for base in BASES} for dt in DTYPES} for role in ('calibration', 'target')}
    preds = {dt: predictions(roles['calibration'][dt]) for dt in DTYPES}
    for dt in DTYPES:
        for name in NAMES:
            written = frozen_c['forecasts'][dt] if name == 'C' else frozen['forecasts'][dt][name]
            for key, value in preds[dt][name].items():
                close(value, written[key], 'calibration-only forecast ' + name)
    gaps = {dt: max(abs(preds[dt]['B_avg'][key]-preds[dt]['P_avg'][key])
                   for key in preds[dt]['B_avg']) for dt in DTYPES}
    sep = gaps['float64'] > .202
    assert sep == (gaps['float32'] > .202) == frozen['separating'] == row['separating']
    close(gaps['float64'], frozen['separation_gap_float64'], 'separation gap')
    signed = {dt: {name: {base: roles['target'][dt][base]-preds[dt][name]['target_'+base]
                          for base in BASES} for name in NAMES} for dt in DTYPES}
    errors = {name: max(abs(x) for x in signed['float64'][name].values()) for name in NAMES}
    numerical = {name: max(abs(signed['float64'][name][base]-signed['float32'][name][base]) for base in BASES)
                 for name in NAMES}
    assert max(numerical[name] for name in NAMES if name != 'C') <= .001
    for name in NAMES:
        if name != 'C':
            close(errors[name], row['analysis']['max_error'][name], 'saved max error '+name)
    definite = {name: errors[name] <= .099 for name in NAMES}
    possible = {name: errors[name] <= .101 for name in NAMES}
    pairgap = {dt: {f'{d}_{p}': roles['target'][dt][f'{d}_{p}_1']-roles['target'][dt][f'{d}_{p}_0']
                    for d, p in GROUPS} for dt in DTYPES}
    definite_w = {cell: all(abs(pairgap[dt][cell]) > .202 for dt in DTYPES) for cell in pairgap['float64']}
    possible_w = {cell: any(abs(pairgap[dt][cell]) > .202 for dt in DTYPES) for cell in pairgap['float64']}
    unresolved_w = [cell for cell in pairgap['float64'] if definite_w[cell] != possible_w[cell]]
    observations = {f'{d}_{p}': [roles[role]['float64'][f'{d}_{p}_{r}']
                                for role in ('calibration', 'target') for r in (0, 1)] for d, p in GROUPS}
    cellmean = {cell: statistics.mean(v) for cell, v in observations.items()}
    grand = statistics.mean(cellmean.values())
    bmean = {d: statistics.mean([cellmean[f'{d}_{p}'] for p in (20, 28)]) for d in ('neg', 'pos')}
    pmean = {p: statistics.mean([cellmean[f'{d}_{p}'] for d in ('neg', 'pos')]) for p in (20, 28)}
    ss = {'balance': 8*sum((v-grand)**2 for v in bmean.values()),
          'position': 8*sum((v-grand)**2 for v in pmean.values()),
          'interaction': 4*sum((cellmean[f'{d}_{p}']-bmean[d]-pmean[p]+grand)**2 for d,p in GROUPS),
          'within': sum((v-cellmean[cell])**2 for cell, values in observations.items() for v in values),
          'total': sum((v-grand)**2 for values in observations.values() for v in values)}
    close(sum(v for k,v in ss.items() if k != 'total'), ss['total'], 'SS decomposition')
    return {'family_id': row['family_id'], 'separating': sep, 'forecast_gap': gaps['float64'],
            'errors': errors, 'definite': definite, 'possible': possible, 'error_precision': numerical,
            'definite_witness': any(definite_w.values()), 'possible_witness': any(possible_w.values()),
            'witness_numerically_unresolved': bool(unresolved_w), 'same_cell_signed_target_gaps': pairgap['float64'],
            'cell_sample_sd': {cell: statistics.stdev(v) for cell,v in observations.items()},
            'cell_mean': cellmean, 'cell_range': {cell: max(v)-min(v) for cell,v in observations.items()},
            'role_mean_differences': {f'{d}_{p}': statistics.mean(roles['target']['float64'][f'{d}_{p}_{r}'] for r in (0,1))-
                statistics.mean(roles['calibration']['float64'][f'{d}_{p}_{r}'] for r in (0,1)) for d,p in GROUPS},
            'sum_squares': ss,
            'worst_balance_B': 'neg' if max(abs(v) for k,v in signed['float64']['B_avg'].items() if k.startswith('neg')) >=
                max(abs(v) for k,v in signed['float64']['B_avg'].items() if k.startswith('pos')) else 'pos'}


def cp(definite, possible, n):
    lo = 0. if definite == 0 else float(beta.ppf(.01, definite, n-definite+1))
    hi = 1. if possible == n else float(beta.ppf(.99, possible+1, n-possible))
    return {'n':n,'definite_hits':definite,'possible_hits':possible,'interval':[lo,hi],
            'status':'adequate' if lo>.9 else 'excluded' if hi<.9 else 'unresolved'}


def group_summary(rows):
    sums = {key: sum(r['sum_squares'][key] for r in rows) for key in ('balance','position','interaction','within','total')}
    return {'n':len(rows),
            'definite_hits':{name:sum(r['definite'][name] for r in rows) for name in NAMES},
            'possible_hits':{name:sum(r['possible'][name] for r in rows) for name in NAMES},
            'C_precision_unresolved':sum(r['error_precision']['C']>.001 for r in rows),
            'B_C_joint':dict(Counter('both' if r['definite']['B_avg'] and r['definite']['C'] else
                                    'B_only' if r['definite']['B_avg'] else 'C_only' if r['definite']['C'] else 'neither' for r in rows)),
            'definite_witness_families':sum(r['definite_witness'] for r in rows),
            'possible_witness_families':sum(r['possible_witness'] for r in rows),
            'witness_numerically_unresolved_families':sum(r['witness_numerically_unresolved'] for r in rows),
            'cell_sd':{f'{d}_{p}':summarize([r['cell_sample_sd'][f'{d}_{p}'] for r in rows]) for d,p in GROUPS},
            'role_mean_difference':{f'{d}_{p}':summarize([r['role_mean_differences'][f'{d}_{p}'] for r in rows]) for d,p in GROUPS},
            'maximum_errors':{name:summarize([r['errors'][name] for r in rows]) for name in NAMES},
            'pooled_recipient_centered_ss':sums,
            'pooled_recipient_centered_ss_shares_descriptive':{k:v/sums['total'] for k,v in sums.items() if k!='total'},
            'worst_balance_B_counts':dict(Counter(r['worst_balance_B'] for r in rows))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    run=args.run
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='completed', 'Wait for a completed run; do not audit incomplete outcomes.'
    for filename, expected in manifest['output_hashes'].items():
        assert hashlib.sha256((run/filename).read_bytes()).hexdigest()==expected, filename
    rows=read_rows(run/'cases.jsonl')
    frozen={r['family_id']:r for r in read_rows(run/'forecasts.jsonl')}
    frozen_c={r['family_id']:r for r in read_rows(run/'descriptive_forecasts.jsonl')}
    assert len(rows)==256 and len({r['family_id'] for r in rows})==256
    result=[family_record(row,frozen[row['family_id']],frozen_c[row['family_id']]) for row in rows]
    eligible=[r for r in result if r['separating']]
    assert len(eligible)>=128
    candidates={name:cp(sum(r['definite'][name] for r in eligible),sum(r['possible'][name] for r in eligible),len(eligible))
                for name in ('B_avg','P_avg')}
    gains=sum(r['definite']['B_avg'] and not r['possible']['B_single4'] for r in result)
    losses=sum(not r['definite']['B_avg'] and r['possible']['B_single4'] for r in result)
    p=float(binom.sf(gains-1,gains+losses,.5)) if gains+losses else 1.
    published=json.loads((run/'analysis_summary.json').read_text())
    for name, own in candidates.items():
        claimed=published['candidates_on_separating_families'][name]
        assert own['definite_hits']==claimed['definite_hits'] and own['possible_hits']==claimed['possible_hits']
        assert own['status']==claimed['status']
        for x,y in zip(own['interval'],claimed['interval']):close(x,y,'independent scipy CP')
    claim=published['paired_averaging_on_all_families']
    assert gains==claim['robust_gains'] and losses==claim['robust_losses']
    close(p,claim['one_sided_exact_mcnemar_p'],'independent scipy paired binomial')
    assert (p<=.01)==claim['improvement_supported']
    for tag,key in (('definite_witness','within_cell_witness_families_descriptive'),
                    ('possible_witness','within_cell_possible_witness_families_descriptive'),
                    ('witness_numerically_unresolved','within_cell_numerically_unresolved_families_descriptive')):
        assert sum(r[tag] for r in result)==published[key]
    descriptive=json.loads((run/'descriptive_summary.json').read_text())
    all_summary, eligible_summary=group_summary(result),group_summary(eligible)
    for name, own in (('all_families',all_summary),('separating_subset',eligible_summary)):
        claim=descriptive[name]
        assert own['definite_hits']['C']==claim['counts']['C_definite_hits']
        assert own['possible_hits']['C']==claim['counts']['C_possible_hits']
        for cell, summary in own['cell_sd'].items():
            for metric in ('median','q1','q3','min','max'):
                close(summary[metric],claim['metric_summaries_fp64']['cell_sample_sd.'+cell][metric],'cell SD '+cell)
    report={'status':'PASS_independent_raw_recount_no_repo_analysis_import',
            'run':run.name,'run_git_head':manifest['git_head'],
            'raw_cases_sha256':hashlib.sha256((run/'cases.jsonl').read_bytes()).hexdigest(),
            'families':256,'separating':len(eligible),'candidate_results':candidates,
            'paired':{'gains':gains,'losses':losses,'robust_point_difference':(gains-losses)/256,
                      'one_sided_exact_p':p,'improvement_supported':p<=.01},
            'all_families':all_summary,'separating_subset':eligible_summary,
            'C_forecasts_verified_from_calibration_only':True,
            'scope':'Read-only arithmetic audit; no model load/forward, no other outcomes opened. Descriptive SS shares are not causal proportions.',
            'families_recomputed':result}
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
