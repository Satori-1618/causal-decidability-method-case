"""Development only: execute the two previously unmeasured token-factor cells."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from qualify import source_rows, dump, digest
from runtime import DyckRuntime, HERE
from token_split import token_factorized_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records, summary = [], {}
    started = time.monotonic()
    for name in ['ood', 'indist']:
        source = source_rows(name)
        strings = [r['string'] for r in source]
        outputs = {}
        for dtype in [torch.float32, torch.float64]:
            tag = str(dtype).split('.')[-1]
            rt = DyckRuntime.from_checkpoint(dtype=dtype)
            native = rt.run(strings)
            outputs[(tag, 'native')] = native
            for mode in ['within_token', 'token_mass', 'both']:
                weights = token_factorized_weights(native.attention, strings, mode)
                result = rt.run(strings, mode='custom_attention', attention_override=weights)
                assert torch.equal(result.values, native.values)
                outputs[(tag, mode)] = result
        data = {m: outputs[('float64',m)].margins.numpy() for m in ['native','within_token','token_mass','both']}
        valid = np.array([r['balanced'] == 'True' for r in source])
        err_within = np.maximum(abs(data['within_token']-data['both']),abs(data['token_mass']-data['native']))
        err_mass = np.maximum(abs(data['within_token']-data['native']),abs(data['token_mass']-data['both']))
        summary[name] = {
            'correct': {m: int(((v < 0) == valid).sum()) for m,v in data.items()},
            'mean_margin_effect': {m: float((v-data['native']).mean()) for m,v in data.items()},
            'context_forecast_maxerror_quantiles': np.quantile(err_within,[0,.25,.5,.75,.9,1]).tolist(),
            'lexical_forecast_maxerror_quantiles': np.quantile(err_mass,[0,.25,.5,.75,.9,1]).tolist(),
            'context_forecast_mae': float(err_within.mean()), 'lexical_forecast_mae': float(err_mass.mean()),
        }
        for i, row in enumerate(source):
            record = {'dataset':name,'index':i,'string':row['string'],'valid':bool(valid[i])}
            for (tag,mode), result in outputs.items():
                record[tag+'_'+mode] = float(result.margins[i])
            records.append(record)
    summary['elapsed_seconds'] = time.monotonic()-started
    summary['status'] = 'development_not_confirmation'
    (args.output/'cases.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    dump(args.output/'summary.json',summary)
    dump(args.output/'manifest.json',{'sources':{n:digest(HERE/n) for n in ['runtime.py','token_split.py','develop_token_split.py','ASSET_LOCK.json']},
                                    'outputs':{n:digest(args.output/n) for n in ['summary.json','cases.jsonl']}})
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
