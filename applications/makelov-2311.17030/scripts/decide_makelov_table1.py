"""Put the reproduced Makelov contrasts through the project's decidability calculator.

    python3 scripts/decide_makelov_table1.py results/makelov_replication_001

This is the same calculation ``scripts/decide.py`` prints, driven from the paired
analysis so the declarations are not retyped. Each contrast is asked twice:

``endpoint``  the pre-experiment question the calculator is built for. The rivals
              are the two idealised endpoints: the sub-direction is inert (0), or
              it carries the whole patch (the full-vs-nullspace effect, which is
              the largest move this operator produces). Nothing here depends on
              what the contrast actually came out at.
``observed``  the post-hoc reading: rivals are 0 and the separation this run
              measured. Flagged because the calculator's contract says its inputs
              are declarations, and this one is not.

and each of those twice again, with the resolution the published means support
(``unpaired``: one arm's spread, the calculator's default sqrt(2) noise factor)
and the one the per-example export supports (``paired``: the spread of the
per-example difference, noise factor 1 because that difference is a single panel).

``readout_scale`` and ``depth`` are declarations about the implementation, not
measurements: one readout is a single answer logit at the final position, whose
magnitude GPT-2 keeps near 20, accumulated over the 768-term unembedding dot
product. On the indicator scale a readout is a 0/1 comparison, so 1 and 1.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import decidability  # noqa: E402

READOUT = {'logit_diff': {'readout_scale': 20.0, 'depth': 768},
           'accuracy': {'readout_scale': 1.0, 'depth': 1}}
ALWAYS_TOGETHER = ()


def sigmas(entry, n):
    """Recover the per-example spreads behind the two resolutions the analysis printed."""
    paired = entry['paired']['resolution'] * math.sqrt(n)
    unpaired = entry['unpaired']['resolution'] * math.sqrt(n) / math.sqrt(2.0)
    return {'paired': {'sigma': paired, 'noise_factor': 1.0},
            'unpaired': {'sigma': unpaired,
                         'noise_factor': decidability.DEFAULT_NOISE_FACTOR}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('directory')
    ap.add_argument('--output', default='decide.json')
    args = ap.parse_args()
    directory = Path(args.directory)
    analysis = json.loads((directory / 'paired_resolution.json').read_text())
    n = analysis['n_pairs']

    out = {'n': n, 'readout_declarations': READOUT, 'runs': {}}
    for contrast, block in analysis['contrasts'].items():
        for scale, entry in block.items():
            endpoint = analysis['contrasts']['full_vs_nullspace'][scale]['separation']
            framings = {
                'endpoint': {'inert_component': 0.0, 'component_carries_the_patch': endpoint},
                'observed': {'components_alike': 0.0,
                             'components_differ': entry['separation']},
            }
            for framing, predictions in framings.items():
                for mode, declared in sigmas(entry, n).items():
                    result = decidability.decidability(
                        predictions=predictions, n=n, sigma=declared['sigma'],
                        dtype='float32', noise_factor=declared['noise_factor'],
                        **READOUT[scale])
                    key = f'{contrast}|{scale}|{framing}|{mode}'
                    out['runs'][key] = {
                        'contrast': contrast, 'scale': scale, 'framing': framing,
                        'resolution_mode': mode, 'declared': declared,
                        'predictions': predictions,
                        'separation': result['separation'],
                        'statistical_floor': result['statistical_floor'],
                        'numerical_floor': result['numerical_floor'],
                        'resolution': result['resolution'],
                        'binding_floor': result['binding_floor'],
                        'ratio': result['ratio'], 'decidable': result['decidable'],
                        'lever': result['lever']['detail'],
                        'post_hoc': framing == 'observed',
                    }
    out['decidability_source_sha256'] = decidability.source_sha256()
    (directory / args.output).write_text(json.dumps(out, indent=1))

    width = max(len(k) for k in out['runs'])
    for key in sorted(out['runs']):
        run = out['runs'][key]
        print(f"{key:<{width}}  sep {run['separation']:+.6f}  res "
              f"{run['resolution']:.6f}  ratio {run['ratio']:7.3f}  "
              f"{'DECIDABLE' if run['decidable'] else 'NOT DECIDABLE'}")
    print('\nwrote', directory / args.output)


if __name__ == '__main__':
    main()
