"""Freeze B and the confirmation scoring for ``PREREG_RESID_MID8.md`` (§8–§10).

``predict`` turns the pilot record into the 28 primary predictions with the pinned
calculator, exactly as §8 calls it, plus the secondary block predictions added at
Freeze B. ``realized`` and ``score`` apply §9 and §10 to the confirmation contrasts.
Everything here is a pure function of its inputs; the scripts do the I/O and gates.
"""
import math

import torch

import decidability as D

LADDER = (20, 50, 100, 250, 500, 1000, 2000)
ALPHA = 0.01
SIGNATURES = 2
Z = D.bonferroni_z(ALPHA, SIGNATURES)
BAND = (0.5, 2.0)
DTYPE = 'float32'
NOISE_FACTOR = 1.0
#: §8, per readout
READOUT_DECLARATION = {'L': {'readout_scale': 20.0, 'depth': 768},
                       'A': {'readout_scale': 1.0, 'depth': 1}}
COMPONENTS = ('row', 'null')
READOUTS = ('L', 'A')

#: Added at Freeze B, after the pilot and before any confirmation pair exists; reported,
#: not judged, and outside §10. On this site every primary cell lands at a ratio above 2,
#: so the primary test can only catch overpromising. Readout A is the one place where the
#: predicted ratio crosses 1 at a usable n, and disjoint blocks of the confirmation order
#: give a realized decision rate there.
SECONDARY_READOUT = 'A'
SECONDARY_LADDER = (2, 3, 4, 6, 8, 12, 16)

CONFIRMATION_SEED = 7013
CONFIRMATION_PER_COMBINATION = 1000


def sigma_for(pilot_summary, readout, component):
    sd = pilot_summary['sd'][readout][component]
    return max(sd['d_inert'], sd['d_all'])


def _cell(pilot_summary, readout, component, n):
    sigma = sigma_for(pilot_summary, readout, component)
    out = D.decidability(
        predictions={'inert': 0.0, 'all': pilot_summary['mean_E_full'][readout]},
        n=n, sigma=sigma, dtype=DTYPE, noise_factor=NOISE_FACTOR, alpha=ALPHA,
        signatures=SIGNATURES, **READOUT_DECLARATION[readout])
    ratio = out['ratio']
    return {'readout': readout, 'component': component, 'n': n, 'sigma': sigma,
            'separation': out['separation'],
            'statistical_floor': out['statistical_floor'],
            'numerical_floor': out['numerical_floor'],
            'binding_floor': out['binding_floor'],
            'ratio': ratio, 'decidable': out['decidable'],
            'outside_band': not (BAND[0] <= ratio <= BAND[1])}


def predict(pilot_summary):
    """§8: 2 components x 2 readouts x 7 sizes, plus predicted crossovers."""
    primary = [_cell(pilot_summary, r, x, n)
               for r in READOUTS for x in COMPONENTS for n in LADDER]
    crossover = {f'{r}/{x}': next((c['n'] for c in primary if c['readout'] == r
                                   and c['component'] == x and c['decidable']), None)
                 for r in READOUTS for x in COMPONENTS}
    secondary = [_cell(pilot_summary, SECONDARY_READOUT, x, n)
                 for x in COMPONENTS for n in SECONDARY_LADDER]
    return {'z': Z, 'primary': primary, 'crossover': crossover, 'secondary': secondary}


def interleave(dataset):
    """Deviation 6: ABB->BAB[0], BAB->ABB[0], ABB->BAB[1], ... over the generated set."""
    first = [r for r in dataset if r['half'] == 'ABB->BAB']
    second = [r for r in dataset if r['half'] == 'BAB->ABB']
    if len(first) != len(second) or len(first) + len(second) != len(dataset):
        raise ValueError('dataset is not two equal halves')
    return [r for pair in zip(first, second) for r in pair]


def _interval(values, z):
    values = torch.as_tensor(values, dtype=torch.float64)
    n = values.numel()
    mean = float(values.mean())
    sd = float(values.std(unbiased=True)) if n > 1 else 0.0
    half = z * sd / math.sqrt(n)
    return {'mean': mean, 'sd': sd, 'half_width': half,
            'contains_zero': abs(mean) <= half}


def realized(d_inert, d_all, z=Z):
    """§9: an endpoint is compatible iff its contrast's interval contains 0;
    decided iff not both are compatible."""
    inert, all_ = _interval(d_inert, z), _interval(d_all, z)
    return {'d_inert': inert, 'd_all': all_,
            'inert_compatible': inert['contains_zero'],
            'all_compatible': all_['contains_zero'],
            'decided': not (inert['contains_zero'] and all_['contains_zero'])}


def score(primary, realized_cells):
    """§10. ``realized_cells`` is keyed like ``(readout, component, n)``."""
    rows = []
    for cell in primary:
        got = realized_cells[(cell['readout'], cell['component'], cell['n'])]
        rows.append(dict(cell, realized_decided=got['decided'],
                         match=cell['decidable'] == got['decided']))
    matches = sum(r['match'] for r in rows)
    outside_misses = [r for r in rows if r['outside_band'] and not r['match']]
    fraction = matches / len(rows)
    if outside_misses:
        verdict = 'contradicted'
    elif fraction >= 0.8:
        verdict = 'supported'
    else:
        verdict = 'inconclusive'
    return {'cells': rows, 'matches': matches, 'total': len(rows),
            'fraction': fraction, 'outside_band_misses': len(outside_misses),
            'verdict': verdict}


def block_rate(d_inert, d_all, n, z=Z):
    """Secondary: share of disjoint consecutive blocks of size n that are decided."""
    d_inert = torch.as_tensor(d_inert, dtype=torch.float64)
    d_all = torch.as_tensor(d_all, dtype=torch.float64)
    blocks = d_inert.numel() // n
    decided = sum(realized(d_inert[i * n:(i + 1) * n], d_all[i * n:(i + 1) * n], z)['decided']
                  for i in range(blocks))
    return {'blocks': blocks, 'decided': decided, 'rate': decided / blocks}
