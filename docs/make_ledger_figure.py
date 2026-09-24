"""Draw docs/figures/decide_ledger.png: the calculator's output on four designs.

    python3 docs/make_ledger_figure.py

A and B are real Freeze B inputs of the resid_mid.8 application (branch
applications/makelov-2311.17030); C and D are declared examples. Needs matplotlib.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from causal_decidability import decidability  # noqa: E402

PRE = dict(noise_factor=1.0, signatures=2)
DESIGNS = [
    ('A  resid_mid.8, logit diff., row, n = 20, fp32',
     decidability({'inert': 0.0, 'all': 4.7958}, n=20, sigma=1.5173, dtype='float32',
                  readout_scale=20.0, depth=768, **PRE)),
    ('B  resid_mid.8, interchange acc., row, n = 3, fp32',
     decidability({'inert': 0.0, 'all': 0.73}, n=3, sigma=0.4970, dtype='float32',
                  readout_scale=1.0, depth=1, **PRE)),
    ('C  declared, n = 24, bf16, 36-step accumulation',
     decidability({'at_node': 0.10, 'after_node': 0.0}, n=24, sigma=0.04, dtype='bfloat16',
                  readout_scale=3.4, depth=36)),
    ('D  declared, operator severs the channel',
     decidability({'at_node': 0.0, 'after_node': 0.0}, n=10_000, sigma=0.04,
                  dtype='float32', readout_scale=3.4, depth=36)),
]
LEVER = {'none': 'no change needed', 'units_or_noise': 'more units: n ≈ {r}',
         'precision': 'more mantissa bits: +{b}',
         'intervention_or_candidates': 'change the intervention or the rivals'}


def main():
    fig, (ax, text) = plt.subplots(1, 2, figsize=(11, 5.2),
                                   gridspec_kw={'width_ratios': [1.6, 1]})
    y = 0
    ticks = []
    for label, out in DESIGNS:
        colour = '#1f7a4a' if out['decidable'] else '#b03a2e'
        if out['separation'] > 0:
            ax.barh(y + 0.28, out['separation'], height=0.26, color=colour)
        else:
            ax.text(2e-7, y + 0.28, 'separation 0', va='center', color=colour,
                    fontsize=9, fontweight='bold')
        ax.barh(y, out['statistical_floor'], height=0.26, color='#8da0cb')
        ax.barh(y - 0.28, out['numerical_floor'], height=0.26, color='#394b6b')
        ticks.append((y, label))
        lever = LEVER[out['lever']['kind']].format(r=out['lever']['replicates_needed'],
                                                   b=out['lever']['extra_mantissa_bits'])
        verdict = 'decidable' if out['decidable'] else 'not decidable'
        text.text(0.0, y + 0.12, f"ratio {out['ratio']:.2f}  →  {verdict}", color=colour,
                  fontsize=10, fontweight='bold', va='center')
        why = ('no n or dtype helps' if out['separation'] == 0
               else f"{out['binding_floor']} floor binds")
        text.text(0.0, y - 0.1, why, fontsize=9, va='center', color='#333333')
        text.text(0.0, y - 0.38, lever, fontsize=9, va='center', color='#333333')
        y -= 1.2
    ax.set_xscale('log')
    ax.set_xlim(1e-7, 30)
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([lab for _, lab in ticks], fontsize=9)
    ax.set_xlabel('readout units (log scale)')
    ax.grid(axis='x', alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#1f7a4a', '#8da0cb', '#394b6b')]
    ax.legend(handles, ['separation of the closest rivals', 'statistical floor',
                        'numerical floor'], fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    text.set_ylim(ax.get_ylim())
    text.axis('off')
    fig.suptitle('What the calculator reports before the experiment runs', fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, 'figures', 'decide_ledger.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    main()
