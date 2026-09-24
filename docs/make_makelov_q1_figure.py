"""Draw docs/figures/makelov_q1.png: where the two read-source explanations separate, and the data.

    python3 docs/make_makelov_q1_figure.py records.jsonl [output.png]

records.jsonl is results/makelov_read_source_q1/records.jsonl on the branch
applications/makelov-2311.17030; the script refuses any other file. To get it from main:

    git show applications/makelov-2311.17030:applications/makelov-2311.17030/results/\
makelov_read_source_q1/records.jsonl > q1_records.jsonl

The figure normalizes each pair's effects by its full-patch effect. The preregistered
comparison used the absolute prediction errors in nats. Needs matplotlib.
"""
import hashlib
import json
import os
import random
import sys
from statistics import fmean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic'})

SURFACE, INK, INK2, MUTED, HAIR = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
A, B = '#2a78d6', '#eb6834'
HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS_SHA256 = 'eb4428e174b67dd65d3a9948a550273dc6fefe6adbfb4c014baadf82542f0ef1'
CONDITIONS = ['full', 'read_row', 'read_null']
PREDICT = {'A': {'full': 1, 'read_row': 1, 'read_null': 0},
           'B': {'full': 1, 'read_row': 0, 'read_null': 1}}


def relative_effects(path):
    """Per base pair: each condition's effect relative to the full patch, mean of both directions."""
    by_pair = {}
    for line in open(path):
        r = json.loads(line)
        m = r['margins']
        gap = m['full'] - m['baseline']
        by_pair.setdefault(r['case_id'], []).append(
            {c: (m[c] - m['baseline']) / gap for c in CONDITIONS})
    return [{c: fmean(d[c] for d in dirs) for c in CONDITIONS} for dirs in by_pair.values()]


def main(records, out):
    with open(records, 'rb') as handle:
        if hashlib.sha256(handle.read()).hexdigest() != RECORDS_SHA256:
            sys.exit('refused: not the Q1 records.jsonl')
    pairs = relative_effects(records)
    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.23)
    ax.set_facecolor(SURFACE)
    rng = random.Random(0)
    half = 0.2
    for i, c in enumerate(CONDITIONS):
        ys = [p[c] for p in pairs]
        if c != 'full':
            xs = [i + rng.uniform(-0.09, 0.09) for _ in ys]
            ax.scatter(xs, ys, s=26, color=INK2, alpha=0.55, edgecolors=SURFACE, linewidths=0.8,
                       zorder=3)
        if c == 'full':
            ax.plot([i - half, i], [1, 1], color=A, linewidth=4, solid_capstyle='round', zorder=4)
            ax.plot([i, i + half], [1, 1], color=B, linewidth=4, solid_capstyle='round', zorder=4)
            ax.text(i, 1.07, 'A and B both predict the full effect,\nby construction: no test',
                    ha='center', va='bottom', fontsize=10.5, color=INK2, linespacing=1.25)
            ax.text(i, 0.93, 'measured: the full effect itself,\nwhich sets the scale',
                    ha='center', va='top', fontsize=10.5, color=INK2, linespacing=1.25)
        else:
            for name, colour in (('A', A), ('B', B)):
                y = PREDICT[name][c]
                ax.plot([i - half, i + half], [y, y], color=colour, linewidth=4,
                        solid_capstyle='round', zorder=4)
                ax.text(i + half + 0.04, y, f'{name} predicts', va='center', fontsize=10.5,
                        color=INK2)
            mean = fmean(ys)
            ax.plot([i - 0.12, i + 0.12], [mean, mean], color=INK, linewidth=2, zorder=5)
            ax.text(i - 0.19, mean, f'measured\nmean {mean:.2f}', ha='right', va='center',
                    fontsize=10.5, color=INK, linespacing=1.2)

    ax.set_xlim(-0.55, 2.75)
    ax.set_ylim(-0.12, 1.22)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['full patch\nread $v$, write $v$',
                        'read only the visible part\nread $v_R$, write $v$',
                        'read only the null part\nread $v_N$, write $v$'], fontsize=11, color=INK)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0  no change', '0.25', '0.5', '0.75', '1  full effect'], fontsize=10,
                       color=INK2)
    ax.set_ylabel('effect relative to the full patch', fontsize=10.5, color=INK2)
    ax.grid(axis='y', color=HAIR, linewidth=1)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(HAIR)
    ax.tick_params(length=0)

    fig.text(0.03, 0.945, 'The same logic on a real model: which part of the patched direction '
             'does the work?', fontsize=16, fontweight='bold', color=INK, va='center')
    fig.text(0.03, 0.895, 'GPT-2 Small, MLP 8, the published direction of Makelov et al. (2023). '
             'A: the visible part $v_R$ carries the signal.', fontsize=11, color=INK2, va='center')
    fig.text(0.03, 0.86, 'B: the null part $v_N$, which the MLP output matrix maps to zero, '
             'carries it. The full patch cannot tell A from B; the two read patches can.',
             fontsize=11, color=INK2, va='center')
    handles = [Line2D([], [], color=A, linewidth=4), Line2D([], [], color=B, linewidth=4),
               Line2D([], [], marker='o', linestyle='', color=INK2, alpha=0.55, markersize=6)]
    ax.legend(handles, ['A predicts (visible part)', 'B predicts (null part)',
                        f'measured, one dot per fresh base pair ({len(pairs)})'],
              loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False,
              fontsize=10, labelcolor=INK2)
    fig.text(0.03, 0.058, 'Preregistered comparison on 64 fresh base pairs: B is closer in '
             '64 of 64 (exact sign test p = 1.1e-19). How close B must be to count as accurate '
             'enough was not declared.', fontsize=9.5, color=MUTED, va='center')
    fig.text(0.03, 0.028, 'Shown normalized: each pair\'s effects divided by its full-patch '
             'effect. The preregistered comparison used the absolute prediction errors in nats.',
             fontsize=9.5, color=MUTED, va='center')
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print('wrote', out, len(pairs), 'pairs')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
         os.path.join(HERE, 'figures', 'makelov_q1.png'))
