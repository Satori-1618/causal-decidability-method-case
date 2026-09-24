"""Draw docs/figures/kaplan_example.png: one patch, two explanations, one deciding condition.

    python3 docs/make_kaplan_figure.py [output.png]

A teaching example built from Kaplan's character/content distinction for "I". No model
is run; the predictions follow from the two stated rules. Needs matplotlib.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic'})

SURFACE, INK, INK2, MUTED, HAIR, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df', '#f0efec'
CONTENT, CHARACTER = '#2a78d6', '#eb6834'
HERE = os.path.dirname(os.path.abspath(__file__))


def run(ax, x, y, parts, size=12, **kw):
    """Draw text pieces one after another on a line; each piece is (text, weight)."""
    renderer = ax.figure.canvas.get_renderer()
    inverse = ax.transData.inverted()
    for text, weight in parts:
        t = ax.text(x, y, text, fontsize=size, fontweight=weight, color=INK, va='center', **kw)
        box = t.get_window_extent(renderer)
        x = inverse.transform((box.x1, box.y0))[0]
    return x


def card(ax, x, y, w, h, face=SURFACE, edge=HAIR):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                facecolor=face, edgecolor=edge, linewidth=1))


def chip(ax, x, y, name, colour):
    ax.add_patch(FancyBboxPatch((x, y - 1.7), 14, 3.4, boxstyle='round,pad=0,rounding_size=1.7',
                                facecolor=colour, alpha=0.16, edgecolor='none'))
    ax.plot([x + 2.0], [y], 'o', markersize=8, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(x + 3.8, y, name, fontsize=12, fontweight='bold', color=INK, va='center')


def main(out):
    fig = plt.figure(figsize=(11, 6.4), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 64)
    ax.axis('off')
    fig.canvas.draw()

    ax.text(3, 60.5, 'A patch that works can fit two explanations', fontsize=17,
            fontweight='bold', color=INK, va='center')
    ax.text(3, 56.6, 'Kaplan: the word “I” has a character, the rule “refers to the speaker”, '
            'and a content, the person who is speaking.', fontsize=11.5, color=INK2, va='center')

    # donor
    card(ax, 3, 40.5, 42, 12.5)
    ax.text(5.5, 50.3, 'DONOR', fontsize=9, fontweight='bold', color=MUTED, va='center')
    run(ax, 5.5, 46.6, [('Alice tells Bob: “', 'normal'), ('I', 'bold'), (' am hungry.”', 'normal')], 13)
    ax.text(5.5, 43.1, 'Here “I” refers to Alice.', fontsize=11, color=INK2, va='center')

    # the two explanations
    for y, colour, name, text in [
            (50.4, CONTENT, 'Content explanation', 'The patch carries the person: Alice.'),
            (44.6, CHARACTER, 'Character explanation',
             'The patch carries the rule “the speaker”,\napplied in the recipient.')]:
        ax.plot([52], [y + 1.2], 'o', markersize=10, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(54, y + 1.2, name, fontsize=12, fontweight='bold', color=INK, va='center')
        ax.text(54, y - 1.3, text, fontsize=11, color=INK2, va='top', linespacing=1.3)

    ax.add_patch(FancyArrowPatch((24, 40.2), (24, 34.6), arrowstyle='-|>', mutation_scale=16,
                                 color=INK2, linewidth=1.5))
    ax.text(26, 37.3, 'patch the representation of “I” into each recipient, in place of “you”',
            fontsize=10.5, color=INK2, va='center', style='italic')

    # table
    cols = [3, 44, 56, 71, 86]
    header_y = 29.5
    for x, label in zip(cols, ['RECIPIENT', 'WITHOUT PATCH', 'CONTENT PREDICTS',
                               'CHARACTER PREDICTS', '']):
        ax.text(x, header_y, label, fontsize=9, fontweight='bold', color=MUTED, va='center')
    ax.text(44, header_y + 2.6, 'Answer to “Who is hungry?”', fontsize=9.5, color=INK2, va='center')
    ax.plot([3, 107], [header_y - 2.2, header_y - 2.2], color=HAIR, linewidth=1)

    rows = [
        (20.5, 'Non-separating design', [('Alice tells Bob: “', 'normal'), ('You', 'bold'), (' are hungry.”', 'normal')],
         'Alice', 'Alice', '=', 'same answer: the patch\nworks and fits both', False),
        (10.0, 'Added condition', [('Carol tells Bob: “', 'normal'), ('You', 'bold'), (' are hungry.”', 'normal')],
         'Alice', 'Carol', '≠', 'different answers: this\ncondition separates\nthe predictions', True),
    ]
    for y, tag, sentence, content, character, sign, verdict, deciding in rows:
        if deciding:
            ax.add_patch(FancyBboxPatch((2, y - 5), 106, 10, boxstyle='round,pad=0,rounding_size=1.2',
                                        facecolor=BAND, edgecolor='none'))
        ax.text(3, y + 2.3, tag, fontsize=10, fontweight='bold', color=INK2, va='center')
        run(ax, 3, y - 1.4, sentence, 12.5)
        ax.text(44, y - 1.4, 'Bob', fontsize=12, color=INK2, va='center')
        chip(ax, 56, y - 1.4, content, CONTENT)
        chip(ax, 71, y - 1.4, character, CHARACTER)
        ax.text(86, y - 1.4, sign, fontsize=20, fontweight='bold', color=INK, va='center')
        ax.text(89.5, y - 1.4, verdict, fontsize=10.5, color=INK, va='center', linespacing=1.25)

    ax.text(3, 2.2, 'Teaching example, not a model run. Step 1: write down what each explanation '
            'predicts. Step 2: find the condition where the predictions differ.',
            fontsize=9.5, color=MUTED, va='center')
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figures', 'kaplan_example.png'))
