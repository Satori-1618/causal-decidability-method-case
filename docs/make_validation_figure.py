"""Draw docs/figures/validation_curve.png: realized decision rate against predicted ratio.

    python3 docs/make_validation_figure.py            # plot from docs/data/validation_points.json
    python3 docs/make_validation_figure.py export \\
        --benchmark-src <benchmark branch>/src \\
        --benchmark-rows <rows_held_out.jsonl.gz> <rows_held_out_seeds_3001_...jsonl.gz> \\
        --resid-result <makelov branch>/results/resid_mid8_confirmation/result.json

Two sources with two different targets, drawn in separate panels on purpose:

  left   synthetic benchmark: share of worlds per sqrt(2) ratio bin in which the full
         method returned exactly the true candidate set (branch
         validation/synthetic-benchmark, rows scored with the corrected calculator)
  right  resid_mid.8 blocks: share of disjoint blocks of n pairs in which at least one of
         the two endpoints was excluded, which includes excluding both (branch
         applications/makelov-2311.17030, secondary analysis). Two rules: the frozen
         normal interval with sample SD, and, post hoc, an exact two-sided sign test on
         the nonzero paired differences at 0.005 per endpoint. The normal rule
         over-excludes on these small discrete samples; the exact test cannot exclude
         anything with fewer than 9 discordant pairs. Neither rate is a reliability.

The similar shape of the two panels does not mean the same performance. ``export``
records the sha256 of every source file, so each point can be traced.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'validation_points.json')


def _sha(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _exact_excludes(values, alpha=0.005):
    """Two-sided exact sign test on the nonzero values; True if it rejects mean zero."""
    plus = sum(1 for v in values if v > 0)
    minus = sum(1 for v in values if v < 0)
    k = plus + minus
    if k == 0:
        return False
    tail = sum(math.comb(k, i) for i in range(min(plus, minus) + 1)) / 2 ** k
    return min(1.0, 2 * tail) <= alpha


def _exact_block_rate(inert, all_, n):
    blocks = len(inert) // n
    hits = sum(_exact_excludes(inert[i * n:(i + 1) * n]) or _exact_excludes(all_[i * n:(i + 1) * n])
               for i in range(blocks))
    return hits / blocks


def export(args):
    sys.path.insert(0, args.benchmark_src)
    import ablation2_metrics as metrics
    out = {'benchmark': [], 'resid_mid8_blocks': [], 'sources': {}}
    for path in args.benchmark_rows:
        with gzip.open(path, 'rt') as handle:
            rows = [json.loads(line) for line in handle]
        bins, zero = metrics.curve(rows, 'full')
        name = os.path.basename(path)
        out['benchmark'].append({
            'rows': name,
            'bins': [{k: b[k] for k in ('ratio_low', 'ratio_high', 'worlds', 'decided',
                                        'decided_rate', 'error_rate')} for b in bins],
            'zero_separation': {k: zero[k] for k in ('worlds', 'decided', 'errors')}})
        out['sources'][name] = _sha(path)
    with open(args.resid_result) as handle:
        result = json.load(handle)
    per_pair_path = os.path.join(os.path.dirname(args.resid_result), 'per_pair.jsonl')
    with open(per_pair_path) as handle:
        pairs = [json.loads(line) for line in handle]
    hit = {c: [p[f'interchange_{c}'] - p['interchange_clean'] for p in pairs]
           for c in ('full', 'row', 'null')}
    for cell in result['secondary']:
        entry = {k: cell[k] for k in (
            'readout', 'component', 'n', 'ratio', 'decidable', 'blocks', 'decided', 'rate')}
        inert = hit[cell['component']]
        all_ = [f - x for f, x in zip(hit['full'], inert)]
        entry['rate_exact_posthoc'] = _exact_block_rate(inert, all_, cell['n'])
        out['resid_mid8_blocks'].append(entry)
    out['sources']['resid_mid8_confirmation/result.json'] = _sha(args.resid_result)
    out['sources']['resid_mid8_confirmation/per_pair.jsonl'] = _sha(per_pair_path)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, 'w') as handle:
        json.dump(out, handle, indent=1)
        handle.write('\n')
    print('wrote', DATA)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(DATA) as handle:
        data = json.load(handle)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    styles = [('#1f4e9c', 'o', '-'), ('#5b8bd9', 's', '--')]
    labels = ['seeds 2003…', 'fresh seeds 3001…']
    for (colour, marker, line), label, series in zip(styles, labels, data['benchmark']):
        bins = [b for b in series['bins'] if b['worlds'] >= 20]
        centres = [(b['ratio_low'] * b['ratio_high']) ** 0.5 for b in bins]
        left.plot(centres, [100 * b['decided_rate'] for b in bins], marker=marker,
                  linestyle=line, color=colour, label=label, markersize=5)
    left.set_title('Synthetic benchmark (truth known by construction)', fontsize=10)
    left.set_ylabel('worlds where the full method returned\nexactly the true candidate set (%)')
    for component, colour in (('row', '#c0392b'), ('null', '#e67e22')):
        pts = [p for p in data['resid_mid8_blocks'] if p['component'] == component]
        right.plot([p['ratio'] for p in pts], [100 * p['rate'] for p in pts], marker='o',
                   linestyle='-', color=colour, label=f'{component}: frozen normal rule')
        right.plot([p['ratio'] for p in pts], [100 * p['rate_exact_posthoc'] for p in pts],
                   marker='o', markerfacecolor='white', linestyle=':', color=colour,
                   label=f'{component}: exact sign test (post hoc)')
    right.set_title('resid_mid.8, interchange accuracy, blocks of n = 2…16', fontsize=10)
    right.set_ylabel('blocks where at least one endpoint\nwas excluded (%)')
    for ax in (left, right):
        ax.axvline(1.0, color='#888888', linewidth=1)
        ax.set_xscale('log')
        ax.set_xlabel('ratio predicted before the run')
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Two different targets. The right panel depends on the inference rule '
                 'used at small n; neither panel is a reliability.', fontsize=10)
    fig.tight_layout()
    path = os.path.join(HERE, 'figures', 'validation_curve.png')
    fig.savefig(path, dpi=150)
    print('wrote', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command')
    ex = sub.add_parser('export')
    ex.add_argument('--benchmark-src', required=True)
    ex.add_argument('--benchmark-rows', nargs=2, required=True)
    ex.add_argument('--resid-result', required=True)
    args = parser.parse_args()
    if args.command == 'export':
        export(args)
    else:
        plot()


if __name__ == '__main__':
    main()
