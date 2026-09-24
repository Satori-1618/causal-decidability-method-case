"""Paired resolution of the MLP8 rowspace-vs-nullspace contrast, and of its two coarser siblings.

    python3 scripts/analyze_makelov_table1.py results/makelov_replication_001

Reads the per-example export written by ``run_makelov_table1.py`` and computes, for
each contrast, the separation and the resolution twice: once treating the two arms
as independent panels (what the published means alone allow) and once using the
fact that both arms are measured on the *same* 2000 base/source pairs.

Two outcome scales, both exported per example:

``accuracy``    the interchange-accuracy indicator. Paired resolution is McNemar's:
               with discordant counts b and c, the difference is (b-c)/n and its
               paired standard error is sqrt(b + c - (b-c)^2/n)/n. The unpaired
               standard error sqrt(p1(1-p1)/n + p2(1-p2)/n) ignores the concordant
               pairs entirely, which is the estimate the published table supports.
``logit_diff`` logit(base IO) - logit(base subject) under the patch, the column the
               published table reports. Paired resolution is the standard error of
               the per-example difference.

``ratio`` throughout is separation / resolution, i.e. the same separation-to-
resolution reading the unpaired 1.61 came from. A paired bootstrap over pairs is
reported alongside the closed forms as an independent check, not as a replacement.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

CONTRASTS = (('rowspace', 'nullspace'), ('full', 'nullspace'), ('full', 'rowspace'))
#: the unpaired reading of the headline contrast that this run is meant to settle,
#: recomputed from the published proportions rather than copied.
PUBLISHED_HEADLINE = {'p_rowspace': 0.0065, 'p_nullspace': 0.0030, 'n': 2000}


def load(path):
    arms = {}
    order = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        arm = arms.setdefault(row['condition'], {'acc': {}, 'ld': {}})
        arm['acc'][row['index']] = row['interchange_correct']
        arm['ld'][row['index']] = row['logit_diff_base_ordering']
        order.setdefault(row['index'], row['pair_id'])
    n = len(order)
    out = {}
    for condition, arm in arms.items():
        if len(arm['acc']) != n:
            raise ValueError(f'condition {condition} is missing pairs')
        out[condition] = {
            'acc': np.array([arm['acc'][i] for i in range(n)], dtype=np.float64),
            'ld': np.array([arm['ld'][i] for i in range(n)], dtype=np.float64)}
    return out, n


def unpaired_reference():
    """The 1.61 the published table supports, recomputed from its own proportions."""
    p1, p2, n = (PUBLISHED_HEADLINE['p_rowspace'], PUBLISHED_HEADLINE['p_nullspace'],
                 PUBLISHED_HEADLINE['n'])
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    return {'separation': p1 - p2, 'resolution': se, 'ratio': (p1 - p2) / se,
            'source': 'published proportions, arms treated as independent'}


def bootstrap(a, b, n_resamples=20000, seed=20260920):
    """Resample whole pairs, so both arms move together exactly as they were measured."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_resamples, n))
    draws = (a[idx] - b[idx]).mean(axis=1)
    return {'resolution': float(draws.std(ddof=1)),
            'ci_lower': float(np.quantile(draws, 0.025)),
            'ci_upper': float(np.quantile(draws, 0.975)),
            'n_resamples': n_resamples, 'seed': seed}


def accuracy_contrast(a, b):
    """McNemar paired vs independent-proportions unpaired, on the same indicators."""
    n = len(a)
    discordant_ab = float(np.sum((a == 1) & (b == 0)))
    discordant_ba = float(np.sum((a == 0) & (b == 1)))
    both = float(np.sum((a == 1) & (b == 1)))
    neither = float(np.sum((a == 0) & (b == 0)))
    separation = (discordant_ab - discordant_ba) / n
    paired_var = (discordant_ab + discordant_ba - (discordant_ab - discordant_ba) ** 2 / n)
    paired_se = math.sqrt(max(paired_var, 0.0)) / n
    p1, p2 = a.mean(), b.mean()
    unpaired_se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    total_discordant = discordant_ab + discordant_ba
    chi2 = ((discordant_ab - discordant_ba) ** 2 / total_discordant
            if total_discordant else float('nan'))
    return {
        'scale': 'interchange accuracy indicator',
        'p_a': float(p1), 'p_b': float(p2), 'separation': float(separation),
        'table': {'both_correct': both, 'a_only': discordant_ab,
                  'b_only': discordant_ba, 'neither': neither},
        'paired': {'resolution': paired_se,
                   'ratio': abs(separation) / paired_se if paired_se else float('inf'),
                   'method': 'McNemar standard error of the paired difference',
                   'mcnemar_chi2': chi2,
                   'exact_binomial_p': exact_mcnemar_p(discordant_ab, discordant_ba)},
        'unpaired': {'resolution': unpaired_se,
                     'ratio': abs(separation) / unpaired_se if unpaired_se else float('inf'),
                     'method': 'independent-proportions standard error'},
        'bootstrap_paired': bootstrap(a, b),
    }


def exact_mcnemar_p(b, c):
    """Two-sided exact binomial test on the discordant pairs only."""
    total = int(b + c)
    if total == 0:
        return float('nan')
    k = int(min(b, c))
    tail = sum(math.comb(total, i) for i in range(0, k + 1)) / 2.0 ** total
    return min(1.0, 2.0 * tail)


def logit_contrast(a, b):
    n = len(a)
    diff = a - b
    separation = float(diff.mean())
    paired_se = float(diff.std(ddof=1) / math.sqrt(n))
    unpaired_se = float(math.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n))
    return {
        'scale': 'logit(base IO) - logit(base subject), the published column',
        'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
        'separation': separation,
        'per_example_difference_sd': float(diff.std(ddof=1)),
        'arm_sd_a': float(a.std(ddof=1)), 'arm_sd_b': float(b.std(ddof=1)),
        'pearson_r_between_arms': float(np.corrcoef(a, b)[0, 1]),
        'paired': {'resolution': paired_se,
                   'ratio': abs(separation) / paired_se if paired_se else float('inf'),
                   'method': 'standard error of the per-example difference'},
        'unpaired': {'resolution': unpaired_se,
                     'ratio': abs(separation) / unpaired_se if unpaired_se else float('inf'),
                     'method': 'independent-arms standard error'},
        'bootstrap_paired': bootstrap(a, b),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('directory')
    ap.add_argument('--output', default='paired_resolution.json')
    args = ap.parse_args()
    directory = Path(args.directory)
    arms, n = load(directory / 'per_example.jsonl')

    report = {'n_pairs': n, 'unpaired_reference_from_published': unpaired_reference(),
              'contrasts': {}}
    for first, second in CONTRASTS:
        key = f'{first}_vs_{second}'
        report['contrasts'][key] = {
            'accuracy': accuracy_contrast(arms[first]['acc'], arms[second]['acc']),
            'logit_diff': logit_contrast(arms[first]['ld'], arms[second]['ld']),
        }
    # clean is carried through the export as the inertness reference for nullspace
    if 'clean' in arms:
        report['nullspace_vs_clean'] = {
            'max_abs_logit_diff_gap': float(np.max(np.abs(
                arms['nullspace']['ld'] - arms['clean']['ld']))),
            'indicators_identical': bool(np.array_equal(
                arms['nullspace']['acc'], arms['clean']['acc'])),
            'mean_logit_diff_gap': float(np.mean(
                arms['nullspace']['ld'] - arms['clean']['ld'])),
        }

    (directory / args.output).write_text(json.dumps(report, indent=1))

    ref = report['unpaired_reference_from_published']
    print(f"published unpaired reference: separation {ref['separation']:.6f} / "
          f"resolution {ref['resolution']:.6f} = ratio {ref['ratio']:.3f}")
    for key, block in report['contrasts'].items():
        print(f'\n{key}  (n={n} pairs)')
        for scale in ('accuracy', 'logit_diff'):
            entry = block[scale]
            print(f"  {scale:<11} separation {entry['separation']:+.6f}")
            for mode in ('unpaired', 'paired'):
                print(f"    {mode:<9} resolution {entry[mode]['resolution']:.6f}"
                      f"   ratio {entry[mode]['ratio']:.3f}")
            print(f"    bootstrap resolution {entry['bootstrap_paired']['resolution']:.6f}"
                  f"   95% CI [{entry['bootstrap_paired']['ci_lower']:+.6f},"
                  f" {entry['bootstrap_paired']['ci_upper']:+.6f}]")
            if scale == 'accuracy':
                table = entry['table']
                print(f"    discordant a-only {table['a_only']:.0f}  b-only "
                      f"{table['b_only']:.0f}  both {table['both_correct']:.0f}"
                      f"   exact McNemar p {entry['paired']['exact_binomial_p']:.3g}")
    print('\nwrote', directory / args.output)


if __name__ == '__main__':
    main()
