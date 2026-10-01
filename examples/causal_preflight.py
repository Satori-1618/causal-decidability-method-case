"""Two pre-experiment checks; standard library only, no model execution.

Run without arguments for three stages of a hypothetical design. Pass --config
for your own fixed mean predictions and separately justified calibration inputs.
Use --structure-only to compare fixed numeric predictions without calibration, or
choice categories declared with "prediction_kind": "category".
See docs/METHOD_PREFLIGHT.md. This is a conditional screen, not a power calculator.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from causal_decidability.design import gains, restrict, signatures  # noqa: E402

EXAMPLE = Path(__file__).parent / 'data' / 'causal_preflight_example.json'


def number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f'{name} must be finite and >= {minimum}')
    return value


def text_field(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')


def category(value):
    text_field(value, 'category prediction')
    if value.strip().upper() == 'UNKNOWN':
        raise ValueError('UNKNOWN predictions block the comparison; they are not a shared category')
    return value


def check(config, selected=None, n=None, structure_only=False):
    """Return mean-signature groups and a conditional resolution screen per pair.

    The interval family covers ALL declared cells, including optional cells.
    A pair clears the screen if any selected cell has gap > 2*(r_stat+b).
    Unknown calibrations are preserved. Predictions are treated as fixed/exact
    inputs; uncertainty in fitted rival predictions is not supported here.
    structure_only needs no sampling or uncertainty inputs and assesses no radius.
    Only structure_only accepts prediction_kind 'category' (e.g. the chosen option);
    labels are compared for equality only and have no numeric gap.
    """
    cells = config['cells']
    if not isinstance(cells, list) or not cells:
        raise ValueError('declare at least one cell')
    names = [c['id'] for c in cells]
    for name in names:
        text_field(name, 'cell id')
    if len(set(names)) != len(names):
        raise ValueError('duplicate cell ids')
    selected = list(names if selected is None else selected)
    if (not selected or any(not isinstance(s, str) for s in selected)
            or len(set(selected)) != len(selected)
            or any(s not in names for s in selected)):
        raise ValueError('selected cells must be distinct declared cell ids')
    indices = [names.index(s) for s in selected]
    predictions = config['predictions']
    if not isinstance(predictions, dict) or len(predictions) < 2:
        raise ValueError('declare at least two rivals')
    kind = config.get('prediction_kind', 'mean')
    if kind not in ('mean', 'category'):
        raise ValueError("prediction_kind must be 'mean' or 'category'")
    if kind == 'category' and not structure_only:
        raise ValueError('category predictions have no numeric gap; use the structure-only check')
    for name, row in predictions.items():
        text_field(name, 'rival name')
        if len(row) != len(cells):
            raise ValueError('every rival must predict every declared cell')
        for value in row:
            category(value) if kind == 'category' else number(value, 'prediction')
    text_field(config.get('prediction_source'), 'prediction_source')
    table = predictions
    if kind == 'category':
        # Equality is all that matters, so any injective coding keeps the grouping exact.
        codes = {label: i for i, label in enumerate(sorted({v for row in predictions.values() for v in row}))}
        table = {name: [codes[v] for v in row] for name, row in predictions.items()}
    extras = [i for i in range(len(cells)) if i not in indices]
    noun = 'means' if kind == 'mean' else 'categories'
    group_key = 'identical_mean_groups' if kind == 'mean' else 'identical_category_groups'
    structure = {
        'selected_cells': selected,
        group_key: signatures(restrict(table, indices)),
        'pairs_separated_by_all_optional_cells': gains(table, indices, extras),
    }
    if structure_only:
        pairs = []
        for a, b in itertools.combinations(sorted(predictions), 2):
            differs = any(predictions[a][i] != predictions[b][i] for i in indices)
            pairs.append({
                'rivals': [a, b], 'cells': [],
                'status': f'different_declared_{noun}' if differs else f'identical_declared_{noun}',
            })
        return {
            **structure,
            'scope': ('Structural check of supplied fixed numeric mean predictions only.'
                      if kind == 'mean' else
                      'Structural check of supplied choice categories only; equality, no gap.'),
            'mode': 'structure_only',
            'prediction_kind': kind,
            'calibration': 'Measurement resolution not assessed; no sample-size conclusion.',
            'all_pairs_clear_planning_screen': None,
            'pairs': pairs,
            'intervention_fidelity': 'Not checked by this calculator; verify separately.',
        }
    text_field(config.get('independent_unit'), 'independent_unit')
    n = config.get('n') if n is None else n
    if n is not None and (type(n) is not int or n < 1):
        raise ValueError('n must be a positive integer count of independent units')
    alpha = number(config['alpha'], 'alpha')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    probability = 1 - alpha / (2 * len(cells))
    if not 0 < probability < 1:
        raise ValueError('alpha/cell count exceeds numerical resolution')
    z = NormalDist().inv_cdf(probability)

    budgets = {}
    for cell in cells:
        sd = cell.get('sd_per_unit')
        numeric = cell.get('numerical_allowance')
        for value, key in ((sd, 'sd_per_unit'), (numeric, 'numerical_allowance')):
            if value is not None:
                number(value, key, 0)
        if sd is not None:
            if cell.get('sd_kind') not in ('known_gaussian', 'pilot_estimate'):
                raise ValueError('sd_kind must be known_gaussian or pilot_estimate')
            text_field(cell.get('sd_source'), 'sd_source')
        if numeric is not None:
            text_field(cell.get('numerical_source'), 'numerical_source')
        stat = None if n is None or sd is None else z * sd / math.sqrt(n)
        if stat is not None:
            number(stat, 'computed statistical radius', 0)
        total = None if stat is None or numeric is None else stat + numeric
        if total is not None:
            number(total, 'computed radius', 0)
        budgets[cell['id']] = {
            'statistical_radius': stat, 'numerical_allowance': numeric,
            'total_radius': total, 'sd_kind': cell.get('sd_kind'),
        }

    pairs = []
    for a, b in itertools.combinations(sorted(predictions), 2):
        rows = []
        for i in indices:
            gap = abs(predictions[a][i] - predictions[b][i])
            number(gap, 'computed prediction gap', 0)
            budget = budgets[names[i]]
            radius = budget['total_radius']
            numeric = budget['numerical_allowance']
            if gap == 0:
                status = 'identical_declared_means'
            elif radius is None:
                status = 'resolution_unknown'
            elif gap > 2 * radius:
                status = 'planning_separated'
            elif gap <= 2 * numeric:
                status = 'numerical_allowance_blocks_this_screen'
            else:
                status = 'planning_overlap'
            rows.append({'cell': names[i], 'gap': gap, 'status': status, **budget})
        statuses = {row['status'] for row in rows}
        if 'planning_separated' in statuses:
            status = 'planning_separated'
        elif 'resolution_unknown' in statuses:
            status = 'resolution_unknown'
        elif statuses == {'identical_declared_means'}:
            status = 'identical_declared_means'
        elif statuses <= {'identical_declared_means', 'numerical_allowance_blocks_this_screen'}:
            status = 'numerical_allowance_blocks_this_screen'
        else:
            status = 'planning_overlap'
        pairs.append({'rivals': [a, b], 'status': status, 'cells': rows})
    return {
        **structure,
        'scope': 'Declared fixed mean predictions; no empirical winner or power claim.',
        'calibration': 'Conditional on SD assumptions and supplied numerical allowances; '
                       'pilot SD uncertainty and fitted-prediction uncertainty are not covered.',
        'n': n, 'independent_unit': config['independent_unit'],
        'simultaneous_cell_count': len(cells), 'alpha': alpha,
        'all_pairs_clear_planning_screen': all(p['status'] == 'planning_separated' for p in pairs),
        'pairs': pairs,
        'intervention_fidelity': 'Not checked by this calculator; verify separately.',
    }


def display(result):
    suffix = ('structural check only' if result.get('mode') == 'structure_only'
              else f"n={result['n']}")
    print(f"\nCells: {', '.join(result['selected_cells'])}; {suffix}")
    print(' ', result['calibration'])
    for pair in result['pairs']:
        print(f"  {' / '.join(pair['rivals'])}: {pair['status']}")
        for cell in pair['cells']:
            radius = cell['total_radius']
            threshold = 'unknown' if radius is None else f'{2 * radius:.4f}'
            label = ('pilot SD: provisional' if cell['sd_kind'] == 'pilot_estimate'
                     else cell['sd_kind'] or 'SD unknown')
            print(f"    {cell['cell']}: gap={cell['gap']:.4f}; 2*radius={threshold}; {label}")
    if result['pairs_separated_by_all_optional_cells']:
        print('  Adding optional cells changes predictions for:',
              result['pairs_separated_by_all_optional_cells'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON template; otherwise show the teaching example')
    parser.add_argument('--n', type=int, help='override independent-unit count')
    parser.add_argument('--cells', nargs='+', help='use these declared cell ids')
    parser.add_argument('--structure-only', action='store_true',
                        help='compare fixed numeric predictions; skip all resolution inputs')
    parser.add_argument('--json', action='store_true', help='emit one full machine-readable report')
    args = parser.parse_args()
    try:
        config = json.loads((args.config or EXAMPLE).read_text())
        if args.config or args.n is not None or args.cells or args.json or args.structure_only:
            result = check(config, args.cells, args.n, args.structure_only)
            print(json.dumps(result, indent=2, allow_nan=False)) if args.json else display(result)
        else:
            print('ILLUSTRATIVE INPUTS, NOT MODEL RESULTS. Conditional mean-resolution screen.')
            for cells, n in ((['full_patch'], 64), (None, 64), (None, 256)):
                display(check(config, cells, n))
        if not args.json:
            print('\nThis screen does not estimate power or identify a mechanism. '
                  'Verify intervention fidelity separately.')
    except (KeyError, TypeError, ValueError, OverflowError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
