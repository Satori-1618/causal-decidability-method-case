"""The twelve-cell example: declare, resolve, decide — checkable by hand.

    python3 examples/twelve_cell.py

Each prompt asks a model to choose Object (O) or Alternative (A) to receive more points;
ties go to Alternative. A donor's activation is patched into a receiver at one declared
site and the choice is read off. Four donors into three receivers give twelve cells.

Nine candidate explanations predict the patched choice. Every entry of the prediction
table is computed from the candidate's rule, not typed in. The script then shows

  * which candidates no outcome of the design can separate (signatures),
  * what the natural six-cell design misses and which donor buys which distinction,
  * the compatible set for a given outcome, with the shift rules decided as one group.

A toy, built to be recomputed by hand. It is not evidence about any model.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from causal_decidability import compatible_set, gains, restrict, signatures  # noqa: E402

DONORS = {'D1': (60, 10), 'D2': (60, 90), 'D3': (30, 10), 'D4': (60, 30)}
RECEIVERS = {'R1': (20, 40), 'R2': (20, 80), 'R3': (80, 20)}
CELLS = [(d, r) for d in DONORS for r in RECEIVERS]


def choice(obj, alt):
    """Strictly more points for Object chooses O; ties choose A."""
    return 'O' if obj > alt else 'A'


def shift(s):
    return lambda d, r: choice(r[0] - r[1] + s, 0)


#: each rule maps (donor points, receiver points) to the predicted patched choice
RULES = {
    'V transfer Object value': lambda d, r: choice(d[0], r[1]),
    'W transfer Alternative value': lambda d, r: choice(r[0], d[1]),
    'D copy donor choice': lambda d, r: choice(*d),
    'F flip receiver choice': lambda d, r: 'A' if choice(*r) == 'O' else 'O',
    'O always Object': lambda d, r: 'O',
    'N preserve receiver choice': lambda d, r: choice(*r),
    'S30 shift point difference by 30': shift(30),
    'S40 shift point difference by 40': shift(40),
    'S50 shift point difference by 50': shift(50),
}
SHIFTS = ('S30 shift point difference by 30', 'S40 shift point difference by 40',
          'S50 shift point difference by 50')


def table():
    """Candidate -> twelve predicted choices, encoded O = 1, A = 0."""
    return {name: tuple(1.0 if rule(DONORS[d], RECEIVERS[r]) == 'O' else 0.0
                        for d, r in CELLS)
            for name, rule in RULES.items()}


def cells_of(*donors):
    return [i for i, (d, _) in enumerate(CELLS) if d in donors]


def render(row):
    letters = ''.join('O' if v else 'A' for v in row)
    return ' | '.join(letters[i:i + 3] for i in range(0, len(letters), 3))


def main():
    predictions = table()
    print('Predicted patched choice, R1R2R3 per donor, D1 | D2 | D3 | D4\n')
    width = max(map(len, predictions))
    for name, row in predictions.items():
        print(f'  {name:<{width}}  {render(row)}')

    full = signatures(predictions)
    print(f'\nFull design, 12 cells: {len(predictions)} candidates, {len(full)} signatures')
    for group in full:
        if len(group) > 1:
            print('  never separable here:', ', '.join(group))

    six = cells_of('D1', 'D2')
    print(f'\nSix-cell design, D1 and D2 only: '
          f'{len(signatures(restrict(predictions, six)))} signatures')
    for donor in ('D3', 'D4'):
        for a, b in gains(predictions, six, cells_of(donor)):
            if not {a, b} <= set(SHIFTS):
                print(f'  adding {donor} separates  {a}  from  {b}')

    groups = [SHIFTS]
    for label, outcome in (('O/A/O from every donor', predictions[SHIFTS[0]]),
                           ("V's row", predictions['V transfer Object value'])):
        result = compatible_set(predictions, outcome, radius=0.0, equivalence_groups=groups)
        print(f'\nOutcome {label}, tolerance 0: {result["outcome"]}')
        print('  retained:', ', '.join(result['retained']))


if __name__ == '__main__':
    main()
