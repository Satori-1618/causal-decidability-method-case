"""Why a fifth condition helps, and why it still cannot identify a unique graph.

Run: python3 -I -S examples/iterative_query_transfer.py [--json]
Constructed scalar worlds only. No LLM outputs, empirical precision or sampling.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'),
               str(ROOT / 'applications/makelov-2311.17030/src')]
from causal_decidability.design import gains, restrict, signatures  # noqa: E402
from query_route_analysis import analyze_records  # noqa: E402

NOTICE = 'CONSTRUCTED TEACHING EXAMPLES — NOT LLM RESULTS'
# X: original patch. M: query value, naturally M=X. C/F/G are identities.
ARMS = {'A': (0, 0), 'B': (1, 1), 'C': (0, 0), 'D': (1, 0),
        'E': (0, 1), 'F': (1, 1), 'G': (0, 0)}


def cancellation(x, m, *, u_override=None):
    """Two active extra paths U=X*M and V=X*M cancel in the tested menu."""
    u, v = x * m if u_override is None else u_override, x * m
    return m + u - v


WORLDS = {'transfer': (lambda x, m: m, 'Y=M'),
          'joint_dependence': (lambda x, m: x * m, 'Y=X*M'),
          'preservation': (lambda x, m: x, 'Y=X'),
          'cancelled_extra_paths': (cancellation, 'U=X*M; V=X*M; Y=M+U-V')}


def analyze_world(name, function, equation):
    cells = {arm: function(x, m) for arm, (x, m) in ARMS.items()}
    # Opposite orientations are components of ONE constructed pair, not two units.
    precision = {'cells': {arm: [y, -y] for arm, y in cells.items()},
                 'controls': {'passed': True, 'identity_tolerance': 0}}
    record = {'pair_id': name, 'precisions': {
        p: copy.deepcopy(precision) for p in ('float32', 'float64')}}
    result = analyze_records([record])
    return {'name': name, 'equation': equation, 'cells': cells,
            'contrasts': result['records'][0]['precisions']['float64']['contrasts'][0],
            'profile_fits': result['records'][0]['profile_successes'],
            'population_status': result['outcome']}


def demonstration():
    worlds = [analyze_world(name, fn, equation) for name, (fn, equation) in WORLDS.items()]
    predictions = {w['name']: [w['cells'][arm] for arm in 'ABCDE'] for w in worlds}
    return {'notice': NOTICE,
            'variables': 'X=original patch; M=query mediator, naturally M=X; Y=output.',
            'arms': {'A': 'patch off, queries free', 'B': 'patch on, queries free',
                     'C': 'patch off, baseline queries', 'D': 'patch on, baseline queries',
                     'E': 'patch off, patched queries'},
            'before_A_to_D': signatures(restrict(predictions, range(4))),
            'after_A_to_E': signatures(predictions),
            'newly_separated_pairs': gains(predictions, range(4), [4]), 'worlds': worlds,
            'outside_menu_counterexample': {
                'intervention': 'At X=M=1, additionally set auxiliary U=0.',
                'transfer_output': 1, 'cancelled_extra_paths_output': cancellation(1, 1, u_override=0)},
            'scope': 'The same shipped analyzer checks the constructed tables. Its precision '
                     'and control inputs are stipulated here, not empirically verified. One '
                     'constructed pair establishes no population coverage. The fifth arm '
                     'separates these response profiles, not all possible causal graphs.'}


def render(result):
    lines = [result['notice'], result['variables'],
             'A-D: removing the query change erases the effect in BOTH Y=M and Y=X*M.',
             'E: transplant changed queries WITHOUT the original patch: Y=M gives 1; Y=X*M gives 0.',
             'Model                         A B C D E | T R S | fitted profile']
    for world in result['worlds']:
        cells = ' '.join(str(world['cells'][arm]) for arm in 'ABCDE')
        contrasts = ' '.join(f'{world["contrasts"][c]:g}' for c in ('T', 'R', 'S'))
        fits = ', '.join(p for p, fits in world['profile_fits'].items() if fits) or 'none'
        lines.append(f'{world["name"]:29} {cells} | {contrasts} | {fits}')
    lines.extend([f'Before: {result["before_A_to_D"]}', f'After: {result["after_A_to_E"]}',
                  'Limit: Y=M+U-V, with U=V=X*M, still looks exactly like Y=M.',
                  'Only a different intervention (for example, set U=0) separates those graphs.',
                  result['scope']])
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = demonstration()
    print(json.dumps(result, indent=2, allow_nan=False) if args.json else render(result))
