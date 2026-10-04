"""Model-free scientific-contract, source, input and release verification."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_contract(plan):
    if (plan['status'], plan['execution_authorized']) not in (
            ('awaiting_final_pre_run_review', False), ('released_for_execution', True)):
        raise ValueError('Inconsistent review/release state')
    if plan['round'] != 'history_005' or plan['model_measurements_performed'] is not False:
        raise ValueError('Keep pre-run plan immutable; outcomes belong in run records')
    model = plan['model']
    if tuple(model[k] for k in ('model', 'layer_one_based', 'head_one_based',
                                'model_heads', 'hidden_width', 'head_width')) != ('a9g0io1r', 2, 1, 4, 64, 16):
        raise ValueError('Reference architecture/head changed')
    if model['sha256'] != 'abec37ff9899e0349de9e19558b61dddadcdf7e15628aaa0fa55e70b435b9f4e':
        raise ValueError('Reference checkpoint changed')
    sample = plan['sampling']
    if tuple(sample[k] for k in ('native_candidates', 'families', 'phase', 'recipient_seed',
                                 'donor_seed')) != (2048, 256, 'confirmation', 26100451, 26100452):
        raise ValueError('Frozen sampling recipe changed')
    if (plan['screen']['strict_cutoff_nat'], plan['target_start']['minimum_separating_families'],
            plan['target_start']['total_families']) != (8., 64, 256):
        raise ValueError('Screen or target-start rule changed')
    intervention = plan['intervention']
    exact = {'balances': [-2], 'positions': [28], 'minimum_prefix_balance': -4,
             'target_symbol': ')', 'recencies': [6, 10],
             'recency_cores': {'t6': '(())((', 't10': '(()()('},
             'suffix_patterns': {'alt': '()()', 'close': '(())'},
             'recency_edit_positions_one_based': [22, 23],
             'suffix_edit_positions_one_based': [26, 27],
             'calibration_transfers_per_family': 8, 'target_transfers_per_family': 8,
             'dtypes': ['float32', 'float64'], 'numerical_allowance_nat': .001,
             'comparison_numerical_guard_nat': .002, 'meaningful_error_advantage_nat': .02,
             'robust_error_advantage_nat': .022}
    if any(intervention.get(k) != v for k, v in exact.items()):
        raise ValueError('History grid or scientific/numerical boundary changed')
    inference = plan['inference']
    if (inference['pairs'] != [['H_recency', 'H_suffix'], ['H_recency', 'H_constant'],
                               ['H_suffix', 'H_constant']]
            or inference['pair_alpha'] != .05 / 3
            or inference['directional_tail_alpha'] != .05 / 6
            or inference['familywise_alpha_upper_bound'] != .05
            or inference['absolute_adequacy_test'] is not False
            or inference['equivalence_test'] is not False):
        raise ValueError('Comparison or error family changed')
    if (plan['cost']['maximum_total_sequence_forwards'],
            plan['cost']['calibration_stop_sequence_forwards'],
            plan['cost']['neural_retries_authorized']) != (36864, 20480, False):
        raise ValueError('Frozen inference budget changed')
    if plan['future_confirmation_start']['automatic_execution'] is not False:
        raise ValueError('No subsequent experiment is authorized')


def require_release(plan, release):
    verify_contract(plan)
    review = release.get('final_review', {})
    commit = review.get('reviewed_commit')
    if not (plan['status'] == 'released_for_execution' and plan['execution_authorized'] is True
            and release.get('round') == 'history_005'
            and release.get('status') == 'approved' and release.get('execution_authorized') is True
            and review.get('status') == 'PASS'
            and isinstance(review.get('reviewer'), str) and review['reviewer'].strip()
            and isinstance(commit, str) and len(commit) == 40
            and all(ch in '0123456789abcdef' for ch in commit)
            and isinstance(release.get('user_execution_authorization'), str)
            and release['user_execution_authorization'].strip()):
        raise ValueError('Execution blocked: independent review and explicit new-round release are pending')


def verify_source_lock():
    path = HERE / 'SOURCE_LOCK.json'
    lock = json.loads(path.read_text())
    files = lock['files']
    required = list(HERE.glob('*.py'))
    required += [HERE / n for n in ('plan.json', 'planning.json', 'PROTOCOL.md',
                                    'REVIEW.md', 'EXECUTION_RELEASE.json')]
    required += [p for p in (HERE / 'inputs').rglob('*') if p.is_file()]
    for item in required:
        if str(item.relative_to(ROOT)) not in files:
            raise ValueError('Unbound source or input: ' + str(item.relative_to(ROOT)))
    for name, expected in files.items():
        item = (ROOT / name).resolve()
        if ROOT not in item.parents or digest(item) != expected:
            raise ValueError('Changed or invalid source: ' + name)
    return dict(files, **{str(path.relative_to(ROOT)): digest(path)})


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    release = json.loads((HERE / 'EXECUTION_RELEASE.json').read_text())
    verify_contract(plan)
    files = verify_source_lock()
    from prepare import validate_inputs
    preparation, candidates, families = validate_inputs(plan, HERE / 'inputs')
    if digest(HERE / 'inputs/preparation.json') != plan['provenance']['prepared_inputs_sha256']:
        raise ValueError('Prepared manifest differs from plan binding')
    if plan['execution_authorized']:
        require_release(plan, release)
    elif (release.get('status'), release.get('execution_authorized'), release.get('round')) != (
            'pending', False, 'history_005'):
        raise ValueError('Plan and release state disagree')
    print('PASS: {} source bindings; {} unscored candidates; {} donor families; execution {}'.format(
        len(files), len(candidates), len(families), 'released' if plan['execution_authorized'] else 'blocked'))


if __name__ == '__main__':
    main()
