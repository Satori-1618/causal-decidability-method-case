"""Model-free source/input verification and one shared execution-release schema."""
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
    if plan['model_measurements_performed'] is not False:
        raise ValueError('Keep pre-run plan immutable; outcomes belong in run records')
    model = plan['model']
    if (model['model'], model['layer_one_based'], model['head_one_based'],
            model['model_heads'], model['hidden_width'], model['head_width']) != ('a9g0io1r', 2, 1, 4, 64, 16):
        raise ValueError('Reference architecture/head differs from the frozen runtime')
    if model['sha256'] != 'abec37ff9899e0349de9e19558b61dddadcdf7e15628aaa0fa55e70b435b9f4e':
        raise ValueError('Reference checkpoint changed')
    sampling = plan['sampling']
    if (sampling['native_candidates'], sampling['families'], sampling['phase'],
            sampling['recipient_seed'], sampling['donor_seed']) != (2048, 256, 'confirmation', 26100341, 26100342):
        raise ValueError('Frozen sampling recipe changed')
    if (plan['screen']['strict_cutoff_nat'], plan['target_start']['minimum_separating_families'],
            plan['target_start']['total_families']) != (8., 128, 256):
        raise ValueError('Screen or target-start rule changed')
    intervention = plan['intervention']
    if (intervention['balances'], intervention['positions'], intervention['minimum_prefix_balance'],
            intervention['target_symbol'], intervention['calibration_transfers_per_family'],
            intervention['target_transfers_per_family'], intervention['dtypes']) != (
            [-2, 2], [20, 28], -4, ')', 8, 8, ['float32', 'float64']):
        raise ValueError('Intervention grid changed')
    expected = {'scientific_tolerance_nat': .1, 'numerical_allowance_nat': .001,
                'definite_hit_limit_nat': .099, 'possible_hit_limit_nat': .101,
                'separation_gap_nat': .202, 'same_cell_witness_gap_nat': .202}
    if any(intervention[k] != v for k, v in expected.items()):
        raise ValueError('Scientific or numerical boundary changed')
    inference = plan['inference']
    if (inference['adequacy_target'], inference['candidate_tail_alpha'], inference['candidate_tail_count'],
            inference['paired_alpha'], inference['familywise_alpha_upper_bound']) != (.9, .01, 4, .01, .05):
        raise ValueError('Error family or adequacy target changed')
    if (plan['cost']['maximum_total_sequence_forwards'], plan['cost']['calibration_stop_sequence_forwards'],
            plan['cost']['neural_retries_authorized']) != (36864, 20480, False):
        raise ValueError('Frozen inference budget changed')
    if plan['future_confirmation_start']['automatic_execution'] is not False:
        raise ValueError('No subsequent experiment is authorized')


def require_release(plan, release):
    verify_contract(plan)
    review = release.get('final_review', {})
    commit = review.get('reviewed_commit')
    if not (plan['status'] == 'released_for_execution' and plan['execution_authorized'] is True
            and release.get('status') == 'approved' and release.get('execution_authorized') is True
            and review.get('status') == 'PASS'
            and isinstance(review.get('reviewer'), str) and review['reviewer'].strip()
            and isinstance(commit, str) and len(commit) == 40
            and all(ch in '0123456789abcdef' for ch in commit)
            and isinstance(release.get('user_execution_authorization'), str)
            and release['user_execution_authorization'].strip()):
        raise ValueError('Execution blocked: final independent review and explicit user release are pending')


def verify_source_lock():
    lock_path = HERE/'SOURCE_LOCK.json'
    lock = json.loads(lock_path.read_text())
    files = lock['files']
    required = list(HERE.glob('*.py')) + [HERE/n for n in ('plan.json', 'PROTOCOL.md', 'EXECUTION_RELEASE.json')]
    required += [p for p in (HERE/'inputs').rglob('*') if p.is_file()]
    for path in required:
        if str(path.relative_to(ROOT)) not in files:
            raise ValueError('Unbound source or input: '+str(path.relative_to(ROOT)))
    for name, expected in files.items():
        path = (ROOT/name).resolve()
        if ROOT not in path.parents or digest(path) != expected:
            raise ValueError('Changed or invalid source: '+name)
    return {**files, str(lock_path.relative_to(ROOT)): digest(lock_path)}


def main():
    plan = json.loads((HERE/'plan.json').read_text())
    release = json.loads((HERE/'EXECUTION_RELEASE.json').read_text())
    verify_contract(plan)
    files = verify_source_lock()
    from prepare import validate_inputs
    preparation, candidates, families = validate_inputs(plan, HERE/'inputs')
    if digest(HERE/'inputs/preparation.json') != plan['provenance']['prepared_inputs_sha256']:
        raise ValueError('Prepared manifest differs from the plan binding')
    if plan['execution_authorized']:
        require_release(plan, release)
    elif release.get('status') != 'pending' or release.get('execution_authorized') is not False:
        raise ValueError('Plan and release state disagree')
    print('PASS: {} source bindings, {} unscored candidates, {} donor families; execution {}'.format(
        len(files), len(candidates), len(families), 'released' if plan['execution_authorized'] else 'blocked'))


if __name__ == '__main__':
    main()
