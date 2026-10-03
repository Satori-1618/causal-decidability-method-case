"""Verify a plan freeze without importing a model or loading outcome records."""
import hashlib
import json
import math
from pathlib import Path

import plan_analysis as analysis

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def verify_hashes(root, files):
    root = Path(root).resolve()
    for name, expected in files.items():
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError('Source path leaves repository: ' + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Source changed: ' + name)


def verify_contract(plan):
    workflow = (plan['status'], plan['execution_authorized'])
    if workflow not in [('awaiting_final_pre_run_review', False), ('released_for_execution', True)]:
        raise ValueError('Inconsistent review-only or released workflow state')
    if plan['model_measurements_performed'] is not False:
        raise ValueError('Pre-run plan must not be relabelled as an outcome record')
    if type(plan['provenance']['new_input_pools_prepared']) is not bool:
        raise ValueError('Input preparation status must be explicit')
    if plan['execution_authorized'] and not plan['provenance']['new_input_pools_prepared']:
        raise ValueError('Cannot release without prepared inputs')
    cohort = plan['cohort']
    if len(cohort) != analysis.HEADS or len({c['model'] for c in cohort}) != analysis.HEADS:
        raise ValueError('Expected exactly six distinct frozen models')
    if any(c['model'] == plan['reference']['model'] for c in cohort):
        raise ValueError('Reference model cannot enter the transfer cohort')
    seeds = [c[k] for c in cohort for k in ('recipient_seed', 'donor_seed')]
    if len(set(seeds)) != 2*analysis.HEADS:
        raise ValueError('Duplicate input seeds')
    checks = [
        (plan['screen']['native_candidates_per_head'], analysis.SCREEN_BUDGET),
        (plan['screen']['families_per_stratum'], analysis.N),
        (plan['inference']['families_per_stratum'], analysis.N),
        (plan['inference']['heads'], analysis.HEADS),
        (plan['inference']['one_sided_cp_bounds'], 4*analysis.HEADS),
        (plan['inference']['tail_alpha'], analysis.TAIL_ALPHA),
        (plan['inference']['minimum_uplift'], analysis.MINIMUM_UPLIFT),
        (plan['inference']['required_substantial_heads'], analysis.REQUIRED_HEADS),
        (plan['cost']['baseline_sequence_forwards'], analysis.BASELINE_SEQUENCE_FORWARDS),
        (plan['cost']['two_anchor_family_sequence_forwards'], analysis.FAMILY_SEQUENCE_FORWARDS),
    ]
    if any(a != b for a, b in checks):
        raise ValueError('Plan and calculator disagree')
    cost = analysis.SCREEN_BUDGET*analysis.BASELINE_SEQUENCE_FORWARDS + 2*analysis.N*analysis.FAMILY_SEQUENCE_FORWARDS
    if (plan['cost']['maximum_scheduled_sequence_forwards_per_head'] != cost
            or plan['cost']['maximum_scheduled_sequence_forwards_cohort'] != analysis.HEADS*cost):
        raise ValueError('Scheduled forward count mismatch')
    ref = plan['reference']
    reconstructed = (ref['old_signed_margin_cutoff']-ref['center'])/ref['radius']
    if not math.isclose(reconstructed, ref['normalized_cutoff'], rel_tol=0., abs_tol=1e-15):
        raise ValueError('Reference normalization changed')
    if (plan['intervention']['secondary_target_transfers'] is not False
            or plan['mechanistic_next']['execution_authorized'] is not False):
        raise ValueError('No secondary model run is authorized')


def require_execution_release(plan, release):
    """No command-line override: final review and explicit user release required."""
    verify_contract(plan)
    review = release.get('final_review', {})
    if (plan['execution_authorized'] is not True
            or plan['status'] != 'released_for_execution'
            or release.get('status') != 'approved'
            or release.get('execution_authorized') is not True
            or review.get('status') != 'PASS'
            or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
            or not isinstance(review.get('reviewed_commit'), str)
            or len(review['reviewed_commit']) != 40
            or any(c not in '0123456789abcdef' for c in review['reviewed_commit'])
            or not isinstance(release.get('user_execution_authorization'), str)
            or not release['user_execution_authorization'].strip()):
        raise ValueError('Execution blocked: final pre-run review and user release are pending')


def main():
    lock = json.loads((HERE/'SOURCE_LOCK.json').read_text())
    verify_hashes(ROOT, lock['files'])
    plan = json.loads((HERE/'plan.json').read_text())
    verify_contract(plan)
    release = json.loads((HERE/'EXECUTION_RELEASE.json').read_text())
    if plan['execution_authorized']:
        require_execution_release(plan, release)
    elif release.get('status') != 'pending' or release.get('execution_authorized') is not False:
        raise ValueError('Plan and execution-release status disagree')
    old = json.loads((HERE.parent.parent/'native_followup/frozen/confirmation_001/COHORT_ASSET_LOCK.json').read_text())
    for head in plan['cohort']:
        if head['sha256'] != old['files'][head['checkpoint']]['sha256']:
            raise ValueError('Checkpoint differs from prior public source lock')
    print('PASS: {} file hashes; six-head plan matches calculator and checkpoint locks; {}'.format(
        len(lock['files']), 'execution released' if plan['execution_authorized'] else 'final review pending; execution not authorized'))


if __name__ == '__main__':
    main()
