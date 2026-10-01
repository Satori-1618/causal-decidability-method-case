"""Public-policy seams and the create-only development evidence chain."""
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
from statistics import NormalDist
import sys

import numpy as np
import pytest

# The archived runner imports and executes tensor-based synthetic worlds.
pytest.importorskip('torch', reason='design-comparison runner tests require .[test-hooks]')

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'applications/design-comparison'
sys.path.insert(0, str(ROOT / 'src'))
from causal_decidability import compatible_set  # noqa: E402

spec = importlib.util.spec_from_file_location('comparison_runner_tests', APP / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
selection = runner.selection


@pytest.fixture(scope='module')
def case():
    return runner.worlds.public_case(192, 'linear', 0.8)


def choose(case, **kwargs):
    return selection.select(case['predictions'], case['equivalence_groups'],
        case['menu'], [0.1] * len(case['menu']), case['numerical_bound'],
        mandatory_count=4, extras=kwargs.pop('extras', 3), samples=8, **kwargs)


@pytest.mark.parametrize('extras', [1, 2, 3, 4])
def test_all_compared_policies_spend_same_budget(case, extras):
    plans = choose(case, extras=extras)['plans']
    budget = (4 + extras) * 8
    for policy in selection.POLICIES:
        plan = plans[policy]
        assert plan['cost'] == sum(plan['counts']) == budget
        assert len(plan['cells']) == len(set(plan['cells']))
        assert plan['cells'][:4] == [0, 1, 2, 3]
        assert all(n >= 1 for n in plan['counts'])
    assert plans['full_only']['cells'] == [0, 1, 2, 3]
    assert plans['full_menu']['cost'] > budget


def test_weak_selected_cells_do_not_redefine_equivalence(case):
    predictions = {name: values[:4] for name, values in case['predictions'].items()}
    result = compatible_set(predictions, predictions['channel_a'], 1e-6,
                            case['equivalence_groups'])
    assert result['outcome'] == 'insufficient_evidence'
    assert len(result['retained_groups']) == 6
    assert result['retained'] == sorted(predictions)


def test_selection_has_no_outcome_input_and_aliases_do_not_reweight(case):
    assert not {'truth', 'observations', 'estimate', 'measurements'} & set(
        inspect.signature(selection.select).parameters)
    with pytest.raises(TypeError):
        choose(case, observations=[1000] * len(case['menu']))
    unaliased = {**case, 'predictions': {k: v for k, v in case['predictions'].items()
                                       if k != 'channel_a_alias'},
                 'equivalence_groups': [[n for n in g if n != 'channel_a_alias']
                                        for g in case['equivalence_groups']]}
    assert choose(case)['plans'] == choose(unaliased)['plans']
    assert choose(case)['plans'] == choose(case)['plans']


def test_known_gaussian_simultaneous_radius_and_boundary_binomial_bound():
    sigmas, counts, alpha, floor = [0.2, 0.5], [4, 25], 0.01, 0.003
    expected = NormalDist().inv_cdf(1 - alpha / 4) * np.array(sigmas) / np.sqrt(counts) + floor
    np.testing.assert_allclose(selection.radii(sigmas, counts, alpha, floor), expected)
    assert runner.binomial_upper(0, 16, .05) == pytest.approx(0.17074972298248095)
    assert runner.binomial_upper(16, 16, .05) == 1.0


def test_nonfinite_candidate_prediction_is_not_a_selection_result(case):
    predictions = {name: list(values) for name, values in case['predictions'].items()}
    predictions['channel_b'][4] = float('inf')
    with pytest.raises(ValueError, match='finite predictions'):
        choose({**case, 'predictions': predictions})


@pytest.mark.parametrize('sigmas,counts,alpha,floor', [
    ([float('nan')], [8], .01, 0), ([0], [8], .01, 0),
    ([.1], [0], .01, 0), ([.1], [1.5], .01, 0),
    ([.1], [8], float('nan'), 0), ([.1], [8], .01, float('inf')),
])
def test_invalid_uncertainty_declarations_fail_closed(sigmas, counts, alpha, floor):
    with pytest.raises(ValueError):
        selection.radii(sigmas, counts, alpha, floor)


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory):
    directory = tmp_path_factory.mktemp('comparison') / 'run'
    runner.generate(directory, cases=8, seed=812, samples=8)
    labels = directory / 'private_labels.jsonl'
    hidden = directory / 'temporarily_unavailable_labels'
    labels.rename(hidden)
    try:
        runner.predict(directory)  # Neither label reading nor hashing is permitted here.
    finally:
        hidden.rename(labels)
    runner.score(directory)
    return directory


def test_small_pipeline_reproduces_and_refuses_overwrite(pipeline):
    verified = runner.verify(pipeline)
    assert verified == {'verified': True, 'cases': 8,
                        'reserved_structures_executed': 0, 'status': 'development_only'}
    assert len(runner.lines(pipeline / 'predictions.jsonl')) == 8 * 4 * 6
    assert runner.read_json(pipeline / 'summary.json')['all_equivalence_groups_preserved']
    for action in (lambda: runner.generate(pipeline, cases=8),
                   lambda: runner.predict(pipeline), lambda: runner.score(pipeline)):
        with pytest.raises(FileExistsError):
            action()


def test_poisoned_public_schema_is_rejected_even_with_matching_hash(pipeline, tmp_path):
    directory = tmp_path / 'poison'
    shutil.copytree(pipeline, directory)
    public = directory / 'public_cases.jsonl'
    records = runner.lines(public)
    records[0]['true_mechanism'] = 'forbidden'
    public.write_text(''.join(json.dumps(row) + '\n' for row in records))
    generation = runner.read_json(directory / 'generation.json')
    generation['artifacts']['public_cases.jsonl'] = runner.sha(public)
    (directory / 'generation.json').write_text(json.dumps(generation))
    with pytest.raises(ValueError, match='Unexpected public case fields'):
        runner.prediction_rows(directory)


def test_edited_predictions_break_the_seal(pipeline, tmp_path):
    directory = tmp_path / 'edited'
    shutil.copytree(pipeline, directory)
    predictions = directory / 'predictions.jsonl'
    predictions.write_text(predictions.read_text() + '\n')
    with pytest.raises(ValueError, match='Artifact hash mismatch: predictions.jsonl'):
        runner.verify(directory)
