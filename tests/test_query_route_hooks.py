"""Tensor tests for actual query-write semantics; optional research runtime."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'applications/makelov-2311.17030/src'))
from query_route import query_hook


def test_only_declared_heads_at_absolute_position_change():
    tensor = torch.arange(2*5*12*4, dtype=torch.float32).reshape(2, 5, 12, 4)
    source = torch.full((2, 2, 4), -31., dtype=torch.float32)
    audit = {}
    changed = query_hook(3, (6, 9), source, audit)(tensor)
    expected = tensor.clone()
    expected[:, 3, (6, 9), :] = source
    assert torch.equal(changed, expected)
    assert audit['calls'] == 1 and audit['passed']
    assert audit['source_sha256'] == audit['inserted_sha256']
    assert not torch.equal(changed[:, -1], source[:, 0:1])


def test_self_insert_is_exact_and_source_immutable():
    torch.manual_seed(17)
    tensor = torch.randn(2, 7, 12, 64, dtype=torch.float64)
    source = tensor[:, 6, (0,), :].clone()
    old = source.clone()
    audit = {}
    result = query_hook(6, (0,), source, audit)(tensor)
    assert torch.equal(result, tensor) and torch.equal(source, old)
    assert audit['actual_change_l2_per_item'] == [0., 0.]


@pytest.mark.parametrize('position,heads,shape', [(7, (0,), (2, 1, 64)),
                                              (6, (12,), (2, 1, 64)),
                                              (6, (0,), (1, 1, 64))])
def test_wrong_position_head_or_batch_fails(position, heads, shape):
    with pytest.raises(ValueError):
        query_hook(position, heads, torch.zeros(shape), {})(torch.zeros(2, 7, 12, 64))


def test_cast_is_audited_and_not_assumed_exact():
    source = torch.tensor([[[1.000000001]]], dtype=torch.float64)
    audit = {}
    result = query_hook(0, (0,), source, audit)(torch.zeros(1, 1, 1, 1, dtype=torch.float32))
    assert result.item() == 1.
    assert audit['passed'] and audit['insertion_error_per_item'][0] > 0
