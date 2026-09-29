"""Development-only qualification of the pinned Tracr replacement adapter.

These fixed sequences are smoke controls, never confirmation families. The
upstream AssembledTransformerModel path is the independent forward reference.
"""
from dataclasses import replace

import numpy as np
import pytest

from tracr_demo.adapter import BOS, VOCABULARY, ReverseAdapter


RECIPIENT = (0, 1, 2, 3)
DONOR = (4, 5, 6, 7)
DTYPES = ("float32", "float64")
SITES = ("site_A", "site_B")


@pytest.fixture(scope="module")
def adapter():
    return ReverseAdapter()


def _coords(adapter, site):
    return adapter.address_indices if site == "site_A" else adapter.output_indices


def _site_state(adapter, native, site):
    if site == "site_B":
        return native["output"]
    if adapter.layer == 0:
        return native["embeddings"]
    return native["residuals"][2 * adapter.layer - 1]


def _assert_same_forward(actual, expected):
    for name in ("scores", "decoded", "output", "embeddings"):
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    assert len(actual["residuals"]) == len(expected["residuals"])
    for index, (left, right) in enumerate(zip(actual["residuals"], expected["residuals"])):
        np.testing.assert_array_equal(left, right, err_msg=f"half-layer {index}")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tokens", (RECIPIENT, (11, 4, 10, 2)))
def test_native_wrapper_matches_upstream_assembled_model(adapter, dtype, tokens):
    # Preserve the upstream class, forward, encoders, and apply method. Only the
    # parameter dtype changes to match the mode being qualified.
    upstream = replace(adapter.native, params=adapter.params[dtype]).apply([BOS, *tokens])
    actual = adapter.native_forward(tokens, dtype=dtype)
    assert upstream.decoded == [BOS, *reversed(tokens)]
    np.testing.assert_array_equal(actual["decoded"][0, 1:], tuple(reversed(tokens)))
    atol = 1e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(actual["output"], upstream.transformer_output, rtol=0, atol=atol)
    np.testing.assert_allclose(actual["embeddings"], upstream.input_embeddings, rtol=0, atol=atol)
    assert len(actual["residuals"]) == 2 * adapter.native.model_config.num_layers
    assert len(actual["residuals"]) == len(upstream.residuals)
    for index, (left, right) in enumerate(zip(actual["residuals"], upstream.residuals)):
        np.testing.assert_allclose(left, right, rtol=0, atol=atol, err_msg=f"half-layer {index}")

    # Independently reconstruct the compiler's projection from named output
    # basis coordinates, including tokens 10/11 to exercise vocabulary order.
    output_coordinates = [adapter.native.residual_labels.index(
        f"{adapter.output_label}:{token}") for token in VOCABULARY]
    expected_scores = np.asarray(upstream.transformer_output)[..., output_coordinates]
    np.testing.assert_allclose(actual["scores"], expected_scores, rtol=0, atol=atol)
    assert actual["output"].dtype == np.dtype(dtype)
    assert actual["embeddings"].dtype == np.dtype(dtype)
    assert all(state.dtype == np.dtype(dtype) for state in actual["residuals"])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("site", SITES)
def test_noop_matches_native_at_every_half_layer(adapter, dtype, site):
    native = adapter.native_forward(RECIPIENT, dtype=dtype)
    noop = adapter.instrumented_forward(RECIPIENT, dtype=dtype, site=site)
    _assert_same_forward(noop, native)
    capture = noop["captured"]
    assert int(capture["count"]) == 1
    np.testing.assert_array_equal(capture["before"], _site_state(adapter, native, site))
    np.testing.assert_array_equal(capture["after"], capture["before"])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("query", (0, 3))
def test_self_patch_is_identity_including_bos_boundaries(adapter, dtype, site, query):
    native = adapter.native_forward(RECIPIENT, dtype=dtype)
    own_values = adapter.donor_values(native, source_query=query, site=site)
    patched = adapter.instrumented_forward(
        RECIPIENT, dtype=dtype, site=site, target_query=query, replacement=own_values)
    _assert_same_forward(patched, native)
    np.testing.assert_array_equal(patched["captured"]["after"], patched["captured"]["before"])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("target_query", (0, 2, 3))
def test_donor_patch_changes_only_requested_site_and_reads_correct_source(
        adapter, dtype, site, target_query):
    recipient = adapter.native_forward(RECIPIENT, dtype=dtype)
    donor = adapter.native_forward(DONOR, dtype=dtype)
    source_query = 1
    values = adapter.donor_values(donor, source_query=source_query, site=site)
    donor_state = _site_state(adapter, donor, site)
    coords = _coords(adapter, site)
    np.testing.assert_array_equal(values, donor_state[0, source_query + 1, list(coords)])
    patched = adapter.instrumented_forward(
        RECIPIENT, dtype=dtype, site=site, target_query=target_query, replacement=values)
    capture = patched["captured"]
    assert int(capture["count"]) == 1
    np.testing.assert_array_equal(capture["before"], _site_state(adapter, recipient, site))
    expected = capture["before"].copy()
    expected[0, target_query + 1, list(coords)] = values.astype(expected.dtype)
    np.testing.assert_array_equal(capture["after"], expected)
    complement = np.ones(expected.shape, dtype=bool)
    complement[0, target_query + 1, list(coords)] = False
    np.testing.assert_array_equal(capture["after"][complement], capture["before"][complement])

    expected_answer = RECIPIENT[2] if site == "site_A" else DONOR[2]
    assert patched["decoded"][0, target_query + 1] == expected_answer
    untouched_queries = [position for position in range(5) if position != target_query + 1]
    np.testing.assert_array_equal(patched["scores"][:, untouched_queries],
                                  recipient["scores"][:, untouched_queries])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("site", SITES)
def test_inserted_values_equal_actual_postcast_payload(adapter, dtype, site):
    coords = _coords(adapter, site)
    # Exercise rounding independently of the donor's nearly one-hot payload.
    values = np.linspace(-1.3, 1.7, len(coords), dtype=np.float64) + 1e-10
    patched = adapter.instrumented_forward(
        RECIPIENT, dtype=dtype, site=site, target_query=2, replacement=values)
    capture = patched["captured"]
    assert capture["after"].dtype == np.dtype(dtype)
    expected = values.astype(dtype)
    np.testing.assert_array_equal(capture["after"][0, 3, list(coords)], expected)
    assert np.isfinite(capture["after"]).all()


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("invalid_kind", ("scalar", "short", "long", "matrix", "nan", "inf"))
def test_rejects_malformed_or_nonfinite_replacement(adapter, site, invalid_kind):
    width = len(_coords(adapter, site))
    values = {
        "scalar": 1.0,
        "short": np.zeros(width - 1),
        "long": np.zeros(width + 1),
        "matrix": np.zeros((1, width)),
        "nan": np.full(width, np.nan),
        "inf": np.full(width, np.inf),
    }[invalid_kind]
    with pytest.raises(ValueError):
        adapter.instrumented_forward(RECIPIENT, site=site, replacement=values)


@pytest.mark.parametrize("site", SITES)
def test_rejects_replacement_that_becomes_nonfinite_after_cast(adapter, site):
    values = np.full(len(_coords(adapter, site)), 1e100, dtype=np.float64)
    with pytest.raises(ValueError):
        adapter.instrumented_forward(RECIPIENT, site=site, replacement=values, dtype="float32")


@pytest.mark.parametrize("invalid_query", (-1, 4, 100, 0.5, True, False))
def test_rejects_invalid_target_and_donor_positions(adapter, invalid_query):
    native = adapter.native_forward(RECIPIENT)
    with pytest.raises(ValueError):
        adapter.instrumented_forward(RECIPIENT, target_query=invalid_query)
    with pytest.raises(ValueError):
        adapter.donor_values(native, source_query=invalid_query)


def test_rejects_unknown_site(adapter):
    native = adapter.native_forward(RECIPIENT)
    with pytest.raises(ValueError):
        adapter.instrumented_forward(RECIPIENT, site="unknown")
    with pytest.raises(ValueError):
        adapter.donor_values(native, site="unknown")


@pytest.mark.parametrize("tokens", ((), (0, 1, 2), (0, 1, 2, 3, 4), (0, 1, 2, 12)))
def test_rejects_sequences_outside_frozen_contract(adapter, tokens):
    with pytest.raises(ValueError):
        adapter.native_forward(tokens)
    with pytest.raises(ValueError):
        adapter.instrumented_forward(tokens)
