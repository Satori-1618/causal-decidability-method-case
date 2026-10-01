# Parts of _instrumented_stack adapt tracr/transformer/model.py.
# Copyright 2022 DeepMind Technologies Limited. Licensed under Apache-2.0.
# See applications/tracr/NOTICE and the upstream LICENSE for the full license.
"""Minimal residual-replacement adapter around the pinned upstream Haiku model.

No weights are trained or changed. An interceptor replaces only Transformer.__call__
with the same stack plus one residual assignment. All embeddings, attention/MLP
modules, parameters, masks and unembedding remain upstream. Native/full-score and
every-half-layer comparisons are required; semantic decoding alone is insufficient.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import re

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from tracr.compiler import compiling, lib
from tracr.rasp import rasp
from tracr.transformer import attention, model

UPSTREAM_COMMIT = "9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd"
BOS = "BOS"
VOCABULARY = tuple(range(12))


def _instrumented_stack(module, embeddings, mask, *, layer, coordinates,
                        target_position, replacement, active, captured):
    """Upstream stack specialized to its compiled, noncausal, no-dropout config."""
    cfg = module.config
    if cfg.layer_norm or cfg.causal or cfg.dropout_rate:
        raise ValueError("Adapter only qualifies the frozen compiled reverse architecture")
    initializer = hk.initializers.VarianceScaling(2 / cfg.num_layers)
    _, seq_len, model_size = embeddings.shape
    mask = mask[:, None, None, :].repeat(seq_len, axis=2)
    residual = embeddings
    residuals, layer_outputs, attn_logits = [], [], []

    def insert(value):
        captured["before"] = value
        updated = value.at[0, target_position, coordinates].set(
            replacement.astype(value.dtype))
        value = jnp.where(active, updated, value)
        captured["after"] = value
        captured["count"] = jnp.asarray(1, dtype=jnp.int32)
        return value

    for index in range(cfg.num_layers):
        if index == layer:
            residual = insert(residual)
        with hk.experimental.name_scope(f"layer_{index}"):
            attn_block = attention.MultiHeadAttention(
                num_heads=cfg.num_heads, key_size=cfg.key_size,
                model_size=model_size, w_init=initializer, name="attn")
            out = attn_block(residual, residual, residual, mask=mask)
            residual = residual + out.out
            residuals.append(residual)
            layer_outputs.append(out.out)
            attn_logits.append(out.logits)
            with hk.experimental.name_scope("mlp"):
                dense_block = hk.Sequential([
                    hk.Linear(cfg.mlp_hidden_size, w_init=initializer, name="linear_1"),
                    cfg.activation_function,
                    hk.Linear(model_size, w_init=initializer, name="linear_2"),
                ])
            dense_out = dense_block(residual)
            residual = residual + dense_out
            residuals.append(residual)
            layer_outputs.append(dense_out)
    if layer == cfg.num_layers:  # explicitly separate answer-copy positive control
        residual = insert(residual)
    return model.TransformerOutput(
        residuals=residuals, layer_outputs=layer_outputs, attn_logits=attn_logits,
        output=residual, input_embeddings=embeddings)


class ReverseAdapter:
    """One compiled parameter source, explicitly cast to fp32 and fp64."""

    def __init__(self):
        if not jax.config.x64_enabled:
            raise ValueError("Set JAX_ENABLE_X64=1 before importing JAX")
        self.program = lib.make_reverse(rasp.tokens)
        self.address_label = self.program.selector.queries.label
        self.output_label = self.program.label
        self.native = compiling.compile_rasp_to_model(
            self.program, set(VOCABULARY), 4, compiler_bos=BOS,
            causal=False, mlp_exactness=100)
        labels = self.native.residual_labels
        self.address_indices = tuple(i for i, label in enumerate(labels)
                                     if label.startswith(self.address_label + ":"))
        self.output_indices = tuple(i for i, label in enumerate(labels)
                                    if label.startswith(self.output_label + ":"))
        if not self.address_indices or len(self.output_indices) != len(VOCABULARY):
            raise ValueError("Cannot resolve complete address/output basis from compiler")
        readers = []
        for name, params in self.native.params.items():
            if name.endswith("/attn/query") and np.any(
                    np.asarray(params["w"])[list(self.address_indices)] != 0):
                readers.append(int(re.search(r"layer_(\d+)", name).group(1)))
        if len(readers) != 1:
            raise ValueError(f"Expected one compiled address reader, found {readers}")
        self.layer = readers[0]
        # Known compiler structure identifies this site, not measured patch outcomes.
        for role in ("key", "value"):
            weights = self.native.params[f"transformer/layer_{self.layer}/attn/{role}"]["w"]
            if np.any(np.asarray(weights)[list(self.address_indices)] != 0):
                raise ValueError("Declared address coordinates also affect K/V")
        output_weights = self.native.params[
            f"transformer/layer_{self.layer}/attn/linear"]["w"]
        if not np.any(np.asarray(output_weights)[:, list(self.output_indices)]):
            raise ValueError("Address reader does not write to the declared output")
        self.params = {dtype: jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=jnp.dtype(dtype)), self.native.params)
            for dtype in ("float32", "float64")}
        self._native_forward = self._make_native()
        self._forwards = {
            "site_A": self._make_instrumented(self.layer, self.address_indices),
            "site_B": self._make_instrumented(self.native.model_config.num_layers,
                                               self.output_indices),
        }

    def _make_native(self):
        @hk.without_apply_rng
        @hk.transform
        def native(tokens):
            compiled = self.native.get_compiled_model()
            out = compiled(tokens, use_dropout=False)
            scores = compiled.unembed(out.transformer_output.output, use_unembed_argmax=False)
            return out.transformer_output, scores
        return jax.jit(native.apply)

    def _make_instrumented(self, layer, coords):
        coordinates = jnp.asarray(coords, dtype=jnp.int32)

        @hk.without_apply_rng
        @hk.transform
        def instrumented(tokens, target_position, replacement, active):
            captured = {}

            def interceptor(next_fun, args, kwargs, context):
                if isinstance(context.module, model.Transformer) and context.method_name == "__call__":
                    return _instrumented_stack(
                        context.module, args[0], args[1], layer=layer,
                        coordinates=coordinates, target_position=target_position,
                        replacement=replacement, active=active, captured=captured)
                return next_fun(*args, **kwargs)

            with hk.intercept_methods(interceptor):
                compiled = self.native.get_compiled_model()
                out = compiled(tokens, use_dropout=False)
                scores = compiled.unembed(out.transformer_output.output, use_unembed_argmax=False)
            return out.transformer_output, scores, captured

        return jax.jit(instrumented.apply)

    def _tokens(self, tokens):
        if len(tokens) != 4 or any(token not in VOCABULARY for token in tokens):
            raise ValueError("This contract only permits four content tokens in 0..11")
        return jnp.asarray([self.native.input_encoder.encode([BOS, *tokens])])

    def _pack(self, result, captured=None):
        out, raw_scores = result
        # Never assume unembedding basis order is the same as vocabulary order.
        order = [self.native.output_encoder.encoding_map[token] for token in VOCABULARY]
        scores = np.asarray(raw_scores)[..., order]
        return {"scores": scores, "decoded": np.argmax(scores, axis=-1),
                "residuals": [np.asarray(x) for x in out.residuals],
                "output": np.asarray(out.output),
                "embeddings": np.asarray(out.input_embeddings),
                "captured": None if captured is None else {
                    key: np.asarray(value) for key, value in captured.items()}}

    def native_forward(self, tokens, dtype="float64"):
        with jax.default_matmul_precision("highest"):
            result = self._native_forward(self.params[dtype], self._tokens(tokens))
        return self._pack(result)

    def instrumented_forward(self, tokens, *, site="site_A", target_query=0,
                             replacement=None, dtype="float64"):
        if (site not in self._forwards or isinstance(target_query, bool)
                or not isinstance(target_query, (int, np.integer)) or not 0 <= target_query < 4):
            raise ValueError("Unknown site or out-of-range content query")
        coords = self.address_indices if site == "site_A" else self.output_indices
        active = replacement is not None
        values = np.zeros(len(coords)) if replacement is None else np.asarray(replacement)
        if values.shape != (len(coords),) or not np.isfinite(values).all():
            raise ValueError("Replacement must be finite and cover the complete site")
        with np.errstate(over="ignore"):
            cast_values = values.astype(dtype)
        if not np.isfinite(cast_values).all():
            raise ValueError("Replacement becomes non-finite after dtype conversion")
        with jax.default_matmul_precision("highest"):
            out, scores, captured = self._forwards[site](
                self.params[dtype], self._tokens(tokens), target_query + 1,
                jnp.asarray(cast_values), jnp.asarray(active))
        return self._pack((out, scores), captured)

    def donor_values(self, donor_native, source_query=1, site="site_A"):
        if (isinstance(source_query, bool) or not isinstance(source_query, (int, np.integer))
                or not 0 <= source_query < 4):
            raise ValueError("Invalid donor content query")
        if site == "site_A":
            state = (donor_native["embeddings"] if self.layer == 0 else
                     donor_native["residuals"][2 * self.layer - 1])
            coords = self.address_indices
        elif site == "site_B":
            state, coords = donor_native["output"], self.output_indices
        else:
            raise ValueError("Unknown site")
        return state[0, source_query + 1, list(coords)]

    def metadata(self):
        hashes = {}
        for dtype, params in self.params.items():
            digest = hashlib.sha256()
            for name in sorted(params):
                for key in sorted(params[name]):
                    value = np.asarray(params[name][key])
                    digest.update(f"{name}/{key}:{value.shape}:{value.dtype}".encode())
                    digest.update(value.tobytes())
            hashes[dtype] = digest.hexdigest()
        cfg = asdict(self.native.model_config)
        cfg["activation_function"] = "jax.nn.relu"
        return {"upstream_commit": UPSTREAM_COMMIT, "config": cfg,
                "parameter_hashes": hashes, "residual_labels": self.native.residual_labels,
                "sites": {
                    "site_A": {"ground_truth": "address", "layer": self.layer,
                               "timing": "before_attention", "indices": self.address_indices},
                    "site_B": {"ground_truth": "donor_answer", "layer": cfg["num_layers"],
                               "timing": "before_unembedding", "indices": self.output_indices}},
                "content_to_model_position": "content_query + 1 (BOS)",
                "model_is_compiled_not_trained": True}
