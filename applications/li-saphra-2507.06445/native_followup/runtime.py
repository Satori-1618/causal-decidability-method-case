"""CPU runtime for the pinned Dyck checkpoint and explicit EOS interventions.

The native arm calls the unmodified upstream forward. Intervention arms replace
only the selected attention module for one call, then restore it in ``finally``.
All query/key indices are absolute token positions; EOS is ``len(text) + 1``.
This file supplies execution and fidelity checks, not a causal interpretation.
"""

from __future__ import annotations

import importlib
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
APPLICATION = HERE.parent
CHECKPOINT = "1aez5d6p"
MAX_LENGTH = 40
TOKEN_IDS = {"(": 3, ")": 4}


def upstream_gpt_class():
    """Import the pinned code without depending on an unrelated ``utils`` package.

    The original model is in the original audit's locked upstream directory.
    Its small missing dependency is fetched separately into the followup cache.
    """
    name = "_li_saphra_pinned_mingpt"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [
            str(APPLICATION / "upstream/utils/minGPT"),
            str(HERE / "cache/utils/minGPT"),
        ]
        sys.modules[name] = package
    return importlib.import_module(name + ".model").GPT


def make_model(*, n_layer=2, n_head=2, n_embd=64, dtype=torch.float32):
    """Build the paper's architecture with dropout disabled, before loading weights."""
    GPT = upstream_gpt_class()
    config = GPT.get_default_config()
    for key, value in {
        "model_type": None, "n_layer": n_layer, "n_head": n_head,
        "n_embd": n_embd, "vocab_size": 5, "block_size": MAX_LENGTH + 2,
        "embd_pdrop": 0.0, "resid_pdrop": 0.0, "attn_pdrop": 0.0,
    }.items():
        setattr(config, key, value)
    return GPT(config).to(device="cpu", dtype=dtype).eval()


def tokenize(strings: Sequence[str]):
    """Exactly BOS=0, PAD=1, EOS=2, (=3, )=4; fixed 42-token rows."""
    if not strings:
        raise ValueError("At least one input is required")
    tokens = torch.full((len(strings), MAX_LENGTH + 2), 1, dtype=torch.long)
    positions = []
    for index, text in enumerate(strings):
        if not isinstance(text, str) or not text or len(text) > MAX_LENGTH:
            raise ValueError("Inputs must be nonempty bracket strings of length <= 40")
        if any(char not in TOKEN_IDS for char in text):
            raise ValueError("Input contains a non-bracket character")
        eos = len(text) + 1
        tokens[index, 0] = 0
        tokens[index, 1:eos] = torch.tensor([TOKEN_IDS[char] for char in text])
        tokens[index, eos] = 2
        positions.append(eos)
    return tokens, torch.tensor(positions, dtype=torch.long)


def _check_rows(rows: torch.Tensor, eos: torch.Tensor):
    if rows.ndim != 2 or eos.shape != (rows.shape[0],):
        raise ValueError("Expected rows [batch, key] and EOS positions [batch]")
    if eos.dtype != torch.long or torch.any(eos < 2) or torch.any(eos >= rows.shape[1]):
        raise ValueError("EOS positions must include at least one bracket token")
    if not rows.is_floating_point() or not torch.isfinite(rows).all():
        raise ValueError("Attention rows must be finite floating-point tensors")
    tol = 32 * torch.finfo(rows.dtype).eps
    positions = torch.arange(rows.shape[1], device=rows.device)[None, :]
    if torch.any(rows < 0) or torch.any(rows[positions > eos[:, None]] != 0):
        raise ValueError("Attention must be nonnegative and zero after EOS")
    if not torch.allclose(rows.sum(-1), torch.ones_like(rows[:, 0]), atol=tol, rtol=0):
        raise ValueError("Each attention row must sum to one within dtype tolerance")


def factorized_eos_weights(native: torch.Tensor, eos: torch.Tensor, mode: str):
    """Factor EOS attention into endpoint mass and within-bracket routing.

    ``routing_only`` preserves both native endpoint weights and total bracket
    mass, spreading that mass uniformly over bracket positions.
    ``gate_only`` assigns BOS and EOS each 1/(L+2), keeps native relative bracket
    weights, and scales their total to L/(L+2).
    ``both`` is uniform over BOS, all L bracket tokens, and EOS.
    """
    _check_rows(native, eos)
    if mode == "identity":
        return native.clone()
    if mode not in {"routing_only", "gate_only", "both"}:
        raise ValueError("Unknown factorized attention mode: " + mode)
    result = native.clone()
    for index, position in enumerate(eos.tolist()):
        length = position - 1
        bracket = native[index, 1:position]
        mass = bracket.sum()
        if mode == "routing_only":
            result[index, 1:position] = mass / length
        elif mode == "gate_only":
            if not mass > 0:
                raise ValueError("Native bracket mass is zero; routing is undefined")
            unit = 1.0 / (length + 2)
            result[index, 0] = unit
            result[index, position] = unit
            result[index, 1:position] = (bracket / mass) * (length * unit)
        else:
            result[index, :position + 1] = 1.0 / (length + 2)
    _check_rows(result, eos)
    return result


@dataclass
class RunResult:
    """CPU tensors in input order; margins > 0 favor the False/reject class."""

    margins: torch.Tensor
    eos_logits: torch.Tensor
    predicted_valid: torch.Tensor
    eos_positions: torch.Tensor
    attention: torch.Tensor
    full_attention: torch.Tensor
    values: torch.Tensor
    node: torch.Tensor
    preprojection: torch.Tensor
    forward_calls: int


Override = Union[torch.Tensor, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]


class DyckRuntime:
    def __init__(self, model, *, layer=1, head=1):
        self.model = model.cpu().eval()
        self.layer = layer
        self.head = head
        if not 0 <= layer < len(model.transformer.h):
            raise ValueError("Target layer outside model")
        self.module = model.transformer.h[layer].attn
        if not 0 <= head < self.module.n_head:
            raise ValueError("Target head outside model")
        if any(block.attn.ablate_heads.count(True) for block in model.transformer.h):
            raise ValueError("Load an unablated model before using the runtime")

    @classmethod
    def from_checkpoint(cls, path: Optional[Path] = None, *, dtype=torch.float32):
        path = Path(path) if path else (
            HERE / "cache/data/model_weights" / ("run_" + CHECKPOINT) / ("run_" + CHECKPOINT + "_checkpoint_5.pt")
        )
        model = make_model(dtype=dtype)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return cls(model)

    @torch.inference_mode()
    def run(
        self, strings: Sequence[str], *, mode="native", batch_size=64,
        attention_override: Optional[Override] = None,
        node_override: Optional[Override] = None,
    ) -> RunResult:
        """Run one declared arm, restoring hooks and head flags even on failure.

        Native and ``uniform_all_queries`` use the unchanged upstream forward;
        the latter activates its score-zeroing ablation on the selected head.
        All other modes reconstruct that forward and affect only target EOS.
        Custom callbacks receive (native EOS row, current head values, EOS
        positions). Tensor overrides have one row per input, in input order.
        ``custom_node`` supplies the selected pre-c_proj head vector directly.
        """
        allowed = {"native", "identity", "routing_only", "gate_only", "both",
                   "uniform_all_queries", "custom_attention", "custom_node"}
        if mode not in allowed:
            raise ValueError("Unknown runtime mode: " + mode)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if (attention_override is not None) != (mode == "custom_attention"):
            raise ValueError("attention_override is required only for custom_attention")
        if (node_override is not None) != (mode == "custom_node"):
            raise ValueError("node_override is required only for custom_node")
        tokens, positions = tokenize(strings)
        result = {key: [] for key in ("margins", "eos_logits", "predicted_valid",
                  "eos_positions", "attention", "full_attention", "values", "node", "preprojection")}
        calls = 0
        module = self.module
        original_forward = module.forward
        # Preserve whether forward was originally an instance attribute, too.
        had_instance_forward = "forward" in module.__dict__
        original_flags = list(module.ablate_heads)
        for start in range(0, len(strings), batch_size):
            stop = min(start + batch_size, len(strings))
            eos = positions[start:stop]
            captured = {}
            handles = []
            batch_calls = 0

            def capture_qkv(_module, _args, output):
                captured["qkv"] = output.detach()

            def capture_preprojection(_module, args):
                captured["preprojection"] = args[0].detach()

            def resolve_override(override, native_row, values):
                replacement = override(native_row, values, eos) if callable(override) else override[start:stop]
                return replacement.to(device=native_row.device, dtype=native_row.dtype)

            def reconstructed(this, x):
                nonlocal batch_calls
                batch_calls += 1
                batch, length, channels = x.shape
                q, k, v = this.c_attn(x).split(this.n_embd, dim=2)
                q = q.view(batch, length, this.n_head, channels // this.n_head).transpose(1, 2)
                k = k.view(batch, length, this.n_head, channels // this.n_head).transpose(1, 2)
                v = v.view(batch, length, this.n_head, channels // this.n_head).transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                this.last_attn_weights_premask = F.softmax(att, dim=-1).detach().cpu()
                att = att.masked_fill(this.bias[:, :, :length, :length] == 0, float("-inf"))
                att = this.attn_dropout(F.softmax(att, dim=-1))
                index = torch.arange(batch)
                native_row = att[index, self.head, eos, :].clone()
                if mode in {"identity", "routing_only", "gate_only", "both"}:
                    replacement = factorized_eos_weights(native_row, eos, mode)
                elif mode == "custom_attention":
                    replacement = resolve_override(attention_override, native_row, v[:, self.head])
                    _check_rows(replacement, eos)
                else:
                    replacement = native_row
                if replacement.shape != native_row.shape:
                    raise ValueError("Attention override has the wrong shape")
                att = att.clone()
                att[index, self.head, eos, :] = replacement
                y = att @ v
                if mode == "custom_node":
                    replacement_node = resolve_override(node_override, native_row, v[:, self.head])
                    if replacement_node.shape != (batch, channels // this.n_head):
                        raise ValueError("Node override has the wrong shape")
                    if not torch.isfinite(replacement_node).all():
                        raise ValueError("Node override must be finite")
                    y = y.clone()
                    y[index, self.head, eos, :] = replacement_node
                y = y.transpose(1, 2).contiguous().view(batch, length, channels)
                this.last_attn_weights = att.detach().cpu()
                return this.resid_dropout(this.c_proj(y))

            def count_native(_module, _args, _output):
                nonlocal batch_calls
                batch_calls += 1

            try:
                handles.append(module.c_attn.register_forward_hook(capture_qkv))
                handles.append(module.c_proj.register_forward_pre_hook(capture_preprojection))
                if mode in {"native", "uniform_all_queries"}:
                    handles.append(module.register_forward_hook(count_native))
                    if mode == "uniform_all_queries":
                        module.ablate_heads[self.head] = True
                else:
                    module.forward = types.MethodType(reconstructed, module)
                logits, _ = self.model(tokens[start:stop])
                if batch_calls != 1:
                    raise RuntimeError("Expected exactly one target attention call per batch")
                index = torch.arange(stop - start)
                eos_logits = logits[index, eos]
                pre = captured["preprojection"]
                channels = module.n_embd
                head_dim = channels // module.n_head
                values = captured["qkv"].split(channels, dim=-1)[2]
                values = values.view(stop - start, tokens.shape[1], module.n_head, head_dim)[:, :, self.head]
                node = pre[index, eos, self.head * head_dim:(self.head + 1) * head_dim]
                margin = eos_logits[:, 0] - eos_logits[:, 1]
                batch_result = {
                    "margins": margin, "eos_logits": eos_logits,
                    "predicted_valid": eos_logits[:, 1] > eos_logits[:, 0],
                    "eos_positions": eos,
                    "attention": module.last_attn_weights[index, self.head, eos, :],
                    "full_attention": module.last_attn_weights[:, self.head, :, :],
                    "values": values, "node": node, "preprojection": pre,
                }
                for key, tensor in batch_result.items():
                    result[key].append(tensor.detach().cpu().clone())
                calls += batch_calls
            finally:
                for handle in handles:
                    handle.remove()
                module.ablate_heads[:] = original_flags
                if had_instance_forward:
                    module.forward = original_forward
                elif "forward" in module.__dict__:
                    delattr(module, "forward")
        return RunResult(**{key: torch.cat(value) for key, value in result.items()}, forward_calls=calls)
