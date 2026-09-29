"""Executed CPU neural circuits for an intervention-design comparison.

Predictions are independent Python formulas; observations run PyTorch graphs
with an actual intervention on a two-unit ReLU hidden layer. These are generic
score units, not logits or a model of GPT-2's native computation. A case reveals
the rival predictions and intervention menu, never which graph generated it.
PyTorch is an optional application dependency, not a package dependency.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

import torch

DEVELOPMENT_FAMILIES = ("linear", "relu_offset")
# Reserved structural checks: development runners/tests must not evaluate these.
HELDOUT_FAMILIES = ("squared", "saturated_relu")
CANDIDATES = (
    "channel_a", "channel_a_alias", "channel_b", "balanced", "joint",
    "context_switch", "context_switch_inverse",
)
OUTSIDE_CANDIDATE = "external_gain"
MANDATORY_COUNT = 4
MENU = (
    "native:c0", "full:c0", "native:c1", "full:c1",
    *(f"{read}:c{context}:d{dose}" for context in (0, 1)
      for read in ("read_a", "read_b") for dose in ("0.5", "1", "2")),
)


def _parse_cell(cell: str) -> tuple[str, int, float]:
    if cell not in MENU:
        raise ValueError(f"Unknown intervention: {cell}")
    parts = cell.split(":")
    return parts[0], int(parts[1][1:]), float(parts[2][1:]) if len(parts) == 3 else 1.0


def _analytic_cell(candidate: str, family: str, amplitude: float,
                   gain: float, cell: str) -> float:
    """Independent formula, with no calls into the executed torch graph."""
    operation, context, dose = _parse_cell(cell)
    a = amplitude * (1.0 if operation == "full" else dose if operation == "read_a" else 0.0)
    b = amplitude * (1.0 if operation == "full" else dose if operation == "read_b" else 0.0)
    if candidate in ("channel_a", "channel_a_alias", OUTSIDE_CANDIDATE):
        z = a
    elif candidate == "channel_b":
        z = b
    elif candidate == "balanced":
        z = (a + b) / 2.0
    elif candidate == "joint":
        z = min(a, b)
    elif candidate == "context_switch":
        z = a if context == 0 else b
    elif candidate == "context_switch_inverse":
        z = b if context == 0 else a
    else:
        raise ValueError(f"Unknown candidate: {candidate}")
    if family == "linear":
        value = z
    elif family == "relu_offset":
        value = max(0.0, z - 0.3 * amplitude)
    elif family == "squared":
        value = z * z / amplitude
    elif family == "saturated_relu":
        value = min(z, 0.75 * amplitude)
    else:
        raise ValueError(f"Unknown family: {family}")
    return gain * value * (1.5 if candidate == OUTSIDE_CANDIDATE else 1.0)


class NeuralCircuit(torch.nn.Module):
    """Two ReLU hidden units followed by candidate-specific computation.

    The alias uses a distinct, algebraically equivalent graph. The joint graph
    computes a minimum using ReLU; the switch graphs gate on a context input.
    All circuits share donor inputs and full-patch endpoints by construction.
    """

    def __init__(self, candidate: str, family: str, amplitude: float, gain: float):
        super().__init__()
        if candidate not in (*CANDIDATES, OUTSIDE_CANDIDATE):
            raise ValueError(f"Unknown candidate: {candidate}")
        if family not in (*DEVELOPMENT_FAMILIES, *HELDOUT_FAMILIES):
            raise ValueError(f"Unknown family: {family}")
        self.candidate, self.family = candidate, family
        self.amplitude, self.gain = amplitude, gain

    def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(inputs)

    def forward(self, inputs: torch.Tensor, context: int,
                patch: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        h = self.hidden(inputs)
        if patch:
            h = h.clone()
            for index, value in patch.items():
                h[index] = value
        a, b = h.unbind()
        c = h.new_tensor(float(context))
        if self.candidate in ("channel_a", OUTSIDE_CANDIDATE):
            z = a
        elif self.candidate == "channel_a_alias":
            z = torch.relu(2.0 * a) - a
        elif self.candidate == "channel_b":
            z = b
        elif self.candidate == "balanced":
            z = 0.5 * a + 0.5 * b
        elif self.candidate == "joint":
            z = a - torch.relu(a - b)
        elif self.candidate == "context_switch":
            z = (1.0 - c) * a + c * b
        else:
            z = c * a + (1.0 - c) * b
        if self.family == "relu_offset":
            z = torch.relu(z - h.new_tensor(0.3 * self.amplitude))
        elif self.family == "squared":
            z = z.square() / h.new_tensor(self.amplitude)
        elif self.family == "saturated_relu":
            z = z - torch.relu(z - h.new_tensor(0.75 * self.amplitude))
        return z * h.new_tensor(self.gain) * (1.5 if self.candidate == OUTSIDE_CANDIDATE else 1.0)


def _execute(case: dict, candidate: str, dtype: torch.dtype) -> list[float]:
    circuit = NeuralCircuit(candidate, case["family"], case["amplitude"],
                            case["parameters"]["output_gain"])
    recipient = torch.zeros(2, dtype=dtype)
    donor = circuit.hidden(torch.full((2,), case["amplitude"], dtype=dtype))
    outputs = []
    with torch.no_grad():
        for cell in case["menu"]:
            operation, context, dose = _parse_cell(cell)
            indices = (0, 1) if operation == "full" else (0,) if operation == "read_a" else (1,) if operation == "read_b" else ()
            patch = {index: donor[index] * dose for index in indices}
            outputs.append(float(circuit(recipient, context, patch)))
    return outputs


def observe_truth(case: dict, truth_name: str) -> list[float]:
    """Execute a candidate (or external_gain negative control) in float32."""
    return _execute(case, truth_name, torch.float32)


def public_case(seed: int, family: str, amplitude: float, *, allow_reserved: bool = False) -> dict:
    """Return JSON-serializable public design information with no truth label.

    numerical_bound is a calibration envelope over this finite menu and the
    declared candidate graphs, not a universal floating-point error guarantee.
    It adds eight float32 epsilons of scale allowance to the measured maximum.
    Higher precision is a reference, not mathematical ground truth; independent
    formulas additionally check the float64 reference in this known circuit.
    """
    if family not in (*DEVELOPMENT_FAMILIES, *HELDOUT_FAMILIES):
        raise ValueError(f"Unknown family: {family}")
    if family in HELDOUT_FAMILIES and not allow_reserved:
        raise ValueError("Reserved structural family: explicit release is required before execution")
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be positive and finite")
    gain = random.Random(seed).uniform(0.8, 1.2)
    payload = json.dumps([seed, family, amplitude], separators=(",", ":"))
    case = {
        "case_id": hashlib.sha256(payload.encode()).hexdigest()[:20],
        "family": family, "amplitude": amplitude, "menu": list(MENU),
        "parameters": {"output_gain": gain},
        "predictions": {name: [_analytic_cell(name, family, amplitude, gain, cell)
                                for cell in MENU] for name in CANDIDATES},
        "equivalence_groups": [["channel_a", "channel_a_alias"],
                               *[[name] for name in CANDIDATES[2:]]],
    }
    max_numeric, max_reference = 0.0, 0.0
    qualification_graph_forward_count = 0
    analytic_formula_evaluation_count = len(CANDIDATES) * len(MENU)
    identity_errors, endpoint_errors = [], []
    all_scale = [1.0]
    for candidate in (*CANDIDATES, OUTSIDE_CANDIDATE):
        low = _execute(case, candidate, torch.float32)
        high = _execute(case, candidate, torch.float64)
        exact = [_analytic_cell(candidate, family, amplitude, gain, cell) for cell in MENU]
        qualification_graph_forward_count += len(low) + len(high)
        analytic_formula_evaluation_count += len(exact)
        max_numeric = max(max_numeric, *(abs(a - b) for a, b in zip(low, high)))
        max_reference = max(max_reference, *(abs(a - b) for a, b in zip(high, exact)))
        all_scale.extend(abs(value) for value in high)
        circuit = NeuralCircuit(candidate, family, amplitude, gain)
        native_input = torch.zeros(2, dtype=torch.float32)
        donor_input = torch.full((2,), amplitude, dtype=torch.float32)
        for context in (0, 1):
            unpatched = float(circuit(native_input, context))
            self_patch = float(circuit(native_input, context, {0: native_input[0], 1: native_input[1]}))
            identity_errors.append(abs(unpatched - self_patch))
            endpoint_errors.append(abs(low[2 * context + 1] - float(circuit(donor_input, context))))
            qualification_graph_forward_count += 3
    scale = max(all_scale)
    reference_pass = max_reference <= 64.0 * torch.finfo(torch.float64).eps * scale
    full_values = [case["predictions"][name][index] for name in CANDIDATES for index in (1, 3)]
    matched_full = max(full_values) - min(full_values) <= 64.0 * torch.finfo(torch.float64).eps * scale
    case["numerical_bound"] = max_numeric + 8.0 * torch.finfo(torch.float32).eps * scale
    case["controls"] = {
        "reference_formula_agreement": reference_pass,
        "identity_patch_exact": max(identity_errors) == 0.0,
        "full_patch_matches_native_donor": max(endpoint_errors) == 0.0,
        "full_patch_matches_across_candidates": matched_full,
        "maximum_fp32_fp64_difference": max_numeric,
        "maximum_fp64_formula_difference": max_reference,
        "qualification_graph_forward_count": qualification_graph_forward_count,
        "analytic_formula_evaluation_count": analytic_formula_evaluation_count,
        "all_passed": reference_pass and matched_full and max(identity_errors + endpoint_errors) == 0.0,
    }
    if not case["controls"]["all_passed"]:
        raise AssertionError("Executed circuit control failed")
    return case
