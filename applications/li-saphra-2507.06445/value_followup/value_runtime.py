"""Single-value contribution transfers at a fixed final-layer EOS query.

For recipient position j and donor position k, deliver
    h_patched = h_recipient + a_recipient[j] * (v_donor[k] - v_recipient[j]).

This replaces one contribution to the head's AV output through the existing
custom-node operator. It does NOT globally overwrite the value projection, the
attention distribution, or every query that reads the same token. Positions are
absolute: BOS=0, bracket tokens=1..L, EOS=L+1. No data are written by this module.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass
class ValueTransferResult:
    recipient_strings: List[str]
    donor_strings: List[str]
    recipient_positions: torch.Tensor
    donor_positions: torch.Tensor
    native_margins: torch.Tensor
    donor_native_margins: torch.Tensor
    patched_margins: torch.Tensor
    a_r: torch.Tensor
    v_r: torch.Tensor
    v_d: torch.Tensor
    h_r: torch.Tensor
    h_patch_intended: torch.Tensor
    h_patch_delivered: torch.Tensor
    requested_node_delta: torch.Tensor
    delivered_node_delta: torch.Tensor
    value_difference_l2: torch.Tensor
    node_delta_l2: torch.Tensor
    construction_roundoff_linf: torch.Tensor
    controls: Dict

    def snapshots(self):
        """Compact JSON-ready records, sufficient to recompute the node formula."""
        records = []
        for i, text in enumerate(self.recipient_strings):
            row = {
                'recipient_string': text, 'donor_string': self.donor_strings[i],
                'recipient_position': int(self.recipient_positions[i]),
                'donor_position': int(self.donor_positions[i]),
                'native_margin': float(self.native_margins[i]),
                'donor_native_margin': float(self.donor_native_margins[i]),
                'patched_margin': float(self.patched_margins[i]),
                'margin_change': float(self.patched_margins[i] - self.native_margins[i]),
                'a_r': float(self.a_r[i]),
                'value_difference_l2': float(self.value_difference_l2[i]),
                'node_delta_l2': float(self.node_delta_l2[i]),
                'construction_roundoff_linf': float(self.construction_roundoff_linf[i]),
                'dtype': str(self.native_margins.dtype),
            }
            for name in ['v_r', 'v_d', 'h_r', 'h_patch_intended', 'h_patch_delivered',
                         'requested_node_delta', 'delivered_node_delta']:
                row[name] = getattr(self, name)[i].tolist()
            records.append(row)
        return records


def _positions(positions, strings, name):
    raw = torch.as_tensor(positions)
    if raw.shape != (len(strings),) or raw.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    }:
        raise ValueError(name + ' must contain one integer absolute position per pair')
    value = raw.to(device='cpu', dtype=torch.long)
    lengths = torch.tensor([len(s) for s in strings])
    if torch.any(value < 1) or torch.any(value > lengths):
        raise ValueError(name + ' must target bracket tokens 1..L, excluding BOS/EOS/padding')
    return value


def _validate_inputs(rt, recipients, donors, recipient_positions, donor_positions):
    if not recipients or len(recipients) != len(donors):
        raise ValueError('Require the same positive number of recipient and donor strings')
    for recipient, donor in zip(recipients, donors):
        if not isinstance(recipient, str) or not isinstance(donor, str):
            raise ValueError('Inputs must be strings')
        if not recipient or len(recipient) > 40 or set(recipient) - {'(', ')'} or set(donor) - {'(', ')'}:
            raise ValueError('Inputs must be nonempty bracket strings of length <=40')
        if len(recipient) != len(donor):
            raise ValueError('Recipient and donor length must match within each pair')
    recipient_positions = _positions(recipient_positions, recipients, 'recipient_positions')
    donor_positions = _positions(donor_positions, donors, 'donor_positions')
    for r, d, j, k in zip(recipients, donors, recipient_positions, donor_positions):
        if r[int(j)-1] != d[int(k)-1]:
            raise ValueError('Transferred positions must have the same token symbol')
    if rt.layer != len(rt.model.transformer.h) - 1:
        raise ValueError('This operator is scoped to the final attention layer')
    return recipient_positions, donor_positions


def _observe(rt, strings, *, batch_size, **kwargs):
    """Capture all heads' attention and V at this layer for operator checks."""
    attention, values, handles = [], [], []
    def capture_attention(module, _args, _output):
        attention.append(module.last_attn_weights.detach().cpu().clone())
    def capture_values(_module, _args, output):
        values.append(output.split(rt.module.n_embd, dim=-1)[2].detach().cpu().clone())
    try:
        handles.append(rt.module.register_forward_hook(capture_attention))
        handles.append(rt.module.c_attn.register_forward_hook(capture_values))
        result = rt.run(strings, batch_size=batch_size, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if len(attention) != result.forward_calls or len(values) != result.forward_calls:
        raise AssertionError('Expected one attention/value observation per target-layer forward')
    return result, torch.cat(attention), torch.cat(values)


def _check_unchanged(rt, native, patched, native_attention, patched_attention, native_values, patched_values):
    if not torch.equal(native_attention, patched_attention):
        raise AssertionError('An attention weight at the target layer changed')
    if not torch.equal(native_values, patched_values):
        raise AssertionError('A native value projection at the target layer changed')
    if not torch.equal(native.eos_positions, patched.eos_positions):
        raise AssertionError('EOS positions changed')
    difference = patched.preprojection - native.preprojection
    head_dim = rt.module.n_embd // rt.module.n_head
    index = torch.arange(len(native.margins))
    start = rt.head * head_dim
    difference[index, native.eos_positions, start:start+head_dim] = 0
    if torch.count_nonzero(difference):
        raise AssertionError('A non-target head/query preprojection entry changed')


@torch.inference_mode()
def run_value_transfers(
    rt, recipient_strings: Sequence[str], donor_strings: Sequence[str],
    recipient_positions, donor_positions, *, batch_size=64,
) -> ValueTransferResult:
    """Execute paired transfers and return observations plus audit snapshots.

    Margins are logit(False)-logit(True). Native-node self replacement is checked
    for every recipient. This function imposes only technical fidelity gates;
    it does not select cases, set scientific thresholds, or interpret effects.
    """
    recipients, donors = list(recipient_strings), list(donor_strings)
    recipient_positions, donor_positions = _validate_inputs(
        rt, recipients, donors, recipient_positions, donor_positions)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    native, native_attention, native_values = _observe(rt, recipients, batch_size=batch_size)
    donor = native if recipients == donors else rt.run(donors, batch_size=batch_size)
    index = torch.arange(len(recipients))
    a_r = native.attention[index, recipient_positions]
    v_r = native.values[index, recipient_positions]
    v_d = donor.values[index, donor_positions]
    if a_r.shape != (len(recipients),) or v_r.shape != v_d.shape or v_r.shape != native.node.shape:
        raise AssertionError('Captured attention/value/node shapes are inconsistent')
    if torch.any(a_r < 0) or not all(torch.isfinite(t).all() for t in [a_r, v_r, v_d, native.node]):
        raise AssertionError('Captured tensors must be finite with nonnegative attention')
    requested_delta = a_r[:, None] * (v_d - v_r)
    intended = native.node + requested_delta
    # Float64 is an arithmetic reference, not exact truth. Record the rounding
    # from constructing the new node in its actual execution dtype.
    reference = native.node.double() + a_r.double()[:, None] * (v_d.double() - v_r.double())
    roundoff = (intended.double() - reference).abs().amax(1)
    identity, identity_attention, identity_values = _observe(
        rt, recipients, batch_size=batch_size, mode='custom_node', node_override=native.node)
    _check_unchanged(rt,native,identity,native_attention,identity_attention,native_values,identity_values)
    identity_margin_error = float((identity.margins-native.margins).abs().max())
    identity_node_error = float((identity.node-native.node).abs().max())
    tolerance = 1e-10 if native.margins.dtype == torch.float64 else 1e-5
    if max(identity_margin_error,identity_node_error) > tolerance:
        raise AssertionError('Native-node identity exceeds the dtype-specific tolerance')
    patched, patched_attention, patched_values = _observe(
        rt, recipients, batch_size=batch_size, mode='custom_node', node_override=intended)
    _check_unchanged(rt,native,patched,native_attention,patched_attention,native_values,patched_values)
    if not torch.equal(patched.node,intended):
        raise AssertionError('The delivered head output differs from the intended post-dtype tensor')
    self_mask = torch.tensor([r == d and int(j) == int(k)
                             for r,d,j,k in zip(recipients,donors,recipient_positions,donor_positions)])
    self_error = float((patched.margins[self_mask]-native.margins[self_mask]).abs().max()) if self_mask.any() else 0.
    if self_mask.any() and (torch.count_nonzero(requested_delta[self_mask]) or self_error > tolerance):
        raise AssertionError('Same-input/same-position transfer failed the identity control')
    controls = {
        'identity_max_margin_error':identity_margin_error,'identity_max_node_error':identity_node_error,
        'identity_tolerance':tolerance,'self_same_position_cases':int(self_mask.sum()),
        'self_same_position_max_margin_error':self_error,
        'inserted_node_exactly_intended_after_dtype':True,
        'all_target_layer_attention_weights_unchanged':True,
        'all_target_layer_value_projections_unchanged':True,
        'nontarget_head_and_query_preprojection_exactly_unchanged':True,
        'recipient_and_donor_native_forward_batches':native.forward_calls + (0 if donor is native else donor.forward_calls),
        'identity_forward_batches':identity.forward_calls,'transfer_forward_batches':patched.forward_calls,
    }
    return ValueTransferResult(
        recipients,donors,recipient_positions,donor_positions,native.margins,donor.margins,patched.margins,
        a_r,v_r,v_d,native.node,intended,patched.node,requested_delta,patched.node-native.node,
        torch.linalg.vector_norm(v_d-v_r,dim=1),torch.linalg.vector_norm(patched.node-native.node,dim=1),roundoff,controls)
