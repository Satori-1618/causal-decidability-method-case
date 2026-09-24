"""Audited query reset and reverse transfer for the fixed read-null intervention.

Queries are projected, per-head tensors; only the selected final-position slices
are written. All other computation remains free. This is not a whole-head clamp.
"""
import hashlib

import torch

from makelov_read_source import patch_deltas, position_hook

MLP_SITE = 'blocks.8.mlp.hook_post'
QUERY_HEADS = {'blocks.9.attn.hook_q': (6, 9), 'blocks.10.attn.hook_q': (0,)}
SITES = (MLP_SITE, *QUERY_HEADS)
CELL_MEANINGS = {
    'A': 'baseline; cache q0',
    'B': 'read_null patch; cache q1 with queries free',
    'C': 'patch off; insert q0',
    'D': 'read_null patch; insert q0',
    'E': 'patch off; insert q1',
    'F': 'read_null patch; insert own q1; identity to B',
    'G': 'zero MLP delta; identity to A',
}


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def query_hook(position, heads, source, audit):
    """Insert cached values and audit the actual tensor including dtype conversion."""
    heads = tuple(heads)
    if not heads or len(heads) != len(set(heads)):
        raise ValueError('heads must be nonempty and unique')

    def replace(activation, hook=None):
        if activation.ndim != 4 or not 0 <= position < activation.shape[1]:
            raise ValueError('invalid query tensor or absolute position')
        if any(h < 0 or h >= activation.shape[2] for h in heads):
            raise ValueError('invalid head index')
        before = activation[:, position, heads, :]
        if source.shape != before.shape or not bool(torch.isfinite(source).all()):
            raise ValueError('invalid query source')
        intended = source.to(device=activation.device, dtype=activation.dtype)
        changed = activation.clone()
        changed[:, position, heads, :] = intended
        actual = changed[:, position, heads, :]
        other_heads = [h for h in range(activation.shape[2]) if h not in heads]
        untouched = (torch.equal(changed[:, :position], activation[:, :position])
                     and torch.equal(changed[:, position+1:], activation[:, position+1:])
                     and torch.equal(changed[:, position, other_heads, :],
                                     activation[:, position, other_heads, :]))
        actual64, source64 = actual.detach().cpu().double(), source.detach().cpu().double()
        delta64 = actual64 - before.detach().cpu().double()
        error = (actual64-source64).abs().flatten(1).amax(-1)
        budget = (8 * torch.finfo(activation.dtype).eps
                  * source64.abs().flatten(1).amax(-1).clamp_min(1))
        audit.update({
            'calls': audit.get('calls', 0) + 1,
            'position': position, 'heads': list(heads),
            'source_sha256': tensor_hash(source), 'inserted_sha256': tensor_hash(actual),
            'insertion_error_per_item': error.tolist(),
            'dtype_budget_per_item': budget.tolist(),
            'actual_change_l2_per_item': delta64.flatten(1).norm(dim=-1).tolist(),
            'actual_change_l2_per_item_head': delta64.norm(dim=-1).tolist(),
            'other_slices_unchanged': untouched,
            'inserted_equals_cast_source': torch.equal(actual, intended),
            'passed': untouched and torch.equal(actual, intended) and bool((error <= budget).all()),
        })
        if not audit['passed']:
            raise RuntimeError('query insertion fidelity failed')
        return changed

    return replace


def run_route_case(model, case, directions):
    """Seven forwards for two reciprocal prompts; no generation or KV caching."""
    if model.cfg.n_layers != 12 or model.cfg.n_heads != 12:
        raise ValueError('this instrument is pinned to GPT-2 Small')
    position = case['position']
    tokens = torch.tensor(case['token_ids'], dtype=torch.long, device=model.cfg.device)
    if tokens.ndim != 2 or tokens.shape[0] != 2 or position != tokens.shape[1]-1:
        raise ValueError('expected two aligned reciprocal prompts at final absolute position')
    answers = case['answer_token_ids']
    values, audits, events = {}, {}, {}

    def collect(output):
        logits = output[:, position, answers].detach().cpu().double()
        if logits.shape != (2, 2) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError('invalid answer logits')
        return logits

    def make_hooks(arm, delta=None, sources=None):
        hooks, audits[arm], events[arm] = [], {}, []

        def trace(site, function):
            def wrapped(activation, hook=None):
                events[arm].append(site)
                return function(activation, hook=hook)
            return wrapped

        if delta is not None:
            audits[arm][MLP_SITE] = {}
            hooks.append((MLP_SITE, trace(MLP_SITE, position_hook(position, delta, audits[arm][MLP_SITE]))))
        if sources is not None:
            for site, heads in QUERY_HEADS.items():
                audits[arm][site] = {}
                hooks.append((site, trace(site, query_hook(position, heads, sources[site], audits[arm][site]))))
        return hooks

    with torch.inference_mode():
        output, cache = model.run_with_cache(tokens, names_filter=list(SITES))
        values['A'] = collect(output)
        receiver = cache[MLP_SITE][:, position, :].detach().clone()
        q0 = {site: cache[site][:, position, heads, :].detach().clone()
              for site, heads in QUERY_HEADS.items()}
        del output, cache
        deltas = patch_deltas(receiver, receiver.flip(0), directions)
        delta = deltas['read_null']
        hooks = make_hooks('B', delta=delta)
        with model.hooks(fwd_hooks=hooks):
            output, cache = model.run_with_cache(tokens, names_filter=list(QUERY_HEADS))
        values['B'] = collect(output)
        q1 = {site: cache[site][:, position, heads, :].detach().clone()
              for site, heads in QUERY_HEADS.items()}
        del output, cache
        for arm, d, q in (('C', None, q0), ('D', delta, q0), ('E', None, q1),
                          ('F', delta, q1), ('G', deltas['identity'], None)):
            output = model.run_with_hooks(tokens, fwd_hooks=make_hooks(arm, delta=d, sources=q))
            values[arm] = collect(output)
            del output

    for arm in audits:
        expected = [site for site in SITES if site in audits[arm]]
        if events[arm] != expected:
            raise RuntimeError(f'{arm}: hook order/count mismatch: {events[arm]} != {expected}')
        if any(a['calls'] != 1 or not a['passed'] for a in audits[arm].values()):
            raise RuntimeError(f'{arm}: invalid hook audit')
    scale = max(1., max(v.abs().max().item() for v in values.values()))
    tolerance = 64 * torch.finfo(receiver.dtype).eps * scale
    errors = {f'{a}_vs_{b}': (values[a]-values[b]).abs().max().item()
              for a, b in (('C', 'A'), ('F', 'B'), ('G', 'A'))}
    if any(e > tolerance for e in errors.values()):
        raise RuntimeError(f'identity control failed: {errors}; budget={tolerance}')
    return {
        'cells': {arm: (v[:, 0]-v[:, 1]).tolist() for arm, v in values.items()},
        'answer_logits': {arm: v.tolist() for arm, v in values.items()},
        'controls': {'passed': True, 'identity_tolerance': 2*tolerance,
                     'answer_logit_identity_tolerance': tolerance, 'identity_errors': errors,
                     'identity_exact': {f'{a}_vs_{b}': torch.equal(values[a], values[b])
                                        for a, b in (('C', 'A'), ('F', 'B'), ('G', 'A'))},
                     'fidelity': audits, 'hook_events': events},
        'query_source_hashes': {name: {site: tensor_hash(q) for site, q in source.items()}
                                for name, source in (('q0', q0), ('q1', q1))},
        'natural_query_change_l2': {site: (q1[site].double()-q0[site].double()).norm(dim=-1).cpu().tolist()
                                   for site in QUERY_HEADS},
        'model_forward_calls': 7,
    }
