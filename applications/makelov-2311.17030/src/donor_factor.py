"""Four-donor factorial intervention, with two fixed recipient orderings.

All arms use one fixed null read and full write direction at MLP8. Name assignment
and mention position are controlled donor factors, not identified semantic payloads.
"""
import hashlib

import torch

from makelov_read_source import patch_deltas, position_hook

SITE = 'blocks.8.mlp.hook_post'
CELLS = ('00', '01', '10', '11')
MAPS = ({'00': 0, '01': 1, '10': 2, '11': 3},
        {'00': 1, '01': 0, '10': 3, '11': 2})


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_family(model, case, directions):
    """Six same-batch-size forwards; only recipient rows 0/1 are changed.

    The four source activations are captured in one baseline forward. Keeping the
    batch size fixed for every arm removes an unnecessary source of identity drift.
    The last two rows have zero interventions and provide additional output identities.
    """
    if (model.cfg.n_layers, model.cfg.d_model) != (12, 768):
        raise ValueError('instrument requires GPT-2 Small')
    if case.get('recipient_indices') != [0, 1] or case.get('donor_indices') != list(MAPS):
        raise ValueError('unexpected factorial source mapping')
    tokens = torch.tensor(case['token_ids'], dtype=torch.long, device=model.cfg.device)
    position, answers = case['position'], case['answer_token_ids']
    if tokens.ndim != 2 or tokens.shape[0] != 4 or position != tokens.shape[1] - 1:
        raise ValueError('expected four token-aligned prompts and an absolute final position')
    if len(answers) != 2 or answers[0] == answers[1]:
        raise ValueError('two distinct answer token IDs are required')

    def collect(output):
        scores = output[:, position, answers].detach().cpu().double()
        if scores.shape != (4, 2) or not bool(torch.isfinite(scores).all()):
            raise RuntimeError('invalid selected answer logits')
        return scores

    with torch.inference_mode():
        output, cache = model.run_with_cache(tokens, names_filter=[SITE])
        baseline = collect(output)
        h = cache[SITE][:, position, :].detach().clone()
        del output, cache
        if not bool(torch.isfinite(h).all()):
            raise RuntimeError('nonfinite source activation')
        receiver = h[:2]
        read = torch.as_tensor(directions['read_null'], dtype=h.dtype, device=h.device)
        write = torch.as_tensor(directions['full'], dtype=h.dtype, device=h.device)
        if read.shape != (h.shape[-1],) or write.shape != read.shape:
            raise ValueError('direction dimension mismatch')
        outputs, audits, alpha_values, norms, sources = {}, {}, {}, {}, {}
        for cell in (*CELLS, 'zero'):
            source_indices = [MAPS[r][cell] for r in (0, 1)] if cell != 'zero' else [0, 1]
            donor = h[source_indices]
            alpha = (donor - receiver) @ read
            delta = patch_deltas(receiver, donor, directions)['read_null']
            if not torch.equal(delta, alpha[:, None] * write):
                raise RuntimeError('declared scalar/write does not reconstruct actual delta')
            if cell in ('00', 'zero') and (torch.count_nonzero(alpha) or torch.count_nonzero(delta)):
                raise RuntimeError('self donor must produce exactly zero delta')
            batch_delta = torch.zeros_like(h)
            batch_delta[:2] = delta
            audit = {}
            output = model.run_with_hooks(
                tokens, fwd_hooks=[(SITE, position_hook(position, batch_delta, audit))])
            outputs[cell] = collect(output)
            del output
            if audit.get('calls') != 1 or not audit.get('passed'):
                raise RuntimeError('local insertion/count check failed')
            if audit['actual_delta_l2_per_item'][2:] != [0.0, 0.0]:
                raise RuntimeError('nonrecipient batch rows were changed')
            audits[cell] = audit
            alpha_values[cell] = alpha.detach().cpu().double().tolist()
            norms[cell] = delta.detach().cpu().double().norm(dim=-1).tolist()
            sources[cell] = source_indices

    scale = max(1.0, baseline.abs().max().item(),
                *(v.abs().max().item() for v in outputs.values()))
    tolerance = 64 * torch.finfo(h.dtype).eps * scale
    errors = {'self_vs_baseline': (outputs['00'] - baseline).abs().max().item(),
              'zero_vs_baseline': (outputs['zero'] - baseline).abs().max().item(),
              'untouched_batch_rows': max((v[2:] - baseline[2:]).abs().max().item()
                                           for v in outputs.values())}
    if any(e > tolerance for e in errors.values()):
        raise RuntimeError(f'identity failed: {errors}; budget={tolerance}')
    margins = baseline[:, 0] - baseline[:, 1]
    panels = []
    for r in (0, 1):
        panels.append({'recipient_index': r, 'baseline_margin': margins[r].item(),
                       'donor_indices': dict(MAPS[r]),
                       'source_indices': [MAPS[r][c] for c in CELLS],
                       'patched_margins': {c: (outputs[c][r, 0]-outputs[c][r, 1]).item()
                                           for c in CELLS},
                       'alphas': {c: alpha_values[c][r] for c in CELLS},
                       'delta_l2': {c: norms[c][r] for c in CELLS},
                       'controls': {'passed': True, 'identity_tolerance': 2*tolerance}})
    return {'baseline_margins': margins.tolist(), 'panels': panels,
            'baseline_answer_logits': baseline.tolist(),
            'patched_answer_logits': {c: v.tolist() for c, v in outputs.items()},
            'source_indices': sources, 'source_activations_sha256': tensor_sha(h),
            'source_activation_sha256_per_prompt': [tensor_sha(row) for row in h],
            'controls': {'passed': True, 'identity_tolerance': 2*tolerance,
                         'answer_logit_identity_tolerance': tolerance,
                         'identity_errors': errors,
                         'identity_exact': all(v == 0 for v in errors.values()),
                         'fidelity': audits, 'site': SITE, 'absolute_position': position,
                         'zero_alphas': alpha_values['zero']},
            'model_forward_calls': 6,
            'model_prompt_evaluations': 24}
