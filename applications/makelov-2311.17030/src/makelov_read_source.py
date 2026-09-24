"""Common-write / split-read intervention at GPT-2 MLP8 post-GELU.

This decomposes the *patch's* read source. It does not identify the naturally used
semantic variable. Row/null components are never independently normalized.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

VECTOR_SHA = '83a3ff420ee50eac7b364fe6d992d0d2a436b147d472cf10bbba92b76bd8739f'
ARMS = ('identity', 'full', 'read_row', 'read_null')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_published_vector(path):
    """Read the pinned float32 payload; never execute a downloaded pickle.

    This exact joblib file has a 224-byte ndarray header, 3072 little-endian
    float32 values, and a STOP byte. Any other file is rejected, not unpickled.
    """
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != VECTOR_SHA:
        raise ValueError('published direction hash mismatch')
    if len(data) != 12513 or data[-1:] != b'.':
        raise ValueError('unexpected pinned ndarray layout')
    v = np.frombuffer(data[224:-1], dtype='<f4').astype(np.float64)
    if v.shape != (3072,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
        raise ValueError('invalid direction')
    return v


def decompose_direction(vector, w_out, weight_epsilon=np.finfo(np.float32).eps):
    """W has shape [d_mlp, d_model]; its columns span visible activation directions.

    Rank-aware SVD avoids inventing a visible direction in the centered output
    matrix's nullspace. The cutoff is tied to the stored weight precision.
    """
    w = np.asarray(w_out, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if w.ndim != 2 or v.shape != (w.shape[0],) or not np.isfinite(w).all():
        raise ValueError('incompatible or nonfinite decomposition inputs')
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('direction must be finite and nonzero')
    v = v / norm
    u, s, _ = np.linalg.svd(w, full_matrices=False)
    cutoff = weight_epsilon * max(w.shape) * s[0]
    rank = int(np.count_nonzero(s > cutoff))
    basis = u[:, :rank]
    row = basis @ (basis.T @ v)
    null = v - row
    audit = {'original_norm': float(norm), 'rank': rank,
             'singular_cutoff': float(cutoff), 'smallest_singular_value': float(s[-1]),
             'row_norm': float(np.linalg.norm(row)), 'null_norm': float(np.linalg.norm(null)),
             'orthogonality_error': float(abs(row @ null)),
             'null_output_norm': float(np.linalg.norm(null @ w)),
             'decomposition_error': float(np.max(np.abs(v-row-null)))}
    if audit['null_output_norm'] > max(cutoff, 1e-12):
        raise RuntimeError('null component fails output-map fidelity')
    return {'full': v, 'read_row': row, 'read_null': null}, audit


def patch_deltas(receiver, donor, directions):
    """All arms write into the SAME normalized full vector, in working dtype."""
    if receiver.shape != donor.shape or receiver.ndim != 2:
        raise ValueError('activation batches must match')
    delta = donor - receiver
    v = torch.as_tensor(directions['full'], device=receiver.device, dtype=receiver.dtype)
    out = {'identity': torch.zeros_like(receiver)}
    for name in ('full', 'read_row', 'read_null'):
        read = torch.as_tensor(directions[name], device=receiver.device, dtype=receiver.dtype)
        out[name] = (delta @ read).unsqueeze(-1) * v
    return out


def position_hook(position, delta, audit):
    """Only replace the absolute manifest position, with post-cast fidelity checks."""
    def hook(activation, hook=None):
        if activation.ndim != 3 or not 0 <= position < activation.shape[1]:
            raise ValueError('invalid absolute intervention position')
        if delta.shape != activation[:, position, :].shape:
            raise ValueError('intervention batch or hidden dimension mismatch')
        before = activation[:, position, :]
        work_delta = delta.to(device=activation.device, dtype=activation.dtype)
        changed = activation.clone()
        changed[:, position, :] = before + work_delta
        # Audit the tensor actually inserted, in higher precision on CPU.
        b = before.detach().cpu().double()
        d = delta.detach().cpu().double()
        actual = changed[:, position, :].detach().cpu().double() - b
        error = (actual-d).abs().amax(dim=-1)
        eps = torch.finfo(activation.dtype).eps
        budget = 8 * eps * (b.abs()+d.abs()).amax(dim=-1).clamp_min(1)
        untouched = (torch.equal(changed[:, :position], activation[:, :position])
                     and torch.equal(changed[:, position+1:], activation[:, position+1:]))
        audit['calls'] = audit.get('calls', 0) + 1
        audit['insertion_error_max_per_item'] = error.tolist()
        audit['rounding_budget_per_item'] = budget.tolist()
        audit['actual_delta_l2_per_item'] = actual.norm(dim=-1).tolist()
        audit['other_positions_unchanged'] = untouched
        audit['passed'] = untouched and bool(torch.all(error <= budget))
        if not audit['passed']:
            raise RuntimeError('intervention fidelity failed')
        return changed
    return hook


def make_cases(source_dir, tokenizer, n=32, seed=20260920):
    """Fresh draws from the authors' held-out vocabulary/template split.

    Both directions of a swap belong to ONE independent base pair. Token-only
    eligibility precedes every model forward; no baseline/effect filtering.
    """
    source = Path(source_dir)
    data = {k: json.loads((source/'data'/f'{k}.json').read_text())
            for k in ('names', 'objects', 'places', 'prefixes', 'templates')}
    names = data['names'][len(data['names'])//2:]
    names = [x for x in names if len(tokenizer.encode(' '+x, add_special_tokens=False)) == 1]
    rng = random.Random(seed)
    cases, seen = [], set()
    for attempt in range(max(1000, n*100)):
        if len(cases) == n:
            return cases
        io, subject = rng.sample(names, 2)
        obj = rng.choice(data['objects'][len(data['objects'])//2:])
        place = rng.choice(data['places'][len(data['places'])//2:])
        template = rng.choice(data['templates'][2:])
        prompts = [data['prefixes'][2]+template.format(name_A=a, name_B=b,
                    name_C=subject, object=obj, place=place)
                   for a, b in ((io, subject), (subject, io))]
        ids = [[tokenizer.bos_token_id]+tokenizer.encode(p, add_special_tokens=False)
               for p in prompts]
        if len(ids[0]) != len(ids[1]) or tuple(prompts) in seen:
            continue
        seen.add(tuple(prompts))
        key = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()[:16]
        cases.append({'case_id': key, 'draw': attempt, 'prompts': prompts,
                      'patterns': ['ABB', 'BAB'], 'token_ids': ids,
                      'position': len(ids[0])-1, 'io': io, 'subject': subject,
                      'answer_token_ids': [tokenizer.encode(' '+x, add_special_tokens=False)[0]
                                           for x in (io, subject)]})
    raise RuntimeError('not enough token-eligible unique cases')


def run_case(model, case, directions, layer=8):
    """Two swap directions, five forwards; all cells exported before averaging."""
    tokens = torch.tensor(case['token_ids'], device=model.cfg.device)
    site = f'blocks.{layer}.mlp.hook_post'
    pos = case['position']
    with torch.inference_mode():
        baseline, cache = model.run_with_cache(tokens, names_filter=[site])
        h = cache[site][:, pos, :].clone()
        del cache
        deltas = patch_deltas(h, h.flip(0), directions)
        split_error = (deltas['full']-deltas['read_row']-deltas['read_null']).abs().max().item()
        scale = max(1., deltas['full'].abs().max().item())
        if split_error > 32*torch.finfo(h.dtype).eps*scale:
            raise RuntimeError('read decomposition failed after dtype conversion')
        values, audits = {'baseline': baseline[:, pos, case['answer_token_ids']].cpu().double()}, {}
        del baseline
        for arm in ARMS:
            audit = {}
            output = model.run_with_hooks(tokens, fwd_hooks=[(site, position_hook(pos, deltas[arm], audit))])
            if audit['calls'] != 1:
                raise RuntimeError('hook count mismatch')
            values[arm] = output[:, pos, case['answer_token_ids']].cpu().double()
            audits[arm] = audit
            del output
        if not torch.equal(values['identity'], values['baseline']):
            raise RuntimeError('identity control changed answer logits')
    records = []
    for i, pattern in enumerate(case['patterns']):
        records.append({'case_id': case['case_id'], 'receiver_pattern': pattern,
            'donor_pattern': case['patterns'][1-i], 'position': pos,
            'answer_logits': {k: v[i].tolist() for k, v in values.items()},
            'margins': {k: float(v[i, 0]-v[i, 1]) for k, v in values.items()},
            'split_delta_max_error': split_error,
            'fidelity': {a: {k: (v[i] if isinstance(v, list) else v) for k, v in x.items()}
                         for a, x in audits.items()}})
    return records
