"""Re-run of three Table 1 rows of Makelov, Lange & Nanda (arXiv:2311.17030).

The published artifact ``patching_metrics_ioi.joblib`` stores thirteen aggregated
means and nothing else, so no paired standard error can be recovered from it. This
module rebuilds their evaluation so that the 2000 per-example cells behind four of
those means (clean, DAS MLP8, its rowspace component, its nullspace component) can
be exported and a paired contrast computed.

Everything here is a port of the upstream code at commit
``e0c465b74561d9c3dd1f2afa770974bf5fcaee01`` (vendored under
``artifacts/makelov_source``), not a reinterpretation of it:

* ``sample_one`` / ``resample_pattern`` / ``sample_das`` follow ``data_utils.py``
  call for call, so the global ``random`` stream after ``random.seed(42)`` is
  consumed in the same order and the same amount.
* ``direction_patch`` follows ``model_utils.DirectionPatch``. It is exactly scale
  invariant in ``v`` (the update divides by ``||v||**2``), so renormalising a
  sub-direction before patching is a mathematical no-op; upstream renormalises,
  and both conventions are runnable here via ``normalise``.
* ``ROW_BASIS_QR`` reproduces upstream's ``Q, _ = torch.linalg.qr(W_out)``, which
  keeps all 768 columns. ``ROW_BASIS_SVD`` is the rank-aware alternative; the two
  differ only in whether the one numerically-dead column of the centered ``W_out``
  is counted as rowspace.

The published direction is read as a pinned raw payload by
``load_published_vector``; no downloaded pickle is executed.
"""
import hashlib
import json
import random
from pathlib import Path

import torch

ROW_BASIS_QR = 'qr'
ROW_BASIS_SVD = 'svd'
CONDITIONS = ('clean', 'full', 'rowspace', 'nullspace')
#: upstream's ``PromptDistribution.prefix_len``
PREFIX_LEN = 2
#: upstream's ``Node('post', layer=8, seq_pos=-1)``
SITE_LAYER = 8
SITE_NAME = f'blocks.{SITE_LAYER}.mlp.hook_post'
#: the four rows of the published table this module targets.
#: ``accuracy_is_placeholder`` marks the clean row: ``ioi_analysis.ipynb`` computes
#: ``clean_accuracy``, prints it, and then appends the literal ``0.0`` to the table
#: instead. The published 0.0000 is therefore not a measurement and cannot be
#: reproduced by measuring anything; it is excluded from the gate.
PUBLISHED_ROWS = {
    'clean': {'accuracy': 0.0000, 'logit_diff': 3.363561,
              'accuracy_is_placeholder': True},
    'full': {'accuracy': 0.0430, 'logit_diff': 1.820645},
    'rowspace': {'accuracy': 0.0065, 'logit_diff': 2.918310},
    'nullspace': {'accuracy': 0.0030, 'logit_diff': 3.363561},
}
PUBLISHED_LABELS = {
    'clean': 'clean',
    'full': 'DAS MLP8 direction',
    'rowspace': 'DAS MLP8 rowspace component',
    'nullspace': 'DAS MLP8 nullspace component',
}


#: SHA-256 of the published ``das_mlp8.joblib``, per ``artifacts/makelov_source``
VECTOR_SHA = '83a3ff420ee50eac7b364fe6d992d0d2a436b147d472cf10bbba92b76bd8739f'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_published_vector(path):
    """Read the pinned float32 payload; never execute a downloaded pickle.

    This exact joblib file is a 224-byte ndarray header, 3072 little-endian
    float32 values and a STOP byte. Any other file is rejected rather than
    unpickled. Kept here rather than imported so that the replication runs from
    this module plus the pinned inputs alone.
    """
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != VECTOR_SHA:
        raise ValueError('published direction hash mismatch')
    if len(data) != 12513 or data[-1:] != b'.':
        raise ValueError('unexpected pinned ndarray layout')
    vector = torch.frombuffer(bytearray(data[224:-1]), dtype=torch.float32)
    if vector.shape != (3072,) or not bool(torch.isfinite(vector).all()):
        raise ValueError('invalid direction')
    if float(vector.norm()) == 0.0:
        raise ValueError('direction must be nonzero')
    return vector.clone()


# --- upstream data_utils.py ---------------------------------------------------

def load_distribution(source_dir):
    """The authors' 1:1 vocabulary split and held-out third template.

    ``test_distribution`` takes the second half of names/objects/places and
    ``TEMPLATES[2:]``; prefixes are shared and indexed by ``prefix_len``.
    """
    source = Path(source_dir)
    raw = {k: json.loads((source / 'data' / f'{k}.json').read_text())
           for k in ('names', 'objects', 'places', 'prefixes', 'templates')}
    return {
        'names': raw['names'][len(raw['names']) // 2:],
        'objects': raw['objects'][len(raw['objects']) // 2:],
        'places': raw['places'][len(raw['places']) // 2:],
        'prefixes': raw['prefixes'],
        'templates': raw['templates'][2:],
        'prefix_len': PREFIX_LEN,
    }


def _ioi_roles(names):
    """Subject is always the third name; the other of the first two is the IO."""
    if names[2] not in names[:2] or len(set(names)) != 2:
        raise ValueError('not an IOI name triple')
    s_name = names[2]
    io_name = [x for x in names[:2] if x != s_name][0]
    return {'s_name': s_name, 'io_name': io_name,
            's1_pos': names[:2].index(s_name), 'io_pos': names[:2].index(io_name)}


def sentence(prompt):
    return prompt['prefix'] + prompt['template'].format(
        name_A=prompt['names'][0], name_B=prompt['names'][1],
        name_C=prompt['names'][2], object=prompt['obj'], place=prompt['place'])


def sample_one(rng, dist, pattern, symbol_order):
    """Port of ``PromptDistribution.sample_one``.

    Upstream writes ``unique_ids = list(set(pattern))``. A set of one-character
    strings has no defined order across processes, so the symbol-to-sampled-name
    assignment upstream used is not recoverable from the source; it is passed in
    explicitly here. The RNG draws are identical either way, so only which of the
    two sampled names plays ``A`` changes.
    """
    unique_ids = list(symbol_order)
    if set(unique_ids) != set(pattern):
        raise ValueError('symbol order does not match the pattern alphabet')
    template = rng.choice(dist['templates'])
    unique_names = rng.sample(dist['names'], len(unique_ids))
    if len(set(unique_names)) != len(unique_names):
        raise ValueError('duplicate names sampled')
    names = tuple(unique_names[unique_ids.index(i)] for i in pattern)
    obj = rng.choice(dist['objects'])
    place = rng.choice(dist['places'])
    return {'names': names, 'template': template, 'obj': obj, 'place': place,
            'prefix': dist['prefixes'][dist['prefix_len']]}


def resample_pattern(prompt, orig_pattern, new_pattern):
    """Port of ``Prompt.resample_pattern`` for the case that draws no new names.

    For ABB -> BAB (and the reverse) every symbol of the new pattern already has a
    name, so upstream's ``random.choice`` branch is never entered and the RNG
    stream is untouched. Anything else is rejected rather than silently guessed.
    """
    if len(orig_pattern) != 3 or len(new_pattern) != 3:
        raise ValueError('patterns must have length 3')
    if len(set(orig_pattern)) != 2 or len(set(new_pattern)) != 2:
        raise ValueError('patterns must use exactly two symbols')
    orig_to_name = {orig_pattern[i]: prompt['names'][i] for i in range(3)}
    if not set(new_pattern) <= set(orig_to_name):
        raise ValueError('this port only covers resampling without new names')
    return dict(prompt, names=tuple(orig_to_name[s] for s in new_pattern))


def sample_das(rng, dist, base_patterns, source_patterns, samples_per_combination,
               symbol_order):
    """Port of ``PromptDistribution.sample_das`` with ``labels='position'``.

    The patched answer pair is ``(base.s_name, base.io_name)`` whenever the subject
    moves position between base and source, which is the case for every ABB/BAB
    pair; the first entry is the interchange target, the second the foil.
    """
    base_prompts, source_prompts = [], []
    for orig in base_patterns:
        for corrupted in source_patterns:
            batch = [sample_one(rng, dist, orig, symbol_order)
                     for _ in range(samples_per_combination)]
            base_prompts.extend(batch)
            source_prompts.extend(resample_pattern(p, orig, corrupted) for p in batch)
    records = []
    for index, (base, source) in enumerate(zip(base_prompts, source_prompts)):
        b, s = _ioi_roles(base['names']), _ioi_roles(source['names'])
        if b['s1_pos'] == s['s1_pos']:
            patched = (b['io_name'], b['s_name'])
        else:
            patched = (b['s_name'], b['io_name'])
        records.append({
            'index': index,
            'base_sentence': sentence(base), 'source_sentence': sentence(source),
            'base_names': list(base['names']), 'source_names': list(source['names']),
            'base_io_name': b['io_name'], 'base_s_name': b['s_name'],
            'base_s1_pos': b['s1_pos'], 'source_s1_pos': s['s1_pos'],
            'obj': base['obj'], 'place': base['place'],
            'template': base['template'], 'prefix': base['prefix'],
            'patched_answer_names': list(patched),
        })
    return records


def build_patching_dataset(dist, symbol_order, seed=42, samples_per_combination=1000):
    """The exact object ``ioi_analysis.ipynb`` calls ``PATCHING_DATASET``.

    The notebook seeds once and then draws TEST_DATASET first, so the patching set
    is the *second and third* ``sample_das`` calls off that stream. Skipping the
    first draw would give a different 2000 prompts.
    """
    rng = random.Random(seed)
    sample_das(rng, dist, ['ABB'], ['BAB'], samples_per_combination, symbol_order)
    first = sample_das(rng, dist, ['ABB'], ['BAB'], samples_per_combination, symbol_order)
    second = sample_das(rng, dist, ['BAB'], ['ABB'], samples_per_combination, symbol_order)
    records = []
    for half, part in (('ABB->BAB', first), ('BAB->ABB', second)):
        for record in part:
            item = dict(record, half=half, index=len(records))
            item['pair_id'] = hashlib.sha256(
                f"{item['base_sentence']}|{item['source_sentence']}".encode()
            ).hexdigest()[:16]
            records.append(item)
    return records


# --- direction decomposition --------------------------------------------------

def decompose(vector, w_out, basis=ROW_BASIS_QR):
    """Split ``v`` into the part ``W_out`` can transmit and the part it cannot.

    ``ROW_BASIS_QR`` is upstream's reduced QR of the (3072, 768) ``W_out``. With
    ``center_writing_weights=True`` that matrix is numerically rank 767, so its
    768th QR column is a direction ``W_out`` maps to roughly zero; counting it as
    rowspace is upstream's convention. ``ROW_BASIS_SVD`` drops it instead.
    """
    w = torch.as_tensor(w_out, dtype=torch.float32)
    v = torch.as_tensor(vector, dtype=torch.float32)
    if w.ndim != 2 or v.shape != (w.shape[0],):
        raise ValueError('incompatible decomposition inputs')
    if basis == ROW_BASIS_QR:
        q, _ = torch.linalg.qr(w)
        rank = int(q.shape[1])
    elif basis == ROW_BASIS_SVD:
        u, s, _ = torch.linalg.svd(w.double(), full_matrices=False)
        cutoff = float(torch.finfo(torch.float32).eps) * max(w.shape) * float(s[0])
        rank = int(torch.count_nonzero(s > cutoff))
        q = u[:, :rank].float()
    else:
        raise ValueError(f'unknown row-space basis {basis!r}')
    row = v @ q @ q.T
    null = v - row
    audit = {'basis': basis, 'rank': rank,
             'vector_norm': float(v.norm()),
             'row_norm': float(row.norm()), 'null_norm': float(null.norm()),
             'row_dot_null': float(row @ null),
             'null_through_w_out': float((null.double() @ w.double()).norm()),
             'row_through_w_out': float((row.double() @ w.double()).norm()),
             'split_error': float((v - row - null).abs().max())}
    return {'full': v, 'rowspace': row, 'nullspace': null}, audit


def direction_patch(base_activation, source_activation, v):
    """Port of ``model_utils.DirectionPatch.__call__``.

    ``a = (<u,v> - <w,v>) / ||v||**2`` makes the patched component along ``v``
    equal the source's, leaving every orthogonal component untouched. Because the
    update divides by ``||v||**2`` it is invariant to rescaling ``v``: upstream's
    ``das_mlp8_row_unit`` and the un-normalised ``das_mlp8_row`` give the same
    patch up to float rounding.
    """
    if base_activation.shape != source_activation.shape:
        raise ValueError('activation batches must match')
    if base_activation.shape[1:] != v.shape:
        raise ValueError('direction does not match the activation width')
    base_proj = base_activation @ v
    source_proj = source_activation @ v
    coefficient = (source_proj - base_proj) / v.norm() ** 2
    return base_activation + coefficient.unsqueeze(-1) * v


# --- evaluation ---------------------------------------------------------------

def answer_token_ids(model, names):
    return torch.tensor([model.to_single_token(f' {name}') for name in names])


def _last_position_logits(model, tokens):
    with torch.inference_mode():
        return model(tokens)[:, -1, :]


def _site_activations(model, tokens):
    with torch.inference_mode():
        _, cache = model.run_with_cache(tokens, names_filter=[SITE_NAME])
    return cache[SITE_NAME][:, -1, :].clone()


def _patched_hook(delta_slice):
    """Replace only the last sequence position, leaving every other one identical."""
    def hook(activation, hook=None):
        if activation.ndim != 3:
            raise ValueError('unexpected activation rank at the patched site')
        if delta_slice.shape != activation[:, -1, :].shape:
            raise ValueError('patch slice does not match the activation batch')
        changed = activation.clone()
        changed[:, -1, :] = delta_slice
        return changed
    return hook


def run_conditions(model, dataset, directions, batch_size=100, normalise=True,
                   progress=None):
    """One pass over the dataset; every per-example cell is kept.

    ``normalise`` renormalises the rowspace/nullspace sub-directions to unit length
    before patching, which is what upstream does and what ``direction_patch`` is
    invariant to. It is exposed so the claim can be checked rather than asserted.
    """
    base_sentences = [r['base_sentence'] for r in dataset]
    source_sentences = [r['source_sentence'] for r in dataset]
    io_ids = answer_token_ids(model, [r['base_io_name'] for r in dataset])
    s_ids = answer_token_ids(model, [r['base_s_name'] for r in dataset])
    target_ids = answer_token_ids(model, [r['patched_answer_names'][0] for r in dataset])
    foil_ids = answer_token_ids(model, [r['patched_answer_names'][1] for r in dataset])

    vectors = {}
    for name in ('full', 'rowspace', 'nullspace'):
        v = directions[name].clone()
        if normalise or name == 'full':
            v = v / v.norm()
        vectors[name] = v

    cells = {c: {'io': [], 's': [], 'target': [], 'foil': [], 'argmax': []}
             for c in CONDITIONS}
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        base_tokens = model.to_tokens(base_sentences[start:stop])
        source_tokens = model.to_tokens(source_sentences[start:stop])
        if base_tokens.shape != source_tokens.shape:
            raise ValueError('base and source prompts must tokenise to equal length')
        base_site = _site_activations(model, base_tokens)
        source_site = _site_activations(model, source_tokens)
        for condition in CONDITIONS:
            if condition == 'clean':
                logits = _last_position_logits(model, base_tokens)
            else:
                patched = direction_patch(base_site, source_site, vectors[condition])
                model.reset_hooks()
                with torch.inference_mode():
                    logits = model.run_with_hooks(
                        base_tokens,
                        fwd_hooks=[(SITE_NAME, _patched_hook(patched))])[:, -1, :]
                model.reset_hooks()
            logits = logits.float().cpu()
            index = torch.arange(stop - start)
            cells[condition]['io'].append(logits[index, io_ids[start:stop]])
            cells[condition]['s'].append(logits[index, s_ids[start:stop]])
            cells[condition]['target'].append(logits[index, target_ids[start:stop]])
            cells[condition]['foil'].append(logits[index, foil_ids[start:stop]])
            cells[condition]['argmax'].append(logits.argmax(dim=-1))
        if progress is not None:
            progress(stop, len(dataset))
    out = {}
    for condition in CONDITIONS:
        out[condition] = {k: torch.cat(v) for k, v in cells[condition].items()}
        out[condition]['interchange'] = (
            out[condition]['argmax'] == target_ids).to(torch.int64)
    return out


def aggregate(results):
    """The two published columns, computed the way ``get_patching_acc_and_ld`` does.

    ``accuracy`` is interchange accuracy: an unrestricted argmax over the whole
    vocabulary compared against the patched-correct name. ``logit_diff`` is the
    mean of ``logit(base IO) - logit(base subject)`` under the patch, i.e. it keeps
    the *base* answer ordering for every row, which is why an inert patch must
    reproduce the clean value exactly rather than its negation.
    """
    rows = []
    for condition in CONDITIONS:
        cell = results[condition]
        rows.append({
            'condition': condition,
            'intervention': PUBLISHED_LABELS[condition],
            'accuracy': float(cell['interchange'].double().mean()),
            'logit_diff': float((cell['io'] - cell['s']).double().mean()),
        })
    return rows
