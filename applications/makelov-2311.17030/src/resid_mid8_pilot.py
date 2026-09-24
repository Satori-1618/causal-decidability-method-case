"""Blinded pilot for ``PREREG_RESID_MID8.md``.

The pilot exists to fix two inputs of the Freeze B calculator call (§8): the size
of the effect to be decomposed, ``mean E(full)``, and the spread of the paired
contrasts, ``sd d_inert`` and ``sd d_all``. It must not reveal where ``E(row)``
or ``E(null)`` lies (§7). Everything that leaves this module therefore goes
through ``blinded_summary``, which is built from an allowlist and checked by
``validate_record``; the per-pair values are never returned to the caller.

The site is ``resid_mid`` of layer 8 at the last position. Everything else is
the validated port in ``makelov_table1``: the same test distribution, dataset
builder, direction patch and hook. Only the site and the decomposition differ;
the decomposition is upstream's ``decompose_resid_along_namemovers``
(``ioi_analysis.ipynb``, cell 8) called with ``das_resid_mid``.
"""
import hashlib
import math
from pathlib import Path

import torch

import makelov_table1 as M

SITE_LAYER = 8
#: ``transformer_lens.utils.get_act_name('resid_mid', layer=8)``
SITE_NAME = f'blocks.{SITE_LAYER}.hook_resid_mid'
#: upstream's ``NAME_MOVERS``
NAME_MOVERS = ((9, 6), (9, 9), (10, 0))
CONDITIONS = ('clean', 'full', 'row', 'null')
COMPONENTS = ('row', 'null')
READOUTS = ('L', 'A')
CONTRASTS = ('d_inert', 'd_all')

#: §6. Nothing in this module can generate the confirmation seed.
PILOT_SEED = 7001
PILOT_PER_COMBINATION = 100
PILOT_N = 2 * PILOT_PER_COMBINATION
#: what upstream's ``list(set(pattern))`` yields for 'ABB' and 'BAB' under
#: ``PYTHONHASHSEED=0`` (hash slots 1 and 6 of 8, no collision). Checked at runtime.
SYMBOL_ORDER = ('B', 'A')
REQUIRED_HASHSEED = '0'

#: upstream ``das_resid_mid.joblib`` at the vendored revision
DIRECTION_SHA = '4e69edf9361c3216d9e6bb753d75de440128858757f93028a2293db0488124e3'
DIRECTION_BYTES = 3297
#: joblib ``NumpyArrayWrapper`` header for a C-ordered '<f4' array of shape (768,),
#: padded to a 16-byte-aligned payload. Byte-identical to the header of the pinned
#: ``das_mlp8.joblib`` except for the two shape bytes (768 instead of 3072).
_DIRECTION_HEADER = (
    b'\x80\x04\x95\xd2\x00\x00\x00\x00\x00\x00\x00\x8c\x13joblib.numpy_pickle\x94'
    b'\x8c\x11NumpyArrayWrapper\x94\x93\x94)\x81\x94}\x94(\x8c\x08subclass\x94'
    b'\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94\x8c\x05shape\x94M\x00\x03\x85\x94'
    b'\x8c\x05order\x94\x8c\x01C\x94\x8c\x05dtype\x94h\x06\x8c\x05dtype\x94\x93\x94'
    b'\x8c\x02f4\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xff'
    b'J\xff\xff\xff\xffK\x00t\x94b\x8c\nallow_mmap\x94\x88\x8c\x1b'
    b'numpy_array_alignment_bytes\x94K\x10ub\x02\xff\xff'
)

#: Freeze A plus the dated pre-run clarification appended under §13
PREREG_SHA = 'b8fc20ad96671f1369cd0e2c95dab7d6a4ba10c11d6af1df3533205ae6fa8d81'
#: §8
CALCULATOR_SHA = 'ab0fd838a9a6b354d72d83c8b9730b095c0a443a4eeae756f2d1491eb17365f8'


def load_direction(path):
    """Read the pinned float32 payload; never execute a downloaded pickle."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != DIRECTION_SHA:
        raise ValueError('das_resid_mid hash mismatch')
    if (len(data) != DIRECTION_BYTES or data[:len(_DIRECTION_HEADER)] != _DIRECTION_HEADER
            or data[-1:] != b'.'):
        raise ValueError('unexpected pinned ndarray layout')
    vector = torch.frombuffer(bytearray(data[len(_DIRECTION_HEADER):-1]),
                              dtype=torch.float32)
    if vector.shape != (768,) or not bool(torch.isfinite(vector).all()):
        raise ValueError('invalid direction')
    if float(vector.norm()) == 0.0:
        raise ValueError('direction must be nonzero')
    return vector.clone()


def upstream_symbol_order(patterns=('ABB', 'BAB')):
    """The order upstream's ``list(set(pattern))`` gives in *this* process.

    Only meaningful under a fixed ``PYTHONHASHSEED``; the runner refuses to go on
    unless it equals ``SYMBOL_ORDER`` for every pattern the builder uses.
    """
    orders = {p: tuple(set(p)) for p in patterns}
    if len(set(orders.values())) != 1:
        raise ValueError(f'patterns disagree on the symbol order: {orders}')
    return next(iter(orders.values()))


def build_pilot_dataset(dist):
    """§6: the replication's builder, seed 7001, 100 per combination, 200 pairs."""
    dataset = M.build_patching_dataset(
        dist, list(SYMBOL_ORDER), seed=PILOT_SEED,
        samples_per_combination=PILOT_PER_COMBINATION)
    if len(dataset) != PILOT_N:
        raise ValueError('pilot dataset has the wrong size')
    return dataset


def decompose_namemovers(vector, w_qs):
    """Port of upstream ``decompose_resid_along_namemovers``.

    ``w_qs`` are the (d_model, d_head) query matrices of the name movers, in
    ``NAME_MOVERS`` order. ``row`` is the projection onto the column span of their
    concatenation, obtained by reduced QR as upstream does; ``null = v - row``.
    """
    v = torch.as_tensor(vector, dtype=torch.float32)
    w = torch.cat([torch.as_tensor(q, dtype=torch.float32) for q in w_qs], dim=1)
    if w.ndim != 2 or v.shape != (w.shape[0],):
        raise ValueError('incompatible decomposition inputs')
    q, _ = torch.linalg.qr(w)
    row = v @ q @ q.T
    null = v - row
    audit = {'basis': 'qr', 'rank': int(q.shape[1]),
             'vector_norm': float(v.norm()),
             'row_norm': float(row.norm()), 'null_norm': float(null.norm()),
             'row_dot_null': float(row @ null),
             'null_through_queries': float((null.double() @ w.double()).norm()),
             'split_error': float((v - row - null).abs().max())}
    return {'full': v, 'row': row, 'null': null}, audit


def weights_fingerprint(w_qs):
    """SHA-256 of the concatenated name-mover queries, so later runs can match them."""
    w = torch.cat([torch.as_tensor(q, dtype=torch.float32) for q in w_qs], dim=1)
    return hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest()


def _site_activations(model, tokens):
    with torch.inference_mode():
        _, cache = model.run_with_cache(tokens, names_filter=[SITE_NAME])
    return cache[SITE_NAME][:, -1, :].clone()


def run_conditions(model, dataset, directions, batch_size=100):
    """One pass over the pilot pairs, returning per-pair cells for the four conditions.

    Sub-directions are renormalised to unit length as upstream does; the patch is
    invariant to that (``makelov_table1.direction_patch``).
    """
    base_sentences = [r['base_sentence'] for r in dataset]
    source_sentences = [r['source_sentence'] for r in dataset]
    io_ids = M.answer_token_ids(model, [r['base_io_name'] for r in dataset])
    s_ids = M.answer_token_ids(model, [r['base_s_name'] for r in dataset])
    target_ids = M.answer_token_ids(model, [r['patched_answer_names'][0] for r in dataset])
    vectors = {name: directions[name] / directions[name].norm()
               for name in ('full', 'row', 'null')}
    cells = {c: {'ld': [], 'argmax': []} for c in CONDITIONS}
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
                logits = M._last_position_logits(model, base_tokens)
            else:
                patched = M.direction_patch(base_site, source_site, vectors[condition])
                model.reset_hooks()
                with torch.inference_mode():
                    logits = model.run_with_hooks(
                        base_tokens,
                        fwd_hooks=[(SITE_NAME, M._patched_hook(patched))])[:, -1, :]
                model.reset_hooks()
            logits = logits.float().cpu()
            index = torch.arange(stop - start)
            cells[condition]['ld'].append(
                logits[index, io_ids[start:stop]] - logits[index, s_ids[start:stop]])
            cells[condition]['argmax'].append(logits.argmax(dim=-1))
    out = {}
    for condition in CONDITIONS:
        ld = torch.cat(cells[condition]['ld'])
        hit = (torch.cat(cells[condition]['argmax']) == target_ids).to(torch.int64)
        out[condition] = {'ld': ld, 'interchange': hit}
    return out


def effects(cells):
    """§3/§4 plus the dated clarification: every effect is relative to clean, per pair.

    ``L``: clean logit difference minus patched logit difference (IO minus S).
    ``A``: patched interchange indicator minus the clean one, so that a patch that
    changes nothing has effect exactly 0, as the inert rival requires.
    """
    out = {'L': {}, 'A': {}}
    for condition in ('full', 'row', 'null'):
        out['L'][condition] = (cells['clean']['ld'].double() - cells[condition]['ld'].double())
        out['A'][condition] = (cells[condition]['interchange'].double()
                               - cells['clean']['interchange'].double())
    return out


def contrasts(effect):
    """§5: ``d_inert = E(X)``, ``d_all = E(full) - E(X)``, per readout and component."""
    return {r: {x: {'d_inert': effect[r][x], 'd_all': effect[r]['full'] - effect[r][x]}
                for x in COMPONENTS}
            for r in READOUTS}


def _sd(values):
    """Sample SD, ddof = 1, in float64."""
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError('need a 1-d sample of at least two pairs')
    if not bool(torch.isfinite(values).all()):
        raise ValueError('non-finite per-pair value')
    return float(values.std(unbiased=True))


def blinded_summary(effect):
    """The only numbers §7 lets out: ``mean E(full)`` per readout, SDs of the contrasts.

    The mean of ``E(row)``, ``E(null)``, ``d_inert`` or ``d_all`` is never computed
    here, so it cannot leak through a later edit of the output code either.
    """
    n = {int(effect[r][c].numel()) for r in READOUTS for c in ('full', 'row', 'null')}
    if len(n) != 1:
        raise ValueError('readouts and conditions must cover the same pairs')
    d = contrasts(effect)
    return {
        'n': n.pop(),
        'mean_E_full': {r: float(effect[r]['full'].double().mean()) for r in READOUTS},
        'sd': {r: {x: {k: _sd(d[r][x][k]) for k in CONTRASTS} for x in COMPONENTS}
               for r in READOUTS},
        'sd_convention': 'sample standard deviation, ddof=1, float64',
    }


#: every key a pilot record may carry; anything else is rejected before writing
RECORD_KEYS = frozenset({'summary', 'provenance', 'decomposition'})
SUMMARY_KEYS = frozenset({'n', 'mean_E_full', 'sd', 'sd_convention'})
DECOMPOSITION_KEYS = frozenset({'basis', 'rank', 'vector_norm', 'row_norm', 'null_norm',
                                'row_dot_null', 'null_through_queries', 'split_error'})


def validate_record(record):
    """Reject any record that carries more than §7 allows."""
    if set(record) != RECORD_KEYS:
        raise ValueError(f'record keys {sorted(record)} are not the allowlist')
    summary = record['summary']
    if set(summary) != SUMMARY_KEYS:
        raise ValueError(f'summary keys {sorted(summary)} are not the allowlist')
    if set(summary['mean_E_full']) != set(READOUTS):
        raise ValueError('mean_E_full must be keyed by readout only')
    if set(summary['sd']) != set(READOUTS):
        raise ValueError('sd must be keyed by readout')
    for r in READOUTS:
        if set(summary['sd'][r]) != set(COMPONENTS):
            raise ValueError('sd must be keyed by component')
        for x in COMPONENTS:
            if set(summary['sd'][r][x]) != set(CONTRASTS):
                raise ValueError('sd must be keyed by contrast')
            for value in summary['sd'][r][x].values():
                if not (isinstance(value, float) and math.isfinite(value) and value >= 0):
                    raise ValueError('sd must be a finite non-negative float')
    if set(record['decomposition']) != DECOMPOSITION_KEYS:
        raise ValueError('decomposition audit carries unexpected keys')
    provenance = record['provenance']
    if not isinstance(provenance, dict) or any(
            isinstance(v, (list, tuple)) and len(v) > 16 for v in provenance.values()):
        raise ValueError('provenance must be flat metadata, not data')
    return record
