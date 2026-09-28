"""Small paired Round 2 runner, reusing the immutable Round 1 hook and readout."""
import random
import time

import torch

from mixing_round1_design import TARGET, agreement_control, design_indices
from mixing_runner import TechnicalError
from mixing_word_structure import swap_pair


def run_pair(runner, matrix, cell, *, case_id, seed, control_target, audit=False):
    start = time.perf_counter()
    cases = swap_pair(matrix, cell)
    arms, active, audit_arrays = [], [], {}
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)
        if not passed:
            raise TechnicalError(name)

    # Qualify on both native conditions before computing any conflict patch.
    for k, case in enumerate(cases):
        prepared, native, vectors = {}, {}, {}
        for role, index in (('recipient', cell['i_N']), ('donor', cell['i_P'])):
            g = case[role]
            prompt, rec, out = runner.native(g, index, g[index][TARGET])
            prepared[role] = prompt
            native[role] = rec
            vectors[role] = out.hidden_states[runner.layer][0, -1].detach().clone()
            if role == 'recipient':
                vectors['native_full_logits'] = out.logits[0, -1].float().detach().cpu()
            check(f'{k}_{role}_bos', prompt['dropped'] == runner.tokenizer.bos_token
                  and int((prompt['input_ids'][0] == runner.tokenizer.bos_token_id).sum()) == 1)
            del out
        arms.append({'matrix': case['recipient'], 'donor_matrix': case['donor'],
                     'design_indices': design_indices(**case), 'native': native})
        active.append((prepared, vectors))

    for role in ('recipient', 'donor'):
        x, y = [a[0][role]['input_ids'][0].tolist() for a in active]
        check(f'{role}_equal_length', len(x) == len(y))
        old = cases[0]['recipient'][cell['i_P']][TARGET]
        new = cases[0]['recipient'][cell['i_L']][TARGET]
        token_map = {runner.context_ids[old]: runner.context_ids[new],
                     runner.context_ids[new]: runner.context_ids[old]}
        check(f'{role}_only_intended_tokens_changed',
              y == [token_map.get(token, token) for token in x])
        check(f'{role}_same_entity_spans',
              arms[0]['native'][role]['entity_positions'] == arms[1]['native'][role]['entity_positions'])

    qualifies = all(a['native'][role]['correct']
                    and a['native'][role]['readout_matches_generation']
                    and a['native'][role]['readout']['answer_mass_full_vocab'] >= 0.5
                    for a in arms for role in ('recipient', 'donor'))
    record = {'case_id': case_id, 'seed': seed, 'cell': cell,
              'qualifies': qualifies, 'qualification_before_conflict_patch': True,
              'arms': arms, 'technical': checks, 'control_target': control_target}
    if not qualifies:
        record['exclusion_reason'] = 'native competence or native answer mass'
        record['runtime_seconds'] = time.perf_counter()-start
        return record, audit_arrays

    for k, (arm, (prepared, vectors)) in enumerate(zip(arms, active)):
        prompt = prepared['recipient']
        length = prompt['input_ids'].shape[1]
        entities = [g[TARGET] for g in arm['matrix']]

        def patch(vector, label, save_full=False):
            out, hook = runner.forward(prompt['input_ids'], (runner.layer, vector))
            check(f'{k}_{label}_hook', runner.check_hook(hook, length, runner.model.config.hidden_size))
            readout = runner.readout(out, entities, full=save_full)
            full = readout.pop('_full_logits', None)
            if full is not None:
                audit_arrays[f'{case_id}_{k}_{label}'] = full.numpy()
            generation, genhook = runner.generate(prompt['input_ids'], (runner.layer, vector))
            check(f'{k}_{label}_generation_hook', genhook['writes'] == 1
                  and genhook['positions'] == [length-1])
            check(f'{k}_{label}_readout_generation', bool(generation['token_ids'])
                  and generation['token_ids'][0] == readout['top_token_id'])
            readout.update(generation=generation, hook=hook,
                           generation_hook={name: genhook[name] for name in ('writes', 'positions')})
            del out
            return readout

        arm['patch'] = patch(vectors['donor'], 'conflict', audit)
        identity, ihook = runner.forward(prompt['input_ids'], (runner.layer, vectors['recipient']))
        check(f'{k}_identity_hook', runner.check_hook(ihook, length, runner.model.config.hidden_size))
        err = float((identity.logits[0, -1].float().cpu()-vectors['native_full_logits']).abs().max())
        arm['identity_max_full_logit_error'] = err
        check(f'{k}_identity', err <= 0.001)
        if audit:
            audit_arrays[f'{case_id}_{k}_native'] = vectors['native_full_logits'].numpy()
        del identity

        # Same deterministic derangement in both arms; positive control selected by
        # generated family parity, never by conflict effects.
        control = agreement_control(arm['matrix'], control_target, cell['i_N'], random.Random(seed+10000000))
        indices = design_indices(**control)
        check(f'{k}_agreement_indices', all(indices[key] == control_target for key in ('i_P','i_L','i_R')))
        _, native_control, out = runner.native(control['donor'], control_target,
                                                control['donor'][control_target][TARGET])
        vector = out.hidden_states[runner.layer][0, -1].detach().clone()
        del out
        readout = patch(vector, 'agreement')
        winner = max(range(len(entities)), key=readout['answer_logits'].__getitem__)
        arm['agreement'] = {'native': native_control, 'readout': readout,
                            'target': control_target, 'transfer': winner == control_target}

    record['runtime_seconds'] = time.perf_counter()-start
    return record, audit_arrays
