"""Second factorial split: token composition versus positions of the same token."""
import torch

from runtime import _check_rows


def token_factorized_weights(native, strings, mode):
    """Preserve BOS/EOS and total bracket mass in every cell.

    within_token: flatten among identical symbols; preserve symbol totals.
    token_mass: use count-proportional symbol totals; preserve conditional rows.
    both: uniform over brackets, identical to runtime's routing_only endpoint.
    """
    if mode not in {'within_token', 'token_mass', 'both', 'identity'}:
        raise ValueError('Unknown token-factor mode')
    if len(strings) != native.shape[0]:
        raise ValueError('One string per row required')
    eos = torch.tensor([len(s) + 1 for s in strings], dtype=torch.long)
    _check_rows(native, eos)
    result = native.clone()
    if mode == 'identity':
        return result
    for i, text in enumerate(strings):
        mass = native[i, 1:len(text)+1].sum()
        for symbol in '()':
            indices = [j + 1 for j, char in enumerate(text) if char == symbol]
            if not indices:
                continue
            current = native[i, indices]
            current_mass = current.sum()
            if mode == 'within_token':
                result[i, indices] = current_mass / len(indices)
            elif mode == 'token_mass':
                if not current_mass > 0:
                    raise ValueError('Zero native group mass: conditional routing undefined')
                result[i, indices] = (current / current_mass) * mass * len(indices) / len(text)
            else:
                result[i, indices] = mass / len(text)
    _check_rows(result, eos)
    return result
