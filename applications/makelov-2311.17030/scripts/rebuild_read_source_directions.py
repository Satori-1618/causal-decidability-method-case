"""Rebuild the read-source runs' directions.npz from the fetched upstream vector.

    python3 scripts/fetch_upstream.py
    python3 scripts/rebuild_read_source_directions.py

The file holds the authors' published MLP8 direction (normalised) and its split into the
output-visible and output-null parts, so it is not redistributed here. This script
recomputes it with the pilot's own functions, from das_mlp8.joblib and GPT-2's W_out, and
keeps it only if its sha256 equals the one the pilot recorded. The verifier
(scripts/verify_makelov_read_source.py) needs it.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import makelov_read_source as experiment  # noqa: E402

#: both runs used the same directions (identical sha256), so both get the rebuilt file
TARGETS = [ROOT / 'results' / run / 'directions.npz'
           for run in ('makelov_read_source_001', 'makelov_read_source_q1')]
EXPECTED = 'a3d41db3ce884de11a35e643d28474ebb1158eb2f597414a7b676fe94706e475'


def main():
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        'gpt2-small', device='cpu', center_unembed=True, center_writing_weights=True,
        fold_ln=True, refactor_factored_attn_matrices=True)
    directions, _ = experiment.decompose_direction(
        experiment.load_published_vector(ROOT / 'artifacts' / 'makelov_source' / 'das_mlp8.joblib'),
        model.W_out[8].detach().cpu().numpy())
    for target in TARGETS:
        np.savez(target, **directions)
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got != EXPECTED:
            target.unlink()
            sys.exit(f'rebuilt directions differ from the recorded ones ({got}); removed')
        print('rebuilt', target.relative_to(ROOT), got)


if __name__ == '__main__':
    main()
