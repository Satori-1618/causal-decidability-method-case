"""Freeze B of ``PREREG_RESID_MID8.md``: the numeric predictions, from the pilot record.

    python3 scripts/freeze_b_resid_mid8.py

Reads ``results/resid_mid8_pilot/pilot.json``, calls the pinned calculator as §8
declares, and writes ``results/resid_mid8_freeze_b/predictions.json``. No model, no
data: the inputs are the pilot's allowlisted numbers only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import makelov_table1 as M  # noqa: E402
import resid_mid8_freeze as F  # noqa: E402
import resid_mid8_pilot as P  # noqa: E402

PILOT = ROOT / 'results' / 'resid_mid8_pilot' / 'pilot.json'
OUTPUT = ROOT / 'results' / 'resid_mid8_freeze_b' / 'predictions.json'
PILOT_HEAD = '47b6e19bdf519e630734fe1868bbe131ada02565'


def main():
    if M.sha256(ROOT / 'src' / 'decidability.py') != P.CALCULATOR_SHA:
        sys.exit('refused: calculator does not match the hash pinned in §8')
    record = P.validate_record(json.loads(PILOT.read_text()))
    if record['provenance']['git_head'] != PILOT_HEAD:
        sys.exit('refused: pilot record is not from the committed pilot')
    if OUTPUT.exists():
        sys.exit(f'refused: {OUTPUT.relative_to(ROOT)} exists')
    out = F.predict(record['summary'])
    out['declaration'] = {
        'prereg': 'PREREG_RESID_MID8.md', 'prereg_sha256': P.PREREG_SHA,
        'calculator_sha256': P.CALCULATOR_SHA, 'pilot_sha256': M.sha256(PILOT),
        'pilot_git_head': PILOT_HEAD, 'ladder': list(F.LADDER), 'alpha': F.ALPHA,
        'signatures': F.SIGNATURES, 'noise_factor': F.NOISE_FACTOR, 'dtype': F.DTYPE,
        'band': list(F.BAND), 'readouts': F.READOUT_DECLARATION,
        'sigma_rule': 'max(sd d_inert, sd d_all) from the pilot',
        'secondary': {'readout': F.SECONDARY_READOUT, 'ladder': list(F.SECONDARY_LADDER),
                      'status': 'added at Freeze B, before any confirmation pair exists; '
                                'reported, not judged; outside section 10'},
        'confirmation': {'seed': F.CONFIRMATION_SEED,
                         'per_combination': F.CONFIRMATION_PER_COMBINATION,
                         'order': 'interleaved (section 13, item 6)'},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    with open(OUTPUT, 'x') as handle:
        handle.write(json.dumps(out, indent=1) + '\n')
    for cell in out['primary']:
        print(f"{cell['readout']} {cell['component']:>4} n={cell['n']:>4}  "
              f"ratio={cell['ratio']:8.2f}  decidable={cell['decidable']}  "
              f"outside_band={cell['outside_band']}")
    print('crossover', out['crossover'])
    for cell in out['secondary']:
        print(f"secondary {cell['readout']} {cell['component']:>4} n={cell['n']:>2}  "
              f"ratio={cell['ratio']:5.2f}  decidable={cell['decidable']}")


if __name__ == '__main__':
    main()
