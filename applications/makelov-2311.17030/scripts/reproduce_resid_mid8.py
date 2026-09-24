"""Re-run the resid_mid.8 pilot and confirmation and compare with the committed results.

    python3 scripts/fetch_upstream.py
    PYTHONHASHSEED=0 python3 scripts/reproduce_resid_mid8.py

The pinned runners refuse to run twice and require every input to be committed; that is
what made the original run credible. This script is for checking afterwards. It calls
the same pinned functions, writes nothing, and reports whether

  * the pilot's blinded record matches ``results/resid_mid8_pilot/pilot.json``,
  * Freeze B follows from it (``results/resid_mid8_freeze_b/predictions.json``),
  * every confirmation pair's cells match ``results/resid_mid8_confirmation/per_pair.jsonl``,
  * the scored outcome matches ``results/resid_mid8_confirmation/result.json``.

Logit differences are compared to 1e-4 (CPU float32 can differ in the last bits across
machines); interchange indicators, pair ids and verdicts must match exactly.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import makelov_table1 as M  # noqa: E402
import resid_mid8_freeze as F  # noqa: E402
import resid_mid8_pilot as P  # noqa: E402

RESULTS = ROOT / 'results'
TOLERANCE = 1e-4


def _runner():
    spec = importlib.util.spec_from_file_location(
        'confirmation', ROOT / 'scripts' / 'run_resid_mid8_confirmation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _close(a, b):
    return abs(a - b) <= TOLERANCE


def main():
    if os.environ.get('PYTHONHASHSEED') != '0' or P.upstream_symbol_order() != P.SYMBOL_ORDER:
        sys.exit('run with PYTHONHASHSEED=0')
    checks = {}
    runner = _runner()
    dist = M.load_distribution(ROOT / 'artifacts' / 'makelov_source')
    model = runner.load_model()
    w_qs = [model.W_Q[layer, h].detach().cpu() for layer, h in P.NAME_MOVERS]
    directions, audit = P.decompose_namemovers(
        P.load_direction(ROOT / 'artifacts' / 'makelov_source' / 'das_resid_mid.joblib'), w_qs)

    pilot = json.loads((RESULTS / 'resid_mid8_pilot' / 'pilot.json').read_text())
    checks['decomposition'] = all(_close(audit[k], pilot['decomposition'][k])
                                  for k in audit if isinstance(audit[k], float))
    summary = P.blinded_summary(P.effects(P.run_conditions(
        model, P.build_pilot_dataset(dist), directions)))
    frozen = pilot['summary']
    checks['pilot'] = (
        all(_close(summary['mean_E_full'][r], frozen['mean_E_full'][r]) for r in P.READOUTS)
        and all(_close(summary['sd'][r][x][k], frozen['sd'][r][x][k])
                for r in P.READOUTS for x in P.COMPONENTS for k in P.CONTRASTS))

    predictions = json.loads((RESULTS / 'resid_mid8_freeze_b' / 'predictions.json').read_text())
    again = json.loads(json.dumps(F.predict(frozen)))
    checks['freeze_b'] = all(again[k] == predictions[k]
                             for k in ('z', 'primary', 'crossover', 'secondary'))

    dataset = runner.build_confirmation(dist)
    cells = P.run_conditions(model, dataset, directions)
    stored = [json.loads(line) for line in
              (RESULTS / 'resid_mid8_confirmation' / 'per_pair.jsonl').read_text().splitlines()]
    same_pairs = [r['pair_id'] for r in dataset] == [s['pair_id'] for s in stored]
    same_cells = same_pairs and all(
        _close(float(cells[c]['ld'][i]), s[f'ld_{c}'])
        and int(cells[c]['interchange'][i]) == s[f'interchange_{c}']
        for i, s in enumerate(stored) for c in P.CONDITIONS)
    checks['confirmation_pairs'] = same_pairs
    checks['confirmation_cells'] = same_cells

    d = P.contrasts(P.effects(cells))
    realized = {(r, x, n): F.realized(d[r][x]['d_inert'][:n], d[r][x]['d_all'][:n])
                for r in F.READOUTS for x in F.COMPONENTS for n in F.LADDER}
    scored = F.score(predictions['primary'], realized)
    result = json.loads((RESULTS / 'resid_mid8_confirmation' / 'result.json').read_text())
    checks['primary_outcome'] = (
        scored['verdict'] == result['primary']['verdict']
        and [c['realized_decided'] for c in scored['cells']]
        == [c['realized_decided'] for c in result['primary']['cells']])
    blocks = [F.block_rate(d['A'][c['component']]['d_inert'], d['A'][c['component']]['d_all'],
                           c['n'])['decided'] for c in predictions['secondary']]
    checks['secondary_blocks'] = blocks == [s['decided'] for s in result['secondary']]

    print(json.dumps({'checks': checks, 'verdict': scored['verdict'],
                      'matches': f"{scored['matches']}/{scored['total']}"}, indent=1))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
