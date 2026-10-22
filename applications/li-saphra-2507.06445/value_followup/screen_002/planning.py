"""Exact finite-sample power under explicit hypothetical Bernoulli rates."""
import json
import math
from analyze_screen import N, primary_counts


def plan():
    decisions = {(a, r): primary_counts(a, r) for a in range(N+1) for r in range(N+1)}
    result = {'n_per_stratum': N, 'conditional_on': 'filled strata and valid technical gates',
              'rates_are_assumptions_not_pilot_predictions': True, 'scenarios': []}
    for pa, pr in [(0.9, 0.05), (0.75, 0.1), (0.5, 0.1)]:
        powers = {'positive_enrichment': 0., 'substantial_enrichment': 0.}
        for (a, r), report in decisions.items():
            probability = (math.comb(N, a)*pa**a*(1-pa)**(N-a)
                           * math.comb(N, r)*pr**r*(1-pr)**(N-r))
            if report['directional_status'] == 'positive_enrichment':
                powers['positive_enrichment'] += probability
            if report['decision'] == 'supports_at_least_25pp_enrichment':
                powers['substantial_enrichment'] += probability
        result['scenarios'].append({'accepted_rate': pa, 'rejected_rate': pr, **powers})
    return result


if __name__ == '__main__':
    print(json.dumps(plan(), indent=2))
