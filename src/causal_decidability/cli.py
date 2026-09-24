"""One command, CPU, no model: can this design decide between its rivals?

    causal-decide \
        --prediction at_node=1.60 --prediction after_node=0.0 \
        --prediction at_node_affine=1.592 --equivalent at_node,at_node_affine \
        --n 24 --sigma 0.04 --dtype bfloat16 --readout-scale 3.4 --depth 36

Predictions are what each rival implies for the estimand UNDER THE OPERATOR YOU ACTUALLY
EXECUTE, not under the one you meant. If you have coefficients instead, pass
`--coefficient name=value` repeatedly together with `--interaction`.

`--readout-scale` is the magnitude of ONE readout value the estimand is built from, and
`--depth` the number of accumulation steps inside it. Both are properties of your
implementation, not measurements. `--json` prints the whole result. The exit status is 0
when the planning ratio exceeds one, 1 when it does not, and 2 when a declaration is
invalid (for example a non-finite sigma). These are heuristic planning statuses, not
power calculations or inference about which explanation is correct. For backwards
compatibility, --json always exits 0 on valid declarations; inspect its fields.
"""
import argparse
import json

from . import calculator as decidability


def _pairs(values, cast=float):
    out = {}
    for item in values or ():
        if '=' not in item:
            raise SystemExit(f'expected name=value, got {item!r}')
        name, raw = item.split('=', 1)
        out[name.strip()] = cast(raw)
    return out


def build(args):
    predictions = _pairs(args.prediction)
    coefficients = _pairs(args.coefficient)
    if coefficients:
        if args.interaction is None:
            raise SystemExit('--coefficient needs --interaction')
        predictions.update({name: a * args.interaction
                            for name, a in coefficients.items()})
    if len(predictions) < 2:
        raise SystemExit('at least two rival predictions are needed')
    groups = [tuple(part.strip() for part in item.split(','))
              for item in args.equivalent or ()]
    for group in groups:
        for name in group:
            if name not in predictions:
                raise SystemExit(f'unknown candidate in --equivalent: {name!r}')
    return decidability.decidability(
        predictions=predictions, n=args.n, sigma=args.sigma, dtype=args.dtype,
        readout_scale=args.readout_scale, depth=args.depth,
        equivalence_groups=groups, noise_factor=args.noise_factor,
        alpha=args.alpha, signatures=args.signatures)


def render(result, predictions):
    width = max(len(name) for name in predictions)
    lines = ['rival predictions under the operator actually executed']
    for name in sorted(predictions, key=lambda k: predictions[k]):
        lines.append(f'  {name:<{width}}  {predictions[name]:+.6g}')
    lines.append('')
    lines.append(f'separation        {result["separation"]:.6g}'
                 + (f'   ({" vs ".join(result["closest_rivals"])})'
                    if result['closest_rivals'] else ''))
    lines.append(f'statistical floor {result["statistical_floor"]:.6g}'
                 f'   (n={result["n"]}, sigma={result["sigma"]})')
    lines.append(f'numerical floor   {result["numerical_floor"]:.6g}'
                 f'   ({result["dtype"]}, scale={result["readout_scale"]:g},'
                 f' depth={result["depth"]})')
    lines.append(f'resolution        {result["resolution"]:.6g}'
                 f'   ({result["binding_floor"]} floor binds)')
    lines.append('')
    ratio = result['ratio']
    lines.append(f'ratio             {ratio:.3g}')
    lines.append('planning heuristic ' + ('PREDICTED DECIDABLE (ratio > 1)' if result['decidable']
                                         else 'NOT PREDICTED DECIDABLE (ratio <= 1)'))
    lines.append('lever             ' + result['lever']['detail'])
    lines.append('warning           ' + result['interpretation'])
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prediction', action='append',
                    help='name=value, what this rival implies for the estimand')
    ap.add_argument('--coefficient', action='append',
                    help='name=value, used with --interaction')
    ap.add_argument('--interaction', type=float)
    ap.add_argument('--equivalent', action='append',
                    help='comma-separated names that may never be told apart')
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--sigma', type=float, required=True)
    ap.add_argument('--dtype', default='float32', choices=sorted(
        k for k in decidability.MANTISSA_BITS if not k.startswith('torch.')))
    ap.add_argument('--readout-scale', type=float, default=1.0)
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--noise-factor', type=float,
                    default=decidability.DEFAULT_NOISE_FACTOR,
                    help='sd multiplier of the estimand; default sqrt(2), a paired contrast')
    ap.add_argument('--alpha', type=float, default=decidability.DEFAULT_ALPHA)
    ap.add_argument('--signatures', type=int, default=3)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    try:
        result = build(args)
    except ValueError as error:
        ap.error(str(error))  # exit status 2: an invalid declaration is not a verdict
    if args.json:
        print(json.dumps(result, indent=1, sort_keys=True))
        return
    predictions = _pairs(args.prediction)
    if args.coefficient:
        predictions.update({name: a * args.interaction
                            for name, a in _pairs(args.coefficient).items()})
    print(render(result, predictions))
    raise SystemExit(0 if result['decidable'] else 1)


if __name__ == '__main__':
    main()
