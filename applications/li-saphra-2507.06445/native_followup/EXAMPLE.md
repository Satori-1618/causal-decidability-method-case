# What changes when the same symbol is read somewhere else?

The model classifies bracket strings. A valid string must never have more closes
than opens while reading left to right, and must finish with equal counts.

This real fresh-test input has 16 opens and 16 closes but fails the first rule:

```text
()))()(((()(())(())((()())))())(
```

The first three symbols, `())`, already contain too many closes. Nevertheless,
checkpoint `1aez5d6p` initially accepts it. The table gives its actual rejection
margin: **positive means reject**, negative means accept.

| Intervention at layer 2/head 2 | What changes | Rejection margin |
|---|---|---:|
| Native | Nothing | −0.788 |
| Full uniform | All attention weights | +1.720 |
| Bracket routing only | Which bracket positions are read; special-token masses fixed | +1.482 |
| Group masses only | BOS/EOS/bracket masses; bracket routing fixed | −0.570 |
| Within-symbol only | Weights between occurrences of `(`, and separately between occurrences of `)`; symbol totals fixed | +1.457 |
| Symbol masses only | Total open/close weights; within-symbol routing fixed | −0.753 |

The full patch alone says only that changing attention changes this answer.
The two additional splits show a more specific fact: **changing which occurrence
of the same symbol is read is sufficient here to flip the answer, while changing
the total open/close weights is not.** Different occurrences carry causally
relevant information beyond their raw character identity; context and position
are not yet separated.

This is an illustration, selected after the run as the first qualifying family
in sorted hash order (`005ea0a6f47f8be4f59f`, `invalid_open`, one of 17 such
examples). The evidence is the complete frozen evaluation, not this selected
answer flip. The [main result](README.md) includes the actual population bounds
and all transfer failures.

## Reuse the procedure

1. **List what your intervention changes.** Here: special-token masses,
   open/close totals, and routing between identical symbols.
2. **Write opposite predictions.** If only symbol totals explain the bracket-
   routing effect, changing within-symbol weights should reproduce native output;
   changing symbol totals should reproduce the full bracket-routing effect.
   The within-symbol account predicts the reverse.
3. **Check separation before the two new interventions.** If the measured
   native and bracket-routing endpoints are closer than the justified tolerance,
   these forecasts cannot be resolved for that case. Do not count such cases as
   successful discrimination.
4. **Execute valid, isolated changes.** Check the actual injected weights,
   untouched values and other heads, numerical precision and hook placement.
5. **Keep all outcomes.** One account can fit, both can fit, both can fail, or
   the evidence can be insufficient. New data, a fixed tolerance and the correct
   independent unit are required for a population-level adequacy claim.

Required data: access to the model's attention and value tensors, exact target
positions, inputs and task labels, native/intervened task margins per input,
and family identifiers. Output labels alone cannot verify this internal
intervention. The implementation is [token_split.py](token_split.py); its first
argument is a native attention row and its output is the explicitly modified row.
