# Worked example: what did the patch transfer?

This is a fully specified toy example, not a measured LLM result. It is implemented in
[`examples/twelve_cell.py`](../examples/twelve_cell.py); every table entry below is
computed from a rule. A real-model application would first have to verify its baseline
behavior and state the predictions of its candidates under the actual internal patch.

Each prompt asks: **Choose Object or Alternative to receive more points.** Object wins
only when its points are strictly greater; ties go to Alternative. An activation from a
donor is patched into a recipient at a declared site.

| Case | Object points | Alternative points | Baseline choice in the toy |
|---|---:|---:|---|
| Donor D1 | 60 | 10 | Object |
| Donor D2 | 60 | 90 | Alternative |
| Donor D3 | 30 | 10 | Object |
| Donor D4 | 60 | 30 | Object |
| Recipient R1 | 20 | 40 | Alternative |
| Recipient R2 | 20 | 80 | Alternative |
| Recipient R3 | 80 | 20 | Object |

Suppose D1's patch changes R1's answer from Alternative to Object. **That observation
alone cannot tell us what moved.** Eight of the nine candidates below predict that
change, among them transferring the value 60, copying the donor's choice, flipping the
recipient's choice and forcing Object. Only "preserve the recipient's choice" does not.

## Write the competing predictions

Each three-letter block below gives the choices for R1, R2 and R3: `O` = Object,
`A` = Alternative. The columns vary the donor.

| Candidate rule | D1 | D2 | D3 | D4 |
|---|---|---|---|---|
| Transfer donor Object value; keep recipient Alternative value | OAO | OAO | AAO | OAO |
| Transfer donor Alternative value; keep recipient Object value | OOO | AAA | OOO | AAO |
| Copy donor choice | OOO | AAA | OOO | OOO |
| Flip recipient choice | OOA | OOA | OOA | OOA |
| Always choose Object | OOO | OOO | OOO | OOO |
| Preserve recipient choice | AAO | AAO | AAO | AAO |
| Add 30, 40 or 50 to the recipient's **point difference** | OAO | OAO | OAO | OAO |

The final row contains three distinct candidate rules. There are therefore nine
candidates but only **seven prediction patterns** in this twelve-cell design.

## See what each new condition buys

With only D1 and D2, the six-cell design has **five** distinct patterns. It cannot separate
Object-value transfer from any of the three point shifts. It also cannot separate
Alternative-value transfer from copying the donor's choice.

- **Add D3.** At R1, Object-value transfer gives 30 versus 40, hence Alternative.
  The fixed shifts change the recipient's difference of −20 to a positive value,
  hence Object. This cell separates the value rule from the shift rules.
- **Add D4.** At R1, Alternative-value transfer gives 20 versus 30, hence Alternative.
  Copying D4's choice gives Object. This cell separates those two explanations.
- **Keep R3.** Its baseline is Object. Copying D1's Object choice preserves that answer,
  whereas flipping the recipient's choice produces Alternative.

These distinctions follow from the candidate rules **before observing a patch outcome**.
The observation then determines which candidates remain compatible.

## Keep the remaining ambiguity visible

The shifts of 30, 40 and 50 produce identical choices in every cell. Repeating this design
cannot distinguish them through this choice readout; report them together. They are
equivalent **in this design**, not under every possible experiment.

Even the pattern OAO for D1 identifies only `40 < transferred value ≤ 80` under the value
rule. It does not identify 60 exactly. Likewise, a shift of points in this toy is not a
shift of an LLM's log-probability margin; that is a different candidate requiring its own
prediction rule and measurements.

Run the example:

```bash
python3 examples/twelve_cell.py
```

The lesson is **effect → competing explanations → separating conditions → compatible
explanation set**. The first arrow does not establish the last.
