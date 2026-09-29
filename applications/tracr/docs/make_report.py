"""Render an evidence-bound one-page figure and report from frozen Tracr records."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

APP = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def main():
    run = APP / "results/confirmation_001"
    dev = APP / "results/development_004"
    summary, verification = load(run / "summary.json"), load(run / "verification.json")
    manifest, trajectories = load(run / "manifest.json"), load(run / "trajectories.json")
    development = load(dev / "summary.json")
    if verification["status"] != "PASS":
        raise ValueError("Report requires records-only verification")
    # Fixed exposition rule: the first frozen family, not the strongest measurement.
    first, trace = manifest["families"][0], trajectories[0]
    by_id = {c["case_id"]: c for c in [first["shared"], *first["discriminators"], *first["followups"]]}
    cases = [by_id[step["case_id"]] for step in trace["trajectory"]]
    token = lambda n: chr(ord("A") + int(n))
    sequence = lambda x: "[" + ", ".join(token(v) for v in x) + "]"
    outputs = [token(max(range(12), key=lambda i: step["observation"][i])) for step in trace["trajectory"]]
    addr = [token(c["recipient"][2]) for c in cases]
    copy = [token(c["donor"][2]) for c in cases]
    decision = summary["population_decision"]
    lower = decision["bounds"]["lower_rate"]
    records = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]
    maximums = {k: max(r["checks"]["diagnostics"][k] for r in records) for k in
                ("insert_error", "complement_error", "native_site_error", "donor_source_error", "noop_score_error", "self_score_error")}
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="#fbfcfe")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    navy, teal, muted = "#18334b", "#087f8c", "#53677b"
    fig.text(.075, .949, "FROM A PATCH EFFECT TO A NARROWER EXPLANATION", fontsize=9, color=teal, weight="bold")
    fig.text(.075, .903, "Two explanations. One separating test.", fontsize=22, color=navy, weight="bold")
    fig.text(.075, .868, "Measured in DeepMind’s Tracr reversal Transformer", fontsize=13, color=muted)
    fig.text(.075, .833, "Known circuit • frozen rule • 128 fresh token-assignment families", fontsize=10, color=muted)

    headers = ["1  Shared outcome", "2  Separate the rivals", "3  Check a new prediction"]
    xs = [.075, .365, .655]
    for i, x in enumerate(xs):
        ax.add_patch(FancyBboxPatch((x, .505), .265, .273,
                                   boxstyle="round,pad=0.007,rounding_size=0.01",
                                   facecolor="#eaf3f7" if i == 0 else "#e6f3ef", edgecolor="none"))
        fig.text(x+.013, .748, headers[i], fontsize=10, color=navy, weight="bold")
        fig.text(x+.013, .714, "Recipient  " + sequence(cases[i]["recipient"]), fontsize=9.3, color=navy)
        fig.text(x+.013, .689, "Donor       " + sequence(cases[i]["donor"]), fontsize=9.3, color=navy)
        fig.text(x+.013, .651, "Address predicts       " + addr[i], fontsize=10, color=navy)
        fig.text(x+.013, .626, "Answer copy predicts  " + copy[i], fontsize=10, color=navy)
        fig.text(x+.013, .583, "Measured: " + outputs[i], fontsize=17, color=teal, weight="bold")
        fig.text(x+.013, .541, "Both remain" if i == 0 else "Address remains", fontsize=11, color=navy, weight="bold")
    for x in (.346, .636):
        fig.text(x, .633, "→", fontsize=19, color=teal, ha="center")
    fig.text(.075, .472,
             "Native first answer: " + token(cases[0]["recipient"][3]) +
             ". The patch copies only the intermediate address (third content position).",
             fontsize=9.7, color=muted)
    fig.text(.075, .449, "Changing donor content separates the explanations; changing recipient content tests the survivor.",
             fontsize=9.5, color=muted)

    fig.text(.075, .393, f"{summary['site_A_successes']}/{summary['family_count']} fresh families passed", fontsize=20, color=navy, weight="bold")
    fig.text(.075, .361, f"One-sided 95% lower bound: {100*lower:.2f}%  •  Frozen adequacy target: 95%", fontsize=11, color=muted)
    fig.text(.075, .335, "The same classifier also recognized deliberate answer copying in 8/8 development controls.",
             fontsize=10, color=muted)
    fig.text(.075, .295, "What makes this more than an output change?", fontsize=13, color=navy, weight="bold")
    fig.text(.075, .266,
             "Declared rival predictions; a separating condition chosen from those predictions;\n"
             "audited tensor replacement; paired precision checks; a frozen fresh-family test.",
             fontsize=11, color=navy, linespacing=1.55)
    fig.text(.075, .191, "What this establishes", fontsize=13, color=navy, weight="bold")
    fig.text(.075, .162,
             "The method reproduces the expected narrowing in an externally built, known circuit.\n"
             "The site was chosen using compiler structure. This validates the implementation\n"
             "and tested candidate comparison; it does not discover an unknown LLM mechanism.",
             fontsize=10.5, color=muted, linespacing=1.5)
    fig.text(.075, .075, "Felix Borck  |  causal-decidability-method-case  |  codex/tracr-iterative-validation", fontsize=8.5, color=muted)
    fig.text(.075, .051, "Source: applications/tracr/results/confirmation_001  •  First frozen family shown; letters map tokens 0–11.",
             fontsize=8.1, color=muted)
    out = APP / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "tracr_narrowing.pdf")
    fig.savefig(out / "tracr_narrowing.png", dpi=165)
    plt.close(fig)
    table = "\n".join(f"| {i+1} | {sequence(c['recipient'])} | {sequence(c['donor'])} | {addr[i]} | {copy[i]} | {outputs[i]} |"
                      for i, c in enumerate(cases))
    report = f"""# Tracr: measured iterative narrowing

**Status: frozen fresh-family confirmation completed; records-only verification PASS.**
The claim concerns one known compiled reversal circuit and the two declared
interpretations of its address-subspace patch. The operator chose the site using
compiler structure, so the outcome was structurally expected. This is external
implementation/method validation, not discovery of an unknown LLM mechanism.

![Measured narrowing](figures/tracr_narrowing.png)

[One-page PDF](figures/tracr_narrowing.pdf).

## Fresh result

- **{summary['site_A_successes']}/{summary['family_count']}** whole fresh families produced the qualified **2 → 1 → 1** trajectory.
- One-sided 95% finite-population lower bound: **{lower:.8f}**; frozen adequacy threshold **0.95**. Decision: **{decision['status']}**.
- Required count was **126/128**. There was one final population test, no optional extension, no replacement families, and no per-stage statistical look.
- All required technical checks passed in **{summary['record_count']}** dtype-specific intervention records. These are not independent samples.
- Maximum paired fp32/fp64 full-score deviation: **{summary['paired_precision_max_error']:.10g}**.
- Frozen numerical allowance **{manifest['numeric_allowance']:.10g}**, separate from scientific tolerance **0.01**; total radius **{manifest['numeric_allowance']+.01:.10g}**.
- Development's deliberate final-output-copy control selected donor-answer copying in **{development['site_B_positive_controls']}/8** families. It is a separate technical positive control, not another fresh population claim.

## The first frozen family (not selected by effect size)

Letters are a display mapping of numeric vocabulary 0..11.

| Step | Recipient | Donor | Address predicts | Answer copy predicts | Measured |
|---|---|---|---|---|---|
{table}

The initial observation leaves both candidates compatible. A donor-content change
preserves the transferred address but changes the donor's answer, separating the
predictions. The final recipient-content change checks the surviving rule. That
third row is a within-family predictive check; the population claim comes from
the 128 entirely fresh families, not from counting three rows per family.

## Controls and provenance

Maximum qualification errors across the fresh run:

```json
{json.dumps(maximums, indent=2)}
```

Residual arithmetic uses fp32/fp64, including parameter rounding. The upstream
unembedding matrix produces float64 scores in both modes. These are compiled
projection scores, not nats. Higher precision is a reference, not exact truth.
The source/donor state, source and target position, full subspace, post-cast values
and unchanged complement are bound to stored tensors. Independent upstream-forward
tests also qualify the wrapper, and a model-free verifier recomputes the decisions.

- Upstream: `9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd`.
- Freeze: [CONFIRMATION_FREEZE.json](CONFIRMATION_FREEZE.json), committed before confirmation.
- Fresh summary: [summary.json](results/confirmation_001/summary.json).
- Full trajectories: [trajectories.json](results/confirmation_001/trajectories.json).
- Verification: [verification.json](results/confirmation_001/verification.json).
- Development selected for freeze: [development_004](results/development_004/summary.json).
- Raw tensor snapshots and scores live alongside each summary; the summary binds them by SHA256.

The runner recorded {sum(summary['forward_counts'].values())} analysis forward calls
({summary['forward_counts']}; compiler initialization is excluded from that count)
in {summary['elapsed_seconds']:.2f} seconds including compilation and controls on the
recorded CPU. This measured small-circuit runtime
is not an estimate for pretrained LLM experiments.

## Development history and limits

The original technical smoke and eight development families were disclosed before
confirmation. Development_001 predates stronger donor/source and inventory audits;
development_002 stopped on an overbroad bytecode/source check before outcomes;
development_003 was re-executed after final pre-freeze execution guards. These are
preserved and not counted as independent confirmations. Development_004 passed the
final unchanged surface. Fresh families were generated once, frozen and committed,
then measured without changing the sources or thresholds.

There are two substantive limits. First, the known address site makes the expected
answer analytically clear to the informed operator. This verifies a controlled
external path through the methodology, not unique natural-mechanism discovery.
Second, fresh token assignments retain the same program, length, intervention and
role structure. They do not establish performance on different algorithms, learned
representations, close continuous rivals, or a generally optimal adaptive policy.
The two-option selection menu ties in separation score; a fixed tie break chooses
one. No advantage over a strong fixed design is claimed.

## Next useful extension

Freeze a second compiled program or multiple opaque intervention sites whose roles
the inference code does not receive, including equivalent and out-of-set cases.
Measure false exclusions, correct retention and abstention by difficulty. A later
pretrained-model application tests transfer beyond the controlled compiler setting.
"""
    (APP / "RESULTS.md").write_text(report)


if __name__ == "__main__":
    main()
