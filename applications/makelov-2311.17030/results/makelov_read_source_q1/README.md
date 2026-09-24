# Q1 confirmation: the null-read explanation predicts fresh read patches better

Preregistration `PREREG_READ_SOURCE_Q1.md`, frozen and pushed in `829228c` before this run;
the run's manifest records that commit as its `git_head`. 64 fresh base pairs (seed
20260922, none shared with the pilot), GPT-2 Small, MLP 8, float32, CPU. Identity and
fidelity controls passed.

**Primary, judged:** B (null read) predicted the two read interventions better than A
(visible read) in **64 of 64** base pairs, with no ties. Exact two-sided sign test
p = 1.1e-19. **Outcome: B predicts better.**

Reported, not judged: mean per-pair loss A 1.223, B 0.238 nats; mean difference 0.984
nats, nominal 95 % bootstrap interval [0.905, 1.064]. The sign-flip p (Monte Carlo floor
5e-5) assumes symmetric differences under the null.

**Reach.** This confirms a comparison between two declared explanations, on the pilot's
case-wise loss, at one site of one model: B wins a fresh case more often than A. Nothing
here shows that B is accurate enough (Q2 was deferred, and no tolerance was declared),
that B's reading is the mechanism, or anything about explanations not declared.

Files: `score.json` (scorer output), plus the runner's `cases.json`, `manifest.json`,
`records.jsonl`, `summary.json`, `reference_cpu32.json`, `reference_cpu64.json` and
`directions.npz`.
