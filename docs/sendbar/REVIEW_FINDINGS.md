# Final repository review of `ca7614a`: findings and actions

Internal working document, 2026-10-22.

A fresh review agent read the candidate at `ca7614a` against the acceptance criteria. Its
findings were checked before acting; B1 and B2 were reproduced from the records and the
GitHub repository metadata. The changes below are in the commit that adds this file.

## Blocking

| ID | Finding | Checked how | Action |
|---|---|---|---|
| B1 | OVERVIEW said the full patch "changes the top answer in 4.3% of pairs". 4.3% is interchange accuracy (86/2000: the source prompt's answer becomes the top prediction). The top token changes in 201/2000. | `describe_claim_inputs.py makelov-table1`; `per_example.jsonl` | Reworded to "makes the source prompt's answer the top prediction in 4.3% of pairs (interchange accuracy; 0.3% without a patch)". The claim now also checks the unpatched 0.3% and n = 2000. |
| B2 | OVERVIEW said the public event log "does not reach back that far" for round 1. In fact the public repository was created on 2026-09-24T10:42:02Z, after that run; `PROVENANCE.md` says the push went to the then-private development repository. | GitHub repository metadata; `PROVENANCE.md` Q1 paragraph | Provenance bullet and claim rewritten: frozen in `829228c`, pushed per the project's records to the then-private repository; no public timestamp exists. The run manifest's `git_head` (`829228c…`) is named as what can be checked. |
| B3 | The Mixing setup table did not say that the pinned upstream commit `c53372c` ("Fix binding info leakage by patching last token positions") makes `[-1, -4, -6, -8]` the helper and notebook default. | `SOURCE_LOCK.json` commit subject; `SOURCES.md` (`tasks/dist.py:198-200`, `example.ipynb:208`) | Row now states that this run follows the paper text and experiment loop (`[-1]`) and that leakage through the other positions is untested here. |

## Should fix

| ID | Finding | Action |
|---|---|---|
| S1 | "Only a proposed third design separates all four rules" reads as if the third design alone did. | CausaLab README, root README and evidence map: "Adding a proposed third design to both published ones separates all four rules." |
| S2 | "The setup does not by itself establish" the caveat understated what follows: the direction follows from Jensen's inequality. | Mixing README, round-2 note, root README, evidence map, result page and website card: the caveat is quoted; its direction follows from Jensen (equality only when all cases share one argmax); its effect on the coverage decision, and whether it is largest in this cell, was never quantified. |
| S3 | "Every value in this table is read from the frozen manifest.json" was not true for the paper and upstream rows. | The note now separates manifest/record values from source readings (`SOURCE_LOCK.json`, `SOURCES.md`). |
| S4 | Several numbers on the Mixing and Tracr pages had no claim entry, and `mixing.cases.labels` cited a page that did not show its numbers. | New claims: unresolved reason (10 by support, 0 by answer mass), both mean vectors, W_T adequacy bound 0.028, between-case decision; Tracr required count 126/128, numerical allowance, scientific tolerance and total radius, and the development copy control 8/8 (verified with the frozen verifier on a temporary copy of the run whose summary hash the freeze records). The result page now states the label counts (157/125/8, no ties). |
| S5 | The "128/128 directed cases" geometry claim checked a derived agreement count, not the count itself. | `describe_post_hoc.py` reports `directed_cases_with_larger_null_coefficient`; the claim and a unit test check it. |
| S6 | Check lines did not say that a full Git clone is needed. | Check lines say so. `check_claims.py` and `verify_release.py` stop with a clear message in a ZIP or `--depth` checkout before a check that reads Git history. |
| S7 | Glossary: "unresolved cases count against adequacy and for exclusion" was ambiguous. | Now: "count as non-matches for adequacy and as matches for exclusion". |
| S8 | `METHOD_PREFLIGHT.md`: an inserted paragraph separated the table from its lead-in sentence. | Lead-in restored; the alignment paragraph follows the table's explanation. |
| S9 | Glossary used "confirmation" both as a phase and as an outcome. | "Confirmation" names a phase whatever the outcome; "confirmed" qualifies the decision a confirmation run reached (including a confirmed exclusion), not a candidate. |

## Optional, also applied

- Root README and OVERVIEW name hypotheses A and B as hypotheses, not as components.
- "as the designs intend" → "by construction" (root README, CausaLab row).
- "under 0.5 MB" → "about 0.5 MB (503,367 bytes)"; the sum of the pinned files in
  `reference_manifest.json`.
- OVERVIEW: the proportional rule scales by the ratio of insertion norms.
- Mixing stored claims read the interval arrays and fields that the records checker
  compares with its recomputation. The mean vectors and the mean-gate limit are recomputed
  from the records and read from `manifest.json`.
- `MODEL_FREE_COMMANDS.md` now lists the interpreter flags actually used (`-E -s -B`) and
  every `describe_claim_inputs.py` subcommand.

## Not changed

- Historical round reports keep their wording ("confirmed exclusion"); the glossary now
  explains that usage.
- Hash-bound files (`release/FROZEN_FILES.json`, round-3B manifest) are untouched. No
  changed file's previous hash is recorded anywhere in the repository.
