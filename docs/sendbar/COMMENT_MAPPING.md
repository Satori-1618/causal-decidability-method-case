# Commented review of plan v2 (36 comments): what was done

Internal working document, 2026-10-22.

**Source:** `PRUEFPAKET_PLAN_V2.zip`, uploaded by Felix, SHA-256 `b6b9c58e…af3`. Its
`ABZEICHNUNG.json` lists every comment as open until signed off. Felix asked in the chat to
incorporate the feedback and start implementing. The entries below record what was done
under that instruction. **Sign-off per ID stays with Felix.** External steps (merge,
website, upload, mail) remain separately unapproved.

## Status values

- **done:** changed in this repository, with the commit named in the history.
- **compliant:** already met by the current text; nothing to change.
- **blocked:** needs private inputs or a decision by Felix.
- **web:** applies to the local website patch.

## Repository and process

| ID | Topic | Status | What was done |
|---|---|---|---|
| K01 | Goodfire label | done | Only labels that meant Mixing Mechanisms were changed. Crosswalk file and manifest are byte-identical to `main`. Remaining `goodfire` hits are explained below. |
| K02 | "independent" | done | The glossary distinguishes statistical, computational and review independence. Entry pages describe the review kind and claim no external review. Archived records unchanged. |
| K03 | W_T caveat | done + web | Result page, case README, round-2 note, root README and evidence map separate three things: the pre-run caveat (quoted as a concern), Jensen's inequality, and the measured result. No "largest" claim is made as a fact. Website card: see the website patch. |
| K04 | Makelov push records | done | `provenance/public_push_events.json` links each event to its activity record via `payload.push_id` (rounds 2, 3A, 3B). OVERVIEW names both records. |
| K05 | Historical approval states | compliant | `PREFLIGHT_PLAN.md` keeps its wording; only a dated "later note" was added. The README table points to `FREEZE.json`. |
| K06 | CausaLab hash | done | `results/summary.json` and `source_manifest.json` are restored to `main`. The re-derivation is in `results/rederivation_20261022/`. `run.py --write` never writes into `results/` and refuses to replace differing files. Cited docs are pinned in `reference_manifest.json`. |
| K07 | 16 % vs 17 % | compliant | Not unified. OVERVIEW states a median over 128 directed cases; the frozen app README keeps its pilot ratio. |
| K08 | Flip rate | compliant | The primary metric (logit difference) is named. The 4.3 % rate is defined (interchange accuracy) and gives its population. |
| K09 | CA novelty | done | No novelty claim. The CausaLab README names the library's own hypothesis comparison and published controls; `top_k` is cited from its guide. |
| K10 | R route | done | The glossary says Tracr documents replay, and that no model was rerun here. |
| K11 | Q1 "pushed" | done | OVERVIEW: a historical statement without a public timestamp available today; not a finding that it is wrong. |
| S01 | Confirmation vs transfer | done | The results block separates the development checkpoint, the illustrated transfer head and the full cohort, with every category. "7 of 35" is labelled as the stage-2 H_W status. |
| S02 | Invalid-only separation | done | Scoped to `a9g0io1r`/1 with member counts. 20 and 15 transfer tasks with valid separating members are named. Claim reads the report fields. |
| S03 | 7.7 s | compliant | Stated for the native round only. |
| S04 | Li/Saphra verifier coverage | done | `verify_release.py` checks the retrospective outputs, the native archive (temporary copy) and the round-004 recount. Not reconstructable: full attention and off-target tensors (stated). |
| S05 | Zero vs mean question | done | README: zero output is "a further question, not a missing control". The mail draft is not part of this work. |
| U01 | Claim coverage and kinds | done | Source/provenance claims are entries of kind `source`, with evidence and a manual-check note; the runner never marks them checked. |
| U02 | Phase vs outcome | done | `CLAIMS.json` schema 2 has separate `phase` and `outcome`. The glossary explains them. |
| U03 | Local candidate first | done | The candidate is verified locally (`VERIFICATION_RUN.md`) before any merge, website or upload request. |
| U04 | Intake | done | `INTAKE.md`. |
| U05 | Model-free commands | done | `MODEL_FREE_COMMANDS.md`, with an allow-list enforced in `check_claims.py`. |
| U06 | Coverage instead of 8/8 | done | `verify_release.py` runs a named list of 13 checks and fails unless exactly that list passed. |
| U07 | Self-contained handover | done | `PLAN_v3.md` lists every ID with its status. This file maps the 36 comments. |
| U08 | Package manifests | blocked | Needs the private TaxiGPT and Parrots inputs. |
| U09 | Adequacy contracts | done | `docs/GLOSSARY.md` gives a table of the frozen parameters for each case. |

## TaxiGPT and Causal Parrots

| ID | Topic | Status | What was done |
|---|---|---|---|
| TP01 | Interval direction | web | Website patch: "likely too narrow" removed; it says donor dependence is not modelled and its effect on coverage was not assessed. |
| TP02 | Exploratory status | web | Landing card: exploratory follow-up; data access stated separately ("records not yet public"). |
| TP03 | No ≈ between V-only and joint | web | The page says 457/512 same correct-or-incorrect readout and that the decoded location is not compared. The open 10 % judgement stays beside it. No outreach text drafted. |
| TP04 | Taxi development branches | blocked | Package scope; the website notes the development files as private and not checkable. |
| TP05 | Position audit | web + blocked | Option (b): the website states that the audit's detail statements are not publicly checkable. Option (a) needs private inputs. |
| TP06 | Three-way comparison verdict | blocked | Needs the original paper (not reachable here). |
| TP07 | Accuracy endpoint | blocked | Needs the private 1,536 rows. |
| TP08 | Commit roles | web | Both commit roles are kept and labelled; the plural "other models" is corrected. |
| TP09 | SVG | compliant | No graphic changed. |
| TP10 | Prompt reconstruction | blocked | Needs the private inputs and producer. |
| TP11 | No invented rationale | compliant | No chat-template rationale or label quota added. |

## Remaining `goodfire` hits after K01

- the CausaLab case folder `applications/goodfire-mcqa-preflight/` and its texts (correct);
- `docs/GOODFIRE_CROSSWALK.md`: file name hash-bound; explained in `docs/EVIDENCE_MAP.md`;
- references to that file name in `ROLE_BASELINE_PROTOCOL.md` (hash-bound),
  `run_role_baseline.py` and the round-3B manifest;
- `applications/design-comparison/results/development_001/manifest.json`: historical branch name;
- `applications/gur-arieh-2510.06182/SOURCE_LOCK.json`: an affiliation listed by the paper;
- the explanatory note in `RELEASE_NOTES.md`.
