# Provenance

## What can be checked from this branch alone

The hashes form a chain. Each step pins the step before it, and the files here reproduce
every pin:

- `results/resid_mid8_pilot/pilot.json` records the sha256 of the pilot code, the
  calculator, the preregistration and the direction it ran with. All of them match the
  copies here.
- `scripts/freeze_b_resid_mid8.py` accepts only the pilot record from commit
  `47b6e19`, and `results/resid_mid8_freeze_b/predictions.json` follows from that record
  through the pinned calculator (`tests/test_resid_mid8_freeze.py`).
- `scripts/run_resid_mid8_confirmation.py` pins the sha256 of `predictions.json`
  (`3cb542b9…`), and it refuses to run if the file differs.
- `scripts/reproduce_resid_mid8.py` reruns everything and compares.

**What a hash chain cannot show is timing:** that each step was committed *before* the
next one ran. That rests on the commit history of the development repository
(`Satori-1618/vagueness-decision`, branch `codex/impossible-rows-abstraction`), which is
private at the time of writing. The relevant commits:

| step | commit | author date |
|---|---|---|
| Freeze A | `6c8cdb83a3ba0aaf3a35f54be3685147ba6634a7` | 2026-09-21T12:03:40+02:00 |
| clarification and pilot code | `47b6e19bdf519e630734fe1868bbe131ada02565` | 2026-09-21T12:21:06+02:00 |
| pilot record, Freeze B, confirmation runner | `4d54a76b1d5994a2f2d5c99aa5b0004641b0a0f5` | 2026-09-21T12:25:20+02:00 |
| confirmation result | `c735b920ecd94b9f8deaf09991fb565a1ef9a778` | 2026-09-21T12:28:08+02:00 |

Author dates can be set by hand, and this chain carries no external timestamp: nothing
was registered outside the private repository at the time of each freeze. Making that
history public later would show the commits, not when they were made. The hashes prove
which versions are bound to which; the claim that each step ran only after the previous
one was committed rests on the author's record. The pilot record also carries the commit
it ran at (`git_head` in `pilot.json`). Future applications timestamp each freeze
externally at the time (see `applications/README.md` on `main`).

## File by file

Each file below is byte-identical to the development repository at `6539ce5`. The last
column is the commit where the file last changed there.

| file | sha256 (first 16) | last changed |
|---|---|---|
| `PREREG_RESID_MID8.md` | `b8fc20ad96671f13` | `47b6e19` 2026-09-21T12:21:06+02:00 |
| `artifacts/makelov_source/source_manifest.json` | `137c8cd9a7640696` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `results/makelov_replication_001/aggregate.json` | `4239cfe529cda25e` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `results/makelov_replication_001/dataset.jsonl` | `075788148299885d` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `results/makelov_replication_001/decide.json` | `8c2e0b1e4a171f2b` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `results/makelov_replication_001/paired_resolution.json` | `b2a6f2b1f3cc1f65` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `results/makelov_replication_001/per_example.jsonl` | `a368886d2c132526` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `results/resid_mid8_confirmation/README.md` | `eff874174b1f8563` | `1d39412` 2026-09-21T13:52:08+02:00 |
| `results/resid_mid8_confirmation/per_pair.jsonl` | `dd2fe962fdb38bcc` | `c735b92` 2026-09-21T12:28:08+02:00 |
| `results/resid_mid8_confirmation/result.json` | `49b7235ea638bc14` | `c735b92` 2026-09-21T12:28:08+02:00 |
| `results/resid_mid8_freeze_b/predictions.json` | `3cb542b9cfd990e5` | `4d54a76` 2026-09-21T12:25:20+02:00 |
| `results/resid_mid8_pilot/pilot.json` | `ec76fccfc2d3c356` | `4d54a76` 2026-09-21T12:25:20+02:00 |
| `scripts/analyze_makelov_table1.py` | `7c69883aa6ee78d2` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `scripts/decide_makelov_table1.py` | `d7a3a69c0b783157` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `scripts/freeze_b_resid_mid8.py` | `439a1751bd36ee19` | `4d54a76` 2026-09-21T12:25:20+02:00 |
| `scripts/run_makelov_table1.py` | `d6321d97c32adf54` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `scripts/run_resid_mid8_confirmation.py` | `402cab5cc4424da3` | `4d54a76` 2026-09-21T12:25:20+02:00 |
| `scripts/run_resid_mid8_pilot.py` | `a008e11fd3280ea5` | `47b6e19` 2026-09-21T12:21:06+02:00 |
| `src/decidability.py` | `ab0fd838a9a6b354` | `2149078` 2026-09-21T11:49:10+02:00 |
| `src/makelov_table1.py` | `435081ba3b0043b6` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `src/resid_mid8_freeze.py` | `aeffe5949089df22` | `4d54a76` 2026-09-21T12:25:20+02:00 |
| `src/resid_mid8_pilot.py` | `4c400fb828083673` | `47b6e19` 2026-09-21T12:21:06+02:00 |
| `tests/test_decidability_dtype.py` | `12105a35f01c1907` | `2149078` 2026-09-21T11:49:10+02:00 |
| `tests/test_makelov_table1.py` | `78dcb83ab30c068c` | `b3ff96b` 2026-09-20T18:53:06+02:00 |
| `tests/test_resid_mid8_freeze.py` | `e780e02895fc1e02` | `4d54a76` 2026-09-21T12:25:20+02:00 |
| `tests/test_resid_mid8_pilot.py` | `a1d6a8489ec55087` | `47b6e19` 2026-09-21T12:21:06+02:00 |
| `MAKELOV_READ_SOURCE_PILOT.md` | `c14f2737848c5763` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `src/makelov_read_source.py` | `71d73bcef3071080` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `scripts/run_makelov_read_source.py` | `af7d9dd561fa8c21` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `scripts/verify_makelov_read_source.py` | `2158b59e2b51e20c` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `tests/test_makelov_read_source.py` | `b66b53df24826770` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `results/makelov_read_source_001/cases.json` | `32ea3f441c863f66` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `results/makelov_read_source_001/manifest.json` | `ab5702c4100f06c0` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `results/makelov_read_source_001/records.jsonl` | `5e63b3b95cfd5032` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `results/makelov_read_source_001/reference_cpu32.json` | `83b45a2075154531` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `results/makelov_read_source_001/reference_cpu64.json` | `bf88632796b6b197` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `results/makelov_read_source_001/summary.json` | `1af0cf135275a845` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `results/makelov_read_source_001/verification.json` | `0d2a87d1ebc18a92` | `7401b78` 2026-09-21T14:23:28+02:00 |
| `PREREG_READ_SOURCE_Q1.md` | `f5bb62caaea253e7` | `829228c` 2026-09-21T18:40:24+02:00 |
| `scripts/score_read_source_q1.py` | `a9a67d9d4aba55e0` | `829228c` 2026-09-21T18:40:24+02:00 |
| `results/makelov_read_source_q1/README.md` | `ddf552dbf9370597` | `6539ce5` 2026-09-21T18:41:40+02:00 |
| `results/makelov_read_source_q1/cases.json` | `386cd48b89b316b7` | `6539ce5` 2026-09-21T18:41:40+02:00 |
| `results/makelov_read_source_q1/manifest.json` | `043a2f02e99b5d22` | `6539ce5` 2026-09-21T18:41:40+02:00 |
| `results/makelov_read_source_q1/records.jsonl` | `eb4428e174b67dd6` | `6539ce5` 2026-09-21T18:41:40+02:00 |
| `results/makelov_read_source_q1/reference_cpu32.json` | `d3c401afd8d98f5f` | `6539ce5` 2026-09-21T18:41:40+02:00 |
| `results/makelov_read_source_q1/reference_cpu64.json` | `0f69d4ba84fb236a` | `6539ce5` 2026-09-21T18:41:40+02:00 |
| `results/makelov_read_source_q1/score.json` | `17b1139642730088` | `6539ce5` 2026-09-21T18:41:40+02:00 |
| `results/makelov_read_source_q1/summary.json` | `6dd5dc2610f8eecc` | `6539ce5` 2026-09-21T18:41:40+02:00 |

`results/makelov_replication_001/decide.json` was computed on 2026-09-20 with the
calculator before its bfloat16 constant was corrected (`2149078`). That correction
changes only bfloat16 floors; the replication is float32, so the file is unaffected.

The Q1 confirmation was frozen in `829228c` (preregistration and scorer) and pushed to
the private development repository before the run; the run's `manifest.json` records
`829228c` as its `git_head`, and the scorer pins the preregistration's sha256. As with
the other steps, there is no public timestamp. Its `directions.npz` is identical to the
pilot's and is rebuilt by the same script.

The read-source pilot was written on 2026-09-20 and committed to the development
repository only on 2026-09-21 (`7401b78`); until then it existed as untracked files, so its
own record (`manifest.json`, `git_head` 1fb9ad2) is the only trace of when it ran. Its
`directions.npz` (sha256 `a3d41db3…`) contains the authors' vector and is not shipped;
`scripts/rebuild_read_source_directions.py` regenerates it and checks that hash.

The files not listed (`README.md`, this file, `requirements.txt`,
`scripts/fetch_upstream.py`, `scripts/reproduce_resid_mid8.py`,
`scripts/rebuild_read_source_directions.py`, `scripts/compare_read_source_runs.py`,
`scripts/check_read_source_q1.py`,
`results/makelov_read_source_001_cpu_rerun/comparison.json`,
`results/makelov_read_source_q1/NOTES.md`, `tests/test_calculator_matches_core.py`,
`tests/test_check_read_source_q1.py`) were written for this repository.
