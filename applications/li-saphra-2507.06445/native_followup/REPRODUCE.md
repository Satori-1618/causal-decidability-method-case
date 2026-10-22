# Reproduction map

Run commands from this directory. The recorded execution used Python 3.10.10,
PyTorch 2.9.0, CPU, one thread, deterministic algorithms. Model execution requires
PyTorch; plotting additionally requires matplotlib. Existing development scripts
also use numpy. The frozen analyzers and raw-artifact verifier use the standard
library. Higher precision is a reference, not exact arithmetic.

## Inspect and independently verify without inference

```bash
python result_bundle.py unpack
python analyze_continued.py \
  --freeze frozen/confirmation_001/freeze.json \
  --run results/confirmation_001_continued \
  --output /tmp/dyck-confirmation-recomputed.json
python verify_confirmation.py --help
```

The committed [verification report](results/verification_report.json) records
the separate verification implementation and hashes. See [VERIFICATION.md](VERIFICATION.md)
for the exact verification command. The archive is a lossless packaging of the
raw continuation tree. It retains forecast files saved before target arms,
per-input margins, controls and manifests; [its index](artifacts/confirmation_001.index.json)
contains a hash for every file. The initial failed run is retained separately.

## Re-run the model work

Fetch assets at the pinned public repository revision; both commands verify
existing caches against committed hashes:

```bash
python fetch_assets.py
python fetch_cohort.py
python -m unittest discover -s tests -v
```

The original strict runner is intentionally unchanged:

```bash
python confirm.py \
  --freeze frozen/confirmation_001/freeze.json \
  --output results/my_original_run
```

It requires frozen files to equal committed HEAD versions, refuses overwrite,
and may stop at the same strict ordinal control. The committed execution did
stop there. To reproduce the documented continuation, with the same source run
and preserved failure:

```bash
python continue_confirmation.py \
  --freeze frozen/confirmation_001/freeze.json \
  --source-run results/confirmation_001 \
  --amendment CONTINUATION.md \
  --output results/my_continued_run
```

The continuation retains the first completed task and failed task from its source
run, then measures the other predefined tasks. It preserves every failure and
does not relax a control. Numerical results can vary across environments; compare
the scientific decisions and gates, rather than promising byte-identical neural
computation on another machine. Original elapsed execution: 17.99 seconds before
the stop, then 492.96 seconds for the continuation; implementation, review and
analysis time are additional.

## Provenance and scope

- Original retrospective application: `../SOURCE_LOCK.json`, unchanged.
- Development and missing model dependency: `ASSET_LOCK.json`.
- Hypotheses and decision rules: [CONFIRMATION.md](CONFIRMATION.md), local freeze
  commit `92860b6`. This was not an externally timestamped preregistration.
- Fixed cases, tasks, weights and tolerances: `frozen/confirmation_001/`.
- Failure-accounting amendment: commits `d50b14a`, `0224056`.
- Frozen decision function: `analyze_confirmation.py`; continuation reporting
  calls it unchanged and adds explicit invalid-task rows and descriptive interactions.
- Independent verification: `verify_confirmation.py` imports neither analyzer
  nor runner. It checks saved results and control records, not every unsaved
  intermediate tensor independently.

Development checkpoint selection was retrospective. Transfer heads were selected
by the published head-type definition before new interventions; all realized
transfer models are two-layer networks. The shared training seeds are not iid
replications. Fresh inputs have equal counts and length 32, with cyclic orbits
disjoint from published evaluation inputs; original-training overlap is unknown.
