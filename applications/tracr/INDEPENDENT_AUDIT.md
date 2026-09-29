# Read-only confirmation audit

Status: PASS. Performed by an independent review agent after confirmation; no
additional model calls and no frozen source edits. This audit additionally checked
raw tensors directly without relying on the producer's verifier or classifier.

- All 11 executable/protocol/dependency files bound by the freeze match current
  files and freeze commit `c64c3af`.
- Committed freeze bytes match the confirmation manifest's SHA256 binding.
- Recorded order: freeze commit 15:26:53 UTC, execution checkout `5c2b1a3` at
  15:27:06 UTC, confirmation started 15:27:08.900885 UTC on 2026-09-29. Hashes and
  local history provide an auditable record, not an independent timestamp authority.
- All 768 dtype-specific records were checked for donor source, native pre-site
  tensor, exact inserted values, unchanged complement, no-op/self-patch all-state
  and score identity, and unchanged non-target outputs.
- All 128 families have the complete three-stage trajectory in both precisions.
- Maximum absolute address-endpoint coordinate error was approximately
  `1.5722e-6`, below the frozen radius `0.01000762939453125`. On separating and
  followup conditions, maximum-coordinate distance from donor-answer copying was
  `1.0`.
- Independent exact-rational evaluation of `comb(K,128)/comb(M,128)` at the
  all-success boundary gave the same first non-rejected population count:
  `K=116980279`, `M=119750392`, lower success fraction `0.9768676080826525`.
  The null tail was approximately `0.001408055359755`; the SciPy value differs
  by about `2.2e-12`, immaterial to the frozen 0.05 decision.

The result supports this controlled known-circuit validation. It does not establish
discovery of unknown native representations, transfer beyond the reversal program,
or superiority of the tied-menu selector over a strong fixed design.
