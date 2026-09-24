# Preregistration: a prospective test of design planning at resid_mid.8

**Frozen before any forward pass at this site.** Two freeze points:

- **Freeze A** — this document, committed before the pilot runs.
- **Freeze B** — the numeric predictions, computed from the pilot by the pinned calculator,
  committed before the confirmation runs. The confirmation script must refuse to run unless
  the Freeze B file exists with the hash recorded in its commit.

Nothing at resid_mid.8 has been computed by anyone, as far as the public record shows: the
direction `das_resid_mid.joblib` is loaded in cell 3 of `ioi_analysis.ipynb` in
amakelov/activation-patching-illusion and used in no computation; it has zero mentions in
arXiv:2311.17030v2, no row in `patching_metrics_ioi.joblib`, and no saved notebook output.

---

## 1. What is being tested

**The planning claim, not a mechanism.** Before running, the procedure predicts whether a
given design — a readout at a sample size — can distinguish two declared endpoint rivals.
This preregistration tests whether those predictions hold on fresh data at a site whose
outcome is unknown.

Which rival turns out to be right is recorded as a secondary result. **It is not used to
judge the method.** A correct prediction that a design cannot decide is a success.

## 2. Site, intervention, model

- Model: GPT-2-small, float32, CPU. Weights as in `results/makelov_replication_001/`.
- Site: the residual stream at the middle of layer 8 (`resid_mid`, layer 8), last prompt
  position — after attention in layer 8, before MLP 8.
- Direction: `v = das_resid_mid.joblib` from the upstream repository, pinned by sha256 in
  Freeze B.
- Decomposition, as the paper does for `resid_post.8`: `v_row` is the projection of `v` onto
  the row space of the concatenated query matrices of the name-mover heads (9.6), (9.9),
  (10.0), obtained by QR; `v_null = v − v_row`.
- Patch: the one-dimensional interchange patch already validated in the replication —
  replace the component of the base activation along a direction with the source's
  component along the same direction.
- Conditions per pair: clean, `full` (along `v`), `row` (along `v_row`), `null` (along
  `v_null`).

Unlike `v_MLP`'s null component at MLP 8, `v_null` here is **not** in the kernel of anything
downstream: it is only orthogonal to the three name-mover queries. Its effect is therefore
not zero by construction, which is part of why the outcome is open.

## 3. Rivals, as endpoints

For each component `X ∈ {row, null}`, two declared endpoint rivals:

| rival | prediction for the effect of patching `X` |
|---|---|
| **inert** | `E(X) = 0` |
| **carries all** | `E(X) = E(full)` |

where `E(·)` is the effect of a patch relative to clean, per pair. The substantive readings
map onto the outcomes: the row component carries the patch (row = all, null = inert), the
null component carries it (the reverse), or both are needed (at least one component
excludes both endpoints).

These are the only rivals declared. Explanations not written here are neither tested nor
excluded.

## 4. Readouts — both declared now

- **L — logit difference**, IO minus S at the final position. Effect of a patch: clean
  logit difference minus patched logit difference, per pair.
- **A — interchange accuracy**: indicator that the patched prediction flips to the patched
  answer. Effect of a patch: the indicator itself.

Neither readout may be dropped or added after the pilot.

## 5. Statistical unit and contrasts

Unit: one base/source pair. All contrasts are paired within a pair. For each component `X`
and readout `r`, two paired contrasts are computed per pair:

- `d_inert = E(X)` — tests the **inert** endpoint
- `d_all   = E(full) − E(X)` — tests the **carries all** endpoint

## 6. Data and seeds

- Distribution: the upstream **test** distribution — names, objects and places split 1:1,
  third template held out — patterns ABB→BAB and BAB→ABB in equal numbers.
- `PYTHONHASHSEED=0` is fixed for every run. Upstream `data_utils.py:317` uses
  `list(set(pattern))`, whose order otherwise varies between processes and changes the
  dataset. This was measured, not assumed.
- **Pilot:** seed 7001, 200 pairs (100 + 100).
- **Confirmation:** seed 7013, 2000 pairs (1000 + 1000), in the fixed order generated.
  Evaluated on **nested prefixes** of size n ∈ {20, 50, 100, 250, 500, 1000, 2000}.
- Neither seed has been used before in this project; the replication used the notebook's
  own seeding. No confirmation pair may be generated before Freeze B.

## 7. The pilot is blinded to the answer

The pilot script outputs **only**:

- the mean of `E(full)` per readout — the size of the effect to be decomposed;
- the standard deviation of each paired contrast `d_inert`, `d_all` per component and
  readout — location-invariant, so it does not reveal where `E(X)` lies.

It must **not** write, print or log the means of `E(row)` or `E(null)`. Those are the answer.
The pilot script is committed and hashed before it runs.

## 8. The prediction, computed at Freeze B

For each component × readout × n — 2 × 2 × 7 = **28 predictions** — the pinned calculator is
called as

```
decidability(predictions = {inert: 0, all: mean E(full) from the pilot},
             n = n, sigma = max(sd d_inert, sd d_all) from the pilot,
             dtype = 'float32', noise_factor = 1.0,
             alpha = 0.01, signatures = 2,
             readout_scale = 20, depth = 768   for L
             readout_scale = 1,  depth = 1     for A)
```

**Pinned, not defaulted.** The calculator's default is `signatures = 3`; here there are two
distinct predictions per contrast, so `signatures = 2`, giving a simultaneous quantile
z = 2.8070 at α = 0.01. §9 uses **the same z**. `noise_factor = 1.0` because `sigma` is
already the SD of a paired per-pair contrast. Calculator: `src/decidability.py`, sha256
`ab0fd838a9a6b354d72d83c8b9730b095c0a443a4eeae756f2d1491eb17365f8` (with the corrected
bfloat16 constant; irrelevant in float32 but pinned). The larger of the two SDs is used, to be
conservative. In float32 the statistical floor binds; **the numerical floor is not tested
here** (see §11).

Output per cell: predicted ratio and verdict (decidable ⟺ ratio > 1). Additionally, for each
component × readout, the **predicted crossover** — the smallest n in the ladder predicted
decidable.

## 9. What "decided" means after the run

For each component × readout × n, on the confirmation prefix of size n, with intervals
mean ± z · sd / √n at the same z = 2.8070:

- **inert** is compatible iff the interval for `d_inert` contains 0;
- **carries all** is compatible iff the interval for `d_all` contains 0;
- **realized decided** ⟺ *not both* endpoints are compatible.

"Neither compatible" counts as decided: it excludes both endpoints and points to a
component that is neither inert nor sufficient.

## 10. Success and failure, declared now

**Primary outcome:** the fraction of the 28 predicted verdicts that match the realized ones.

- **Supported:** at least 80 % match, **and** every prediction whose predicted ratio lies
  outside [0.5, 2] matches.
- **Contradicted:** any prediction with a predicted ratio outside [0.5, 2] fails.
- **Inconclusive:** otherwise.

The band [0.5, 2] is set from the synthetic benchmark, where the transition in decidability
was soft around a ratio of 1. Mismatches inside the band are reported and expected; mismatches
outside it are what the claim forbids.

**Also reported, not judged:** the observed crossover against the predicted one per component
and readout; and the secondary result — which endpoints each component excludes at n = 2000.

## 11. What this does not test

- **The numerical floor.** The run is float32, so the statistical floor binds. The synthetic
  benchmark showed the numerical formula over-predicting measured error by a median factor of
  about 27 in bfloat16; that is a separate question for a separate, separately frozen test.
- **Generality.** One model, one site, one paper. A success is one prospective confirmation,
  not a validation across designs.
- **Rivals not declared here.**

## 12. Prior knowledge, disclosed

Priors are permitted; hidden outcomes are not. Known before this document: the published
Table 1 of arXiv:2311.17030 (resid_post.8: the name-mover row space carries most of the
patch; MLP 8: both components needed), and this project's exact replication of the three MLP8
rows with per-example values. Nothing at resid_mid.8 is known.

## 13. Deviations

Any deviation from this document is recorded in a dated section appended below, with its
reason, **before** the confirmation outcome is read. A deviation discovered after reading it
makes the affected predictions exploratory, and they are reported as such.

### 2026-09-21 — clarifications and one deviation, before any forward pass at the site

Recorded before the pilot runs; no per-pair or aggregate value at `resid_mid.8` has been
computed. Items 1–5 fix choices the text above leaves open; item 6 is a deviation; item 7
discloses a limitation of §7.

1. **Effect on readout A is clean-relative.** `E_A(X) = 1{patched argmax = patched answer}
   − 1{clean argmax = patched answer}`, per pair. §3 defines every `E(·)` relative to clean;
   §4's "the indicator itself" is read accordingly, so that a patch that changes nothing has
   `E = 0` exactly, as the inert rival requires. `d_all` is unaffected by this choice.
2. **Symbol order.** Upstream's `list(set(pattern))` under `PYTHONHASHSEED=0` gives
   `['B', 'A']` for both `'ABB'` and `'BAB'` (hash slots 1 and 6 of 8, no collision). The
   pilot and the confirmation use that order and check it at runtime. The replication used
   `'AB'`, the order that reproduced the notebook's own dataset.
3. **Data construction.** The replication's `build_patching_dataset`, including its discarded
   first draw of one combination's worth of pairs: seed 7001 with 100 per combination for the
   pilot, seed 7013 with 1000 per combination for the confirmation.
4. **Standard deviation.** Sample SD with ddof = 1, computed in float64 over the per-pair
   values; the same convention in §8 and §9.
5. **Direction pinned now.** `das_resid_mid.joblib` at the vendored revision, sha256
   `4e69edf9361c3216d9e6bb753d75de440128858757f93028a2293db0488124e3`, 3297 bytes, read as a
   raw float32 payload and never unpickled. §2 deferred the pin to Freeze B; pinning it
   before the pilot is stricter.
6. **Deviation: confirmation order is interleaved.** The builder returns the 1000 ABB→BAB
   pairs first, then the 1000 BAB→ABB pairs. Taken "in the fixed order generated" (§6), every
   prefix up to n = 1000 would contain one half only, so six of the seven prefixes would not
   be draws from the declared 1:1 distribution the pilot estimates. The confirmation
   therefore evaluates the generated set in the order ABB→BAB[0], BAB→ABB[0], ABB→BAB[1], …,
   so every prefix in the ladder holds equal numbers of both. The pairs themselves are
   unchanged. Found on reading the builder, before the pilot.
7. **Limitation of the blinding for readout A.** §7 calls the SD location-invariant. For L it
   is. For A the contrasts take values in {−1, 0, 1}, and the SD of such a variable is not
   location-invariant: with clean interchange near zero, `sd d_inert ≈ √(p(1−p))` for the
   pilot rate p of `E_A(X) = 1`, so the pilot reveals that rate up to p ↔ 1 − p. Because §8
   fixes every input to the calculator and item 6 settles the one open design choice before
   the pilot, this knowledge leaves nothing to adjust. The pilot is blinded for L and only
   nominally for A, and is reported that way.
