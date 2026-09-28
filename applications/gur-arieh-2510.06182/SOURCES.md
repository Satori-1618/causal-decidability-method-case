# Sources and upstream facts

Recorded on 28 September 2026. The machine-readable record is
[SOURCE_LOCK.json](SOURCE_LOCK.json); `scripts/lock_sources.py --check` confirms every
hash and every cited line below against a clone of the upstream repository.

## Pinned sources

| source | pin | checked |
|---|---|---|
| Paper | arXiv:2510.06182**v2** (28 May 2026), CC BY 4.0; arXiv comment: accepted to ICLR 2026 | HTML sha256 recorded |
| Code | `yoavgur/mixing-mechs` at `c53372c606e7cadf2494d2ac7b08e466042052df` (8 Aug 2026), the default branch head when retrieved | commit read from the clone |
| Code licence | MIT, "Copyright (c) 2025 yoavgur" | sha256 of `LICENSE` |
| Blog post | <https://yoav.ml/blog/2025/mixing-mechs/> | HTML sha256 recorded |
| Model | `google/gemma-2-2b-it` at revision `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` (last modified 27 Aug 2024), **not declared upstream** | from Hugging Face API metadata only |

The model repository is gated. Its revision and file sizes were readable without
authentication; the sha256 of the weight and tokenizer files were masked. They will be
recorded at gate 1, after the user accepts the Gemma licence and before any model run.
No file of the model repository was downloaded.

The upstream code is a separate read-only clone outside this repository. Nothing is
vendored.

## Task registry

`tasks/dist.py:88-103` lists 14 schema constants, but `grammar/schemas.py` at `c53372c`
defines only ten: filling liquids, music performance, people and objects, programming
dictionary, lab experiments, chemistry experiments, transportation, sports events, space
observations and boxes. These are the paper's ten tasks (App. A.1, Table 1). The other
four constants were removed upstream in commit `15c1a4c`; importing them raises
`ImportError`, so `tasks/dist.py` cannot be imported unmodified at this commit. The
adapter reads `grammar/schemas.py` only, which needs the standard library alone.
Categories, entity counts, templates and queries of the ten tasks are in the lock.

## Chat-template format

`format_prompt` (`tasks/dist.py:127-136`) applies the tokenizer's chat template to a
single user message with a generation prompt and drops the first five characters; the
string is then tokenized again with default special tokens. For Gemma the answer is read
after `Answer:\nmodel\n` (`tasks/dist.py:106-117`). That the five dropped characters are
the literal `<bos>` is an inference from the code: the rendered template was not checked,
because the tokenizer files are gated. This is deferred to gate 3.

## Upstream facts from the brief

**Patch positions — verified.** The experiment loop fixes `token_positions = [-1]`
(`tasks/dist.py:312`). The helper `run_with_cf_hf` falls back to `[-1, -4, -6, -8]` when
called without positions (`tasks/dist.py:198-200`); commit `c53372c` introduced that
default and the same list in the notebook (`example.ipynb:208`). The paper patches the
last-token residual stream vector (Sec. 3, Sec. 3.3). The patched tensor is the input of
decoder block `layer`, i.e. `hidden_states[layer]` (`tasks/dist.py:221`).

**Readout — verified, with a qualification.** Logits at the last position are read at
the ids of the n in-context entity tokens (`tasks/dist.py:328-329, 358`) and stored as raw
logits (`tasks/dist.py:393`); the prediction is their argmax (`tasks/dist.py:360`). The
paper's softmax over entity indices is applied later, when averaging (Sec. 4).
Generation is off by default on the command line (`tasks/dist.py:424`) and in the
notebook (`example.ipynb:100`); the function `get_dist` itself defaults to
`generate=True` (`tasks/dist.py:288`), which the command line overrides. Entities must be
single tokens (`tasks/dist.py:271-273`).

**Classification — verified.** Labels are assigned by an ordered if/elif chain:
positional, lexical (`keyload`), reflexive (`payload`), no effect, otherwise unknown
(`tasks/dist.py:369-378`; the notebook uses the same order, `example.ipynb:263-271`).
When two candidates predict the same entity the earlier label wins, and the argmax breaks
ties towards the lowest index. This application does not use the classification and
reports ties as equivalence.

**Index collisions — verified.** In the main template (messiness 0,
`tasks/dist.py:487-488`, used by the notebook, `example.ipynb:136`) the positional index
is `cf_query_index` (`training.py:1226`), the reflexive index `swap_index`
(`training.py:1236`) and the lexical index `swappy_index` (`training.py:1249`). Both of
the latter are drawn independently from all indices except the two query indices, so
they coincide with probability 1/(n−2). Upstream locates them afterwards by substring
matching of tokens (`tasks/dist.py:331-335`). Nothing keeps the lexical, reflexive or
native index away from the positional index's neighbours. This application checks
admissibility on design indices.

**Case filtering — verified, with a qualification.** `--do-filter` defaults to off
(`tasks/dist.py:435`), and the notebook sets `do_filter=False` (`example.ipynb:146`);
without it no case is filtered for native correctness (`training.py:376`). The function
`get_counterfactual_datasets` itself defaults to `do_filter=True` (`training.py:308`);
both entry points override it.

**Agreement conditions — verified for the paper, qualified by the blog.** App. D.1 and
Figure 18 align only two mechanisms, on the music task: positional with reflexive at
t_entity = 1, positional with lexical at t_entity = 3. The paper has no three-way
agreement experiment, and the released template cannot put the lexical or reflexive
index on the positional one. The blog post, however, shows probabilities with all three
mechanisms pointing to one entity (its Figure 1), and its interactive figure holds one
mean vector for every index combination, including coinciding ones. How those cells were
built is not documented. Round 1's agreement control stays this application's own
construction; public text must not call three-way agreement unprecedented.

**Environment — verified.** Upstream pins no versions. `CausalAbstraction/requirements.txt`
lists bare package names and contains an unresolved merge-conflict block (lines 14-18).
The model id is a command-line default without a revision (`tasks/dist.py:411`); the
command line loads `torch_dtype="auto"` (`tasks/dist.py:470`), the notebook bfloat16
(`example.ipynb:152`).

## Layer ℓ

The paper defines ℓ as the last layer before retrieval (Sec. 3.3, App. D.2) but does not
print its value for gemma-2-2b-it. The Figure 2 caption shows layer 18 and names layers
16–18 as carrying binding information; the blog post says layers 19–25 already carry the
retrieved entity. The notebook uses `layer = 18` (`example.ipynb:98`); the command-line
default is 17 (`tasks/dist.py:415`). On 28 September 2026 the user fixed the study site
by declaration at the input of decoder block 18 (`hidden_states[18]`, last token, as
`tasks/dist.py:221`). A pilot "answer copy" check between layers 18 and 19 was dropped
as a gate: in the Round 1 conflict design the completed-answer copy and the reflexive
pointer predict the same token (Round 0), so it cannot separate anything. The layer-19
patch is kept only as a labelled diagnostic.

## Openness check

**Question.** Do the paper, its appendix or the blog post already test, case by case,
whether single cases look like the per-cell average?

**Where I searched.** The full v2 HTML text, all appendices (A–G) and every table and
figure caption, with a keyword sweep for case-level terms; the blog text and captions, the
interactive figure's script and one of its data files (n = 7, t = 2); the upstream code
at `c53372c` (the repository contains no result data). Figure images were not inspected,
only their captions and the text.

**What I found.** Sec. 4 averages 150 interventions per (i_P, i_L, i_R) cell after a
softmax over entity indices and fits the mixture model to these means. The figures show
mean distributions per cell or per index, argmax labels pooled into shares per single
index (pooling over the other indices), and confusion matrices of predicted against
positional index. The blog's interactive data hold one mean vector per cell, with no
case-level values or spread.

**Result.** No case-level test of the Round 1 question was found. No STOP.

## Discrepancies between the brief and the sources

1. **Three-way agreement.** True for the paper (App. D.1), but the blog displays mean
   probabilities for cells where all three indices coincide. Handling: the agreement
   control is still built and verified by this application; no novelty claim.
2. **`tasks/dist.py` is not importable at `c53372c`.** Not in the brief. Handling: the
   adapter reads `grammar/schemas.py` only; a runner (later, after authorization) must
   not import `tasks/dist.py` unmodified.
3. **Two function defaults differ from the entry-point defaults** (`generate=True` at
   `tasks/dist.py:288`, `do_filter=True` at `training.py:308`). The brief's statements
   hold for the command line and the notebook.
4. **ℓ is not printed in the paper**, and upstream uses 18 (notebook) and 17 (command
   line). Handling: fixed by declaration at 18 (user, 28 September 2026); 19 is a
   diagnostic only.
5. **The paper states n³ = 8,000 distributions** for its Sec. 4 data, while the released
   template never places the lexical or reflexive index on the positional one. The code
   that produced coinciding cells is not in the repository. Recorded only; Round 1 does
   not depend on it.
