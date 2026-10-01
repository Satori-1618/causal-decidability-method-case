"""Round 1 model runner (protocol v2): prompts, last-token residual patching, readouts.

Requires torch and transformers (pinned in requirements-model.txt); the analyzer and the
records-only checker do not import this module. Upstream ``tasks/dist.py`` is never
imported (it does not import at c53372c); prompts come from ``mixing_prompts`` through
the adapter, and the patch reproduces upstream's ``run_with_cf_hf`` for one position:

- the donor's residual stream entering decoder block ``layer`` is Hugging Face
  ``hidden_states[layer]`` of the donor prompt (``tasks/dist.py:221``), taken at the
  donor's last token;
- a forward-pre hook on block ``layer`` writes it into the recipient's residual stream
  at the recipient's last prompt token (absolute position T - 1), whenever the block
  sees at least T positions (``tasks/dist.py:239-246``; incremental decoding steps see
  one position and are left alone).

Readouts at the last position:

- **Primary (answer form).** The logits of the n in-context entities in the form the
  model answers with (capitalised, no leading space, exactly one token), and the
  answer-token mass: the full-vocabulary probability on those n tokens. Recorded as
  ``answer_logits`` and ``answer_mass_full_vocab``.
- **Descriptive only (the paper's readout).** The logits of the n in-context entity
  tokens as they occur in the context (' country', ``tasks/dist.py:358``) and their mass,
  kept under ``descriptive.paper_readout``; nothing that decides reads it.

Pools (protocol v2): every entity is one token with a leading space; entities of the
answered category are also one token in the answer form, and both forms' ids are unique
across those entities. Greedy generation under every patch is recorded as a diagnostic.
"""
import hashlib
import math
import time
from pathlib import Path

import torch

from mixing_prompts import chat_prompt, raw_prompt
from mixing_round1_design import TARGET, QUERY, agreement_control, design_indices, target_rebind

PUNCTUATION = ".,;:!?\"'()[]*`"
REQUIRED_SNAPSHOT_FILES = (
    "config.json", "generation_config.json", "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors", "model.safetensors.index.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
)


class SnapshotIncomplete(RuntimeError):
    """The local snapshot lacks a required file; the run must STOP, never download."""


class TechnicalError(RuntimeError):
    """A technical failure (alignment, hook, shape, non-finite readout): INVALID."""


# ---- model files --------------------------------------------------------------------

def snapshot_path(repo_id, revision, cache=None):
    cache = Path(cache or Path.home() / ".cache/huggingface/hub")
    path = cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
    missing = [name for name in REQUIRED_SNAPSHOT_FILES if not (path / name).is_file()]
    if missing:
        raise SnapshotIncomplete(f"snapshot {path} lacks {', '.join(missing)}")
    return path


def _digests(path, chunk=1 << 24):
    sha256, size = hashlib.sha256(), 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            sha256.update(block)
            size += len(block)
    git = hashlib.sha1(b"blob %d\0" % size)
    if size < (1 << 26):
        git.update(Path(path).read_bytes())
        git_oid = git.hexdigest()
    else:
        git_oid = None
    return sha256.hexdigest(), size, git_oid


def snapshot_hashes(path, locked_files):
    """sha256 of every snapshot file, matched to the cache's content address and to the
    sizes and git blob ids recorded from the hub API (SOURCE_LOCK.json).

    The Hugging Face cache stores LFS files under their sha256 and other files under
    their git blob id; both must agree with a fresh digest of the file.
    """
    locked = {entry["path"]: entry for entry in locked_files}
    files, problems = {}, []
    for name in sorted(p.name for p in Path(path).iterdir() if p.is_file()):
        target = (Path(path) / name).resolve()
        sha256, size, git_oid = _digests(target)
        entry = locked.get(name)
        record = {"sha256": sha256, "size": size, "cache_blob": target.name}
        if entry is None:
            problems.append(f"{name}: not in the locked file list")
        else:
            if entry["size"] != size:
                problems.append(f"{name}: size {size} differs from locked {entry['size']}")
            if entry.get("lfs"):
                record["content_address_matches"] = target.name == sha256
            else:
                record["git_oid"] = git_oid
                record["content_address_matches"] = target.name == git_oid == entry["git_oid"]
            if not record["content_address_matches"]:
                problems.append(f"{name}: content address does not match the digest")
        files[name] = record
    missing = [name for name in REQUIRED_SNAPSHOT_FILES if name not in files]
    problems += [f"{name}: missing" for name in missing]
    return {"files": files, "passed": not problems, "problems": problems}


def load_model(path, dtype, device, attention="eager"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=dtype,
                                                 attn_implementation=attention)
    return model.to(device).eval(), tokenizer


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---- tokens and pools ---------------------------------------------------------------

def answer_form(entity):
    """The form the model answers with: first letter capitalised, no leading space."""
    return entity[:1].upper() + entity[1:]


def one_token(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def round1_pools(tokenizer, spec, target=TARGET):
    """Protocol v2 pools, applied mechanically in upstream order.

    - Every entity: its context form (' entity') is exactly one token.
    - Entities of the answered category: also their answer form ('Entity') is exactly
      one token, and the context-form ids and the answer-form ids are each unique across
      those entities (every entity in a collision is dropped).

    Returns (pools, dropped with reasons, context-form ids {(category, entity): id},
    answer-form ids {entity: id} of the answered category)."""
    target_category = spec["categories"][target]
    pools, dropped, context_ids, answer_ids = {}, {}, {}, {}
    for category in spec["categories"]:
        kept = []
        for entity in spec["items"][category]:
            context = one_token(tokenizer, " " + entity)
            if context is None:
                dropped.setdefault(category, []).append({"entity": entity, "reason": "context form not one token"})
                continue
            if category == target_category:
                answer = one_token(tokenizer, answer_form(entity))
                if answer is None:
                    dropped.setdefault(category, []).append({"entity": entity, "reason": "answer form not one token"})
                    continue
                answer_ids[entity] = answer
            context_ids[(category, entity)] = context
            kept.append(entity)
        pools[category] = kept
    kept = pools[target_category]
    context_counts, answer_counts = {}, {}
    for entity in kept:
        context_counts[context_ids[(target_category, entity)]] = context_counts.get(context_ids[(target_category, entity)], 0) + 1
        answer_counts[answer_ids[entity]] = answer_counts.get(answer_ids[entity], 0) + 1
    colliding = [e for e in kept if context_counts[context_ids[(target_category, e)]] > 1
                 or answer_counts[answer_ids[e]] > 1]
    for entity in colliding:
        dropped.setdefault(target_category, []).append({"entity": entity, "reason": "token id not unique"})
        kept.remove(entity)
        del answer_ids[entity]
        del context_ids[(target_category, entity)]
    return pools, dropped, context_ids, answer_ids


def pools_record(pools, dropped, context_ids, answer_ids, spec, target=TARGET):
    """The pools and token ids as plain data (for the lock and the manifest)."""
    target_category = spec["categories"][target]
    return {
        "rule": ("every entity one token with a leading space; entities of the answered category "
                 "also one token in the answer form (capitalised, no leading space), with context-form "
                 "and answer-form ids each unique across them"),
        "answered_category": target_category,
        "pools": pools, "dropped": dropped,
        "context_form_ids": {c: {e: context_ids[(c, e)] for e in pools[c]} for c in spec["categories"]},
        "answer_form_ids": {e: answer_ids[e] for e in pools[target_category]},
    }


def first_word(text):
    words = text.strip().split()
    return words[0].strip(PUNCTUATION).lower() if words else ""


# ---- patching -----------------------------------------------------------------------

class LastTokenPatch:
    """Forward-pre hook writing one vector at absolute position ``length - 1``.

    Counts every call and every write, and records the input shapes, so that the hook
    gate (exactly one write per patched forward, at the last prompt position) can be
    checked per run.
    """

    def __init__(self, block, vector, length):
        self.block, self.vector, self.length = block, vector.detach(), int(length)
        self.calls, self.writes, self.shapes, self.positions = 0, 0, [], []
        self.handle = None

    def __call__(self, module, args, kwargs):
        from_kwargs = not args
        hidden = kwargs["hidden_states"] if from_kwargs else args[0]
        self.calls += 1
        self.shapes.append(list(hidden.shape))
        if hidden.dim() != 3 or hidden.shape[1] < self.length:
            return None
        patched = hidden.clone()
        position = self.length - 1
        patched[:, position, :] = self.vector.to(device=patched.device, dtype=patched.dtype)
        self.writes += 1
        self.positions.append(position)
        if from_kwargs:
            return args, {**kwargs, "hidden_states": patched}
        return (patched, *args[1:]), kwargs

    def __enter__(self):
        self.handle = self.block.register_forward_pre_hook(self, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        return False

    def report(self):
        return {"calls": self.calls, "writes": self.writes, "shapes": self.shapes,
                "positions": self.positions, "vector_shape": list(self.vector.shape)}


def blocks(model):
    return model.model.layers


# ---- runner -------------------------------------------------------------------------

class Runner:
    """One model, tokenizer, task spec, frozen pools and token ids, and study layer."""

    def __init__(self, model, tokenizer, spec, pools, context_ids, answer_ids, layer, *,
                 diagnostic_layer=None, query=QUERY, target=TARGET, max_new_tokens=None):
        self.model, self.tokenizer, self.spec = model, tokenizer, spec
        self.pools = pools
        self.layer, self.diagnostic_layer = layer, diagnostic_layer
        self.categories = spec["categories"]
        self.target_category = self.categories[target]
        self.query_categories = [self.categories[i] for i in query]
        self.target = target
        self.max_new_tokens = max_new_tokens or spec["max_new_tokens"]
        self.device = next(model.parameters()).device
        self.vocab_size = model.get_output_embeddings().weight.shape[0]
        self.context_ids = {e: context_ids[(self.target_category, e)] for e in pools[self.target_category]}
        self.answer_ids = {e: answer_ids[e] for e in pools[self.target_category]}

    # -- prompts
    def prompt(self, matrix, query_group):
        raw = raw_prompt(self.spec, matrix, query_group, self.query_categories, self.target_category)
        chat, dropped = chat_prompt(self.tokenizer, raw)
        ids = self.tokenizer(chat, return_tensors="pt")["input_ids"].to(self.device)
        return {"raw": raw, "chat": chat, "dropped": dropped, "input_ids": ids}

    def aligned_entity_ids(self, matrix, input_ids):
        """Context-form ids of the n target entities in group order, checked against the
        prompt: the target pool's context tokens occur exactly once each, in group order.
        Located by design, verified on tokens; a mismatch is a technical failure."""
        expected = [self.context_ids[g[self.target]] for g in matrix]
        pool = set(self.context_ids.values())
        sequence = input_ids[0].tolist()
        found = [(i, t) for i, t in enumerate(sequence) if t in pool]
        if [t for _, t in found] != expected:
            raise TechnicalError("target-entity tokens in the prompt differ from the design")
        return expected, [i for i, _ in found]

    # -- forwards
    @torch.no_grad()
    def forward(self, input_ids, patch=None):
        """One forward pass; ``patch`` is (layer, vector) or None."""
        if patch is None:
            out = self.model(input_ids, output_hidden_states=True, use_cache=False)
            return out, None
        layer, vector = patch
        with LastTokenPatch(blocks(self.model)[layer], vector, input_ids.shape[1]) as hook:
            out = self.model(input_ids, output_hidden_states=True, use_cache=False)
        return out, hook.report()

    @torch.no_grad()
    def generate(self, input_ids, patch=None):
        kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False,
                      attention_mask=torch.ones_like(input_ids),
                      pad_token_id=self.tokenizer.pad_token_id)
        if patch is None:
            ids = self.model.generate(input_ids, **kwargs)
            report = None
        else:
            layer, vector = patch
            with LastTokenPatch(blocks(self.model)[layer], vector, input_ids.shape[1]) as hook:
                ids = self.model.generate(input_ids, **kwargs)
            report = hook.report()
        new = ids[0, input_ids.shape[1]:].tolist()
        return {"token_ids": new,
                "text": self.tokenizer.decode(new, skip_special_tokens=True)}, report

    def readout(self, out, entities, full=False):
        """Primary answer-form readout and the descriptive paper readout for the n
        entities (in group order) of the prompt that was run."""
        logits = out.logits[0, -1].float()
        if logits.shape[0] != self.vocab_size:
            raise TechnicalError("logit vector length differs from the vocabulary")
        lse_full = torch.logsumexp(logits, 0)

        def scored(ids):
            index = torch.tensor(ids, device=logits.device)
            values = logits[index]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[index] = False
            lse = torch.logsumexp(values, 0)
            return values, lse, torch.logsumexp(logits[mask], 0)

        answer, lse_answer, lse_answer_complement = scored([self.answer_ids[e] for e in entities])
        paper, lse_paper, lse_paper_complement = scored([self.context_ids[e] for e in entities])
        top = int(torch.argmax(logits))
        result = {
            "answer_logits": [float(x) for x in answer.tolist()],
            "answer_mass_full_vocab": float(torch.exp(lse_answer - lse_full)),
            "logsumexp_full": float(lse_full),
            "logsumexp_answer_tokens": float(lse_answer),
            "logsumexp_answer_complement": float(lse_answer_complement),
            "top_token_id": top,
            "top_token": self.tokenizer.convert_ids_to_tokens(top),
            "top_token_logit": float(logits[top]),
            "descriptive": {"paper_readout": {
                "entity_logits": [float(x) for x in paper.tolist()],
                "entity_mass_full_vocab": float(torch.exp(lse_paper - lse_full)),
                "logsumexp_complement": float(lse_paper_complement)}},
        }
        primary = result["answer_logits"] + [result["answer_mass_full_vocab"], result["logsumexp_full"]]
        if not all(math.isfinite(v) for v in primary):
            raise TechnicalError("non-finite primary readout")
        if full:
            result["_full_logits"] = logits.detach().cpu()
        return result

    @staticmethod
    def check_hook(report, length, hidden_size):
        """Gate 4 per patched forward: one call, one write, at the last prompt position,
        on a [1, T, D] input and a [D] vector."""
        return (report["calls"] == 1 and report["writes"] == 1
                and report["positions"] == [length - 1]
                and report["shapes"] == [[1, length, hidden_size]]
                and report["vector_shape"] == [hidden_size])

    # -- one unpatched run of a prompt, with hidden states and greedy generation
    def native(self, matrix, query_group, answer, generate=True):
        prompt = self.prompt(matrix, query_group)
        _, positions = self.aligned_entity_ids(matrix, prompt["input_ids"])
        out, _ = self.forward(prompt["input_ids"])
        entities = [g[self.target] for g in matrix]
        readout = self.readout(out, entities)
        argmax = max(range(len(entities)), key=lambda i: readout["answer_logits"][i])
        paper = readout["descriptive"]["paper_readout"]["entity_logits"]
        record = {"raw_prompt": prompt["raw"], "dropped_prefix": prompt["dropped"],
                  "prompt_tokens": int(prompt["input_ids"].shape[1]),
                  "entity_positions": positions, "entities": entities, "answer": answer,
                  "readout": readout, "readout_argmax": argmax,
                  "readout_argmax_entity": entities[argmax],
                  "descriptive_paper_argmax_entity": entities[max(range(len(entities)), key=paper.__getitem__)]}
        if generate:
            generation, _ = self.generate(prompt["input_ids"])
            word = first_word(generation["text"])
            first_token = generation["token_ids"][0] if generation["token_ids"] else None
            first_token_ok = first_token == self.answer_ids[answer]
            record.update(generation=generation["text"], generation_ids=generation["token_ids"],
                          first_word=word, first_token_is_answer_form=first_token_ok,
                          correct=word == answer.lower() and first_token_ok,
                          readout_matches_generation=word == entities[argmax].lower())
        return prompt, record, out

    def family(self, *, case_id, draw_index, seed, cell_key, cell, rng, n, audit=False,
               identity=True):
        """One base context: conflict donor, three agreement donors, the no-patch run,
        the conflict patch at the study layer (and the diagnostic layer), the agreement
        patches and the identity self-patch. Returns one record in the checker's schema
        plus the measurements the pilot reports."""
        from mixing_round1_design import random_matrix
        start = time.perf_counter()
        hidden_size = self.model.config.hidden_size
        G = random_matrix(n, [self.pools[c] for c in self.categories], rng)
        conflict = target_rebind(G, cell, n)
        technical = {"checks": {}, "failures": []}

        def check(name, ok, detail=None):
            technical["checks"][name] = bool(ok)
            if not ok:
                technical["failures"].append(f"{name}: {detail}" if detail else name)

        indices = design_indices(**conflict)
        check("conflict_design_indices", indices == cell, indices)
        i_N = cell["i_N"]
        agreement_cases = []
        for key in ("i_P", "i_L", "i_R"):
            j = cell[key]
            case = agreement_control(G, j, i_N, rng)
            a_indices = design_indices(**case)
            check(f"agreement_{key}_design_indices",
                  (a_indices["i_P"], a_indices["i_L"], a_indices["i_R"], a_indices["i_N"]) == (j, j, j, i_N),
                  a_indices)
            agreement_cases.append((key, j, case, a_indices))

        recipient_answer = G[i_N][self.target]
        rec_prompt, recipient, rec_out = self.native(G, i_N, recipient_answer)
        L = rec_prompt["input_ids"].shape[1]
        donor_answer = conflict["donor"][cell["i_P"]][self.target]
        don_prompt, donor, don_out = self.native(conflict["donor"], cell["i_P"], donor_answer)
        check("dropped_prefix_is_bos", rec_prompt["dropped"] == self.tokenizer.bos_token
              and don_prompt["dropped"] == self.tokenizer.bos_token, rec_prompt["dropped"])
        check("single_bos", int((rec_prompt["input_ids"][0] == self.tokenizer.bos_token_id).sum()) == 1
              and int(rec_prompt["input_ids"][0, 0]) == self.tokenizer.bos_token_id)
        recipient_entities = [g[self.target] for g in G]
        lowered = [e.lower() for e in recipient_entities]

        def patched(layer, hidden, full=False):
            vector = hidden[layer][0, -1]
            out, report = self.forward(rec_prompt["input_ids"], patch=(layer, vector))
            hook_ok = self.check_hook(report, L, hidden_size)
            readout = self.readout(out, recipient_entities, full=full)
            generation, gen_report = self.generate(rec_prompt["input_ids"], patch=(layer, vector))
            word = first_word(generation["text"])
            readout["patched_generation"] = {
                "text": generation["text"], "token_ids": generation["token_ids"], "first_word": word,
                "entity_index": lowered.index(word) if word in lowered else None,
                "hook_writes": gen_report["writes"], "hook_positions": gen_report["positions"]}
            hook_ok = hook_ok and gen_report["writes"] == 1 and gen_report["positions"] == [L - 1]
            return readout, report, hook_ok

        conflict_readout, conflict_hook, ok = patched(self.layer, don_out.hidden_states, full=audit)
        check("conflict_hook", ok, conflict_hook)
        full_logits = conflict_readout.pop("_full_logits", None)
        diagnostic = None
        if self.diagnostic_layer is not None:
            d_readout, d_hook, d_ok = patched(self.diagnostic_layer, don_out.hidden_states)
            check("diagnostic_hook", d_ok, d_hook)
            diagnostic = {"layer": self.diagnostic_layer, "readout": d_readout, "hook": d_hook}

        agreement = []
        for key, j, case, a_indices in agreement_cases:
            a_answer = case["donor"][j][self.target]
            _, a_donor, a_out = self.native(case["donor"], j, a_answer)
            a_readout, a_hook, a_ok = patched(self.layer, a_out.hidden_states)
            check(f"agreement_{key}_hook", a_ok, a_hook)
            logits = a_readout["answer_logits"]
            argmax = max(range(n), key=logits.__getitem__)
            agreement.append({"target": key, "j": j, "design_indices": a_indices,
                              "donor_native": a_donor, "readout": a_readout, "hook": a_hook,
                              "answer_logits": logits,
                              "answer_mass_full_vocab": a_readout["answer_mass_full_vocab"],
                              "argmax": argmax, "transfer": argmax == j})

        identity_record = None
        if identity:
            vector = rec_out.hidden_states[self.layer][0, -1]
            out_id, id_hook = self.forward(rec_prompt["input_ids"], patch=(self.layer, vector))
            gen_id, gen_hook = self.generate(rec_prompt["input_ids"], patch=(self.layer, vector))
            base_logits, id_logits = rec_out.logits[0, -1].float(), out_id.logits[0, -1].float()
            index = torch.tensor([self.answer_ids[e] for e in recipient_entities], device=base_logits.device)
            identity_record = {
                "max_abs_answer_logit_difference": float((id_logits[index] - base_logits[index]).abs().max()),
                "max_abs_logit_difference": float((id_logits - base_logits).abs().max()),
                "same_answer_argmax": int(torch.argmax(id_logits[index])) == recipient["readout_argmax"],
                "same_generation": gen_id["token_ids"] == recipient["generation_ids"],
                "generation": gen_id["text"],
                "hook": id_hook, "generation_hook": {k: gen_hook[k] for k in ("calls", "writes", "positions")},
            }
            check("identity_hook", self.check_hook(id_hook, L, hidden_size), id_hook)
            check("identity_generation_hook", gen_hook["writes"] == 1 and gen_hook["positions"] == [L - 1],
                  gen_hook)

        qualifies = bool(recipient["correct"] and donor["correct"]
                         and recipient["readout_matches_generation"]
                         and donor["readout_matches_generation"])
        technical["passed"] = not technical["failures"]
        record = {
            "schema": "round1 record v2 (answer-form primary readout)",
            "case_id": case_id, "draw_index": draw_index, "seed": seed, "cell_key": cell_key,
            "cell": dict(cell), "qualifies": qualifies,
            "design_indices": indices, "technical": technical,
            "answer_logits": conflict_readout["answer_logits"],
            "answer_mass_full_vocab": conflict_readout["answer_mass_full_vocab"],
            "descriptive": {"paper_readout": conflict_readout["descriptive"]["paper_readout"]},
            "readout": conflict_readout, "hook": conflict_hook,
            "matrix": [list(g) for g in G],
            "donor_matrix": [list(g) for g in conflict["donor"]],
            "donor_answer": {"entity": donor_answer, "in_recipient": donor_answer in recipient_entities,
                             "recipient_index": (recipient_entities.index(donor_answer)
                                                 if donor_answer in recipient_entities else None)},
            "native": {"recipient": recipient, "donor": donor},
            "nopatch": {"answer_logits": recipient["readout"]["answer_logits"],
                        "answer_mass_full_vocab": recipient["readout"]["answer_mass_full_vocab"],
                        "descriptive": recipient["readout"]["descriptive"]},
            "agreement": agreement,
            "identity": identity_record,
            "diagnostic": diagnostic,
            "runtime_seconds": time.perf_counter() - start,
        }
        return record, full_logits

    def conflict_only(self, record):
        """Re-run the no-patch readout and the conflict patch of a stored family on this
        runner's device and dtype (the gate-7 reference)."""
        G = [tuple(g) for g in record["matrix"]]
        donor = [tuple(g) for g in record["donor_matrix"]]
        cell = record["cell"]
        entities = [g[self.target] for g in G]
        rec_prompt = self.prompt(G, cell["i_N"])
        self.aligned_entity_ids(G, rec_prompt["input_ids"])
        base, _ = self.forward(rec_prompt["input_ids"])
        don_out, _ = self.forward(self.prompt(donor, cell["i_P"])["input_ids"])
        vector = don_out.hidden_states[self.layer][0, -1]
        out, report = self.forward(rec_prompt["input_ids"], patch=(self.layer, vector))
        ok = self.check_hook(report, rec_prompt["input_ids"].shape[1], self.model.config.hidden_size)
        keep = ("answer_logits", "answer_mass_full_vocab", "logsumexp_full", "descriptive")
        conflict = self.readout(out, entities)
        nopatch = self.readout(base, entities)
        return {"case_id": record["case_id"], "device": str(self.device),
                "dtype": str(next(self.model.parameters()).dtype),
                "conflict": {k: conflict[k] for k in keep}, "nopatch": {k: nopatch[k] for k in keep},
                "hook_ok": ok}
