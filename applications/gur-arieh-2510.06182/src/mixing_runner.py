"""Round 1 model runner: prompts, last-token residual patching and the entity readout.

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

The readout at the last position records the n in-context entity-token logits (ids of
the entity tokens as they occur in the context, with their leading space, as
``tasks/dist.py:358``), logsumexp over the full vocabulary and over its complement, the
absolute entity-token mass, and a float32-unembedding copy of the entity logits as a
precision diagnostic. Full-vocabulary logits are returned only on request (audit sample).

Readout-validity diagnostics, recorded and never used by the declared statistic: the
logits of each entity's answer forms without a leading space ('Country', 'country'; None
where the form is not one token), and greedy generation under every patch, so that the
pilot can report how often the in-context entity readout names the entity the patched
model actually generates.
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
    """A technical failure (alignment, hook, shape): INVALID, never unresolved."""


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
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=dtype,
                                                 attn_implementation=attention)
    return model.to(device).eval(), tokenizer


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---- tokens -------------------------------------------------------------------------

def single_token_id(tokenizer, entity):
    """Id of ' entity' if it is one token with a leading space, else None."""
    ids = tokenizer.encode(" " + entity, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def single_token_pools(tokenizer, spec):
    """Keep only single-token entities (with a leading space), in upstream order."""
    pools, dropped, ids = {}, {}, {}
    for category in spec["categories"]:
        kept = []
        for entity in spec["items"][category]:
            token = single_token_id(tokenizer, entity)
            if token is None:
                dropped.setdefault(category, []).append(entity)
            else:
                kept.append(entity)
                ids[(category, entity)] = token
        pools[category] = kept
    return pools, dropped, ids


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
    """One model, tokenizer, task spec and study layer."""

    def __init__(self, model, tokenizer, spec, pools, entity_ids, layer, *,
                 diagnostic_layer=None, query=QUERY, target=TARGET, max_new_tokens=None):
        self.model, self.tokenizer, self.spec = model, tokenizer, spec
        self.pools, self.entity_ids = pools, entity_ids
        self.layer, self.diagnostic_layer = layer, diagnostic_layer
        self.categories = spec["categories"]
        self.target_category = self.categories[target]
        self.query_categories = [self.categories[i] for i in query]
        self.target = target
        self.max_new_tokens = max_new_tokens or spec["max_new_tokens"]
        self.device = next(model.parameters()).device
        self.vocab_size = model.get_output_embeddings().weight.shape[0]
        self.target_ids = {e: entity_ids[(self.target_category, e)] for e in pools[self.target_category]}
        self.answer_form_ids = {e: {"capitalized": self._one_token(e[:1].upper() + e[1:]),
                                    "lowercase": self._one_token(e)}
                                for e in pools[self.target_category]}
        self._unembed32 = None

    def _one_token(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    # -- prompts
    def prompt(self, matrix, query_group):
        raw = raw_prompt(self.spec, matrix, query_group, self.query_categories, self.target_category)
        chat, dropped = chat_prompt(self.tokenizer, raw)
        ids = self.tokenizer(chat, return_tensors="pt")["input_ids"].to(self.device)
        return {"raw": raw, "chat": chat, "dropped": dropped, "input_ids": ids}

    def aligned_entity_ids(self, matrix, input_ids):
        """Ids of the n target entities in group order, checked against the prompt: the
        tokens of the target pool occur in the prompt exactly once each, in group order.
        Located by design, verified on tokens; a mismatch is a technical failure."""
        expected = [self.target_ids[g[self.target]] for g in matrix]
        pool = set(self.target_ids.values())
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

    def unembed32(self, final_hidden_last):
        """Entity-independent float32 unembedding of the final (normed) hidden state,
        with the model's final soft-capping; a precision diagnostic only."""
        if self._unembed32 is None:
            self._unembed32 = self.model.get_output_embeddings().weight.detach().float()
        logits = final_hidden_last.float() @ self._unembed32.T
        cap = getattr(self.model.config, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def readout(self, out, entity_ids, extra_ids=None, full=False, entities=None):
        logits = out.logits[0, -1].float()
        if logits.shape[0] != self.vocab_size:
            raise TechnicalError("logit vector length differs from the vocabulary")
        index = torch.tensor(entity_ids, device=logits.device)
        entity = logits[index]
        lse_full = torch.logsumexp(logits, 0)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[index] = False
        lse_complement = torch.logsumexp(logits[mask], 0)
        lse_entity = torch.logsumexp(entity, 0)
        top = int(torch.argmax(logits))
        final = out.hidden_states[-1][0, -1]
        entity32 = self.unembed32(final)[index]
        result = {
            "entity_logits": [float(x) for x in entity.tolist()],
            "entity_mass_full_vocab": float(torch.exp(lse_entity - lse_full)),
            "logsumexp_full": float(lse_full),
            "logsumexp_complement": float(lse_complement),
            "logsumexp_entities": float(lse_entity),
            "top_token_id": top,
            "top_token": self.tokenizer.convert_ids_to_tokens(top),
            "top_token_logit": float(logits[top]),
            "entity_logits_fp32_unembed": [float(x) for x in entity32.tolist()],
        }
        if extra_ids:
            result["extra_logits"] = {name: float(logits[i]) for name, i in extra_ids.items()}
        if entities is not None:
            result["answer_form_logits"] = {
                form: [(float(logits[self.answer_form_ids[e][form]])
                        if self.answer_form_ids[e][form] is not None else None) for e in entities]
                for form in ("capitalized", "lowercase")}
        if not all(math.isfinite(v) for v in result["entity_logits"] + [result["logsumexp_full"],
                                                                        result["logsumexp_complement"]]):
            raise TechnicalError("non-finite readout")
        if full:
            result["_full_logits"] = logits.detach().cpu()
        return result

    @staticmethod
    def check_hook(report, length, hidden_size):
        """Gate 4 per patched forward: one call, one write, at the last prompt position,
        on a [1, T, D] input and a [D] vector."""
        ok = (report["calls"] == 1 and report["writes"] == 1
              and report["positions"] == [length - 1]
              and report["shapes"] == [[1, length, hidden_size]]
              and report["vector_shape"] == [hidden_size])
        return ok

    # -- one run of a prompt, unpatched, with hidden states and greedy generation
    def native(self, matrix, query_group, answer, generate=True):
        prompt = self.prompt(matrix, query_group)
        entity_ids, positions = self.aligned_entity_ids(matrix, prompt["input_ids"])
        out, _ = self.forward(prompt["input_ids"])
        entities = [g[self.target] for g in matrix]
        readout = self.readout(out, entity_ids, entities=entities)
        argmax = max(range(len(entities)), key=lambda i: readout["entity_logits"][i])
        record = {"raw_prompt": prompt["raw"], "dropped_prefix": prompt["dropped"],
                  "prompt_tokens": int(prompt["input_ids"].shape[1]),
                  "entity_positions": positions, "answer": answer,
                  "readout": readout, "readout_argmax": argmax,
                  "readout_argmax_entity": entities[argmax]}
        if generate:
            generation, _ = self.generate(prompt["input_ids"])
            word = first_word(generation["text"])
            record.update(generation=generation["text"], generation_ids=generation["token_ids"],
                          first_word=word, correct=word == answer.lower(),
                          readout_matches_generation=word == entities[argmax].lower())
        return prompt, entity_ids, record, out

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
        rec_prompt, rec_ids, recipient, rec_out = self.native(G, i_N, recipient_answer)
        L = rec_prompt["input_ids"].shape[1]
        donor_answer = conflict["donor"][cell["i_P"]][self.target]
        don_prompt, _, donor, don_out = self.native(conflict["donor"], cell["i_P"], donor_answer)
        don_hidden = don_out.hidden_states
        check("dropped_prefix_is_bos", rec_prompt["dropped"] == self.tokenizer.bos_token
              and don_prompt["dropped"] == self.tokenizer.bos_token, rec_prompt["dropped"])
        check("single_bos", int((rec_prompt["input_ids"][0] == self.tokenizer.bos_token_id).sum()) == 1
              and int(rec_prompt["input_ids"][0, 0]) == self.tokenizer.bos_token_id)

        donor_in_recipient = donor_answer in [g[self.target] for g in G]
        extra = None
        if not donor_in_recipient:
            token = self.target_ids.get(donor_answer)
            extra = {"absent_donor_answer": token} if token is not None else None

        recipient_entities = [g[self.target] for g in G]

        def patched(layer, hidden, extra_ids=None, full=False):
            vector = hidden[layer][0, -1]
            out, report = self.forward(rec_prompt["input_ids"], patch=(layer, vector))
            hook_ok = self.check_hook(report, L, hidden_size)
            readout = self.readout(out, rec_ids, extra_ids, full=full, entities=recipient_entities)
            generation, gen_report = self.generate(rec_prompt["input_ids"], patch=(layer, vector))
            word = first_word(generation["text"])
            lowered = [e.lower() for e in recipient_entities]
            readout["patched_generation"] = {
                "text": generation["text"], "token_ids": generation["token_ids"], "first_word": word,
                "entity_index": lowered.index(word) if word in lowered else None,
                "hook_writes": gen_report["writes"], "hook_positions": gen_report["positions"]}
            hook_ok = hook_ok and gen_report["writes"] == 1 and gen_report["positions"] == [L - 1]
            return readout, report, hook_ok

        conflict_readout, conflict_hook, ok = patched(self.layer, don_hidden, extra, full=audit)
        check("conflict_hook", ok, conflict_hook)
        full_logits = conflict_readout.pop("_full_logits", None)
        diagnostic = None
        if self.diagnostic_layer is not None:
            d_readout, d_hook, d_ok = patched(self.diagnostic_layer, don_hidden, extra)
            check("diagnostic_hook", d_ok, d_hook)
            diagnostic = {"layer": self.diagnostic_layer, "readout": d_readout, "hook": d_hook}

        agreement = []
        for key, j, case, a_indices in agreement_cases:
            a_answer = case["donor"][j][self.target]
            _, _, a_donor, a_out = self.native(case["donor"], j, a_answer)
            a_readout, a_hook, a_ok = patched(self.layer, a_out.hidden_states)
            check(f"agreement_{key}_hook", a_ok, a_hook)
            logits = a_readout["entity_logits"]
            argmax = max(range(n), key=logits.__getitem__)
            agreement.append({"target": key, "j": j, "design_indices": a_indices,
                              "donor_native": a_donor, "readout": a_readout, "hook": a_hook,
                              "entity_logits": logits,
                              "entity_mass_full_vocab": a_readout["entity_mass_full_vocab"],
                              "argmax": argmax, "transfer": argmax == j})

        identity_record = None
        if identity:
            vector = rec_out.hidden_states[self.layer][0, -1]
            out_id, id_hook = self.forward(rec_prompt["input_ids"], patch=(self.layer, vector))
            gen_id, gen_hook = self.generate(rec_prompt["input_ids"], patch=(self.layer, vector))
            base_logits, id_logits = rec_out.logits[0, -1].float(), out_id.logits[0, -1].float()
            index = torch.tensor(rec_ids, device=base_logits.device)
            identity_record = {
                "max_abs_entity_logit_difference": float((id_logits[index] - base_logits[index]).abs().max()),
                "max_abs_logit_difference": float((id_logits - base_logits).abs().max()),
                "same_entity_argmax": int(torch.argmax(id_logits[index])) == recipient["readout_argmax"],
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
            "case_id": case_id, "draw_index": draw_index, "seed": seed, "cell_key": cell_key,
            "cell": dict(cell), "qualifies": qualifies,
            "design_indices": indices, "technical": technical,
            "entity_logits": conflict_readout["entity_logits"],
            "entity_mass_full_vocab": conflict_readout["entity_mass_full_vocab"],
            "readout": conflict_readout, "hook": conflict_hook,
            "matrix": [list(g) for g in G],
            "donor_matrix": [list(g) for g in conflict["donor"]],
            "donor_answer": {"entity": donor_answer, "in_recipient": donor_in_recipient,
                             "recipient_index": ([g[self.target] for g in G].index(donor_answer)
                                                 if donor_in_recipient else None)},
            "native": {"recipient": recipient, "donor": donor},
            "nopatch": {"entity_logits": recipient["readout"]["entity_logits"],
                        "entity_mass_full_vocab": recipient["readout"]["entity_mass_full_vocab"]},
            "agreement": agreement,
            "identity": identity_record,
            "diagnostic": diagnostic,
            "runtime_seconds": time.perf_counter() - start,
        }
        return record, full_logits

    def conflict_only(self, record, n):
        """Re-run the no-patch readout and the conflict patch of a stored family (used
        for the float32 reference on the same prompts)."""
        G = [tuple(g) for g in record["matrix"]]
        donor = [tuple(g) for g in record["donor_matrix"]]
        cell = record["cell"]
        rec_prompt = self.prompt(G, cell["i_N"])
        rec_ids, _ = self.aligned_entity_ids(G, rec_prompt["input_ids"])
        base, _ = self.forward(rec_prompt["input_ids"])
        don_prompt = self.prompt(donor, cell["i_P"])
        don_out, _ = self.forward(don_prompt["input_ids"])
        vector = don_out.hidden_states[self.layer][0, -1]
        out, report = self.forward(rec_prompt["input_ids"], patch=(self.layer, vector))
        ok = self.check_hook(report, rec_prompt["input_ids"].shape[1], self.model.config.hidden_size)
        return {"case_id": record["case_id"],
                "nopatch": self.readout(base, rec_ids)["entity_logits"],
                "conflict": self.readout(out, rec_ids)["entity_logits"],
                "hook_ok": ok}
