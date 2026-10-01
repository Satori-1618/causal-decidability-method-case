"""Prompt construction for Round 1, exactly as upstream formats prompts. Standard library.

The task spec (entity pools, templates, queries) comes from the pinned upstream clone
through the adapter ``scripts/lock_sources.py`` (``schema_spec``); nothing is vendored and
``tasks/dist.py`` is never imported. Rendering reproduces, for a row-style definition:

- ``_format_list`` and ``define_by_key`` (``grammar/task_to_causal_model.py:1159-1221``);
- ``raw_input``: ``f"{prefix}{context} {question} Answer:"`` with the query key built from
  the sorted query categories (``grammar/task_to_causal_model.py:1922-1980``, the causal
  model the command line uses; its ``ordering_012`` definition equals ``row_default``
  for the music task);
- ``format_prompt`` (``tasks/dist.py:127-136``): the tokenizer's chat template applied to
  one user message with a generation prompt, first five characters dropped; the string
  is then tokenized with default special tokens (``tasks/dist.py:356``).

Binding matrices are lists of groups, one entity per category in the spec's category
order (for music: Musician, Genre, Instrument).
"""
import importlib.util
from pathlib import Path

APPLICATION = Path(__file__).resolve().parents[1]
ANSWER_SUFFIX = " Answer:"
DROPPED_PREFIX_LENGTH = 5


def load_adapter():
    """The existing adapter module (scripts/lock_sources.py)."""
    spec = importlib.util.spec_from_file_location("mixing_lock_sources",
                                                  APPLICATION / "scripts/lock_sources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_list(items):
    """Upstream ``_format_list``: 'a', 'a and b', 'a, b, and c'."""
    items = list(items)
    if len(items) < 2:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _mapping(spec, group):
    return {category: group[i] for i, category in enumerate(spec["categories"])}


def query_key(spec, query_categories, answer_category):
    """Upstream selects the query template by the sorted query categories."""
    return f"Q:{'_'.join(sorted(query_categories))} A:{answer_category}"


def raw_prompt(spec, matrix, query_group, query_categories, answer_category,
               definition="row_default"):
    """The raw prompt of one binding matrix, asking about group ``query_group``."""
    categories = spec["categories"]
    if len(matrix) < 2 or any(len(g) != len(categories) for g in matrix):
        raise ValueError("matrix must hold at least two groups of one entity per category")
    template = spec["definitions"][definition]
    if any(f"{{{c}_list}}" in template for c in categories):
        raise ValueError("column-style definitions are not used in Round 1")
    if spec["prefix"]:
        lists = {f"{c}_list": format_list(g[i] for g in matrix) for i, c in enumerate(categories)}
        prefix = spec["prefix"].format(**lists)
    else:
        prefix = ""
    clauses = [template.format(**_mapping(spec, g)) for g in matrix]
    if spec["capitalize_first_clause"]:
        clauses[0] = clauses[0][0].upper() + clauses[0][1:]
    context = format_list(clauses) + "."
    key = query_key(spec, query_categories, answer_category)
    question = spec["queries"][key]["question"].format(**_mapping(spec, matrix[query_group]))
    if spec["queries"][key]["answer_category"] != answer_category:
        raise ValueError("query template answers a different category")
    return f"{prefix}{context} {question}{ANSWER_SUFFIX}"


def chat_prompt(tokenizer, raw):
    """Upstream ``format_prompt``: chat template with a generation prompt, first five
    characters dropped. Returns the prompt and the dropped characters (expected
    '<bos>', which the tokenizer adds back as a token)."""
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": raw}], tokenize=False,
                                             add_generation_prompt=True)
    return rendered[DROPPED_PREFIX_LENGTH:], rendered[:DROPPED_PREFIX_LENGTH]
