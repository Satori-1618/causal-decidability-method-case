"""Check the pinned upstream clone against SOURCE_LOCK.json (standard library only).

The upstream repository (yoavgur/mixing-mechs) is a separate, read-only dependency kept
outside this repository. Nothing is vendored. This small adapter reads the clone and:

- confirms the checked-out commit and the licence text;
- confirms the sha256 of every upstream file this application relies on;
- confirms that every cited upstream line still reads as recorded;
- rebuilds the task registry by importing ``grammar/schemas.py``, which needs only the
  standard library (no torch, no model, no download).

Usage::

    python scripts/lock_sources.py --upstream /path/to/mixing-mechs --check
    python scripts/lock_sources.py --upstream /path/to/mixing-mechs --registry

``--upstream`` defaults to the environment variable MIXING_MECHS_UPSTREAM.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

APPLICATION = Path(__file__).resolve().parents[1]
LOCK = APPLICATION / "SOURCE_LOCK.json"


class LockError(ValueError):
    """The clone differs from the recorded lock."""


def require(condition, message):
    if not condition:
        raise LockError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head(root):
    """Commit of a clone's HEAD, read from .git without running git."""
    git = Path(root) / ".git"
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[len("ref: "):]
    if (git / ref).is_file():
        return (git / ref).read_text().strip()
    packed = git / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    raise LockError(f"cannot resolve {ref} in {git}")


def cited_line(root, relative, number):
    lines = (Path(root) / relative).read_text(encoding="utf-8").splitlines()
    require(1 <= number <= len(lines), f"{relative}:{number} is out of range")
    return lines[number - 1]


def registry_assignment(root):
    """Names and line span of the module-level ``schemas = [...]`` list in tasks/dist.py."""
    tree = ast.parse((Path(root) / "tasks/dist.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "schemas"
                and isinstance(node.value, ast.List)):
            names = [element.id for element in node.value.elts]
            return names, node.lineno, node.end_lineno
    raise LockError("tasks/dist.py has no module-level schemas list")


@contextlib.contextmanager
def upstream_grammar(root):
    """Import ``grammar.schemas`` (and ``grammar.grammar``) from the clone for the
    duration of the block, then restore ``sys.path`` and ``sys.modules``. Both modules
    need only the standard library. No bytecode is written into the clone."""
    saved_path, saved_flag = list(sys.path), sys.dont_write_bytecode
    saved_modules = {k: v for k, v in sys.modules.items() if k == "grammar" or k.startswith("grammar.")}
    for key in saved_modules:
        del sys.modules[key]
    sys.path.insert(0, str(Path(root).resolve()))
    sys.dont_write_bytecode = True
    try:
        yield importlib.import_module("grammar.schemas")
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_flag
        for key in [k for k in sys.modules if k == "grammar" or k.startswith("grammar.")]:
            del sys.modules[key]
        sys.modules.update(saved_modules)


def schema_spec(root, name):
    """One upstream task as plain data (entity pools, templates, queries), read from the
    clone's ``grammar/schemas.py``. Prompt rendering from this spec lives in
    ``src/mixing_prompts.py``; nothing is vendored."""
    names, _, _ = registry_assignment(root)
    with upstream_grammar(root) as module:
        for constant in names:
            schema = getattr(module, constant, None)
            if schema is not None and schema.name == name:
                templates = schema.templates
                return {
                    "constant": constant,
                    "name": schema.name,
                    "categories": list(schema.categories),
                    "items": {c: list(schema.items[c]) for c in schema.categories},
                    "definitions": dict(templates.definitions),
                    "prefix": templates.prefix,
                    "capitalize_first_clause": bool(templates.capitalize_first_clause),
                    "queries": {key: {"question": q.question, "answer_category": q.answer_category}
                                for key, q in templates.queries.items()},
                    "max_new_tokens": schema.max_new_tokens,
                }
    raise LockError(f"task {name} is not in the upstream registry")


def task_registry(root):
    """The upstream task registry, read from the clone without vendoring it.

    ``tasks/dist.py`` lists schema constants; ``grammar/schemas.py`` defines the tasks.
    Constants listed but not defined are reported, not skipped silently. No bytecode is
    written into the clone.
    """
    names, start, end = registry_assignment(root)
    with upstream_grammar(root) as module:
        tasks, missing = [], []
        for constant in names:
            schema = getattr(module, constant, None)
            if schema is None:
                missing.append(constant)
                continue
            tasks.append({
                "constant": constant,
                "name": schema.name,
                "categories": list(schema.categories),
                "entities_per_category": {c: len(schema.items[c]) for c in schema.categories},
                "row_template": schema.templates.definitions.get("row_default"),
                "prefix": schema.templates.prefix,
                "queries": {key: query.answer_category
                            for key, query in schema.templates.queries.items()},
                "max_new_tokens": schema.max_new_tokens,
            })
    return {"source": f"tasks/dist.py:{start}-{end}", "schema_module": "grammar/schemas.py",
            "tasks": tasks, "listed_but_not_defined": missing}


def check(root, lock):
    """Compare a clone with the lock; return a short report or raise LockError."""
    root = Path(root)
    upstream = lock["upstream"]
    require(git_head(root) == upstream["commit"],
            f"clone is at {git_head(root)}, lock pins {upstream['commit']}")
    licence = upstream["license"]
    require(sha256(root / licence["file"]) == licence["sha256"], "licence text differs")
    first = (root / licence["file"]).read_text().splitlines()[0].strip()
    require(first == licence["first_line"], "licence is not the recorded MIT text")
    for relative, digest in upstream["files_sha256"].items():
        require((root / relative).is_file(), f"missing upstream file {relative}")
        require(sha256(root / relative) == digest, f"upstream file changed: {relative}")
    cited = 0
    for fact in lock["upstream_facts"]:
        for evidence in fact.get("code_evidence", []):
            actual = cited_line(root, evidence["file"], evidence["line"])
            require(actual.rstrip() == evidence["text"].rstrip(),
                    f"{fact['id']}: {evidence['file']}:{evidence['line']} reads {actual!r}")
            cited += 1
    require(task_registry(root) == lock["task_registry"], "task registry differs from the lock")
    return {"status": "clone matches lock", "commit": upstream["commit"],
            "files": len(upstream["files_sha256"]), "cited_lines": cited,
            "tasks": len(lock["task_registry"]["tasks"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", default=os.environ.get("MIXING_MECHS_UPSTREAM"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--registry", action="store_true")
    args = parser.parse_args()
    if not args.upstream:
        parser.exit(2, "no clone given: pass --upstream or set MIXING_MECHS_UPSTREAM\n")
    try:
        if args.registry:
            print(json.dumps(task_registry(args.upstream), indent=2))
        else:
            print(json.dumps(check(args.upstream, json.loads(LOCK.read_text())), indent=2))
    except (OSError, LockError, KeyError) as error:
        parser.exit(1, f"FAILED: {error}\n")


if __name__ == "__main__":
    main()
