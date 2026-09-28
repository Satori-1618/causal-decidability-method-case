"""Round 1 development pilot on the locally cached model (authorized 28 September 2026).

Offline only: HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are set before transformers is
imported, the model is read from the local Hugging Face cache at the pinned revision,
and an incomplete snapshot STOPs the run instead of downloading. Pilot data are
development data: they are never used to estimate the frozen anchors or delta.

    python scripts/run_mixing_pilot.py run --upstream /path/to/mixing-mechs --output results/pilot
    python scripts/run_mixing_pilot.py summarize --output results/pilot

``run`` writes manifest.json, RUN_STARTED.json (the manifest's hash), records.jsonl (one
family per line, bfloat16), fp32_reference.jsonl, audit_full_logits.npz and
artifact_hashes.json. ``summarize`` (standard library and the analyzer only) computes
the pilot gates from those files into summary.json.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

APPLICATION = Path(__file__).resolve().parents[1]
REPOSITORY = APPLICATION.parents[1]
sys.path.insert(0, str(APPLICATION / "src"))

LOCK = APPLICATION / "SOURCE_LOCK.json"
VALUES = APPLICATION / "PROPOSED_VALUES.json"
PILOT = {
    "task": "music_performance", "n": 7, "layer": 18, "diagnostic_layer": 19,
    "families": 80, "seed_base": 1000000,
    "cells": [["c1", {"i_P": 3, "i_L": 1, "i_R": 5, "i_N": 0}],
              ["c2", {"i_P": 3, "i_L": 5, "i_R": 1, "i_N": 0}],
              ["c3", {"i_P": 3, "i_L": 1, "i_R": 5, "i_N": 6}],
              ["c4", {"i_P": 3, "i_L": 5, "i_R": 1, "i_N": 6}]],
    "cell_assignment": "family i uses cells[i mod 4]",
    "dtype": "bfloat16", "attention": "eager", "reference_dtype": "float32",
    "fp32_reference_families": 32, "audit_full_logit_families": 4,
    "s_min_quantile": 0.99, "s_min_floor": 0.10, "d_min": 0.20,
    "identity_tolerance": 0.001, "dtype_tolerance_T": 0.01,
    "agreement_transfer_floor": 0.90, "agreement_resolution_floor": 0.90,
    "yield_floor": 0.50, "resolution_rate_floor": 0.90,
}
CODE = ["src/mixing_prompts.py", "src/mixing_runner.py", "src/mixing_round1_design.py",
        "src/mixing_round1_analysis.py", "scripts/lock_sources.py", "scripts/run_mixing_pilot.py",
        "scripts/check_mixing_round1_records.py"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, exclusive=True):
    with Path(path).open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def environment(device):
    import importlib.metadata as md
    import torch
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy", "huggingface-hub",
                 "jinja2", "sentencepiece", "protobuf", "accelerate"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "device": device, "mps_available": bool(torch.backends.mps.is_available()),
            "packages": packages,
            "offline": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}}


def run(args):
    import torch
    import mixing_runner as mr
    from mixing_prompts import load_adapter

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output exists and is not empty; earlier artifacts are preserved")
    output.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK.read_text())
    adapter = load_adapter()
    clone = adapter.check(args.upstream, lock)
    spec = adapter.schema_spec(args.upstream, PILOT["task"])
    model_lock = lock["model"]
    try:
        snapshot = mr.snapshot_path(model_lock["id"], model_lock["revision"])
    except mr.SnapshotIncomplete as error:
        save(output / "STOP.json", {"gate": 1, "reason": str(error), "action": "no download attempted"})
        raise SystemExit(f"STOP: {error}")
    hashes = mr.snapshot_hashes(snapshot, model_lock["files_at_revision"])
    device = args.device or mr.pick_device()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    pools, dropped, ids = mr.single_token_pools(tokenizer, spec)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False,
                                             add_generation_prompt=True)
    gate3 = {"pools_kept": {c: len(v) for c, v in pools.items()}, "pools_dropped": dropped,
             "chat_template_rendered_for_X": rendered, "dropped_prefix": rendered[:5],
             "passed": all(len(v) >= PILOT["n"] + 2 for v in pools.values()) and rendered[:5] == "<bos>"}
    manifest = {
        "schema_version": 1, "application": "gur-arieh-2510.06182", "round": 1, "stage": "pilot",
        "label": "DEVELOPMENT PILOT: not used to estimate frozen anchors or delta",
        "authorized": "2026-09-28, by the user, for the pilot only",
        "pilot": PILOT, "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "code_files_sha256": {f"applications/gur-arieh-2510.06182/{c}": sha256(APPLICATION / c) for c in CODE},
        "proposed_values_sha256": sha256(VALUES),
        "upstream": {"path": str(Path(args.upstream).resolve()), "check": clone},
        "task_spec": spec, "single_token_pools": pools,
        "model": {"id": model_lock["id"], "revision": model_lock["revision"],
                  "snapshot": str(snapshot), "hashes": hashes},
        "gate3_tokens": gate3,
        "environment": environment(device),
    }
    save(output / "manifest.json", manifest)
    save(output / "RUN_STARTED.json", {"manifest_sha256": sha256(output / "manifest.json"),
                                      "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if not hashes["passed"]:
        save(output / "STOP.json", {"gate": 1, "reason": hashes["problems"]})
        raise SystemExit("STOP: model or tokenizer files differ from the lock")
    if not gate3["passed"]:
        save(output / "STOP.json", {"gate": 3, "reason": gate3})
        raise SystemExit("STOP: token gate failed")

    timings = {}
    t0 = time.perf_counter()
    model, _ = mr.load_model(snapshot, torch.bfloat16, device, PILOT["attention"])
    timings["load_bf16_seconds"] = time.perf_counter() - t0
    runner = mr.Runner(model, tokenizer, spec, pools, ids, PILOT["layer"],
                       diagnostic_layer=PILOT["diagnostic_layer"])
    families = args.limit or PILOT["families"]
    audit = {}
    records_path = output / "records.jsonl"
    t0 = time.perf_counter()
    with records_path.open("x") as handle:
        for i in range(families):
            key, cell = PILOT["cells"][i % len(PILOT["cells"])]
            seed = PILOT["seed_base"] + i
            try:
                record, full = runner.family(case_id=f"pilot-{i:04d}", draw_index=i, seed=seed,
                                             cell_key=key, cell=cell, rng=random.Random(seed),
                                             n=PILOT["n"], audit=i < PILOT["audit_full_logit_families"])
            except mr.TechnicalError as error:
                record, full = {"case_id": f"pilot-{i:04d}", "draw_index": i, "seed": seed,
                                "cell_key": key, "cell": cell, "qualifies": False,
                                "technical": {"passed": False, "failures": [str(error)], "checks": {}}}, None
            if full is not None:
                audit[record["case_id"]] = full.numpy()
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            print(f"{record['case_id']} {key} qualifies={record['qualifies']} "
                  f"technical={record['technical']['passed']} "
                  f"{record.get('runtime_seconds', 0):.2f}s", flush=True)
    timings["families_bf16_seconds"] = time.perf_counter() - t0
    if audit:
        import numpy as np
        np.savez_compressed(output / "audit_full_logits.npz", **audit)
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    del runner, model
    if device == "mps":
        torch.mps.empty_cache()

    t0 = time.perf_counter()
    model32, _ = mr.load_model(snapshot, torch.float32, device, PILOT["attention"])
    timings["load_fp32_seconds"] = time.perf_counter() - t0
    runner32 = mr.Runner(model32, tokenizer, spec, pools, ids, PILOT["layer"])
    t0 = time.perf_counter()
    with (output / "fp32_reference.jsonl").open("x") as handle:
        for record in records[:PILOT["fp32_reference_families"]]:
            if "matrix" not in record:
                continue
            handle.write(json.dumps(runner32.conflict_only(record, PILOT["n"]), allow_nan=False) + "\n")
    timings["fp32_reference_seconds"] = time.perf_counter() - t0
    save(output / "timings.json", timings)
    rehash(output)
    print(json.dumps(timings, indent=2))


DATA_FILES = ("manifest.json", "RUN_STARTED.json", "records.jsonl", "fp32_reference.jsonl",
              "audit_full_logits.npz", "timings.json", "summary.json", "STOP.json")


def rehash(output):
    """sha256 of the pilot's data files (the report and this index are not included)."""
    save(Path(output) / "artifact_hashes.json",
         {name: sha256(Path(output) / name) for name in DATA_FILES if (Path(output) / name).exists()},
         exclusive=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--upstream", default=os.environ.get("MIXING_MECHS_UPSTREAM"), required=False)
    r.add_argument("--output", required=True)
    r.add_argument("--device", default=None)
    r.add_argument("--limit", type=int, default=None, help="smoke test: fewer families")
    s = sub.add_parser("summarize")
    s.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not args.upstream:
            parser.exit(2, "pass --upstream or set MIXING_MECHS_UPSTREAM\n")
        run(args)
    else:
        from mixing_pilot_summary import summarize
        output = Path(args.output)
        summary = summarize(output, PILOT)
        save(output / "summary.json", summary, exclusive=False)
        rehash(output)
        print(json.dumps({"gates": {k: v.get("passed") for k, v in summary["gates"].items()},
                          "unresolved_rate": summary["unresolved_rate_for_N_rule"],
                          "N": summary["n_rule"] and summary["n_rule"]["N"]}, indent=2))


if __name__ == "__main__":
    main()
