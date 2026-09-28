"""Round 1 development pilot, protocol v2 (pilot 2, authorized 28 September 2026).

Offline only: HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are set before transformers is
imported, the model is read from the local Hugging Face cache at the pinned revision,
and an incomplete snapshot STOPs the run instead of downloading. Pilot data are
development data: they never estimate the frozen anchors or delta; their unresolved
rate gives only a provisional planning N.

Protocol v2: float32 execution on MPS; the primary readout is the answer form; a case is
resolved iff S >= s_min and the answer-token mass >= 0.5; pools restricted as locked in
SOURCE_LOCK.json (``round1_entity_pools``); gate 7 compares MPS float32 with CPU float32
on the 32 declared families below (8 per cell).

    python scripts/run_mixing_pilot.py run --upstream /path/to/clone --output results/pilot2
    python scripts/run_mixing_pilot.py summarize --output results/pilot2
    python scripts/run_mixing_pilot.py index --output results/pilot2      # after adding the report
    python scripts/run_mixing_pilot.py run --upstream ... --output /tmp/smoke --smoke 4

``--smoke K`` runs K families from the declared smoke seed block (1,900,000 + i) and
refuses any output directory inside results/. Every step ends by rewriting
artifact_hashes.json: the sha256 of every file in the output directory and of every
producer script, generated, never edited by hand.

The first pilot (protocol v1, results/pilot/) was produced by this script at commit
a8e4ea1; its code is kept in the repository's history.
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
INDEX = "artifact_hashes.json"
PILOT = {
    "name": "pilot2", "protocol": "v2", "task": "music_performance", "n": 7,
    "layer": 18, "diagnostic_layer": 19,
    "families": 80, "seed_base": 1100000, "smoke_seed_base": 1900000,
    "cells": [["c1", {"i_P": 3, "i_L": 1, "i_R": 5, "i_N": 0}],
              ["c2", {"i_P": 3, "i_L": 5, "i_R": 1, "i_N": 0}],
              ["c3", {"i_P": 3, "i_L": 1, "i_R": 5, "i_N": 6}],
              ["c4", {"i_P": 3, "i_L": 5, "i_R": 1, "i_N": 6}]],
    "cell_assignment": "family i uses cells[i mod 4]",
    "dtype": "float32", "device": "mps", "attention": "eager",
    "gate7_reference": {"device": "cpu", "dtype": "float32",
                        "families": list(range(32)),
                        "note": "families 0-31: 8 per cell under the i mod 4 assignment; fixed before the run"},
    "audit_full_logit_families": 4,
    "s_min_quantile": 0.99, "s_min_floor": 0.10, "d_min": 0.20,
    "answer_mass_floor": 0.5,
    "identity_tolerance": 0.001, "gate7_tolerance_T": 0.01,
    "agreement_transfer_floor": 0.90, "agreement_resolution_floor": 0.90,
    "yield_floor": 0.50, "resolution_rate_floor": 0.90,
}
PRODUCERS = [
    "applications/gur-arieh-2510.06182/scripts/run_mixing_pilot.py",
    "applications/gur-arieh-2510.06182/src/mixing_runner.py",
    "applications/gur-arieh-2510.06182/src/mixing_prompts.py",
    "applications/gur-arieh-2510.06182/src/mixing_pilot_summary.py",
    "applications/gur-arieh-2510.06182/src/mixing_round1_analysis.py",
    "applications/gur-arieh-2510.06182/src/mixing_round1_design.py",
    "applications/gur-arieh-2510.06182/scripts/lock_sources.py",
    "applications/gur-arieh-2510.06182/scripts/check_mixing_round1_records.py",
    "applications/makelov-2311.17030/src/query_route_analysis.py",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, exclusive=True):
    with Path(path).open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def write_index(output):
    """artifact_hashes.json: every file in ``output`` (except this index) and every
    producer script, hashed; generated only by this function."""
    output = Path(output)
    data = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != INDEX}
    index = {"schema": "artifact index v2", "generated_by": "scripts/run_mixing_pilot.py",
             "git_head": git("rev-parse", "HEAD"),
             "data": data,
             "producers": {path: sha256(REPOSITORY / path) for path in PRODUCERS}}
    save(output / INDEX, index, exclusive=False)
    return index


def environment(device):
    import importlib.metadata as md
    import torch
    packages = sorted({(d.metadata["Name"], d.version) for d in md.distributions()},
                      key=lambda x: x[0].lower())
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "device": device, "mps_available": bool(torch.backends.mps.is_available()),
            "packages": {name: version for name, version in packages},
            "offline": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}}


def run(args):
    import torch
    import mixing_runner as mr
    from mixing_prompts import load_adapter

    output = Path(args.output).resolve()
    smoke = args.smoke is not None
    if smoke and (APPLICATION / "results") in output.parents:
        raise SystemExit("smoke runs must write outside results/")
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output exists and is not empty; earlier artifacts are preserved")
    output.mkdir(parents=True, exist_ok=True)
    seed_base = PILOT["smoke_seed_base"] if smoke else PILOT["seed_base"]
    families = args.smoke if smoke else PILOT["families"]

    lock = json.loads(LOCK.read_text())
    adapter = load_adapter()
    clone = adapter.check(args.upstream, lock)
    spec = adapter.schema_spec(args.upstream, PILOT["task"])
    model_lock = lock["model"]
    try:
        snapshot = mr.snapshot_path(model_lock["id"], model_lock["revision"])
    except mr.SnapshotIncomplete as error:
        save(output / "STOP.json", {"gate": 1, "reason": str(error), "action": "no download attempted"})
        write_index(output)
        raise SystemExit(f"STOP: {error}")
    hashes = mr.snapshot_hashes(snapshot, model_lock["files_at_revision"])
    device = PILOT["device"] if torch.backends.mps.is_available() else "cpu"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    pools, dropped, context_ids, answer_ids = mr.round1_pools(tokenizer, spec)
    pool_record = mr.pools_record(pools, dropped, context_ids, answer_ids, spec)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False,
                                             add_generation_prompt=True)
    gate3 = {"pools_kept": {c: len(v) for c, v in pools.items()}, "pools_dropped": dropped,
             "pools_equal_the_lock": json.loads(json.dumps(pool_record)) == lock.get("round1_entity_pools"),
             "chat_template_rendered_for_X": rendered, "dropped_prefix": rendered[:5]}
    gate3["passed"] = (gate3["pools_equal_the_lock"] and rendered[:5] == "<bos>"
                       and all(len(v) >= PILOT["n"] + 2 for v in pools.values()))
    manifest = {
        "schema_version": 2, "application": "gur-arieh-2510.06182", "round": 1,
        "stage": "smoke" if smoke else "pilot2",
        "label": ("SMOKE RUN: not a result" if smoke else
                  "DEVELOPMENT PILOT 2: descriptive; never used to estimate frozen anchors or delta"),
        "authorized": "2026-09-28, by the user: protocol v2 and a second pilot, no split A or B",
        "pilot": PILOT, "seed_base_used": seed_base, "families_run": families,
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "producers_sha256": {path: sha256(REPOSITORY / path) for path in PRODUCERS},
        "proposed_values_sha256": sha256(VALUES), "source_lock_sha256": sha256(LOCK),
        "upstream": {"path": str(Path(args.upstream).resolve()), "check": clone},
        "task_spec": spec, "entity_pools": pool_record,
        "model": {"id": model_lock["id"], "revision": model_lock["revision"],
                  "snapshot": str(snapshot), "hashes": hashes},
        "gate3_tokens": gate3,
        "environment": environment(device),
    }
    save(output / "manifest.json", manifest)
    save(output / "RUN_STARTED.json", {"manifest_sha256": sha256(output / "manifest.json"),
                                      "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    for gate, ok, reason in ((1, hashes["passed"], hashes["problems"]), (3, gate3["passed"], gate3)):
        if not ok:
            save(output / "STOP.json", {"gate": gate, "reason": reason})
            write_index(output)
            raise SystemExit(f"STOP at gate {gate}")

    timings = {}
    t0 = time.perf_counter()
    model, _ = mr.load_model(snapshot, torch.float32, device, PILOT["attention"])
    timings["load_fp32_mps_seconds"] = time.perf_counter() - t0
    runner = mr.Runner(model, tokenizer, spec, pools, context_ids, answer_ids, PILOT["layer"],
                       diagnostic_layer=PILOT["diagnostic_layer"])
    audit = {}
    records_path = output / "records.jsonl"
    prefix = "smoke" if smoke else "pilot2"
    t0 = time.perf_counter()
    with records_path.open("x") as handle:
        for i in range(families):
            key, cell = PILOT["cells"][i % len(PILOT["cells"])]
            seed = seed_base + i
            try:
                record, full = runner.family(case_id=f"{prefix}-{i:04d}", draw_index=i, seed=seed,
                                             cell_key=key, cell=cell, rng=random.Random(seed),
                                             n=PILOT["n"], audit=i < PILOT["audit_full_logit_families"])
            except mr.TechnicalError as error:
                record, full = {"case_id": f"{prefix}-{i:04d}", "draw_index": i, "seed": seed,
                                "cell_key": key, "cell": cell, "qualifies": False,
                                "technical": {"passed": False, "failures": [str(error)], "checks": {}}}, None
            if full is not None:
                audit[record["case_id"]] = full.numpy()
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            print(f"{record['case_id']} {key} qualifies={record['qualifies']} "
                  f"technical={record['technical']['passed']} {record.get('runtime_seconds', 0):.2f}s",
                  flush=True)
    timings["families_fp32_mps_seconds"] = time.perf_counter() - t0
    if audit:
        import numpy as np
        np.savez_compressed(output / "audit_full_logits.npz", **audit)
    if device == "mps":
        timings["mps_driver_gib_after_families"] = torch.mps.driver_allocated_memory() / 2 ** 30
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    del runner, model
    if device == "mps":
        torch.mps.empty_cache()

    t0 = time.perf_counter()
    reference_model, _ = mr.load_model(snapshot, torch.float32, "cpu", PILOT["attention"])
    timings["load_fp32_cpu_seconds"] = time.perf_counter() - t0
    reference = mr.Runner(reference_model, tokenizer, spec, pools, context_ids, answer_ids, PILOT["layer"])
    declared = [i for i in PILOT["gate7_reference"]["families"] if i < families]
    t0 = time.perf_counter()
    with (output / "gate7_cpu_reference.jsonl").open("x") as handle:
        for i in declared:
            record = records[i]
            if "matrix" not in record:
                continue
            handle.write(json.dumps(reference.conflict_only(record), allow_nan=False) + "\n")
    timings["gate7_cpu_reference_seconds"] = time.perf_counter() - t0
    save(output / "timings.json", timings)
    write_index(output)
    print(json.dumps(timings, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--upstream", default=os.environ.get("MIXING_MECHS_UPSTREAM"))
    r.add_argument("--output", required=True)
    r.add_argument("--smoke", type=int, default=None,
                   help="run this many families from the smoke seed block (1,900,000 + i), outside results/")
    s = sub.add_parser("summarize")
    s.add_argument("--output", required=True)
    x = sub.add_parser("index")
    x.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not args.upstream:
            parser.exit(2, "pass --upstream or set MIXING_MECHS_UPSTREAM\n")
        run(args)
    elif args.command == "summarize":
        from mixing_pilot_summary import summarize
        output = Path(args.output)
        summary = summarize(output, PILOT)
        save(output / "summary.json", summary, exclusive=False)
        write_index(output)
        gates = {k: v.get("passed") for k, v in summary["gates"].items()}
        print(json.dumps({"gates": gates, "unresolved_rate": summary["unresolved_rate_for_planning_N"],
                          "planning_N": summary["planning_N"] and summary["planning_N"]["N"]}, indent=2))
    else:
        write_index(Path(args.output))


if __name__ == "__main__":
    main()
