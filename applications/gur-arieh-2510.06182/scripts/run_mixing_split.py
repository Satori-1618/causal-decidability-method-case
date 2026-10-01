"""Round 1 development splits A and B under protocol v2 (authorized 28 September 2026).

Offline only (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 before transformers is imported;
the cached snapshot at the pinned revision; an incomplete snapshot STOPs). DEVELOPMENT
DATA. The protocol, declared before either run, is SPLIT_A_B_PROTOCOL.md; the specs
below are its machine-readable form. Nothing here freezes a value or touches the
confirmation seed block (4,000,000 + i).

    python scripts/run_mixing_split.py run --split A --upstream /path/to/clone --output results/split_A
    python scripts/run_mixing_split.py decide --split A --output results/split_A
    python scripts/run_mixing_split.py run --split B --upstream /path/to/clone --output results/split_B
    python scripts/run_mixing_split.py decide --split B --output results/split_B
    python scripts/run_mixing_split.py index --output results/split_A        # after adding a report
    python scripts/run_mixing_split.py run --split A --smoke 2 --upstream ... --output /tmp/smoke

``run --split B`` reads the committed split-A decision and runs only if its status is
PROCEED, in the selected cell. ``--smoke K`` uses the smoke seed block (1,900,000 + i),
a quota of K per cell, and refuses any output inside results/. Every step rewrites the
generated artifact index (every file of the output directory and every producer).
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

APPLICATION = Path(__file__).resolve().parents[1]
REPOSITORY = APPLICATION.parents[1]
sys.path.insert(0, str(APPLICATION / "src"))
_spec = importlib.util.spec_from_file_location("run_mixing_pilot", APPLICATION / "scripts/run_mixing_pilot.py")
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

CELLS = [["c1", {"i_P": 3, "i_L": 1, "i_R": 5, "i_N": 0}],
         ["c2", {"i_P": 3, "i_L": 5, "i_R": 1, "i_N": 0}],
         ["c3", {"i_P": 3, "i_L": 1, "i_R": 5, "i_N": 6}],
         ["c4", {"i_P": 3, "i_L": 5, "i_R": 1, "i_N": 6}]]
COMMON = {
    "protocol": "v2", "task": "music_performance", "n": 7, "layer": 18, "diagnostic_layer": 19,
    "dtype": "float32", "device": "mps", "attention": "eager",
    "s_min_quantile": 0.99, "s_min_floor": 0.10, "d_min": 0.20, "answer_mass_floor": 0.5,
    "identity_tolerance": 0.001, "gate7_tolerance_T": 0.01,
    "agreement_transfer_floor": 0.90, "agreement_resolution_floor": 0.90,
    "yield_floor": 0.50, "resolution_rate_floor": 0.90,
    "delta": {"false_invalid_rate": 0.05, "resamples": 10000, "seed": 251006182},
    "smoke_seed_base": 1900000, "confirmation_seed_base_untouched": 4000000,
}
SPLIT_A = {
    **COMMON, "name": "split_A", "seed_base": 2000000, "cells": CELLS,
    "qualifying_per_cell": 50, "cap_per_cell": 100,
    "order": "family i = 4j + k uses cell k (c1..c4) and seed 2,000,000 + i; a cell stops at its 50th "
             "qualifying family; at most 100 families per cell",
    "gate7_reference": {"device": "cpu", "dtype": "float32", "families": list(range(32)),
                        "note": "draw indices 0-31 = the first 8 generated families of each cell (j = 0..7)"},
    "audit_full_logit_families": [0, 1, 2, 3],
    "selection": "largest d on split A; ties c1, c2, c3, c4; subject to d >= d_min, else "
                 "NOT_DECIDABLE_WITH_CURRENT_INTERVENTIONS (S1)",
}
SPLIT_B = {
    **COMMON, "name": "split_B", "seed_base": 3000000, "cells": "the cell selected on split A",
    "qualifying": 200, "cap": 400,
    "order": "family i uses the selected cell and seed 3,000,000 + i; generation stops at the 200th "
             "qualifying family; at most 400 families",
    "gate7_reference": {"device": "cpu", "dtype": "float32", "families": list(range(32)),
                        "note": "draw indices 0-31 = the first 32 generated families of the selected cell"},
    "audit_full_logit_families": [0, 1, 2, 3],
}
PRODUCERS = pilot.PRODUCERS + ["applications/gur-arieh-2510.06182/scripts/run_mixing_split.py",
                               "applications/gur-arieh-2510.06182/src/mixing_splits.py"]
SPLIT_A_DECISION = APPLICATION / "results/split_A/split_A_decision.json"


def write_index(output):
    return pilot.write_index(output, PRODUCERS, "scripts/run_mixing_split.py")


def spec_for(split, smoke=None):
    if split == "A":
        spec = json.loads(json.dumps(SPLIT_A))
    else:
        decision = json.loads(SPLIT_A_DECISION.read_text())
        if decision["status"] != "PROCEED" or not decision["selected"]:
            raise SystemExit("split A did not select a cell (or stopped): split B is not run")
        spec = json.loads(json.dumps(SPLIT_B))
        spec["cells"] = [[key, cell] for key, cell in CELLS if key == decision["selected"]]
        spec["selected_on_split_A"] = {"cell": decision["selected"],
                                       "decision_sha256": pilot.sha256(SPLIT_A_DECISION)}
    if smoke is not None:
        spec["seed_base"] = COMMON["smoke_seed_base"]
        if split == "A":
            spec["qualifying_per_cell"], spec["cap_per_cell"] = smoke, 2 * smoke
        else:
            spec["qualifying"], spec["cap"] = smoke, 2 * smoke
        spec["gate7_reference"]["families"] = list(range(min(4, 4 * smoke)))
        spec["smoke"] = True
    return spec


def run(args):
    import torch
    import mixing_runner as mr
    from mixing_prompts import load_adapter
    from mixing_splits import generate_families

    output = Path(args.output).resolve()
    smoke = args.smoke is not None
    if smoke and (APPLICATION / "results") in output.parents:
        raise SystemExit("smoke runs must write outside results/")
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output exists and is not empty; earlier artifacts are preserved")
    spec = spec_for(args.split, args.smoke)
    if spec["seed_base"] == COMMON["confirmation_seed_base_untouched"]:
        raise SystemExit("the confirmation seed block is not used here")
    output.mkdir(parents=True, exist_ok=True)

    lock = json.loads(pilot.LOCK.read_text())
    adapter = load_adapter()
    clone = adapter.check(args.upstream, lock)
    task = adapter.schema_spec(args.upstream, spec["task"])
    model_lock = lock["model"]
    try:
        snapshot = mr.snapshot_path(model_lock["id"], model_lock["revision"])
    except mr.SnapshotIncomplete as error:
        pilot.save(output / "STOP.json", {"gate": 1, "reason": str(error), "action": "no download attempted"})
        write_index(output)
        raise SystemExit(f"STOP: {error}")
    hashes = mr.snapshot_hashes(snapshot, model_lock["files_at_revision"])
    device = spec["device"] if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    pools, dropped, context_ids, answer_ids = mr.round1_pools(tokenizer, task)
    pool_record = mr.pools_record(pools, dropped, context_ids, answer_ids, task)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False,
                                             add_generation_prompt=True)
    gate3 = {"pools_kept": {c: len(v) for c, v in pools.items()}, "pools_dropped": dropped,
             "pools_equal_the_lock": json.loads(json.dumps(pool_record)) == lock.get("round1_entity_pools"),
             "chat_template_rendered_for_X": rendered, "dropped_prefix": rendered[:5]}
    gate3["passed"] = (gate3["pools_equal_the_lock"] and rendered[:5] == "<bos>"
                       and all(len(v) >= spec["n"] + 2 for v in pools.values()))
    manifest = {
        "schema_version": 2, "application": "gur-arieh-2510.06182", "round": 1,
        "stage": ("smoke " if smoke else "") + spec["name"],
        "label": ("SMOKE RUN: not a result" if smoke else
                  f"{spec['name'].upper()}: development data (protocol v2); nothing is frozen"),
        "authorized": "2026-09-28, by the user: split A and split B; no freeze, no confirmation",
        "spec": spec, "git_head": pilot.git("rev-parse", "HEAD"),
        "git_dirty": bool(pilot.git("status", "--porcelain", "--untracked-files=no")),
        "producers_sha256": {path: pilot.sha256(REPOSITORY / path) for path in PRODUCERS},
        "proposed_values_sha256": pilot.sha256(pilot.VALUES), "source_lock_sha256": pilot.sha256(pilot.LOCK),
        "protocol_sha256": pilot.sha256(APPLICATION / "SPLIT_A_B_PROTOCOL.md"),
        "upstream": {"path": str(Path(args.upstream).resolve()), "check": clone},
        "task_spec": task, "entity_pools": pool_record,
        "model": {"id": model_lock["id"], "revision": model_lock["revision"],
                  "snapshot": str(snapshot), "hashes": hashes},
        "gate3_tokens": gate3, "environment": pilot.environment(device),
    }
    pilot.save(output / "manifest.json", manifest)
    pilot.save(output / "RUN_STARTED.json", {"manifest_sha256": pilot.sha256(output / "manifest.json"),
                                            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    for gate, ok, reason in ((1, hashes["passed"], hashes["problems"]), (3, gate3["passed"], gate3)):
        if not ok:
            pilot.save(output / "STOP.json", {"gate": gate, "reason": reason})
            write_index(output)
            raise SystemExit(f"STOP at gate {gate}")

    timings = {}
    t0 = time.perf_counter()
    model, _ = mr.load_model(snapshot, torch.float32, device, spec["attention"])
    timings["load_fp32_mps_seconds"] = time.perf_counter() - t0
    runner = mr.Runner(model, tokenizer, task, pools, context_ids, answer_ids, spec["layer"],
                       diagnostic_layer=spec["diagnostic_layer"])
    audit = {}
    stored = []
    prefix = ("smoke-" if smoke else "") + spec["name"]
    handle = (output / "records.jsonl").open("x")

    def run_family(case_id, draw_index, seed, cell_key, cell):
        try:
            return runner.family(case_id=case_id, draw_index=draw_index, seed=seed, cell_key=cell_key, cell=cell,
                                 rng=random.Random(seed), n=spec["n"],
                                 audit=draw_index in spec["audit_full_logit_families"])
        except mr.TechnicalError as error:
            return {"case_id": case_id, "draw_index": draw_index, "seed": seed, "cell_key": cell_key,
                    "cell": cell, "qualifies": False,
                    "technical": {"passed": False, "failures": [str(error)], "checks": {}}}, None

    def on_record(record, full):
        if full is not None:
            audit[record["case_id"]] = full.numpy()
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        stored.append(record)
        print(f"{record['case_id']} {record['cell_key']} qualifies={record['qualifies']} "
              f"technical={record['technical']['passed']} {record.get('runtime_seconds', 0):.2f}s", flush=True)

    t0 = time.perf_counter()
    per_cell = args.split == "A"
    counts, met = generate_families(
        run_family, spec["cells"], seed_base=spec["seed_base"],
        quota=spec["qualifying_per_cell"] if per_cell else spec["qualifying"],
        cap=spec["cap_per_cell"] if per_cell else spec["cap"], per_cell=per_cell, prefix=prefix,
        on_record=on_record)
    handle.close()
    timings["families_fp32_mps_seconds"] = time.perf_counter() - t0
    timings["counts"], timings["quota_met"] = counts, met
    if audit:
        import numpy as np
        np.savez_compressed(output / "audit_full_logits.npz", **audit)
    if device == "mps":
        timings["mps_driver_gib_after_families"] = torch.mps.driver_allocated_memory() / 2 ** 30
    del runner, model
    if device == "mps":
        torch.mps.empty_cache()

    t0 = time.perf_counter()
    reference_model, _ = mr.load_model(snapshot, torch.float32, "cpu", spec["attention"])
    timings["load_fp32_cpu_seconds"] = time.perf_counter() - t0
    reference = mr.Runner(reference_model, tokenizer, task, pools, context_ids, answer_ids, spec["layer"])
    declared = set(spec["gate7_reference"]["families"])
    t0 = time.perf_counter()
    with (output / "gate7_cpu_reference.jsonl").open("x") as handle:
        for record in stored:
            if record["draw_index"] in declared and "matrix" in record:
                handle.write(json.dumps(reference.conflict_only(record), allow_nan=False) + "\n")
    timings["gate7_cpu_reference_seconds"] = time.perf_counter() - t0
    pilot.save(output / "timings.json", timings)
    write_index(output)
    print(json.dumps(timings, indent=2))


def decide(args):
    from mixing_splits import split_a_decision, split_b_decision
    output = Path(args.output)
    spec = json.loads((output / "manifest.json").read_text())["spec"]
    if args.split == "A":
        decision, name = split_a_decision(output, spec), "split_A_decision.json"
    else:
        decision, name = split_b_decision(output, spec), "split_B_decision.json"
    pilot.save(output / name, decision, exclusive=False)
    write_index(output)
    print(json.dumps({k: decision[k] for k in ("status", "stops")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=("A", "B"), required=True)
    r.add_argument("--upstream", default=os.environ.get("MIXING_MECHS_UPSTREAM"))
    r.add_argument("--output", required=True)
    r.add_argument("--smoke", type=int, default=None)
    d = sub.add_parser("decide")
    d.add_argument("--split", choices=("A", "B"), required=True)
    d.add_argument("--output", required=True)
    x = sub.add_parser("index")
    x.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not args.upstream:
            parser.exit(2, "pass --upstream or set MIXING_MECHS_UPSTREAM\n")
        run(args)
    elif args.command == "decide":
        decide(args)
    else:
        write_index(Path(args.output))


if __name__ == "__main__":
    main()
