"""DIAGNOSTIC for the dtype decision (pilot/development, descriptive; decides nothing).

Pilot-1 (protocol v1) producer of results/pilot/diagnostic_fp32_*. It needs the v1
runner (``single_token_pools``, in-context primary fields) of commit 3cdaffd and refuses
to run against the protocol-v2 runner; check out that commit to rerun it.

What would the pilot look like with float32 as the execution dtype? Offline, cached
snapshot only. It reruns the first 32 pilot families (the gate-7 sample: same seeds,
same cells, same code path ``Runner.family``) with the model in float32 on MPS, and
reports:

- runtime per family and memory (MPS allocations and the process's peak resident size);
- |T(bfloat16) - T(float32)| for the conflict patch, the agreement patches and the
  no-patch runs, under the declared in-context readout and, where all seven answer forms
  are single tokens, under the answer-form readout (logits of 'Country'-style tokens);
- reproducibility: the float32 family rerun against the stored float32 reference;
- the residual that a gate would still see with float32 execution: float32 on MPS
  against float32 on CPU, for the conflict patch of the first 8 families.

    python scripts/diagnose_fp32_execution.py --results results/pilot --upstream /path/to/clone
"""
import argparse
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
APPLICATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APPLICATION / "src"))

FAMILIES, CPU_FAMILIES, S_MIN = 32, 8, 0.10


def gib(x):
    return x / 2 ** 30


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--upstream", default=os.environ.get("MIXING_MECHS_UPSTREAM"))
    args = parser.parse_args()
    import torch
    import mixing_runner as mr
    import mixing_round1_analysis as ra
    from mixing_prompts import load_adapter
    if not hasattr(mr, "single_token_pools"):
        raise SystemExit("pilot-1 producer: it needs the protocol-v1 runner; check out commit 3cdaffd to rerun it")

    results = Path(args.results)
    manifest = json.loads((results / "manifest.json").read_text())
    records = {json.loads(l)["case_id"]: json.loads(l) for l in (results / "records.jsonl").read_text().splitlines()}
    reference = {json.loads(l)["case_id"]: json.loads(l)
                 for l in (results / "fp32_reference.jsonl").read_text().splitlines()}
    pilot = manifest["pilot"]
    adapter = load_adapter()
    adapter.check(args.upstream, json.loads((APPLICATION / "SOURCE_LOCK.json").read_text()))
    spec = adapter.schema_spec(args.upstream, pilot["task"])
    snapshot = mr.snapshot_path(manifest["model"]["id"], manifest["model"]["revision"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    pools, _, ids = mr.single_token_pools(tokenizer, spec)
    if pools != manifest["single_token_pools"]:
        raise SystemExit("pools differ from the pilot manifest")

    memory = {}
    t0 = time.perf_counter()
    model, _ = mr.load_model(snapshot, torch.float32, "mps", pilot["attention"])
    memory["load_seconds"] = time.perf_counter() - t0
    memory["after_load_mps_current_gib"] = gib(torch.mps.current_allocated_memory())
    memory["after_load_mps_driver_gib"] = gib(torch.mps.driver_allocated_memory())
    runner = mr.Runner(model, tokenizer, spec, pools, ids, pilot["layer"],
                       diagnostic_layer=pilot["diagnostic_layer"])
    fp32_records, runtimes, driver_peak = [], [], 0.0
    out_path = results / "diagnostic_fp32_records.jsonl"
    with out_path.open("w") as handle:
        for i in range(FAMILIES):
            key, cell = pilot["cells"][i % len(pilot["cells"])]
            seed = pilot["seed_base"] + i
            record, _ = runner.family(case_id=f"pilot-{i:04d}", draw_index=i, seed=seed, cell_key=key,
                                      cell=cell, rng=random.Random(seed), n=pilot["n"])
            runtimes.append(record["runtime_seconds"])
            driver_peak = max(driver_peak, gib(torch.mps.driver_allocated_memory()))
            fp32_records.append(record)
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(f"{record['case_id']} {record['runtime_seconds']:.2f}s", flush=True)
    memory["max_mps_driver_gib_after_a_family"] = driver_peak
    memory["process_peak_rss_gib"] = gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    memory["machine_memory_gib"] = gib(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))

    def T(logits, cell):
        return ra.case_measures(logits, cell, ra.CONTRACT["w"], S_MIN)

    rows = {"declared_in_context": [], "answer_form": []}
    same_matrix = True
    for fp in fp32_records:
        bf = records[fp["case_id"]]
        same_matrix &= bf["matrix"] == fp["matrix"]
        cell = bf["cell"]
        pairs = [("conflict_layer18", bf["readout"], fp["readout"]),
                 ("no_patch", bf["native"]["recipient"]["readout"], fp["native"]["recipient"]["readout"])]
        pairs += [(f"agreement_{a['target']}", a["readout"], b["readout"])
                  for a, b in zip(bf["agreement"], fp["agreement"])]
        for run, x, y in pairs:
            for which, get in (("declared_in_context", lambda r: r["entity_logits"]),
                               ("answer_form", lambda r: r["answer_form_logits"]["capitalized"])):
                a, b = get(x), get(y)
                if any(v is None for v in a + b):
                    continue
                ma, mb = T(a, cell), T(b, cell)
                rows[which].append({"case_id": fp["case_id"], "run": run, "qualifies": bf["qualifies"],
                                    "T_bf16": ma["T"], "T_fp32": mb["T"],
                                    "abs_T_difference": abs(ma["T"] - mb["T"]),
                                    "same_resolution": ma["resolved"] == mb["resolved"],
                                    "same_labels": ma["labels"] == mb["labels"]})

    def describe(values, runs=None):
        chosen = [r for r in values if runs is None or r["run"] in runs]
        if not chosen:
            return None
        diffs = sorted(r["abs_T_difference"] for r in chosen)
        return {"runs": len(chosen), "max_abs_T_difference": diffs[-1],
                "median_abs_T_difference": diffs[len(diffs) // 2],
                "above_0.01": sum(d > 0.01 for d in diffs),
                "same_resolution": sum(r["same_resolution"] for r in chosen),
                "same_labels": sum(r["same_labels"] for r in chosen)}

    comparison = {which: {"conflict_layer18": describe(v, {"conflict_layer18"}),
                          "no_patch": describe(v, {"no_patch"}),
                          "agreement": describe(v, {"agreement_i_P", "agreement_i_L", "agreement_i_R"})}
                  for which, v in rows.items()}
    reproducible = [fp["entity_logits"] == reference[fp["case_id"]]["conflict"] for fp in fp32_records
                    if fp["case_id"] in reference]

    del runner, model
    torch.mps.empty_cache()
    cpu_model, _ = mr.load_model(snapshot, torch.float32, "cpu", pilot["attention"])
    cpu = mr.Runner(cpu_model, tokenizer, spec, pools, ids, pilot["layer"])
    cpu_rows = []
    for fp in fp32_records[:CPU_FAMILIES]:
        cell = fp["cell"]
        G = [tuple(g) for g in fp["matrix"]]
        donor = [tuple(g) for g in fp["donor_matrix"]]
        start = time.perf_counter()
        rec = cpu.prompt(G, cell["i_N"])
        rec_ids, _ = cpu.aligned_entity_ids(G, rec["input_ids"])
        don_out, _ = cpu.forward(cpu.prompt(donor, cell["i_P"])["input_ids"])
        out, _ = cpu.forward(rec["input_ids"], patch=(pilot["layer"], don_out.hidden_states[pilot["layer"]][0, -1]))
        readout = cpu.readout(out, rec_ids, entities=[g[1] for g in G])
        row = {"case_id": fp["case_id"], "seconds_cpu_two_forwards": time.perf_counter() - start}
        for which, get in (("declared_in_context", lambda r: r["entity_logits"]),
                           ("answer_form", lambda r: r["answer_form_logits"]["capitalized"])):
            a, b = get(fp["readout"]), get(readout)
            if any(v is None for v in a + b):
                row[which] = None
                continue
            row[which] = {"abs_T_difference_mps_vs_cpu": abs(T(a, cell)["T"] - T(b, cell)["T"]),
                          "max_abs_logit_difference": max(abs(x - y) for x, y in zip(a, b))}
        cpu_rows.append(row)

    out = {
        "label": "DIAGNOSTIC, pilot/development, descriptive; decides nothing",
        "families": FAMILIES, "same_families_as_pilot": same_matrix,
        "s_min_used": S_MIN, "s_min_note": "the pilot's s_min was the 0.10 floor in every cell for both readouts",
        "fp32_mps_runtime_per_family_seconds": {
            "mean": sum(runtimes) / len(runtimes), "median": sorted(runtimes)[len(runtimes) // 2],
            "max": max(runtimes)},
        "bf16_mps_runtime_per_family_seconds_same_families": {
            "mean": sum(records[r["case_id"]]["runtime_seconds"] for r in fp32_records) / len(fp32_records)},
        "memory": memory,
        "bf16_vs_fp32_on_mps": comparison,
        "fp32_family_rerun_equals_stored_fp32_reference": {"equal": sum(reproducible), "of": len(reproducible)},
        "fp32_mps_vs_fp32_cpu_conflict_layer18": cpu_rows,
        "rows": rows,
    }
    (results / "diagnostic_fp32_execution.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: out[k] for k in ("fp32_mps_runtime_per_family_seconds", "memory",
                                         "bf16_vs_fp32_on_mps",
                                         "fp32_family_rerun_equals_stored_fp32_reference")}, indent=2))


if __name__ == "__main__":
    main()
