"""Development, freeze, fresh execution and records-only audit for the Tracr case."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .design import (CANDIDATES, CONFIRMATION_SEED, DEV_SEED, Case, Family, Tolerance,
                     cumulative_retention, generate_families, predictions,
                     select_discriminator)

APP = Path(__file__).resolve().parents[2]
ROOT = APP.parents[1]
SCIENTIFIC_TOLERANCE = 0.01
NUMERIC_CAP = 0.01
N_DEV = 8
N_CONFIRM = 128


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    files = list((APP / "src").rglob("*.py")) + list((APP / "scripts").glob("*.py"))
    files += [ROOT / "src/causal_decidability/compatible_set.py", APP / "requirements.txt",
              APP / "requirements.lock.txt",
              APP / "DEVELOPMENT_PROTOCOL.md"]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}


def upstream_provenance():
    upstream = APP / "artifacts/upstream"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", "tracr/*.py"],
                                    cwd=upstream, text=True)
    from tracr.transformer import model as imported_model
    if Path(imported_model.__file__).resolve() != (upstream / "tracr/transformer/model.py").resolve():
        raise ValueError("Imported Tracr is not the pinned checkout")
    from .adapter import UPSTREAM_COMMIT
    if revision != UPSTREAM_COMMIT or dirty:
        raise ValueError("Upstream revision or source differs from the pinned input")
    return {"revision": revision, "python_sources": {
        str(p.relative_to(upstream)): sha(p) for p in sorted((upstream / "tracr").rglob("*.py"))}}


def environment():
    import jax
    return {"python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "jax_devices": [str(x) for x in jax.devices()],
            "jax_enable_x64": bool(jax.config.x64_enabled), "matmul_precision": "highest",
            "packages": dict(sorted((dist.metadata["Name"], dist.version)
                                    for dist in importlib.metadata.distributions()))}


def load_family(value):
    def case(v):
        return Case(**{k: tuple(v[k]) if k in ("recipient", "donor") else v[k]
                       for k in ("case_id", "family_id", "stage", "recipient", "donor",
                                 "source_query", "target_query")})
    result = Family(value["family_id"], value["split"], value["seed"], case(value["shared"]),
                    tuple(case(v) for v in value["discriminators"]),
                    tuple(case(v) for v in value["followups"]))
    if result.signature != value["signature"]:
        raise ValueError("Family content signature mismatch")
    return result


def max_abs(value):
    return float(np.max(np.abs(value)))


def check_arrays(arrays, *, dtype, target, coordinates, recipient, donor,
                 source_position, site_layer, site_timing):
    """Recomputable checks on stored tensors; no trust in a producer's PASS flag."""
    eps = np.finfo(dtype).eps
    native, noop, self_scores, patched = (arrays[key] for key in
                                          ("native_scores", "noop_scores", "self_scores", "patch_scores"))
    before, after = arrays["before"], arrays["after"]
    if site_timing == "before_unembedding":
        expected_before, donor_site = arrays["native_states"][-1], arrays["donor_states"][-1]
    elif site_timing == "before_attention":
        expected_before = (arrays["native_embeddings"] if site_layer == 0 else
                           arrays["native_states"][2 * site_layer - 1])
        donor_site = (arrays["donor_embeddings"] if site_layer == 0 else
                      arrays["donor_states"][2 * site_layer - 1])
    else:
        raise ValueError("Unknown site timing")
    mask = np.zeros(before.shape, dtype=bool)
    mask[0, target, coordinates] = True
    output_mask = np.ones(native.shape, dtype=bool)
    output_mask[0, target, :] = False
    score_tol = 64 * eps * max(1., max_abs(native))
    state_tol = 64 * eps * max(1., max_abs(arrays["native_states"]))
    diagnostics = {
        "noop_score_error": max_abs(native - noop),
        "noop_state_error": max_abs(arrays["native_states"] - arrays["noop_states"]),
        "self_score_error": max_abs(native - self_scores),
        "self_state_error": max_abs(arrays["native_states"] - arrays["self_states"]),
        "non_target_output_error": max_abs((native - patched)[output_mask]),
        "insert_error": max_abs(after[0, target, coordinates] - arrays["donor_values"].astype(dtype)),
        "complement_error": max_abs((before - after)[~mask]),
        "site_change": max_abs(before - after), "score_tolerance": float(score_tol),
        "state_tolerance": float(state_tol),
        "native_site_error": max_abs(before - expected_before),
        "donor_source_error": max_abs(arrays["donor_values"] - donor_site[0, source_position, coordinates]),
    }
    gates = {
        "finite": all(np.isfinite(v).all() for v in arrays.values()),
        "native_recipient_correct": np.argmax(native[0, 1:], axis=-1).tolist() == list(reversed(recipient)),
        "native_donor_correct": np.argmax(arrays["donor_scores"][0, 1:], axis=-1).tolist() == list(reversed(donor)),
        "single_site": int(arrays["count"]) == 1,
        "noop_scores": diagnostics["noop_score_error"] <= score_tol,
        "noop_states": diagnostics["noop_state_error"] <= state_tol,
        "self_scores": diagnostics["self_score_error"] <= score_tol,
        "self_states": diagnostics["self_state_error"] <= state_tol,
        "replacement_exact_after_cast": diagnostics["insert_error"] == 0,
        "complement_exact": diagnostics["complement_error"] == 0,
        "other_output_positions_unchanged": diagnostics["non_target_output_error"] <= score_tol,
        "nontrivial_intervention": diagnostics["site_change"] > state_tol,
        "site_matches_native": diagnostics["native_site_error"] <= state_tol,
        "donor_source_exact": diagnostics["donor_source_error"] == 0,
    }
    return {"gates": {k: bool(v) for k, v in gates.items()}, "diagnostics": diagnostics,
            "passed": bool(all(gates.values()))}


class Recorder:
    def __init__(self, adapter, output, numeric_allowance):
        self.adapter, self.output = adapter, output
        self.numeric_allowance = numeric_allowance
        self.records, self.arrays, self.native_cache = [], {}, {}
        self.forward_counts = {"native": 0, "noop": 0, "self_patch": 0, "patch": 0}

    def native(self, tokens, dtype):
        key = (tuple(tokens), dtype)
        if key not in self.native_cache:
            self.native_cache[key] = self.adapter.native_forward(tokens, dtype)
            self.forward_counts["native"] += 1
        return self.native_cache[key]

    def measure(self, case, site):
        pair, scores = [], {}
        for dtype in ("float32", "float64"):
            native, donor = self.native(case.recipient, dtype), self.native(case.donor, dtype)
            noop = self.adapter.instrumented_forward(case.recipient, site=site, dtype=dtype)
            own = self.adapter.donor_values(native, case.target_query, site)
            identity = self.adapter.instrumented_forward(case.recipient, site=site, dtype=dtype,
                                                        target_query=case.target_query, replacement=own)
            values = self.adapter.donor_values(donor, case.source_query, site)
            patched = self.adapter.instrumented_forward(case.recipient, site=site, dtype=dtype,
                                                       target_query=case.target_query, replacement=values)
            for key in ("noop", "self_patch", "patch"):
                self.forward_counts[key] += 1
            arrays = {"native_scores": native["scores"], "donor_scores": donor["scores"],
                      "noop_scores": noop["scores"], "self_scores": identity["scores"],
                      "patch_scores": patched["scores"], "donor_values": values,
                      "native_states": np.stack(native["residuals"] + [native["output"]]),
                      "donor_states": np.stack(donor["residuals"] + [donor["output"]]),
                      "native_embeddings": native["embeddings"], "donor_embeddings": donor["embeddings"],
                      "noop_states": np.stack(noop["residuals"] + [noop["output"]]),
                      "self_states": np.stack(identity["residuals"] + [identity["output"]]),
                      **patched["captured"]}
            record_id = f"r{len(self.records):06d}"
            coords = self.adapter.address_indices if site == "site_A" else self.adapter.output_indices
            checks = check_arrays(arrays, dtype=dtype, target=case.target_query + 1,
                                  coordinates=list(coords), recipient=case.recipient, donor=case.donor,
                                  source_position=case.source_query + 1,
                                  site_layer=self.adapter.layer if site == "site_A" else self.adapter.native.model_config.num_layers,
                                  site_timing="before_attention" if site == "site_A" else "before_unembedding")
            row = {"record_id": record_id, "case": case.to_dict(), "site": site, "dtype": dtype,
                   "state_dtype": str(arrays["before"].dtype), "score_dtype": str(arrays["patch_scores"].dtype),
                   "coordinates": list(coords), "arrays": sorted(arrays), "checks": checks,
                   "target_scores": patched["scores"][0, case.target_query + 1].tolist()}
            self.records.append(row)
            self.arrays.update({f"{record_id}__{k}": v for k, v in arrays.items()})
            pair.append(row)
            scores[dtype] = arrays["patch_scores"]
        numeric_error = max(max_abs(scores["float32"] - scores["float64"]),
                            max_abs(self.native(case.recipient, "float32")["scores"] -
                                    self.native(case.recipient, "float64")["scores"]),
                            max_abs(self.native(case.donor, "float32")["scores"] -
                                    self.native(case.donor, "float64")["scores"]))
        for row in pair:
            row["paired_precision_error"] = numeric_error
            row["precision_passed"] = numeric_error <= self.numeric_allowance
        qualified = all(r["checks"]["passed"] and r["precision_passed"] for r in pair)
        # Raw records are checkpointed after each condition; interrupted data are retained.
        with (self.output / "records.jsonl").open("a") as handle:
            for row in pair:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        return pair[-1]["target_scores"], qualified, pair

    def save(self):
        np.savez_compressed(self.output / "tensors.npz", **self.arrays)


def run_family(recorder, family, site, tolerance):
    cases, values, measurements = [], [], []
    observations_by_dtype = {"float32": [], "float64": []}

    def add(case):
        value, qualified, pair = recorder.measure(case, site)
        cases.append(case)
        values.append(value)
        measurements.extend(row["record_id"] for row in pair)
        for row in pair:
            observations_by_dtype[row["dtype"]].append(row["target_scores"])
        return qualified

    if not add(family.shared):
        return {"family_id": family.family_id, "site": site, "success": False,
                "status": "qualification_failed", "records": measurements}
    trajectory = cumulative_retention(cases, values, tolerance, controls_passed=True)
    selection = select_discriminator(family, trajectory[-1]["retained_after"], tolerance)
    if selection["selected_case_id"] is None:
        return {"family_id": family.family_id, "site": site, "success": False,
                "status": "no_discriminator", "trajectory": trajectory, "selection": selection,
                "records": measurements}
    chosen = next(c for c in family.discriminators if c.case_id == selection["selected_case_id"])
    if not add(chosen):
        return {"family_id": family.family_id, "site": site, "success": False,
                "status": "qualification_failed", "records": measurements, "selection": selection}
    trajectory = cumulative_retention(cases, values, tolerance, controls_passed=True)
    # Final row is required even if neither/ both remain: it cannot rescue earlier exclusion.
    if not add(family.followup_for(chosen.case_id)):
        return {"family_id": family.family_id, "site": site, "success": False,
                "status": "qualification_failed", "records": measurements, "selection": selection}
    trajectories = {dtype: cumulative_retention(cases, observations, tolerance, controls_passed=True)
                    for dtype, observations in observations_by_dtype.items()}
    trajectory = trajectories["float64"]
    expected = "address" if site == "site_A" else "donor_answer"
    pattern = [sorted(CANDIDATES), [expected], [expected]]
    passed = all([step["retained_after"] for step in t] == pattern for t in trajectories.values())
    passed &= all(not step["native_null_compatible"] for t in trajectories.values() for step in t)
    return {"family_id": family.family_id, "site": site, "success": bool(passed),
            "status": trajectory[-1]["outcome"], "trajectory": trajectory,
            "float32_trajectory": trajectories["float32"], "selection": selection,
            "records": measurements, "expected_control_label_for_scoring_only": expected}


def execute(output, families, *, phase, allowance, freeze_path=None):
    from .adapter import ReverseAdapter
    if output.exists():
        raise ValueError("Refusing to overwrite a run; use a new output path")
    output.mkdir(parents=True)
    started = time.monotonic()
    adapter = ReverseAdapter()
    manifest = {"phase": phase, "started_at": timestamp(), "source_hashes": source_hashes(),
                "upstream": upstream_provenance(), "environment": environment(),
                "model": adapter.metadata(), "numeric_allowance": allowance,
                "scientific_tolerance": SCIENTIFIC_TOLERANCE,
                "families": [f.to_dict() for f in families],
                "freeze_sha256": sha(freeze_path) if freeze_path else None}
    if freeze_path:
        frozen = read(freeze_path)
        for field in ("families", "numeric_allowance", "scientific_tolerance"):
            if json.loads(json.dumps(manifest[field])) != frozen[field]:
                raise ValueError(f"Execution does not match frozen {field}")
        if len(families) != frozen["sample_size"]:
            raise ValueError("Execution sample size differs from freeze")
        if manifest["source_hashes"] != frozen["source_hashes"]:
            raise ValueError("Executable surface changed after freeze")
        if manifest["model"] != frozen["model"]:
            # JSON turns tuples into lists, compare canonically.
            if json.loads(json.dumps(manifest["model"])) != frozen["model"]:
                raise ValueError("Model/parameter/site provenance changed after freeze")
        if manifest["upstream"] != frozen["upstream"]:
            raise ValueError("Upstream sources changed after freeze")
        if manifest["environment"]["packages"] != frozen["packages"]:
            raise ValueError("Numerical dependencies changed after freeze")
    dump(output / "manifest.json", manifest)
    recorder = Recorder(adapter, output, allowance)
    tolerance = Tolerance(SCIENTIFIC_TOLERANCE, allowance)
    trajectories = []
    try:
        for index, family in enumerate(families):
            trajectories.append(run_family(recorder, family, "site_A", tolerance))
            if phase == "development":
                trajectories.append(run_family(recorder, family, "site_B", tolerance))
            if (index + 1) % 16 == 0 or index + 1 == len(families):
                print(f"{phase}: {index + 1}/{len(families)} families", flush=True)
    finally:
        recorder.save()
        dump(output / "trajectories.json", trajectories)
    precision_error = max(r["paired_precision_error"] for r in recorder.records)
    all_qualified = all(r["checks"]["passed"] and r["precision_passed"] for r in recorder.records)
    summary = {"phase": phase, "family_count": len(families),
               "site_A_successes": sum(t["success"] for t in trajectories if t["site"] == "site_A"),
               "site_B_positive_controls": sum(t["success"] for t in trajectories if t["site"] == "site_B"),
               "all_qualified": all_qualified, "paired_precision_max_error": precision_error,
               "record_count": len(recorder.records), "forward_counts": recorder.forward_counts,
               "elapsed_seconds": time.monotonic() - started,
               "hashes": {name: sha(output / name) for name in
                          ("manifest.json", "records.jsonl", "tensors.npz", "trajectories.json")}}
    if phase == "development":
        summary["proposed_numeric_allowance"] = max(64 * np.finfo("float32").eps, 4 * precision_error)
        summary["confirmation_start_eligible"] = bool(all_qualified and
            summary["proposed_numeric_allowance"] <= NUMERIC_CAP and
            summary["site_A_successes"] == len(families) and
            summary["site_B_positive_controls"] == len(families))
    else:
        from .statistics import evaluate_confirmation
        summary["population_decision"] = evaluate_confirmation(
            summary["site_A_successes"], len(families), population_size=frozen["population_size"],
            planned_sample_size=N_CONFIRM, alpha=0.05, minimum_success_rate=0.95)
        summary["mechanistic_claim_qualified"] = bool(all_qualified)
    dump(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def freeze(development, destination):
    from .statistics import plan_confirmation
    dev, manifest = read(development / "summary.json"), read(development / "manifest.json")
    if not dev["confirmation_start_eligible"] or destination.exists():
        raise ValueError("Development not qualified or freeze already exists")
    for name, value in dev["hashes"].items():
        if sha(development / name) != value:
            raise ValueError("Development artifact changed")
    if source_hashes() != manifest["source_hashes"]:
        raise ValueError("Rerun development after executable changes")
    subprocess.run([sys.executable, str(APP / "scripts/verify.py"), str(development)],
                   check=True, capture_output=True, text=True)
    old = tuple(load_family(f) for f in manifest["families"])
    fresh = generate_families(N_CONFIRM, seed=CONFIRMATION_SEED, split="confirmation", excluded_families=old)
    old_cases = {case.signature for f in old for case in (f.shared, *f.discriminators, *f.followups)}
    old_cases.add(((0, 1, 2, 3), (4, 5, 6, 7), 1, 0))  # disclosed development smoke
    new_cases = {case.signature for f in fresh for case in (f.shared, *f.discriminators, *f.followups)}
    if old_cases & new_cases:
        raise ValueError("Fresh manifest overlaps previously measured cases; no silent redraw")
    population_size = 119750400 - len({f.signature for f in old})
    value = {"status": "frozen_before_confirmation_outcomes", "created_at": timestamp(),
             "development_summary_sha256": sha(development / "summary.json"),
             "source_hashes": source_hashes(), "upstream": manifest["upstream"],
             "model": manifest["model"], "packages": manifest["environment"]["packages"],
             "numeric_allowance": dev["proposed_numeric_allowance"],
             "scientific_tolerance": SCIENTIFIC_TOLERANCE,
             "sample_size": N_CONFIRM, "seed": CONFIRMATION_SEED,
             "population_size": population_size,
             "precision_plan": plan_confirmation(population_size, sample_size=N_CONFIRM,
                                                   alpha=0.05, minimum_success_rate=0.95,
                                                   planning_success_rate=0.995),
             "families": [f.to_dict() for f in fresh],
             "stop_rule": "One final population decision; no extension/replacement; technical failure blocks claim",
             "scope": "Within one known compiled reversal circuit; no adaptive-policy superiority"}
    dump(destination, value)
    print(json.dumps({k: value[k] for k in ("status", "sample_size", "numeric_allowance", "precision_plan")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("develop")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("confirm")
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "develop":
        families = generate_families(N_DEV, seed=DEV_SEED, split="development")
        execute(args.out, families, phase="development", allowance=NUMERIC_CAP)
    elif args.command == "freeze":
        freeze(args.development, args.out)
    else:
        frozen = read(args.freeze)
        if frozen["status"] != "frozen_before_confirmation_outcomes" or source_hashes() != frozen["source_hashes"]:
            raise ValueError("Unfrozen or modified experiment")
        frozen_relative = args.freeze.resolve().relative_to(ROOT)
        committed = subprocess.check_output(["git", "show", f"HEAD:{frozen_relative}"], cwd=ROOT)
        if committed != args.freeze.read_bytes():
            raise ValueError("Freeze must be committed before the first confirmation outcome")
        execute(args.out, tuple(load_family(f) for f in frozen["families"]), phase="confirmation",
                allowance=frozen["numeric_allowance"], freeze_path=args.freeze)


if __name__ == "__main__":
    main()
