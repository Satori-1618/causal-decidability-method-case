"""Adversarial runner/verifier tests on the declared development surface only."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from tracr_demo import experiment
from tracr_demo.design import DEV_SEED, Tolerance, generate_families, predictions


@pytest.fixture(scope="module")
def verifier():
    script = Path(experiment.APP) / "scripts/verify.py"
    spec = importlib.util.spec_from_file_location("tracr_test_records_verifier", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def development_run(tmp_path_factory):
    # Reuse precisely the already declared DEV_SEED surface. Never draw a new
    # confirmation family or exercise freeze/confirm from these tests.
    output = tmp_path_factory.mktemp("tracr-development") / "run"
    families = generate_families(experiment.N_DEV, seed=DEV_SEED, split="development")
    experiment.execute(output, families, phase="development", allowance=experiment.NUMERIC_CAP)
    return output


@pytest.fixture
def run_copy(development_run, tmp_path):
    output = tmp_path / "run"
    shutil.copytree(development_run, output)
    return output


def _rows(run):
    return [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]


def _write_rows(run, rows):
    (run / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def _rehash(run, *names):
    summary = experiment.read(run / "summary.json")
    for name in names:
        summary["hashes"][name] = experiment.sha(run / name)
    experiment.dump(run / "summary.json", summary)


def _check_row(run):
    row = next(row for row in _rows(run) if row["site"] == "site_A" and row["dtype"] == "float64")
    with np.load(run / "tensors.npz", allow_pickle=False) as archive:
        arrays = {name: archive[f"{row['record_id']}__{name}"].copy() for name in row["arrays"]}
    site = experiment.read(run / "manifest.json")["model"]["sites"][row["site"]]
    case = row["case"]
    kwargs = dict(dtype=row["dtype"], target=case["target_model_position"],
                  coordinates=row["coordinates"], recipient=case["recipient"], donor=case["donor"],
                  source_position=case["source_model_position"], site_layer=site["layer"],
                  site_timing=site["timing"])
    return arrays, kwargs


def test_unmodified_development_artifact_verifies_without_forward_calls(development_run, verifier, monkeypatch):
    from tracr_demo.adapter import ReverseAdapter

    def refuse_forward(*args, **kwargs):
        raise AssertionError("Records-only verification must never run a model")

    monkeypatch.setattr(ReverseAdapter, "native_forward", refuse_forward)
    monkeypatch.setattr(ReverseAdapter, "instrumented_forward", refuse_forward)
    result = verifier.verify(development_run)
    assert result["status"] == "PASS"
    assert result["model_forwards"] == 0
    assert result["families"] == experiment.N_DEV


def test_verifier_rejects_changed_record_hash(run_copy, verifier):
    with (run_copy / "records.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Changed artifact"):
        verifier.verify(run_copy)


@pytest.mark.parametrize("artifact", (
    "manifest.json", "records.jsonl", "tensors.npz", "trajectories.json",
))
def test_verifier_requires_every_artifact_digest(run_copy, verifier, artifact):
    summary = experiment.read(run_copy / "summary.json")
    del summary["hashes"][artifact]
    experiment.dump(run_copy / "summary.json", summary)
    with pytest.raises(ValueError):
        verifier.verify(run_copy)


def test_verifier_rejects_readout_tamper_even_after_rehash(run_copy, verifier):
    rows = _rows(run_copy)
    rows[0]["target_scores"][0] += 0.25
    _write_rows(run_copy, rows)
    _rehash(run_copy, "records.jsonl")
    with pytest.raises(ValueError, match="readout differs"):
        verifier.verify(run_copy)


def test_verifier_rejects_current_source_mismatch(development_run, verifier, monkeypatch):
    monkeypatch.setattr(verifier, "source_hashes", lambda: {"changed_source.py": "0" * 64})
    with pytest.raises(ValueError, match="source"):
        verifier.verify(development_run)


def test_verifier_rejects_duplicate_trajectory_even_after_rehash(run_copy, verifier):
    trajectories = experiment.read(run_copy / "trajectories.json")
    trajectories.append(trajectories[0])
    experiment.dump(run_copy / "trajectories.json", trajectories)
    _rehash(run_copy, "trajectories.json")
    with pytest.raises(ValueError, match="trajectory"):
        verifier.verify(run_copy)


def test_verifier_rejects_orphan_precision_record_even_after_rehash(run_copy, verifier):
    rows = _rows(run_copy)
    removed = next(row for row in rows if row["dtype"] == "float64")
    rows.remove(removed)
    _write_rows(run_copy, rows)
    with np.load(run_copy / "tensors.npz", allow_pickle=False) as archive:
        tensors = {name: archive[name] for name in archive.files
                   if not name.startswith(removed["record_id"] + "__")}
    np.savez_compressed(run_copy / "tensors.npz", **tensors)
    summary = experiment.read(run_copy / "summary.json")
    summary["record_count"] = len(rows)
    experiment.dump(run_copy / "summary.json", summary)
    _rehash(run_copy, "records.jsonl", "tensors.npz")
    with pytest.raises(ValueError, match="fp32.*fp64"):
        verifier.verify(run_copy)


@pytest.mark.parametrize("field,value", (
    ("family_count", -1),
    ("site_B_positive_controls", -1),
    ("confirmation_start_eligible", False),
))
def test_verifier_recomputes_development_summary_flags(run_copy, verifier, field, value):
    summary = experiment.read(run_copy / "summary.json")
    assert summary[field] != value
    summary[field] = value
    experiment.dump(run_copy / "summary.json", summary)
    with pytest.raises(ValueError):
        verifier.verify(run_copy)


def test_qualification_binds_payload_to_declared_donor_source(development_run):
    arrays, kwargs = _check_row(development_run)
    assert experiment.check_arrays(arrays, **kwargs)["passed"]
    arrays["donor_values"] = np.roll(arrays["donor_values"], 1)
    arrays["after"][0, kwargs["target"], kwargs["coordinates"]] = arrays["donor_values"]
    checks = experiment.check_arrays(arrays, **kwargs)
    assert checks["gates"]["replacement_exact_after_cast"]
    assert checks["gates"]["complement_exact"]
    assert not checks["gates"]["donor_source_exact"]
    assert not checks["passed"]


def test_qualification_binds_before_tensor_to_native_site(development_run):
    arrays, kwargs = _check_row(development_run)
    assert experiment.check_arrays(arrays, **kwargs)["passed"]
    coordinate = next(index for index in range(arrays["before"].shape[-1])
                      if index not in kwargs["coordinates"])
    arrays["before"][0, 0, coordinate] += 1
    arrays["after"][0, 0, coordinate] += 1
    checks = experiment.check_arrays(arrays, **kwargs)
    assert checks["gates"]["replacement_exact_after_cast"]
    assert checks["gates"]["complement_exact"]
    assert not checks["gates"]["site_matches_native"]
    assert not checks["passed"]


class SymbolicRecorder:
    def __init__(self, fail_at=None, wrong_followup=False):
        self.fail_at = fail_at
        self.wrong_followup = wrong_followup
        self.calls = []

    def measure(self, case, site):
        index = len(self.calls)
        self.calls.append(case.case_id)
        candidate = "address" if site == "site_A" else "donor_answer"
        values = list(predictions(case)[candidate])
        if self.wrong_followup and case.stage == "followup":
            values = [0.0] * len(values)
        pair = [dict(record_id=f"r{index}_{dtype}", dtype=dtype, target_scores=values)
                for dtype in ("float32", "float64")]
        return values, index != self.fail_at, pair


@pytest.mark.parametrize("fail_at", (0, 1, 2))
def test_failed_controls_stop_before_dependent_classification(monkeypatch, fail_at):
    family = generate_families(1, seed=DEV_SEED, split="development")[0]
    recorder = SymbolicRecorder(fail_at=fail_at)
    real_classifier = experiment.cumulative_retention
    classified_lengths = []

    def guarded_classifier(cases, *args, **kwargs):
        classified_lengths.append(len(cases))
        assert len(cases) <= fail_at
        return real_classifier(cases, *args, **kwargs)

    monkeypatch.setattr(experiment, "cumulative_retention", guarded_classifier)
    outcome = experiment.run_family(recorder, family, "site_A", Tolerance(0.01, 0.001))
    assert not outcome["success"]
    assert outcome["status"] == "qualification_failed"
    assert len(recorder.calls) == fail_at + 1
    assert classified_lengths == list(range(1, fail_at + 1))


def test_shared_and_discriminator_success_cannot_replace_failed_final_prediction():
    family = generate_families(1, seed=DEV_SEED, split="development")[0]
    recorder = SymbolicRecorder(wrong_followup=True)
    outcome = experiment.run_family(recorder, family, "site_A", Tolerance(0.01, 0.001))
    assert len(recorder.calls) == 3
    assert not outcome["success"]
    assert outcome["status"] == "no_candidate_fits"
    assert outcome["trajectory"][-1]["retained_after"] == []
