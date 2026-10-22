"""Adversarial tests for the independent records-only verifier; no model import."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_value_verifier", HERE / "verify.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))


def snapshot(dtype, recipient, position, donor):
    direction = -1.0 if donor["balance"] < 0 else 1.0
    result = {"dtype": "torch." + dtype, "recipient_string": recipient,
              "donor_string": donor["string"], "recipient_position": position,
              "donor_position": donor["position"], "native_margin": .25,
              "donor_native_margin": .25, "patched_margin": 0. if direction < 0 else .5,
              "margin_change": -.25 if direction < 0 else .25, "a_r": .125,
              "value_difference_l2": 4., "node_delta_l2": .5,
              "construction_roundoff_linf": 0., "v_r": [0.]*16,
              "v_d": [direction]*16, "h_r": [0.]*16}
    for field in ("h_patch_intended", "h_patch_delivered", "requested_node_delta", "delivered_node_delta"):
        result[field] = [.125*direction]*16
    return result


def discrepancies(row):
    result = {}
    for candidate in v.CANDIDATES:
        for cell in v.TARGETS:
            first = ((candidate == "H_state" and cell.startswith("neg_"))
                     or (candidate == "H_position" and cell.split("_")[1] == "20"))
            anchor = v.ANCHORS[0 if first else 1]
            x = row["cells"]
            e32 = x[cell]["float32"]["patched_margin"] - x[anchor]["float32"]["patched_margin"]
            e64 = x[cell]["float64"]["patched_margin"] - x[anchor]["float64"]["patched_margin"]
            result[candidate + ":" + cell] = abs(e32-e64)
    return result


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.run = self.root / "run"
        self.inputs = self.root / "inputs"
        self.run.mkdir()
        self.inputs.mkdir()
        for name in ("cases.jsonl", "preparation.json", "exclusions.json"):
            (self.inputs / name).write_bytes((HERE / "inputs/development_001" / name).read_bytes())
        self.cases = self.inputs / "cases.jsonl"
        self.families = v.read_jsonl(self.cases)
        self.rows, self.anchors, self.selections = [], [], []
        for family in self.families:
            j = family["recipient"].index(")") + 1
            cells = {d["cell"]: {"donor_metadata": d,
                     **{tag: snapshot(tag, family["recipient"], j, d) for tag in ("float32", "float64")}}
                     for d in family["donors"]}
            predicted = v.forecast({a: cells[a]["float64"]["patched_margin"] for a in v.ANCHORS})
            row = {"family_id": family["family_id"], "phase": "development", "recipient": family["recipient"],
                   "recipient_position": j, "cells": cells, "predictions": predicted}
            row["prediction_error_dtype_discrepancies"] = discrepancies(row)
            self.rows.append(row)
            self.anchors.append({k: copy.deepcopy(row[k]) for k in ("family_id", "recipient", "recipient_position", "predictions")})
            self.anchors[-1]["anchors"] = {a: copy.deepcopy(cells[a]) for a in v.ANCHORS}
            a = [0.]*42
            a[0], a[j] = .875, .125
            self.selections.append({"family_id": family["family_id"], "recipient": family["recipient"],
                                    "recipient_position": j, "native_margin": .25, "native_eos_attention": a})
        (self.root / "source.txt").write_text("frozen source\n")
        sources = {"source.txt": v.digest(self.root / "source.txt")}
        self.receipt = {"written_before_target_execution_at": "2026-10-02T10:00:01+00:00",
                        "source_hashes": sources}
        c = {"identity_max_margin_error": 0., "identity_max_node_error": 0.,
             "self_same_position_max_margin_error": 0., "inserted_node_exactly_intended_after_dtype": True,
             "all_target_layer_attention_weights_unchanged": True,
             "all_target_layer_value_projections_unchanged": True,
             "nontarget_head_and_query_preprojection_exactly_unchanged": True}
        self.controls = {stage: {tag: {**c, "identity_tolerance": 1e-5 if tag == "float32" else 1e-10}
                                for tag in ("float32", "float64")}
                         for stage in ("anchors", "remaining_targets")}
        self.controls.update(maximum_prediction_error_dtype_discrepancy=0.,
                             prediction_error_dtype_allowance=.001, precision_gate_passed=True)
        self.manifest = {"status": "completed", "phase": "development",
                         "task": {"model_id": "a9g0io1r", "n_layer": 2, "n_head": 4, "head": 1},
                         "numerical_allowance_on_prediction_error": .001,
                         "started_at": "2026-10-02T10:00:00+00:00",
                         "target_measurement_started_at": "2026-10-02T10:00:02+00:00",
                         "finished_at": "2026-10-02T10:00:03+00:00",
                         "input_cases_sha256": v.digest(self.cases), "source_hashes": sources}
        self.save()

    def save(self):
        write_rows(self.run / "cases.jsonl", self.rows)
        write_rows(self.run / "anchor_forecasts.jsonl", self.anchors)
        write_json(self.run / "recipient_selection.json", self.selections)
        self.receipt["anchor_forecasts_sha256"] = v.digest(self.run / "anchor_forecasts.jsonl")
        self.manifest["anchor_forecasts_sha256"] = self.receipt["anchor_forecasts_sha256"]
        write_json(self.run / "anchor_receipt.json", self.receipt)
        write_json(self.run / "controls.json", self.controls)
        self.manifest["output_hashes"] = {p.name: v.digest(p) for p in self.run.iterdir() if p.name != "manifest.json"}
        write_json(self.run / "manifest.json", self.manifest)

    def audit(self, report=None):
        return v.audit(self.run, self.cases, report, root=self.root)


class IndependentVerifierTests(unittest.TestCase):
    def test_fixed_forecasts_have_six_targets(self):
        p = v.forecast({"neg_20_0": 1., "pos_28_0": 3.})
        self.assertEqual(len(p["H_state"]), 6)
        self.assertEqual(p["H_state"]["neg_28_1"], 1.)
        self.assertEqual(p["H_position"]["neg_28_1"], 3.)
        self.assertEqual(p["H_state"]["pos_20_0"], 3.)
        self.assertEqual(p["H_position"]["pos_20_0"], 1.)

    def test_clean_independent_fixture_and_optional_report(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            result = f.audit()
            self.assertEqual(result["verification"], "PASS")
            self.assertEqual(result["snapshots_checked"], 512)
            self.assertTrue(result["recomputed"]["confirmation_start_gate"])
            self.assertEqual(result["recomputed"]["candidates"]["H_state"]["definite_hits"], 32)
            report = result["recomputed"]
            report["provenance"] = {n: v.digest(f.run/n) for n in ("manifest.json", "cases.jsonl", "anchor_forecasts.jsonl")}
            path = f.root / "report.json"
            write_json(path, report)
            self.assertTrue(f.audit(path)["analysis_report_checked"])
            report["confirmation_start_gate"] = False
            write_json(path, report)
            with self.assertRaisesRegex(ValueError, "confirmation_start_gate"):
                f.audit(path)

    def test_changed_forecast_fails_even_with_updated_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            f.rows[0]["predictions"]["H_state"]["neg_20_1"] = .01
            f.anchors[0]["predictions"] = copy.deepcopy(f.rows[0]["predictions"])
            f.save()
            with self.assertRaisesRegex(ValueError, "fixed anchor rules"):
                f.audit()

    def test_changed_source_fails(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            (f.root / "source.txt").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Current source"):
                f.audit()

    def test_false_chronology_fails_even_with_updated_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            f.receipt["written_before_target_execution_at"] = "2026-10-02T10:00:04+00:00"
            f.save()
            with self.assertRaisesRegex(ValueError, "timestamp ordering"):
                f.audit()

    def test_delivered_tensor_edit_fails(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            f.rows[0]["cells"]["neg_20_1"]["float32"]["h_patch_delivered"][0] = 0.
            f.save()
            with self.assertRaisesRegex(ValueError, "Delivered node"):
                f.audit()

    def test_saved_selection_checks_first_argmax(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            s = f.selections[0]
            old = s["recipient_position"]
            other = next(i+1 for i,c in enumerate(s["recipient"]) if c == ")" and i+1 != old)
            s["native_eos_attention"][0] -= .25
            s["native_eos_attention"][other] = .25
            f.save()
            with self.assertRaisesRegex(ValueError, "highest-attention"):
                f.audit()

    def test_prefix_metadata_is_recomputed(self):
        family = copy.deepcopy(v.read_jsonl(HERE / "inputs/development_001/cases.jsonl")[0])
        exclusions = json.loads((HERE / "inputs/development_001/exclusions.json").read_text())
        family["donors"][0]["balance"] = 2
        with self.assertRaisesRegex(ValueError, "metadata"):
            v.validate_input(family, 0, exclusions)

    def test_largest_of_six_errors_and_same_label_replicate(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            row = f.rows[0]
            for tag in ("float32", "float64"):
                row["cells"]["neg_20_1"][tag]["patched_margin"] = .25
            row["prediction_error_dtype_discrepancies"] = discrepancies(row)
            result = v.family_result(row)
            self.assertEqual(result["max_prediction_error"]["H_state"], .25)
            self.assertEqual(result["same_label_prefix_differences"]["neg_20"], .25)

    def test_error_precision_not_individual_margin_precision(self):
        with tempfile.TemporaryDirectory() as d:
            row = Fixture(d).rows[0]
            row["cells"]["neg_20_0"]["float32"]["patched_margin"] -= .0009
            row["cells"]["neg_20_1"]["float32"]["patched_margin"] += .0009
            row["prediction_error_dtype_discrepancies"] = discrepancies(row)
            with self.assertRaisesRegex(ValueError, "precision allowance"):
                v.family_result(row)

    def test_strict_gap_and_start_gate(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            row = f.rows[0]
            for tag in ("float32", "float64"):
                row["cells"]["pos_28_0"][tag]["patched_margin"] = .202
            row["predictions"] = v.forecast({a: row["cells"][a]["float64"]["patched_margin"] for a in v.ANCHORS})
            row["prediction_error_dtype_discrepancies"] = discrepancies(row)
            self.assertFalse(v.family_result(row)["eligible"])
            results = [v.family_result(r) for r in f.rows]
            for i,r in enumerate(results):
                r["eligible"] = i < 16
                r["max_prediction_error"]["H_state"] = .099 if i < 15 else .101
                r["max_prediction_error"]["H_position"] = .5
            self.assertTrue(v.summarize(results)["confirmation_start_gate"])
            results[14]["max_prediction_error"]["H_state"] = .101
            self.assertFalse(v.summarize(results)["confirmation_start_gate"])
            self.assertEqual(v.summarize(results)["candidates"]["H_state"]["possible_hits"], 16)
            for i,r in enumerate(results):
                r["eligible"] = i < 15
                r["max_prediction_error"]["H_state"] = 0.
            self.assertFalse(v.summarize(results)["confirmation_start_gate"])


if __name__ == "__main__":
    unittest.main()
