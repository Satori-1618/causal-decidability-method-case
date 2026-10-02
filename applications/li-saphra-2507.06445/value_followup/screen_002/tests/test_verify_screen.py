"""Independent verifier tests; synthetic measurements, no model execution."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("screen_independent_verify", HERE / "verify_screen.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def synthetic_family(index, group, separating=True):
    native = 5. if group == "accepted" else 10.
    gap = .5 if separating else .01
    names = v.CELLS if group == "accepted" else v.ANCHORS
    values = {name: native + (gap if name.startswith("pos_") else 0.) for name in names}
    cells = {name: {tag: {"native_margin": native, "patched_margin": value,
                         "margin_change": value-native, "a_r": .03}
                    for tag in ("float32", "float64")}
             for name, value in values.items()}
    result = {"family_id": f"{group}_{index}", "stratum": group, "cells": cells}
    if group == "accepted":
        result["predictions"] = v.old.forecast({a: values[a] for a in v.ANCHORS})
        result["prediction_error_dtype_discrepancies"] = {
            candidate+":"+cell: 0. for candidate in ("H_state", "H_position") for cell in v.TARGETS}
    return result


class VerifyScreen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepared inputs are outcome-free and frozen separately from these tests.
        cls.inputs = HERE / "inputs"
        cls.candidates = v.read_jsonl(cls.inputs / "candidates.jsonl")
        cls.exclusions = json.loads((cls.inputs / "exclusions.json").read_text())

    def screen(self):
        observed = []
        for i, candidate in enumerate(self.candidates):
            margin = 7. if i % 2 == 0 else 9.
            group = "accepted" if margin < 8 else "rejected"
            observed.append({**candidate, "margin_float32": margin, "margin_float64": margin,
                             "screen_accept_float32": margin < 8, "screen_accept_float64": margin < 8,
                             "stratum": group, "selected": i < 64})
        return observed

    def test_exact_extremes_are_nondegenerate(self):
        upper = 1-.0125**(1/32)
        self.assertAlmostEqual(v.exact_interval(0)[1], upper, places=14)
        self.assertAlmostEqual(v.exact_interval(32)[0], .0125**(1/32), places=14)
        self.assertEqual(v.exact_interval(0)[0], 0.)
        self.assertEqual(v.exact_interval(32)[1], 1.)

    def test_exact_interval_symmetry_and_monotonicity(self):
        intervals = [v.exact_interval(k) for k in range(33)]
        for k, bounds in enumerate(intervals):
            self.assertAlmostEqual(bounds[0], 1-intervals[32-k][1], places=13)
        self.assertTrue(all(intervals[k][0] <= intervals[k+1][0] for k in range(32)))
        with self.assertRaises(ValueError):
            v.exact_interval(True)

    def test_primary_joint_bounds_and_decisions(self):
        result = v.primary(32, 0)
        self.assertGreater(result["simultaneous_interval"][0], .25)
        self.assertEqual(result["decision"], "supports_at_least_25pp_enrichment")
        self.assertEqual(v.primary(0, 32)["directional_status"], "negative_enrichment")
        self.assertEqual(v.primary(16, 16)["directional_status"], "unresolved")

    def test_first32_selection_all_native_records(self):
        chosen, counts, error = v.check_screen(self.candidates, self.screen(), self.exclusions)
        self.assertEqual(chosen["accepted"], list(range(0, 64, 2)))
        self.assertEqual(chosen["rejected"], list(range(1, 64, 2)))
        self.assertEqual(counts, {"accepted": 512, "rejected": 512})
        self.assertEqual(error, 0.)

    def test_selection_cannot_skip_earlier_recipient(self):
        rows = self.screen()
        rows[0]["selected"], rows[64]["selected"] = False, True
        with self.assertRaisesRegex(ValueError, "first 32"):
            v.check_screen(self.candidates, rows, self.exclusions)

    def test_native_classification_crossing_fails_even_tiny(self):
        rows = self.screen()
        rows[0].update(margin_float32=7.9999, margin_float64=8.0001)
        with self.assertRaisesRegex(ValueError, "precision"):
            v.check_screen(self.candidates, rows, self.exclusions)

    def test_signed_not_absolute_margin(self):
        rows = self.screen()
        rows[0].update(margin_float32=-12., margin_float64=-12.)
        v.check_screen(self.candidates, rows, self.exclusions)

    def test_recounts_primary_and_secondary_separately(self):
        rows = [synthetic_family(i, g, separating=i < (20 if g == "accepted" else 2))
                for g in ("accepted", "rejected") for i in range(32)]
        result = v.recount(rows)
        self.assertEqual(result["primary"]["strata"]["accepted"]["separating"], 20)
        self.assertEqual(result["primary"]["strata"]["rejected"]["separating"], 2)
        self.assertEqual(result["secondary"]["candidates"]["H_state"]["definite_hits"], 20)
        self.assertEqual(result["secondary"]["candidates"]["H_position"]["definite_hits"], 0)
        self.assertTrue(result["secondary"]["confirmation_start_gate"])

    def test_secondary_gate_not_replaced_by_primary_success(self):
        rows = [synthetic_family(i, g, separating=i < (15 if g == "accepted" else 0))
                for g in ("accepted", "rejected") for i in range(32)]
        self.assertFalse(v.recount(rows)["secondary"]["confirmation_start_gate"])

    def test_gap_boundary_dtype_disagreement(self):
        rows = [synthetic_family(i, g) for g in ("accepted", "rejected") for i in range(32)]
        for tag, delta in (("float32", .2019), ("float64", .2021)):
            rows[0]["cells"][v.ANCHORS[1]][tag]["patched_margin"] = 5.+delta
        with self.assertRaisesRegex(ValueError, "anchor-gap"):
            v.recount(rows)

    def test_unmeasured_rejected_targets_not_accepted(self):
        rows = [synthetic_family(i, g) for g in ("accepted", "rejected") for i in range(32)]
        rows[-1]["cells"]["neg_20_1"] = rows[-1]["cells"][v.ANCHORS[0]]
        with self.assertRaisesRegex(ValueError, "inventory"):
            v.recount(rows)

    def test_freshness_cross_role_and_position_reconstructed(self):
        v.check_exclusions(self.exclusions, v.ROOT)
        corrupted = copy.deepcopy(self.exclusions)
        corrupted["prefixes"]["20"].pop()
        with self.assertRaisesRegex(ValueError, "prefix bans"):
            v.check_exclusions(corrupted, v.ROOT)

    def test_freshness_clean_export_needs_no_ignored_csv(self):
        relative = Path("applications/li-saphra-2507.06445/value_followup/inputs/development_001")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/relative).mkdir(parents=True)
            for name in ("cases.jsonl", "exclusions.json"):
                shutil.copyfile(v.ROOT/relative/name, root/relative/name)
            v.check_exclusions(self.exclusions, root)
            historical = json.loads((root/relative/"exclusions.json").read_text())
            historical["source_hashes"]["cache/data/model_preds/ood_data_preds.csv"] = "changed"
            (root/relative/"exclusions.json").write_text(json.dumps(historical))
            with self.assertRaisesRegex(ValueError, "source history"):
                v.check_exclusions(self.exclusions, root)

    def test_all_prepared_donor_grids(self):
        for family in v.read_jsonl(self.inputs / "donor_families.jsonl"):
            v.check_input_donors(family, self.exclusions)

    def test_first_tie_and_closing_only(self):
        text = "()"*16
        attention = [0.]*42
        attention[1], attention[2], attention[4] = .5, .25, .25
        row = {"recipient": text, "recipient_position": 2, "native_eos_attention": attention}
        self.assertEqual(v.check_selection_row(row)[0], 2)
        row["recipient_position"] = 4
        with self.assertRaisesRegex(ValueError, "first maximum"):
            v.check_selection_row(row)

    def test_receipt_ordering(self):
        manifest = {"started_at": "2026-10-02T12:00:00+00:00",
                    "anchors_measurement_started_at": "2026-10-02T12:02:00+00:00",
                    "target_measurement_started_at": "2026-10-02T12:04:00+00:00",
                    "finished_at": "2026-10-02T12:05:00+00:00"}
        screen = {"written_before_transfers_at": "2026-10-02T12:01:00+00:00"}
        anchors = {"written_before_target_execution_at": "2026-10-02T12:03:00+00:00"}
        v.check_chronology(manifest, screen, anchors)
        screen["written_before_transfers_at"] = "2026-10-02T12:02:00+00:00"
        with self.assertRaisesRegex(ValueError, "chronology"):
            v.check_chronology(manifest, screen, anchors)


if __name__ == "__main__":
    unittest.main()
