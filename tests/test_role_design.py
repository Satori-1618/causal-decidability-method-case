"""Counterexamples and structural accounting for the prospective three-role task."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples"), str(ROOT / "applications/makelov-2311.17030/src")]
import role_design as design
import role_transfer_design as demo
from role_geometry import closest_line_fit


class RoleDesignTests(unittest.TestCase):
    def test_full_grid_and_scientific_denominator(self):
        cells = design.grid()
        self.assertEqual(len(cells), 108)
        self.assertEqual(len(design.primary_cells()), 72)
        self.assertEqual(len(set(c["cell_id"] for c in cells)), 108)
        self.assertEqual(sum(c["exact_self"] for c in cells), 3)
        self.assertTrue(all(c["donor_query"] != c["recipient_query"] for c in design.primary_cells()))
        groups = {}
        for c in design.primary_cells():
            groups.setdefault(c["group"], []).append(c)
        self.assertEqual(len(groups), 36)
        self.assertTrue(all(len(g) == 2 for g in groups.values()))

    def test_identical_donor_answer_and_position_do_not_fix_role_target(self):
        example = design.explanation_example()
        a, b = example["rows"]
        self.assertEqual([r["donor_answer"] for r in (a, b)], ["Alice", "Alice"])
        self.assertEqual([r["donor_mention_position"] for r in (a, b)], [1, 1])
        self.assertEqual([r["predicted_targets"]["role"] for r in (a, b)], ["Bob", "Carol"])
        self.assertEqual([r["predicted_targets"]["position"] for r in (a, b)], ["Bob", "Bob"])
        self.assertEqual([r["predicted_targets"]["name"] for r in (a, b)], ["Alice", "Alice"])

    def test_third_query_breaks_boolean_changed_query_rule(self):
        rows = {r["cell_id"]: r for r in design.primary_cells()}
        a, b = [rows[c] for c in design.DIAGNOSTICS["switch"]]
        self.assertEqual(a["group"], b["group"])
        self.assertNotEqual(a["targets"]["role"], b["targets"]["role"])
        self.assertNotEqual(a["donor_query"], b["donor_query"])
        # This is exactly what two queried roles cannot provide.
        for qr in range(2):
            self.assertEqual(len([qd for qd in range(2) if qd != qr]), 1)

    def test_absolute_oracle_geometry_allows_a_real_positive_witness(self):
        result = demo.demonstration()
        self.assertTrue(result["all_gaps_exceed_proposed_0_52"])
        self.assertEqual(result["target_signature_groups"], [["name"], ["no_op"], ["position"], ["role"]])

    def test_collinear_endpoints_destroy_positional_separation(self):
        # All three native answers differ only along one output direction:
        # extra labels alone cannot rescue identification.
        endpoints = {rid: {q: [float(j), 0., 0.] for j, q in enumerate(design.ROLES)}
                     for rid in design.RECIPIENTS}
        rows = {r["cell_id"]: r for r in design.directions(endpoints)}
        row = rows[design.DIAGNOSTICS["position"][0]]
        self.assertAlmostEqual(closest_line_fit(row["role_direction"], row["position_direction"])["distance"], 0.)

    def test_gains_collapsing_to_zero_are_not_separation(self):
        rows = {r["cell_id"]: r for r in design.directions(demo.synthetic_endpoints())}
        row = rows[design.DIAGNOSTICS["position"][0]]
        self.assertEqual(closest_line_fit((0., 0., 0.), row["position_direction"])["distance"], 0.)

    def test_missing_native_endpoint_fails_closed(self):
        endpoints = demo.synthetic_endpoints()
        del endpoints["b0_gave"]["observer"]
        with self.assertRaises(ValueError):
            design.directions(endpoints)

    def test_query_wording_does_not_change_ground_truth_or_invent_a_result(self):
        for context in (*design.DONORS.values(), *design.RECIPIENTS.values()):
            for query in design.ROLES:
                a = design.render(context, query, wording="fit")
                b = design.render(context, query, wording="heldout")
                self.assertNotEqual(a, b)
                self.assertTrue(a.endswith(" was") and b.endswith(" was"))
        self.assertIn("unmeasured", demo.demonstration()["scope"])

    def test_demo_runs_isolated_without_dependencies(self):
        proc = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "examples/role_transfer_design.py"), "--json"],
                              cwd="/", text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("NOT LLM RESULTS", json.loads(proc.stdout)["notice"])


if __name__ == "__main__":
    unittest.main()
