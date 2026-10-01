"""The amendment changes only the cap; original scientific gates stay binding."""

import ast
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "applications/makelov-2311.17030/src"
sys.path.insert(0, str(SRC))
import donor_factor_planning as original
import donor_factor_planning_512 as amended
from test_donor_factor_analysis import world


class Planning512AmendmentTests(unittest.TestCase):
    def test_only_cap_schema_and_module_documentation_change(self):
        # A whole-module AST comparison catches changes to sampling, rescaling,
        # strict boundaries, seeds, precision checks, and the selection rule.
        source = (SRC / "donor_factor_planning_512.py").read_text()
        source = source.replace("CANDIDATE_SIZES = (128, 192, 256, 384, 512)",
                                "CANDIDATE_SIZES = (128, 192, 256, 384)")
        source = source.replace("donor-factor-planning-512-v1", "donor-factor-planning-v1")
        source = source.replace("no prespecified n <=512", "no prespecified n <=384")
        a = ast.parse(source)
        b = ast.parse((SRC / "donor_factor_planning.py").read_text())
        a.body.pop(0)  # Only the module docstring is allowed to differ.
        b.body.pop(0)
        self.assertEqual(ast.dump(a), ast.dump(b))
        self.assertEqual(amended.CANDIDATE_SIZES, original.CANDIDATE_SIZES + (512,))
        for name, value in vars(original).items():
            if name.isupper() and name != "CANDIDATE_SIZES":
                with self.subTest(constant=name):
                    self.assertEqual(getattr(amended, name), value)

    def test_first_four_candidate_calculations_are_identical(self):
        root = np.diag([.09, .57, .36, .09, .57, .36])
        empirical = np.random.default_rng(57).normal(size=(32, 6)) @ root.T
        for index, n in enumerate(original.CANDIDATE_SIZES):
            with self.subTest(n=n):
                self.assertEqual(amended.exact_coverage_plan(n), original.exact_coverage_plan(n))
                seed = original.PLANNING_SEED + 100 * index
                # Small software fixtures, not an actual planning run.
                self.assertEqual(amended._normal_power(n, root, 16, seed),
                                 original._normal_power(n, root, 16, seed))
                for offset, distribution in ((1, "gaussian"), (2, "empirical")):
                    self.assertEqual(
                        amended._bootstrap_power(n, root, empirical, 8, 16, seed + offset, distribution),
                        original._bootstrap_power(n, root, empirical, 8, 16, seed + offset, distribution))

    def test_512_coverage_matches_independent_direct_binomial_sums(self):
        n = 512
        def tail(k, p):
            return math.fsum(math.comb(n, j) * p**j * (1-p)**(n-j)
                             for j in range(k, n+1))
        threshold = next(k for k in range(n+1) if tail(k, .8) < .025 / 4)
        observed = amended.exact_coverage_plan(n)
        self.assertEqual(observed["required_successes"], threshold)
        self.assertGreaterEqual(tail(threshold-1, .8), .025 / 4)
        self.assertAlmostEqual(observed["power_at_90_percent_success"], tail(threshold, .9), places=11)
        self.assertTrue(observed["passes"])

    def _mocked_candidate_run(self, count=32, fail_distribution=None, **kwargs):
        records = [world(lambda i, p: 0., str(j)) for j in range(count)]
        def normal(n, *args):
            return {"passes": n == 512}
        def bootstrap(n, *args):
            return {"passes": args[-1] != fail_distribution}
        with patch.object(amended, "_normal_power", side_effect=normal) as normal_call, \
                patch.object(amended, "_bootstrap_power", side_effect=bootstrap) as bootstrap_call:
            result = amended.plan_confirmation(records, **kwargs)
        return result, normal_call.call_args_list, bootstrap_call.call_args_list

    def test_512_can_be_selected_only_after_both_bootstrap_checks(self):
        result, normals, bootstraps = self._mocked_candidate_run()
        self.assertEqual(result["schema_version"], "donor-factor-planning-512-v1")
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected_n"], 512)
        self.assertFalse(result["confirmation_authorized"])
        self.assertEqual([call.args[0] for call in normals], [128, 192, 256, 384, 512])
        self.assertEqual([call.args[-1] for call in normals],
                         [original.PLANNING_SEED + 100*i for i in range(5)])
        self.assertEqual([call.args[-2:] for call in bootstraps],
                         [(original.PLANNING_SEED + 401, "gaussian"),
                          (original.PLANNING_SEED + 402, "empirical")])
        self.assertTrue(all(c["bootstrap_checks"] is None for c in result["candidates"][:4]))

    def test_either_bootstrap_failure_retains_stop_at_512(self):
        for distribution in ("gaussian", "empirical"):
            with self.subTest(distribution=distribution):
                result, _, _ = self._mocked_candidate_run(fail_distribution=distribution)
                self.assertEqual(result["status"], "STOP")
                self.assertIsNone(result["selected_n"])
                self.assertIn("<=512", result["stop_reason"])
                self.assertFalse(result["candidates"][-1]["passes"])

    def test_reduced_counts_altered_seed_or_wrong_development_size_are_test_only(self):
        settings = ({"gaussian_simulations": 16}, {"bootstrap_simulations": 16},
                    {"bootstrap_draws": 16}, {"seed": original.PLANNING_SEED + 1},
                    {"count": 31}, {"count": 33})
        for settings_item in settings:
            with self.subTest(settings=settings_item):
                result, _, _ = self._mocked_candidate_run(**settings_item)
                self.assertEqual(result["status"], "TEST_ONLY")
                self.assertEqual(result["selected_n"], 512)
                self.assertFalse(result["confirmation_authorized"])


if __name__ == "__main__":
    unittest.main()
