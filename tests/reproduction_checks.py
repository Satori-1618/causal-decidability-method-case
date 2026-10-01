"""Tightly scoped portability allowances for archived Goodfire development replay.

These are test-comparison tolerances, not revised scientific decision rules.
Python 3.10/3.11 planning-power differences were at most 2.3e-13; NumPy
1.26/2.4 full-logit diagnostics differed by at most 3.6e-15. All decisions,
sample sizes, thresholds, other measurements and artifact hashes remain exact.
"""
import math


def tolerance(path):
    audit_roots = (("audit_full_logits",), ("gate_table", "audit_full_logits"))
    diagnostics = ("max_abs_answer_mass_difference", "max_abs_logsumexp_difference")
    if any(path == root + (field,) for root in audit_roots for field in diagnostics):
        return 1e-14
    power_roots = (("n_rule",), ("planning_N",), ("split_B", "N_rule"),
                   ("split_B", "freeze", "N_rule"))
    for root in power_roots:
        if path[:len(root)] != root:
            continue
        suffix = path[len(root):]
        if (len(suffix) == 1 or
                (len(suffix) == 3 and suffix[0] == "table" and type(suffix[1]) is int)):
            if suffix[-1] in ("adequacy_power", "exclusion_power"):
                return 1e-12
    return None


def assert_records_match(test, actual, expected, path=()):
    test.assertIs(type(actual), type(expected), str(path))
    if isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys(), str(path))
        for key in expected:
            assert_records_match(test, actual[key], expected[key], path + (key,))
    elif isinstance(expected, list):
        test.assertEqual(len(actual), len(expected), str(path))
        for index, (value, reference) in enumerate(zip(actual, expected)):
            assert_records_match(test, value, reference, path + (index,))
    elif tolerance(path) is not None:
        test.assertIsInstance(actual, float, str(path))
        test.assertTrue(math.isfinite(actual) and math.isfinite(expected), str(path))
        test.assertAlmostEqual(actual, expected, delta=tolerance(path), msg=str(path))
    else:
        test.assertEqual(actual, expected, str(path))
