"""Round 1 analyzer unit tests (Gur-Arieh et al. application). Constructed data; no model."""
import copy
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
if str(APP / "src") not in sys.path:
    sys.path.insert(0, str(APP / "src"))
import mixing_round1_analysis as ra  # noqa: E402
import query_route_analysis  # noqa: E402  (made importable by the analyzer)

CELL = {"i_P": 3, "i_L": 1, "i_R": 5, "i_N": 0}
N_GROUPS = 7


def logits_from(p):
    return [math.log(x) for x in p]


def distribution(positional, lexical, reflexive, native, cell=CELL, n=N_GROUPS):
    """p with the given masses at i_P, i_L, i_R and i_N; the rest spread thinly."""
    p = [1e-6] * n
    p[cell["i_P"]] += positional
    p[cell["i_L"]] += lexical
    p[cell["i_R"]] += reflexive
    p[cell["i_N"]] += native
    total = math.fsum(p)
    return [x / total for x in p]


def manifest(N, t_w=0.45, t_a=0.97, q_bar=(0.45, 0.30, 0.25), delta=1.0):
    return {
        "n_groups": N_GROUPS, "cell": dict(CELL), "N": N,
        "rule": {**ra.CONTRACT, "s_min": 0.10, "d_min": 0.20, "agreement_transfer_floor": 0.9},
        "anchors": {"T_W": max(q_bar), "T_A": t_a, "d": t_a - max(q_bar), "q_bar_B": list(q_bar), "m_B": 100},
        "mean_gate": {"delta": delta},
        "development_gates": {"resolution_rate_B": 0.95, "agreement_transfer_rate_B": 0.95},
    }


def record(i, p, **extra):
    base = {"case_id": f"case-{i}", "draw_index": i, "qualifies": True,
            "design_indices": dict(CELL), "technical": {"passed": True},
            "entity_logits": logits_from(p), "entity_mass_full_vocab": 0.9}
    base.update(extra)
    return base


W_LIKE = distribution(0.45 * 0.8, 0.30 * 0.8, 0.25 * 0.8, 0.2)
UNRESOLVED = distribution(0.01, 0.01, 0.01, 0.97)


class SupportWeightingTrapTest(unittest.TestCase):
    """The analyzer averages q, not p: every resolved case has equal weight."""

    def test_anchor_is_concentration_of_mean_q_not_of_mean_p(self):
        strong_positional = distribution(0.9, 0.0, 0.0, 0.1)
        weak_lexical = distribution(0.0, 0.1, 0.0, 0.9)
        agreement = [(0, CELL["i_P"], logits_from(distribution(0.97, 0.0, 0.0, 0.03)))]
        result = ra.anchors([logits_from(strong_positional), logits_from(weak_lexical)],
                            agreement, CELL, 1, s_min=0.05)
        self.assertAlmostEqual(result["T_W"], 0.5, places=4)
        mean_p = [(a + b) / 2 for a, b in zip(strong_positional, weak_lexical)]
        _, q_of_mean_p = ra.q_map(mean_p, CELL, 1)
        self.assertAlmostEqual(ra.concentration(q_of_mean_p), 0.9, places=4)

    def test_case_concentration_ignores_support(self):
        high = ra.case_measures(logits_from(distribution(0.72, 0.09, 0.09, 0.10)), CELL, 1, 0.05)
        low = ra.case_measures(logits_from(distribution(0.16, 0.02, 0.02, 0.80)), CELL, 1, 0.05)
        self.assertAlmostEqual(high["T"], low["T"], places=4)
        self.assertGreater(high["S"], 3 * low["S"])

    def test_a_weakly_supported_concentrated_case_conforms_to_A(self):
        records = [record(i, W_LIKE) for i in range(99)]
        records.append(record(99, distribution(0.19, 0.005, 0.005, 0.80)))
        summary = ra.analyze_confirmation(manifest(100), records)
        last = summary["cases"][-1]
        self.assertTrue(last["resolved"])
        self.assertTrue(last["A_T"])


class UnresolvedRuleTests(unittest.TestCase):
    def test_unresolved_never_causes_an_exclusion(self):
        for u in (0, 10, 30, 60, 99):
            records = [record(i, W_LIKE if i >= u else UNRESOLVED) for i in range(100)]
            summary = ra.analyze_confirmation(manifest(100), records)
            with self.subTest(u=u):
                self.assertEqual(summary["u"], u)
                self.assertNotEqual(summary["statuses"]["W_T"], "excluded")

    def test_old_rule_would_have_excluded_what_the_new_rule_leaves_undecided(self):
        records = [record(i, W_LIKE if i >= 40 else UNRESOLVED) for i in range(100)]
        summary = ra.analyze_confirmation(manifest(100), records)
        self.assertEqual(summary["statuses"]["W_T"], "undecided")
        old_upper = query_route_analysis.clopper_pearson(60, 100, 0.05, 2)[1]
        self.assertLess(old_upper, 0.8)

    def test_unresolved_count_as_non_matches_for_adequacy(self):
        records = [record(i, W_LIKE if i >= 10 else UNRESOLVED) for i in range(200)]
        summary = ra.analyze_confirmation(manifest(200), records)
        self.assertEqual(summary["profiles"]["W_T"]["k_match"], 190)
        self.assertEqual(summary["statuses"]["W_T"], "adequate")


class TechnicalFailureTests(unittest.TestCase):
    def assert_invalid(self, records):
        summary = ra.analyze_confirmation(manifest(len(records)), records)
        self.assertEqual(summary["run_status"], "INVALID")
        self.assertTrue(summary["invalid_reason"].startswith("technical failure"))
        self.assertEqual(summary["level"], "S1")
        self.assertNotIn("u", summary)

    def test_failed_control_is_invalid_not_unresolved(self):
        records = [record(i, W_LIKE) for i in range(20)]
        records[3]["technical"] = {"passed": False}
        self.assert_invalid(records)

    def test_non_finite_readout_is_invalid(self):
        records = [record(i, W_LIKE) for i in range(20)]
        records[5]["entity_logits"][2] = float("nan")
        self.assert_invalid(records)

    def test_design_index_mismatch_is_invalid(self):
        records = [record(i, W_LIKE) for i in range(20)]
        records[7]["design_indices"] = {**CELL, "i_L": 6}
        self.assert_invalid(records)


class ContractAndDesignTests(unittest.TestCase):
    def test_kappa_of_one_half_or_more_is_refused(self):
        bad = manifest(20)
        bad["rule"]["kappa"] = 0.5
        with self.assertRaisesRegex(ValueError, "contract"):
            ra.analyze_confirmation(bad, [record(i, W_LIKE) for i in range(20)])

    def test_inadmissible_cells_are_refused_on_design_indices(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            ra.check_cell({**CELL, "i_R": CELL["i_L"]}, N_GROUPS, 1)
        with self.assertRaisesRegex(ValueError, "more than w"):
            ra.check_cell({**CELL, "i_N": CELL["i_P"] + 1}, N_GROUPS, 1)

    def test_intervals_come_from_the_existing_helper_with_alpha_over_four(self):
        self.assertIs(ra.clopper_pearson, query_route_analysis.clopper_pearson)
        lower, _ = ra.clopper_pearson(50, 50, ra.CONTRACT["alpha"], ra.CONTRACT["family_size"])
        self.assertAlmostEqual(lower ** 50, ra.CONTRACT["label_tail"], places=12)

    def test_records_must_be_complete_and_unique(self):
        records = [record(i, W_LIKE) for i in range(20)]
        with self.assertRaisesRegex(ValueError, "expected N"):
            ra.analyze_confirmation(manifest(21), records)
        duplicate = copy.deepcopy(records)
        duplicate[4]["case_id"] = duplicate[3]["case_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            ra.analyze_confirmation(manifest(20), duplicate)
        gap = records[:5] + records[6:] + [record(20, W_LIKE)]
        with self.assertRaisesRegex(ValueError, "draw_index"):
            ra.analyze_confirmation(manifest(20), gap)

    def test_level_names_the_excluded_profile(self):
        self.assertEqual(ra.level("VALID", {"W_T": "excluded", "A_T": "adequate"}), "S3 (W_T excluded only)")
        self.assertEqual(ra.level("VALID", {"W_T": "undecided", "A_T": "excluded"}), "S3 (A_T excluded)")
        self.assertEqual(ra.level("VALID", {"W_T": "excluded", "A_T": "excluded"}), "S3 (both excluded)")
        self.assertEqual(ra.level("VALID", {"W_T": "adequate", "A_T": "undecided"}), "S2")
        self.assertEqual(ra.level("INVALID", {"W_T": "INVALID", "A_T": "INVALID"}), "S1")


class LabelAndSelectionTests(unittest.TestCase):
    def test_ties_are_reported_as_equivalence(self):
        p = distribution(0.05, 0.40, 0.40, 0.15)
        labels, background = ra.background_corrected_labels(p, CELL, 1)
        self.assertEqual(labels, ["lexical", "reflexive"])
        self.assertEqual(background, 1)

    def test_selection_takes_the_largest_d_and_breaks_ties_by_declared_order(self):
        chosen = ra.select_cell([("c1", 0.3), ("c2", 0.4), ("c3", 0.4)], d_min=0.2)
        self.assertEqual(chosen["selected"], "c2")
        stop = ra.select_cell([("c1", 0.1), ("c2", 0.15)], d_min=0.2)
        self.assertEqual(stop["status"], ra.NOT_DECIDABLE)
        self.assertEqual(stop["level"], "S1")

    def test_s_min_rule_uses_an_upper_order_statistic_and_a_floor(self):
        values = [i / 1000 for i in range(1, 101)]
        self.assertEqual(ra.anchored_s_min(values, 0.99, 0.0), 0.099)
        self.assertEqual(ra.anchored_s_min(values, 0.99, 0.2), 0.2)

    def test_delta_is_seeded_and_grows_as_the_false_invalid_rate_falls(self):
        pool = [(0.5 + 0.01 * (i % 7), 0.3, 0.2 - 0.01 * (i % 7)) for i in range(50)]
        a = ra.delta_threshold(pool, 50, 80, 0.05, 400, seed=7)
        self.assertEqual(a, ra.delta_threshold(pool, 50, 80, 0.05, 400, seed=7))
        self.assertLessEqual(a, ra.delta_threshold(pool, 50, 80, 0.01, 400, seed=7))
        self.assertGreater(a, 0)

    def test_post_hoc_strictness_reproduces_the_frozen_row_and_decides_nothing(self):
        records = [record(i, W_LIKE if i % 3 else UNRESOLVED) for i in range(30)]
        summary = ra.analyze_confirmation(manifest(30), records)
        table = ra.post_hoc_strictness(summary)
        self.assertEqual(table["matches_by_kappa"]["0.25"]["W_T"], summary["profiles"]["W_T"]["k_match"])
        self.assertIn("decides nothing", table["status"])
        self.assertNotIn("statuses", table)


if __name__ == "__main__":
    unittest.main()
