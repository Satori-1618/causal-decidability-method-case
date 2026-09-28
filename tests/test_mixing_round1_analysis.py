"""Round 1 analyzer unit tests (Gur-Arieh et al. application), protocol v2. No model.

Primary readout: answer-form logits and the answer-token mass; resolved iff S >= s_min
and mass >= 0.5; the paper's in-context readout is descriptive and never read.
"""
import copy
import math
import random
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
N = 200  # the N rule at split B's unresolved rate 0 (resolution_rate_B = 1.0)


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


def measurement(p, mass=0.9):
    return {"answer_logits": logits_from(p), "answer_mass_full_vocab": mass}


def manifest(N=N, t_a=0.97, q_bar=(0.45, 0.30, 0.25), delta=1.0, resolution_rate_B=1.0):
    sizing = ra.n_rule(1 - resolution_rate_B)
    return {
        "n_groups": N_GROUPS, "cell": dict(CELL), "N": N,
        "N_rule": {"N": sizing["N"], "adequacy_powered": sizing["adequacy_powered"]},
        "rule": {**ra.CONTRACT, "s_min": 0.10, "d_min": 0.20, "agreement_transfer_floor": 0.9},
        "anchors": {"T_W": max(q_bar), "T_A": t_a, "d": t_a - max(q_bar), "q_bar_B": list(q_bar), "m_B": 100},
        "mean_gate": {"delta": delta},
        "development_gates": {"resolution_rate_B": resolution_rate_B, "agreement_resolution_rate_B": 0.95,
                              "agreement_transfer_rate_B": 0.95},
    }


def record(i, p, mass=0.9, **extra):
    base = {"case_id": f"case-{i}", "draw_index": i, "qualifies": True,
            "design_indices": dict(CELL), "technical": {"passed": True},
            "answer_logits": logits_from(p), "answer_mass_full_vocab": mass,
            "descriptive": {"paper_readout": {"entity_logits": logits_from(p), "entity_mass_full_vocab": 3e-5}}}
    base.update(extra)
    return base


W_LIKE = distribution(0.45 * 0.8, 0.30 * 0.8, 0.25 * 0.8, 0.2)
UNRESOLVED = distribution(0.01, 0.01, 0.01, 0.97)
CONCENTRATED_L = distribution(0.01, 0.95, 0.01, 0.03)


class SupportWeightingTrapTest(unittest.TestCase):
    """The analyzer averages q, not p: every resolved case has equal weight."""

    def test_anchor_is_concentration_of_mean_q_not_of_mean_p(self):
        strong_positional = distribution(0.9, 0.0, 0.0, 0.1)
        weak_lexical = distribution(0.0, 0.1, 0.0, 0.9)
        agreement = [(0, CELL["i_P"], measurement(distribution(0.97, 0.0, 0.0, 0.03))),
                     (0, CELL["i_L"], measurement(distribution(0.0, 0.97, 0.0, 0.03))),
                     (0, CELL["i_R"], measurement(distribution(0.0, 0.0, 0.97, 0.03)))]
        result = ra.anchors([measurement(strong_positional), measurement(weak_lexical)],
                            agreement, CELL, 1, s_min=0.05)
        self.assertAlmostEqual(result["T_W"], 0.5, places=4)
        mean_p = [(a + b) / 2 for a, b in zip(strong_positional, weak_lexical)]
        _, q_of_mean_p = ra.q_map(mean_p, CELL, 1)
        self.assertAlmostEqual(ra.concentration(q_of_mean_p), 0.9, places=4)

    def test_case_concentration_ignores_support(self):
        high = ra.case_measures(measurement(distribution(0.72, 0.09, 0.09, 0.10)), CELL, 1, 0.05)
        low = ra.case_measures(measurement(distribution(0.16, 0.02, 0.02, 0.80)), CELL, 1, 0.05)
        self.assertAlmostEqual(high["T"], low["T"], places=4)
        self.assertGreater(high["S"], 3 * low["S"])

    def test_a_weakly_supported_concentrated_case_conforms_to_A(self):
        records = [record(i, W_LIKE) for i in range(N - 1)]
        records.append(record(N - 1, distribution(0.19, 0.005, 0.005, 0.80)))
        summary = ra.analyze_confirmation(manifest(), records)
        last = summary["cases"][-1]
        self.assertTrue(last["resolved"])
        self.assertTrue(last["A_T"])


class UnresolvedRuleTests(unittest.TestCase):
    def test_unresolved_never_causes_an_exclusion(self):
        for u in (0, 20, 60, 120, 199):
            records = [record(i, W_LIKE if i >= u else UNRESOLVED) for i in range(N)]
            summary = ra.analyze_confirmation(manifest(), records)
            with self.subTest(u=u):
                self.assertEqual(summary["u"], u)
                self.assertNotEqual(summary["statuses"]["W_T"], "excluded")

    def test_old_rule_would_have_excluded_what_the_new_rule_leaves_undecided(self):
        records = [record(i, W_LIKE if i >= 80 else UNRESOLVED) for i in range(N)]
        summary = ra.analyze_confirmation(manifest(), records)
        self.assertEqual(summary["statuses"]["W_T"], "undecided")
        old_upper = query_route_analysis.clopper_pearson(120, N, 0.05, 2)[1]
        self.assertLess(old_upper, 0.8)

    def test_unresolved_count_as_non_matches_for_adequacy(self):
        records = [record(i, W_LIKE if i >= 10 else UNRESOLVED) for i in range(N)]
        summary = ra.analyze_confirmation(manifest(), records)
        self.assertEqual(summary["profiles"]["W_T"]["k_match"], 190)
        self.assertEqual(summary["statuses"]["W_T"], "adequate")


class AnswerMassGateTests(unittest.TestCase):
    """Protocol v2: resolved iff S >= s_min and answer-token mass >= 0.5."""

    def test_low_answer_mass_is_unresolved_whatever_the_support(self):
        high = ra.case_measures(measurement(W_LIKE, mass=0.49), CELL, 1, 0.10)
        self.assertTrue(high["S_ok"])
        self.assertFalse(high["mass_ok"])
        self.assertFalse(high["resolved"])
        self.assertEqual(high["labels"], [])
        self.assertTrue(ra.case_measures(measurement(W_LIKE, mass=0.5), CELL, 1, 0.10)["resolved"])

    def test_low_mass_cases_count_like_unresolved_cases(self):
        records = [record(i, W_LIKE, mass=0.2 if i < 30 else 0.9) for i in range(N)]
        summary = ra.analyze_confirmation(manifest(), records)
        self.assertEqual(summary["u"], 30)
        self.assertEqual(summary["shares_vs_T_W"]["unresolved_by_answer_mass"], 30)
        self.assertEqual(summary["profiles"]["W_T"]["k_match"], 170)
        self.assertEqual(summary["profiles"]["W_T"]["k_match_plus_unresolved"], 200)

    def test_missing_or_non_finite_mass_is_invalid(self):
        for bad in (None, float("nan"), float("inf"), "0.9", 1.5):
            records = [record(i, W_LIKE) for i in range(N)]
            records[7]["answer_mass_full_vocab"] = bad
            summary = ra.analyze_confirmation(manifest(), records)
            with self.subTest(bad=bad):
                self.assertEqual(summary["run_status"], "INVALID")
                self.assertTrue(summary["invalid_reason"].startswith("technical failure"))
        records = [record(i, W_LIKE) for i in range(N)]
        del records[3]["answer_mass_full_vocab"]
        self.assertEqual(ra.analyze_confirmation(manifest(), records)["run_status"], "INVALID")


class PaperReadoutIsDescriptiveOnlyTests(unittest.TestCase):
    """The in-context readout lives under 'descriptive' and changes no status."""

    def build(self):
        return [record(i, W_LIKE if i % 3 else CONCENTRATED_L, mass=0.3 if i % 17 == 0 else 0.9)
                for i in range(N)]

    def test_changing_the_paper_readout_changes_nothing(self):
        records = self.build()
        before = ra.analyze_confirmation(manifest(), records)
        rng = random.Random(3)
        altered = copy.deepcopy(records)
        for r in altered:
            r["descriptive"]["paper_readout"]["entity_logits"] = [rng.uniform(-20, 20) for _ in range(N_GROUPS)]
            r["descriptive"]["paper_readout"]["entity_mass_full_vocab"] = rng.random()
            r["entity_logits"] = [rng.uniform(-20, 20) for _ in range(N_GROUPS)]  # the old field name
            r["entity_mass_full_vocab"] = 0.99
        self.assertEqual(ra.analyze_confirmation(manifest(), altered), before)
        for r in altered:
            del r["descriptive"]
        self.assertEqual(ra.analyze_confirmation(manifest(), altered), before)

    def test_decision_functions_cannot_consume_the_paper_readout(self):
        paper = {"entity_logits": logits_from(W_LIKE), "entity_mass_full_vocab": 0.9}
        with self.assertRaises(ra.TechnicalFailure):
            ra.case_measures(paper, CELL, 1, 0.1)
        with self.assertRaises(ra.TechnicalFailure):
            ra.nopatch_supports([paper], CELL, 1)
        with self.assertRaises(ra.TechnicalFailure):
            ra.anchors([paper], [(0, 3, paper)], CELL, 1, 0.1)
        records = [record(i, W_LIKE) for i in range(N)]
        for r in records:
            r["entity_logits"] = r.pop("answer_logits")
        self.assertEqual(ra.analyze_confirmation(manifest(), records)["run_status"], "INVALID")

    def test_descriptive_helpers_match_the_primary_shape(self):
        described = ra.describe_logits(logits_from(W_LIKE), CELL, 1, 0.1)
        primary = ra.case_measures(measurement(W_LIKE), CELL, 1, 0.1)
        for key in ("S", "q", "T", "resolved", "labels"):
            self.assertEqual(described[key], primary[key])


class TechnicalFailureTests(unittest.TestCase):
    def assert_invalid(self, records):
        summary = ra.analyze_confirmation(manifest(), records)
        self.assertEqual(summary["run_status"], "INVALID")
        self.assertTrue(summary["invalid_reason"].startswith("technical failure"))
        self.assertEqual(summary["level"], "S1")
        self.assertNotIn("u", summary)

    def test_failed_control_is_invalid_not_unresolved(self):
        records = [record(i, W_LIKE) for i in range(N)]
        records[3]["technical"] = {"passed": False}
        self.assert_invalid(records)

    def test_non_finite_readout_is_invalid(self):
        records = [record(i, W_LIKE) for i in range(N)]
        records[5]["answer_logits"][2] = float("nan")
        self.assert_invalid(records)

    def test_design_index_mismatch_is_invalid(self):
        records = [record(i, W_LIKE) for i in range(N)]
        records[7]["design_indices"] = {**CELL, "i_L": 6}
        self.assert_invalid(records)


class NRuleAtFreezeTests(unittest.TestCase):
    def test_frozen_N_must_follow_the_rule_at_split_B(self):
        self.assertEqual(ra.n_rule(0.0)["N"], 200)
        self.assertEqual(manifest(resolution_rate_B=0.95)["N_rule"]["N"], 500)
        bad = manifest(N=200, resolution_rate_B=0.95)
        bad["N_rule"]["N"] = 200
        with self.assertRaisesRegex(ValueError, "N rule"):
            ra.analyze_confirmation(bad, [record(i, W_LIKE) for i in range(200)])
        wrong_label = manifest()
        wrong_label["N_rule"]["adequacy_powered"] = False
        with self.assertRaisesRegex(ValueError, "N rule"):
            ra.analyze_confirmation(wrong_label, [record(i, W_LIKE) for i in range(N)])

    def test_development_sets_N_from_split_B(self):
        import mixing_synthetic_worlds as worlds
        rng = random.Random(9)
        a = worlds.development_split("w_inside", 60, rng)
        b = worlds.development_split("w_inside", 200, rng)
        b["conflict"] = [dict(x, answer_mass_full_vocab=worlds.LOW_MASS) if k < 8 else x
                         for k, x in enumerate(b["conflict"])]
        decision = ra.development_decision(
            {"c1": a}, {"c1": b}, [("c1", CELL)], n=N_GROUPS, s_min_quantile=0.99, s_min_floor=0.10,
            d_min=0.20, agreement_transfer_floor=0.9, false_invalid_rate=0.05, resamples=50, seed=1)
        self.assertEqual(decision["status"], "PROCEED")
        self.assertAlmostEqual(decision["freeze"]["N_rule"]["unresolved_rate_B"], 0.04)
        self.assertEqual(decision["freeze"]["N"], 400)
        self.assertEqual(decision["freeze"]["mean_gate"]["N"], 400)

    def test_bisection_power_equals_the_scan(self):
        import mixing_round1_design as design
        for N_, coverage, u in ((150, 0.9, 0.0), (200, 0.7, 0.02), (300, 0.9, 0.05), (500, 0.7, 0.1)):
            a, b = ra.power(N_, coverage, u), design.power_by_scan(N_, coverage, u)
            self.assertAlmostEqual(a["adequate"], b["adequate"], places=12)
            self.assertAlmostEqual(a["excluded"], b["excluded"], places=12)


class ContractAndDesignTests(unittest.TestCase):
    def test_kappa_of_one_half_or_more_is_refused(self):
        bad = manifest()
        bad["rule"]["kappa"] = 0.5
        with self.assertRaisesRegex(ValueError, "contract"):
            ra.analyze_confirmation(bad, [record(i, W_LIKE) for i in range(N)])

    def test_mass_floor_and_readout_are_part_of_the_contract(self):
        self.assertEqual(ra.CONTRACT["answer_mass_floor"], 0.5)
        bad = manifest()
        bad["rule"]["answer_mass_floor"] = 0.1
        with self.assertRaisesRegex(ValueError, "contract"):
            ra.analyze_confirmation(bad, [record(i, W_LIKE) for i in range(N)])

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
        records = [record(i, W_LIKE) for i in range(N)]
        with self.assertRaisesRegex(ValueError, "expected N"):
            ra.analyze_confirmation(manifest(), records[:-1])
        duplicate = copy.deepcopy(records)
        duplicate[4]["case_id"] = duplicate[3]["case_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            ra.analyze_confirmation(manifest(), duplicate)
        gap = records[:5] + records[6:] + [record(N, W_LIKE)]
        with self.assertRaisesRegex(ValueError, "draw_index"):
            ra.analyze_confirmation(manifest(), gap)

    def test_level_names_the_excluded_profile(self):
        self.assertEqual(ra.level("VALID", {"W_T": "excluded", "A_T": "adequate"}), "S3 (W_T excluded only)")
        self.assertEqual(ra.level("VALID", {"W_T": "undecided", "A_T": "excluded"}), "S3 (A_T excluded)")
        self.assertEqual(ra.level("VALID", {"W_T": "excluded", "A_T": "excluded"}), "S3 (both excluded)")
        self.assertEqual(ra.level("VALID", {"W_T": "adequate", "A_T": "undecided"}), "S2")
        self.assertEqual(ra.level("INVALID", {"W_T": "INVALID", "A_T": "INVALID"}), "S1")


class AgreementAnchorTests(unittest.TestCase):
    """P, L and R weigh equally in T_A, and at least 90% of agreement runs must resolve."""

    @staticmethod
    def agreement(j, concentrated, mass=0.9):
        if concentrated:
            masses = {j: 0.97}
        else:
            other = CELL["i_R"] if j != CELL["i_R"] else CELL["i_L"]
            masses = {j: 0.582, other: 0.388}
        p = [1e-6] * N_GROUPS
        for index, value in masses.items():
            p[index] += value
        p[CELL["i_N"]] += 0.03
        total = math.fsum(p)
        return measurement([x / total for x in p], mass)

    def test_targets_weigh_equally_whatever_their_run_counts(self):
        runs = ([(k, CELL["i_P"], self.agreement(CELL["i_P"], True)) for k in range(6)]
                + [(6, CELL["i_L"], self.agreement(CELL["i_L"], False)),
                   (7, CELL["i_R"], self.agreement(CELL["i_R"], False))])
        result = ra.agreement_anchor(runs, CELL, 1, s_min=0.10)
        by_target = result["T_A_by_target"]
        self.assertAlmostEqual(result["T_A"], math.fsum(by_target.values()) / 3, places=12)
        pooled = math.fsum([by_target["i_P"]] * 6 + [by_target["i_L"], by_target["i_R"]]) / 8
        self.assertGreater(pooled - result["T_A"], 0.05)
        self.assertEqual(result["agreement_runs_by_target"], {"i_P": 6, "i_L": 1, "i_R": 1})

    def test_a_target_without_a_resolved_run_leaves_T_A_undefined(self):
        runs = [(k, CELL["i_P"], self.agreement(CELL["i_P"], True)) for k in range(3)]
        runs.append((3, CELL["i_L"], self.agreement(CELL["i_L"], True)))
        runs.append((4, CELL["i_R"], self.agreement(CELL["i_R"], True, mass=0.1)))
        with self.assertRaisesRegex(ValueError, "i_R"):
            ra.anchors([measurement(W_LIKE)], runs, CELL, 1, s_min=0.10)
        with self.assertRaisesRegex(ValueError, "not one of"):
            ra.agreement_anchor([(0, CELL["i_N"], measurement(W_LIKE))], CELL, 1, 0.10)

    def test_agreement_resolution_below_ninety_percent_stops(self):
        import mixing_synthetic_worlds as worlds
        rng = random.Random(5)
        split = worlds.development_split("w_inside", 60, rng)
        split["agreement"] = [(k, j, dict(x, answer_mass_full_vocab=0.1) if k % 5 == 0 else x)
                              for k, j, x in split["agreement"]]
        decision = ra.development_decision(
            {"c1": split}, {"c1": split}, [("c1", CELL)], n=N_GROUPS, s_min_quantile=0.99,
            s_min_floor=0.10, d_min=0.20, agreement_transfer_floor=0.0,
            false_invalid_rate=0.05, resamples=50, seed=1)
        self.assertEqual((decision["status"], decision["level"]), ("STOP", "S1"))
        self.assertIn("agreement-control runs", decision["reason"])
        self.assertAlmostEqual(decision["anchors"]["agreement_resolution_rate"], 0.8)

    def test_frozen_agreement_resolution_below_the_floor_is_refused(self):
        bad = manifest()
        bad["development_gates"]["agreement_resolution_rate_B"] = 0.89
        with self.assertRaisesRegex(ValueError, "agreement-control resolution"):
            ra.analyze_confirmation(bad, [record(i, W_LIKE) for i in range(N)])


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
        records = [record(i, W_LIKE if i % 3 else UNRESOLVED) for i in range(N)]
        summary = ra.analyze_confirmation(manifest(), records)
        table = ra.post_hoc_strictness(summary)
        self.assertEqual(table["matches_by_kappa"]["0.25"]["W_T"], summary["profiles"]["W_T"]["k_match"])
        self.assertIn("decides nothing", table["status"])
        self.assertNotIn("statuses", table)


if __name__ == "__main__":
    unittest.main()
