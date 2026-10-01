"""Symbolic contract tests only; production confirmation seed is never sampled."""
from dataclasses import replace
import inspect
import math
from pathlib import Path
import sys
import unittest

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(APP / "src"))

from tracr_demo import design as d  # noqa: E402


class FamilyDesign(unittest.TestCase):
    def setUp(self):
        self.family = d.generate_families(1, seed=71, split="development")[0]

    def test_reproducible_unique_families_and_fixed_integer_vocabulary(self):
        first = d.generate_families(30, seed=71, split="development")
        self.assertEqual(first, d.generate_families(30, seed=71, split="development"))
        self.assertEqual(len({family.family_id for family in first}), 30)
        self.assertEqual(len({family.signature for family in first}), 30)
        self.assertEqual(d.VOCABULARY, tuple(range(12)))
        for family in first:
            cases = (family.shared,) + family.discriminators + family.followups
            for case in cases:
                self.assertEqual((case.source_query, case.target_query), (1, 0))
                self.assertEqual((case.address, case.native_index), (2, 3))
                self.assertEqual(case.to_dict()["source_model_position"], 2)
                self.assertEqual(case.to_dict()["target_model_position"], 1)
                self.assertTrue(set(case.recipient + case.donor) <= set(d.VOCABULARY))
                self.assertTrue(all(type(token) is int for token in case.recipient + case.donor))

    def test_exact_shared_discriminator_followup_contrasts(self):
        family = self.family
        shared_pred = d.predictions(family.shared)
        self.assertEqual(shared_pred["address"], shared_pred["donor_answer"])
        for case in family.discriminators:
            pred = d.predictions(case)
            followup = family.followup_for(case.case_id)
            next_pred = d.predictions(followup)
            self.assertNotEqual(pred["address"], pred["donor_answer"])
            self.assertEqual(case.recipient, family.shared.recipient)
            self.assertEqual(case.donor[:2] + case.donor[3:],
                             family.shared.donor[:2] + family.shared.donor[3:])
            self.assertEqual(case.donor, followup.donor)
            self.assertEqual(pred["donor_answer"], next_pred["donor_answer"])
            self.assertNotIn(next_pred["address"], tuple(pred.values()))
            self.assertEqual(d.native_prediction(case), d.native_prediction(followup))
            self.assertNotIn(d.native_prediction(case), tuple(pred.values()))
            self.assertNotIn(d.native_prediction(followup), tuple(next_pred.values()))

    def test_fresh_split_excludes_signatures_and_ids_without_using_production_seed(self):
        # Alternate fixture seeds exercise split logic, never the held-out seed.
        dev = d.generate_families(20, seed=111, split="development")
        fresh = d.generate_families(20, seed=222, split="confirmation", excluded_families=dev)
        self.assertFalse({f.signature for f in dev} & {f.signature for f in fresh})
        self.assertFalse({f.family_id for f in dev} & {f.family_id for f in fresh})
        later = d.generate_families(20, seed=111, split="development", excluded_families=dev)
        self.assertFalse({f.signature for f in dev} & {f.signature for f in later})
        self.assertFalse({f.family_id for f in dev} & {f.family_id for f in later})
        self.assertNotEqual(d.DEV_SEED, d.CONFIRMATION_SEED)

    def test_signature_ignores_split_seed_ids_and_menu_order(self):
        family = self.family
        renamed = replace(family, split="confirmation", seed=999)
        reordered = replace(family, discriminators=family.discriminators[::-1],
                            followups=family.followups[::-1])
        self.assertEqual(family.signature, renamed.signature)
        self.assertEqual(family.signature, reordered.signature)

    def test_invalid_families_cases_and_generation_are_refused(self):
        family = self.family
        for kwargs in (dict(count=0, seed=1, split="development"),
                       dict(count=True, seed=1, split="development"),
                       dict(count=1, seed=True, split="development"),
                       dict(count=1, seed=-1, split="development"),
                       dict(count=1, seed=1, split="unknown"),
                       dict(count=1, seed=d.CONFIRMATION_SEED, split="development"),
                       dict(count=1, seed=222, split="confirmation"),
                       dict(count=1, seed=71, split="confirmation", excluded_families=[family])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                d.generate_families(**kwargs)
        for kwargs in (dict(recipient=(1, 2, 3)), dict(donor=(1, 1, 2, 3)),
                       dict(source_query=2), dict(source_query=True),
                       dict(recipient=(0, 1, 2, 12)), dict(stage="unknown")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(family.shared, **kwargs)
        with self.assertRaises(ValueError):
            replace(family, discriminators=family.discriminators[:1])
        with self.assertRaises(ValueError):
            family.followup_for("not-allowed")


class PredictionAndRetention(unittest.TestCase):
    def setUp(self):
        self.family = d.generate_families(1, seed=71, split="development")[0]
        self.tolerance = d.Tolerance(scientific_tolerance=.1, numeric_allowance=.01)
        choice = self.family.discriminators[0]
        self.cases = (self.family.shared, choice, self.family.followup_for(choice.case_id))

    def trajectory(self, observations, **kwargs):
        return d.cumulative_retention(self.cases[:len(observations)], observations,
                                      self.tolerance, controls_passed=True, **kwargs)

    def test_one_hot_vectors_cover_every_fixed_vocab_coordinate(self):
        for token in d.VOCABULARY:
            result = d.one_hot(token)
            self.assertEqual(len(result), 12)
            self.assertEqual(sum(result), 1.0)
            self.assertEqual(result[token], 1.0)
        for token in (-1, 12, True, 1.0):
            with self.assertRaises(ValueError):
                d.one_hot(token)

    def test_shared_both_then_address_only_and_followup_preserves_it(self):
        observations = [d.predictions(case)["address"] for case in self.cases]
        trajectory = self.trajectory(observations)
        self.assertEqual(trajectory[0]["retained_after"], list(d.CANDIDATES))
        self.assertEqual(trajectory[0]["outcome"], "insufficient_evidence")
        for step in trajectory[1:]:
            self.assertEqual(step["retained_after"], ["address"])
            self.assertEqual(step["outcome"], "resolved")
            self.assertFalse(step["native_null_compatible"])
            self.assertEqual(step["scientific_tolerance"], .1)
            self.assertEqual(step["numeric_allowance"], .01)
        self.assertEqual(trajectory[2]["retained_before"], ["address"])

    def test_donor_candidate_can_survive_without_ground_truth_override(self):
        observations = [d.predictions(case)["donor_answer"] for case in self.cases]
        self.assertEqual(self.trajectory(observations)[-1]["retained_after"], ["donor_answer"])

    def test_native_null_and_mixed_scores_are_neither_candidate(self):
        common = d.predictions(self.cases[0])["address"]
        native = self.trajectory([common, d.native_prediction(self.cases[1])])[-1]
        self.assertEqual(native["retained_after"], [])
        self.assertEqual(native["outcome"], "no_candidate_fits")
        self.assertTrue(native["native_null_compatible"])
        pred = d.predictions(self.cases[1])
        mixture = tuple((a + b) / 2 for a, b in zip(pred["address"], pred["donor_answer"]))
        mixed = self.trajectory([common, mixture])[-1]
        self.assertEqual(mixed["outcome"], "no_candidate_fits")
        self.assertFalse(mixed["native_null_compatible"])

    def test_excluded_candidate_cannot_reappear(self):
        observations = [d.predictions(case)[name] for case, name in
                        zip(self.cases, ("address", "address", "donor_answer"))]
        trajectory = self.trajectory(observations)
        self.assertEqual(trajectory[1]["retained_after"], ["address"])
        self.assertEqual(trajectory[2]["retained_after"], [])

    def test_equivalence_group_is_preserved_and_cannot_switch_member_between_rows(self):
        group = [d.CANDIDATES]
        observations = [d.predictions(case)["address"] for case in self.cases]
        trajectory = self.trajectory(observations, equivalence_groups=group)
        self.assertTrue(all(step["retained_after"] == list(d.CANDIDATES) for step in trajectory))
        observations[-1] = d.predictions(self.cases[-1])["donor_answer"]
        contradictory = self.trajectory(observations, equivalence_groups=group)
        self.assertEqual(contradictory[-1]["outcome"], "no_candidate_fits")

    def test_radius_boundary_and_calibration_components_are_explicit(self):
        self.assertAlmostEqual(self.tolerance.radius, .11)
        self.assertEqual(d.Tolerance(0, 0).radius, 0)
        self.assertLess(d.Tolerance(.49, .009).radius, .5)
        for scientific, numeric in ((.5, 0), (.4, .1), (-.1, 0), (0, -.1),
                                    (math.nan, 0), (0, math.inf), (True, 0)):
            with self.subTest(scientific=scientific, numeric=numeric), self.assertRaises(ValueError):
                d.Tolerance(scientific, numeric)
        # compatible_set uses <= at the valid radius; <.5 is the design gate.
        scores = list(d.predictions(self.cases[0])["address"])
        scores[0] += .125
        result = d.cumulative_retention(self.cases[:1], [scores], d.Tolerance(.125, 0),
                                        controls_passed=True)
        self.assertEqual(result[0]["retained_after"], list(d.CANDIDATES))

    def test_invalid_measurements_and_controls_are_errors_not_exclusion(self):
        common = d.predictions(self.cases[0])["address"]
        for value in (math.nan, math.inf, -math.inf, True, "1"):
            invalid = list(common)
            invalid[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.trajectory([invalid])
        for invalid in (common[:-1], common + (0.0,)):
            with self.assertRaises(ValueError):
                self.trajectory([invalid])
        for valid in (False, None, 1, "yes"):
            with self.subTest(valid=valid), self.assertRaises(ValueError):
                d.cumulative_retention(self.cases[:1], [common], self.tolerance,
                                        controls_passed=valid)
        with self.assertRaises(ValueError):
            self.trajectory([common], equivalence_groups=[("address", "unknown")])

    def test_invalid_or_cross_family_trajectories_are_refused(self):
        common = d.predictions(self.cases[0])["address"]
        other = d.generate_families(1, seed=72, split="development")[0]
        invalid_cases = [(), self.cases[1:], (self.cases[0], self.cases[0]),
                         (self.cases[0], self.cases[-1]),
                         (self.cases[0], other.discriminators[0]),
                         (self.cases[0], self.cases[1], self.family.followups[1])]
        for cases in invalid_cases:
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                d.cumulative_retention(cases, [common] * len(cases), self.tolerance,
                                        controls_passed=True)
        with self.assertRaises(ValueError):
            d.cumulative_retention(self.cases, [common], self.tolerance, controls_passed=True)


class DiscriminatorSelection(unittest.TestCase):
    def setUp(self):
        self.family = d.generate_families(1, seed=71, split="development")[0]
        self.tolerance = d.Tolerance(.1, .01)

    def test_two_allowed_choices_tie_break_without_outcomes(self):
        result = d.select_discriminator(self.family, d.CANDIDATES, self.tolerance)
        self.assertEqual(len(result["menu"]), 2)
        self.assertEqual([choice["separated_pairs"] for choice in result["menu"]], [1, 1])
        self.assertEqual(result["selected_case_id"], min(c.case_id for c in self.family.discriminators))
        self.assertEqual([choice["cost_units"] for choice in result["menu"]], [1, 1])
        signature = inspect.signature(d.select_discriminator)
        self.assertEqual(set(signature.parameters), {"family", "retained", "tolerance",
                                                     "used_case_ids", "equivalence_groups"})
        with self.assertRaises(TypeError):
            d.select_discriminator(self.family, d.CANDIDATES, self.tolerance, observations=[])

    def test_remaining_choice_and_no_separable_rival_stop(self):
        first = self.family.discriminators[0]
        result = d.select_discriminator(self.family, d.CANDIDATES, self.tolerance,
                                         used_case_ids=[first.case_id])
        self.assertEqual(result["selected_case_id"], self.family.discriminators[1].case_id)
        for retained in ([], ["address"], ["donor_answer"]):
            self.assertIsNone(d.select_discriminator(self.family, retained,
                                                     self.tolerance)["selected_case_id"])
        exhausted = d.select_discriminator(self.family, d.CANDIDATES, self.tolerance,
                                            used_case_ids=[c.case_id for c in self.family.discriminators])
        self.assertIsNone(exhausted["selected_case_id"])

    def test_equivalence_groups_and_invalid_retained_sets_are_preserved(self):
        result = d.select_discriminator(self.family, d.CANDIDATES, self.tolerance,
                                         equivalence_groups=[d.CANDIDATES])
        self.assertIsNone(result["selected_case_id"])
        for retained in (["unknown"], ["address", "address"]):
            with self.assertRaises(ValueError):
                d.select_discriminator(self.family, retained, self.tolerance)
        with self.assertRaises(ValueError):
            d.select_discriminator(self.family, ["address"], self.tolerance,
                                   equivalence_groups=[d.CANDIDATES])
        with self.assertRaises(ValueError):
            d.select_discriminator(self.family, d.CANDIDATES, self.tolerance,
                                   used_case_ids=["unknown"])


if __name__ == "__main__":
    unittest.main()
