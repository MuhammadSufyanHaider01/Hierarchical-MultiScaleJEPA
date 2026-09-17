from __future__ import annotations

import unittest

from evaluation import select_pareto_knee_exit


class ParetoKneeSelectorTest(unittest.TestCase):
    def test_canonical_l4_knee(self):
        result = select_pareto_knee_exit(
            {2: 0.840, 4: 0.920, 6: 0.935, 8: 0.940},
            {2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0},
            higher_is_better=True,
        )
        self.assertEqual(result.selected_exit, 4)
        self.assertGreater(result.knee_scores[4], result.knee_scores[6])
        self.assertFalse(result.used_fallback)

    def test_lower_is_better_metrics_are_oriented(self):
        result = select_pareto_knee_exit(
            {2: 10.0, 4: 5.0, 6: 4.25, 8: 4.0},
            {2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0},
            higher_is_better=False,
        )
        self.assertEqual(result.selected_exit, 4)
        self.assertEqual(result.oriented_utility[2], -10.0)
        self.assertEqual(result.oriented_utility[8], -4.0)

    def test_dominated_exit_is_never_selectable(self):
        result = select_pareto_knee_exit(
            {2: 0.84, 4: 0.92, 6: 0.91, 8: 0.94},
            {2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0},
            higher_is_better=True,
        )
        self.assertEqual(result.dominated_exits, (6,))
        self.assertNotIn(6, result.pareto_exits)
        self.assertNotEqual(result.selected_exit, 6)

    def test_performance_endpoint_is_derived_not_forced_to_l8(self):
        result = select_pareto_knee_exit(
            {2: 0.84, 4: 0.92, 6: 0.91, 8: 0.90},
            {2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0},
            higher_is_better=True,
        )
        self.assertEqual(result.pareto_exits, (2, 4))
        self.assertEqual(result.performance_endpoint, 4)
        self.assertEqual(result.selected_exit, 4)
        self.assertEqual(result.fallback_reason, "no_intermediate_pareto_point")

    def test_no_positive_knee_falls_back_to_performance_endpoint(self):
        result = select_pareto_knee_exit(
            {2: 0.80, 4: 0.805, 6: 0.83, 8: 0.90},
            {2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0},
            higher_is_better=True,
        )
        self.assertEqual(result.selected_exit, 8)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.fallback_reason, "no_positive_knee")

    def test_two_pareto_points_fall_back_to_performance_endpoint(self):
        result = select_pareto_knee_exit(
            {2: 0.80, 4: 0.80, 6: 0.80, 8: 0.90},
            {2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0},
            higher_is_better=True,
        )
        self.assertEqual(result.pareto_exits, (2, 8))
        self.assertEqual(result.selected_exit, 8)
        self.assertEqual(result.fallback_reason, "no_intermediate_pareto_point")

    def test_exact_knee_tie_prefers_shallower_exit(self):
        result = select_pareto_knee_exit(
            {2: 0.0, 4: 0.5, 6: 0.75, 8: 1.0},
            {2: 1.0, 4: 2.0, 6: 3.0, 8: 5.0},
            higher_is_better=True,
        )
        self.assertEqual(result.knee_scores[4], result.knee_scores[6])
        self.assertEqual(result.selected_exit, 4)

    def test_single_pareto_exit_has_explicit_fallback(self):
        result = select_pareto_knee_exit(
            {2: 0.90, 4: 0.89, 6: 0.88, 8: 0.87},
            {2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0},
            higher_is_better=True,
        )
        self.assertEqual(result.selected_exit, 2)
        self.assertEqual(result.fallback_reason, "single_pareto_exit")

    def test_zero_frontier_compute_range_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "zero compute range"):
            select_pareto_knee_exit(
                {2: 0.8, 4: 0.8},
                {2: 2.0, 4: 2.0},
                higher_is_better=True,
            )

    def test_mismatched_keys_and_nonfinite_values_fail(self):
        with self.assertRaisesRegex(ValueError, "same exit layers"):
            select_pareto_knee_exit({2: 0.8}, {4: 2.0}, higher_is_better=True)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "not finite"):
                select_pareto_knee_exit({2: invalid}, {2: 2.0}, higher_is_better=True)
            with self.assertRaisesRegex(ValueError, "not finite"):
                select_pareto_knee_exit({2: 0.8}, {2: invalid}, higher_is_better=True)
        with self.assertRaisesRegex(ValueError, "positive"):
            select_pareto_knee_exit({2: 0.8}, {2: 0.0}, higher_is_better=True)

    def test_repeated_calls_are_deterministic(self):
        metrics = {2: 0.840, 4: 0.920, 6: 0.935, 8: 0.940}
        costs = {2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0}
        self.assertEqual(
            select_pareto_knee_exit(metrics, costs, higher_is_better=True),
            select_pareto_knee_exit(metrics, costs, higher_is_better=True),
        )


if __name__ == "__main__":
    unittest.main()
