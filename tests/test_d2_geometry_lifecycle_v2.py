"""Pure-Python regression tests for D2 geometry/lifecycle v2."""

import importlib.util
import inspect
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMSOLRunner


CylinderRunner = load_runner(
    "d2_cylinder_runner",
    ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d2_zigzag_runner",
    ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")


class CylinderErosionTests(unittest.TestCase):
    def setUp(self):
        self.runner = CylinderRunner()

    def test_uniform_profile_has_no_internal_shoulders(self):
        radii = [2.5e-3] * self.runner.seg_count
        self.assertEqual(self.runner._shoulder_areas(radii), [0.0] * 8)
        self.assertAlmostEqual(
            self.runner._feature_volume(radii, self.runner.Lseg),
            self.runner.reference_volume, places=20)

    def test_step_shoulder_is_owned_by_larger_segment(self):
        radii = [2.0e-3, 3.0e-3, 1.0e-3]
        areas = self.runner._shoulder_areas(radii)
        expected = math.pi * ((3.0e-3) ** 2 - (2.0e-3) ** 2)
        expected += math.pi * ((3.0e-3) ** 2 - (1.0e-3) ** 2)
        self.assertAlmostEqual(areas[0], 0.0)
        self.assertAlmostEqual(areas[1], expected)
        self.assertAlmostEqual(areas[2], 0.0)

    def test_segment_temperature_mask_has_local_analytic_area(self):
        condition, area = self.runner._segment_lateral_mask(3, 2.5e-3)
        self.assertIn("z>", condition)
        self.assertIn("&&z<", condition)
        self.assertAlmostEqual(
            area, 2.0 * math.pi * 2.5e-3 * 0.8 * self.runner.Lseg)

    def test_mapped_radius_loss_conserves_surface_flux_volume(self):
        radii = [2.0e-3, 3.0e-3, 1.0e-3]
        temperatures = [2600.0, 2800.0, 2500.0]
        rates, shoulder_areas, volume_rates = (
            self.runner._cylinder_erosion_rates(radii, temperatures))
        mapped = sum(
            2.0 * math.pi * radius * self.runner.Lseg * rate
            for radius, rate in zip(radii, rates))
        self.assertAlmostEqual(mapped, sum(volume_rates), places=20)
        self.assertGreater(sum(shoulder_areas), 0.0)


class ZigzagErosionTests(unittest.TestCase):
    def setUp(self):
        self.runner = ZigzagRunner()

    def test_turn_caps_are_zero_for_equal_sides_and_local_for_mismatch(self):
        self.assertEqual(
            self.runner._turn_cap_areas([0.5e-3] * 4), [0.0] * 4)
        sides = [0.4e-3, 0.6e-3, 0.5e-3]
        areas = self.runner._turn_cap_areas(sides)
        self.assertEqual(areas[0], 0.0)
        self.assertGreater(areas[1], 0.0)
        self.assertEqual(areas[2], 0.0)

    def test_local_block_rates_conserve_all_exposed_surface_volume(self):
        sides = [0.4e-3, 0.6e-3, 0.5e-3]
        lengths = [10e-3, 4e-3, 12e-3]
        temperatures = [2700.0, 2850.0, 2600.0]
        rates, cap_areas, volume_rates = self.runner._block_erosion_rates(
            sides, temperatures, lengths)
        derivatives = self.runner._block_volume_derivatives(sides, lengths)
        mapped = sum(
            derivative * rate
            for derivative, rate in zip(derivatives, rates))
        self.assertAlmostEqual(mapped, sum(volume_rates), places=20)
        self.assertGreater(sum(cap_areas), 0.0)
        self.assertGreater(rates[1] / rates[0], 2.0)

    def test_equal_sides_and_temperature_recede_uniformly_at_turns(self):
        sides = [0.5e-3] * 4
        lengths = [2e-3, 5e-3, 3e-3, 4e-3]
        rates, cap_areas, _ = self.runner._block_erosion_rates(
            sides, [2800.0] * 4, lengths)
        self.assertEqual(cap_areas, [0.0] * 4)
        for rate in rates[1:]:
            self.assertAlmostEqual(rate / rates[0], 1.0, places=12)

    def test_subquantum_state_does_not_create_nonexistent_shoulder(self):
        state_sides = [0.500e-3, 0.498e-3]
        represented_sides = [0.500e-3, 0.500e-3]
        _, cap_areas, _ = self.runner._block_erosion_rates(
            state_sides, [2700.0, 2800.0], [10e-3, 10e-3],
            surface_sides=represented_sides)
        self.assertEqual(cap_areas, [0.0, 0.0])

    def test_exposed_lengths_remove_half_connector_at_each_turn(self):
        sides = [0.4e-3, 0.6e-3, 0.5e-3]
        lengths = [2e-3, 3e-3, 4e-3]
        exposed = self.runner._exposed_segment_lengths(sides, lengths)
        self.assertAlmostEqual(exposed[0], lengths[0] - 0.3e-3)
        self.assertAlmostEqual(exposed[1], lengths[1] - 0.6e-3)
        self.assertAlmostEqual(exposed[2], lengths[2] - 0.3e-3)

        turn_areas = self.runner._turn_surface_areas(sides)
        self.assertAlmostEqual(turn_areas[0][0], 4.0 * (0.6e-3) ** 2)
        self.assertAlmostEqual(
            turn_areas[0][1], (0.6e-3) ** 2 - (0.4e-3) ** 2)

    def test_turn_connector_volume_and_derivative_are_consistent(self):
        sides = [0.4e-3, 0.6e-3, 0.5e-3]
        lengths = [10e-3, 4e-3, 12e-3]
        derivatives = self.runner._block_volume_derivatives(sides, lengths)
        base_stub_radii = [self.runner.R0, self.runner.R0]
        step = 1.0e-10
        for index, derivative in enumerate(derivatives):
            plus = list(sides)
            minus = list(sides)
            plus[index] += step
            minus[index] -= step
            v_plus = self.runner._erosion_state_volume(
                plus, lengths, base_stub_radii, self.runner.STUB_LEN)
            v_minus = self.runner._erosion_state_volume(
                minus, lengths, base_stub_radii, self.runner.STUB_LEN)
            finite_difference = (v_plus - v_minus) / (2.0 * step)
            self.assertAlmostEqual(
                finite_difference / derivative, 1.0, places=6)

    def test_equal_side_initial_state_keeps_exact_target_volume(self):
        side, _, _ = self.runner.compute_side_and_blocks(
            8, 104e-3, 0.8e-3)
        points = self.runner.build_full_path(
            8, 104e-3, 0.8e-3, self.runner.L0 - 0.8e-3,
            self.runner.STUB_LEN, self.runner.L0)
        lengths = self.runner.path_segment_lengths(points)
        volume = self.runner._erosion_state_volume(
            [side] * len(lengths), lengths,
            [self.runner.R0, self.runner.R0], self.runner.STUB_LEN)
        self.assertAlmostEqual(volume, self.runner.V0, places=20)
        self.assertEqual(self.runner.vol_tol, 1.0e-4)

    def test_geometry_projection_is_local_bounded_and_hits_failure(self):
        self.runner._init_side = 0.5e-3
        exact = [0.499e-3, 0.497e-3, 0.492e-3, 0.4e-3]
        projected = self.runner._project_block_sides_to_geometry(exact)
        quantum = (self.runner.geometry_side_quantum_fraction
                   * self.runner._init_side)
        self.assertEqual(len(set(projected)), 4)
        self.assertLessEqual(
            max(abs(a - b) for a, b in zip(exact, projected)),
            0.5 * quantum + 1.0e-15)
        self.assertAlmostEqual(projected[-1], 0.4e-3)

    def test_terminal_stub_sidewall_and_inner_shoulder_both_erode(self):
        radii = [2.5e-3, 2.5e-3]
        stub_temperatures = [800.0, 850.0]
        block_sides = [0.5e-3, 0.45e-3]
        block_temperatures = [2400.0, 2500.0]
        rates, shoulder_areas, volume_rates = self.runner._stub_erosion_rates(
            radii, stub_temperatures, block_sides, block_temperatures)
        self.assertTrue(all(area > 0.0 for area in shoulder_areas))
        mapped = sum(
            2.0 * math.pi * radius * self.runner.STUB_LEN * rate
            for radius, rate in zip(radii, rates))
        self.assertAlmostEqual(mapped, sum(volume_rates), places=20)
        self.assertTrue(all(rate > 0.0 for rate in rates))

    def test_per_block_geometry_preserves_distinct_sides_and_connectivity(self):
        points = self.runner.build_full_path(
            4, 12e-3, 2e-3, 13e-3,
            self.runner.STUB_LEN, self.runner.L0)
        lengths = self.runner.path_segment_lengths(points)
        sides = [0.35e-3 + index * 0.01e-3
                 for index in range(len(lengths))]
        blocks = self.runner.build_blocks_with_sides(points, sides)
        turns = self.runner.build_turn_blocks(points, sides)
        self.assertEqual(len(blocks), len(sides))
        self.assertEqual(len(turns), len(blocks) - 1)

        for block, side in zip(blocks, sides):
            _, _, _, _, sx, sy, sz = block
            cross_section = sorted((sx, sy, sz))[:2]
            self.assertAlmostEqual(cross_section[0], side)
            self.assertAlmostEqual(cross_section[1], side)

        for left, right in zip(blocks[:-1], blocks[1:]):
            _, lx, ly, lz, lsx, lsy, lsz = left
            _, rx, ry, rz, rsx, rsy, rsz = right
            overlaps = (
                min(lx + lsx, rx + rsx) - max(lx, rx),
                min(ly + lsy, ry + rsy) - max(ly, ry),
                min(lz + lsz, rz + rsz) - max(lz, rz),
            )
            self.assertTrue(all(value > 0.0 for value in overlaps), overlaps)

        for turn, left_side, right_side in zip(turns, sides[:-1], sides[1:]):
            _, _, _, _, sx, sy, sz = turn
            expected = max(left_side, right_side)
            self.assertEqual((sx, sy, sz), (expected, expected, expected))

        self.runner._path_points = points
        self.runner._segment_endpoints = self.runner.path_segments(points)
        self.runner._turn_flags = self.runner.turn_flags(
            self.runner._segment_endpoints)
        condition, active_area = self.runner._block_lateral_mask(
            1, blocks[1], sides)
        self.assertIn("&&", condition)
        self.assertGreater(active_area, 0.0)
        self.assertLess(active_area, 4.0 * sides[1] * lengths[1])

    def test_short_terminal_segment_mask_uses_exposed_endpoint_sliver(self):
        n_runs, run_length, z_first = 8, 104e-3, 0.8e-3
        side, blocks, _ = self.runner.compute_side_and_blocks(
            n_runs, run_length, z_first)
        points = self.runner.build_full_path(
            n_runs, run_length, z_first, self.runner.L0 - z_first,
            self.runner.STUB_LEN, self.runner.L0)
        self.runner._path_points = points
        self.runner._segment_endpoints = self.runner.path_segments(points)
        self.runner._turn_flags = self.runner.turn_flags(
            self.runner._segment_endpoints)
        sides = [side] * len(blocks)
        axis, low, high, exposed = self.runner._block_exposed_interval(
            0, sides)
        self.assertEqual(axis, "z")
        self.assertAlmostEqual(
            exposed, z_first - self.runner.STUB_LEN - 0.5 * side)
        self.assertGreater(high, low)
        condition, active_area = self.runner._block_lateral_mask(
            0, blocks[0], sides)
        self.assertIn("z>", condition)
        self.assertGreater(active_area, 0.0)


class LifecycleStateTests(unittest.TestCase):
    def test_both_runners_advertise_v2_contract(self):
        for runner in (CylinderRunner(), ZigzagRunner()):
            self.assertEqual(runner.lifecycle_version, "lifecycle_v2")
            self.assertIn("v2", runner.geometry_version)
            self.assertTrue(runner.erosion_model)
            annotated = runner._annotate_voltage_result(
                {}, "max_safe", "lifeTotalP03sphere_J")
            self.assertEqual(annotated["failureFraction"], 0.20)
            self.assertGreater(annotated["maxErosionStep_s"], 0.0)
            self.assertLessEqual(
                annotated["geometryVolumeTolerance_rel"], 1.0e-4)
        self.assertEqual(
            ZigzagRunner().turn_connector_rule,
            "max_adjacent_side_cube_split_external_faces")

    def test_failure_timestep_is_not_forced_to_one_second(self):
        for runner in (CylinderRunner(), ZigzagRunner()):
            runner.max_lifetime_h = 10.0
            dt = runner._next_erosion_timestep(
                current=[0.80025], limits=[0.8], rates=[0.001],
                resolution_delta=0.1, time_s=0.0)
            self.assertAlmostEqual(dt, 0.25)
            updated, losses, failed = runner._advance_erosion_features(
                [0.80025], [1.0], [0.8], [0.001], dt)
            self.assertEqual(updated, [0.8])
            self.assertAlmostEqual(losses[0], 0.2)
            self.assertEqual(failed, [0])

    def test_lifetime_cap_truncates_the_last_interval_exactly(self):
        for runner in (CylinderRunner(), ZigzagRunner()):
            runner.max_lifetime_h = 1.0
            dt = runner._next_erosion_timestep(
                current=[1.0], limits=[0.8], rates=[1.0e-9],
                resolution_delta=0.1, time_s=3599.75)
            self.assertAlmostEqual(dt, 0.25)

    def test_terminal_statuses_distinguish_failure_and_all_censors(self):
        cases = (
            ((True, False, False, "feature_loss_20pct"), "OK"),
            ((False, True, False, "lifetime_cap"),
             "CENSORED_LIFETIME_CAP"),
            ((False, False, True, "step_limit"),
             "CENSORED_STEP_LIMIT"),
            ((False, False, False, "negligible_erosion"),
             "CENSORED_NEGLIGIBLE_EROSION"),
            ((False, False, False, ""), "CENSORED_UNRESOLVED"),
        )
        for runner in (CylinderRunner(), ZigzagRunner()):
            for arguments, expected in cases:
                self.assertEqual(
                    runner._lifecycle_terminal_status(*arguments), expected)

    def test_overtemperature_crossing_is_interpolated(self):
        for runner in (CylinderRunner(), ZigzagRunner()):
            fraction = runner._overtemperature_fraction(
                3200.0, 3300.0, 3273.15)
            self.assertAlmostEqual(fraction, 0.7315)

    def test_internal_model_retry_never_disconnects_mph_client(self):
        methods = (
            CylinderRunner._restart_at_radii,
            ZigzagRunner._restart_at_geometry,
        )
        for method in methods:
            source = inspect.getsource(method)
            self.assertNotIn("self.stop()", source)
            self.assertNotIn("self.start()", source)
            self.assertIn("self.client.remove(self.model)", source)

    def test_legacy_java_paths_cannot_be_mistaken_for_d2_authority(self):
        paths = (
            ROOT / "cylinder_family" / "src" / "cylinder_baseline.java",
            ROOT / "zigzag_family" / "zigzag_baseline.java",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn("MODEL_AUTHORITY=LEGACY_D1_ONLY", source)
            self.assertIn("D2_RUNNER=", source)


if __name__ == "__main__":
    unittest.main()
