"""Pure-Python regression tests for the D3 operating-point contract."""

import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE = "lifeTotalP03escape_J"


def load_runner(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMSOLRunner


CylinderRunner = load_runner(
    "d3_cylinder_runner",
    ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d3_zigzag_runner",
    ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")


def exact_candidate(voltage, energy, uniformity=20.0, lifetime_h=10.0):
    return {
        "status": "OK",
        "Vwork_V": voltage,
        OBJECTIVE: energy,
        "U_pct": uniformity,
        "lifetimeH": lifetime_h,
        "maxErosionTmax_K": 3200.0,
        "failureReached": True,
        "lifetimeExact": True,
        "censored": False,
        "capLimited": False,
        "stepLimited": False,
    }


class RatedOperatingPointTests(unittest.TestCase):
    def runners(self):
        return (CylinderRunner(), ZigzagRunner())

    def test_both_families_publish_one_formal_contract(self):
        for runner in self.runners():
            self.assertEqual(runner.voltage_policy, "rated_lifecycle_scan")
            self.assertEqual(runner.voltage_objective, OBJECTIVE)
            self.assertEqual(
                runner.operating_point_version,
                "rated_lifecycle_energy_v1")
            self.assertEqual(
                runner.temperature_primary_domain,
                "all_tungsten_volume")
            self.assertEqual(
                runner.electrode_boundary_mode, "fixed_temperature")
            self.assertEqual(runner.max_lifetime_h, 1000.0)
            self.assertEqual(runner.max_erosion_steps, 150)

    def test_default_evaluate_delegates_to_the_formal_scan(self):
        cylinder = CylinderRunner()
        cylinder.evaluate_voltage_candidates = lambda *args, **kwargs: kwargs
        cylinder_call = cylinder.evaluate([2.5e-3] * cylinder.seg_count)
        self.assertEqual(cylinder_call["objective"], OBJECTIVE)
        self.assertEqual(
            cylinder_call["electrode_boundary_mode"], "fixed_temperature")

        zigzag = ZigzagRunner()
        zigzag.evaluate_voltage_candidates = lambda *args, **kwargs: kwargs
        zigzag_call = zigzag.evaluate(8, 104.0e-3, 0.8e-3)
        self.assertEqual(zigzag_call["objective"], OBJECTIVE)
        self.assertEqual(
            zigzag_call["electrode_boundary_mode"], "fixed_temperature")

    def test_highest_exact_lifecycle_energy_wins(self):
        for runner in self.runners():
            low = exact_candidate(80.0, 8.0e8)
            high = exact_candidate(70.0, 9.0e8)
            selected = runner._select_voltage_scan_result(
                [low, high], OBJECTIVE)
            self.assertEqual(selected["Vwork_V"], 70.0)
            self.assertTrue(selected["ratedVoltageEligible"])
            self.assertEqual(selected["ratedVoltageExactCandidateCount"], 2)
            self.assertEqual(selected["status"], "OK")

    def test_tie_break_is_lower_u_then_longer_life_then_lower_voltage(self):
        for runner in self.runners():
            candidates = [
                exact_candidate(90.0, 1.0e9, 12.0, 20.0),
                exact_candidate(80.0, 1.0e9, 10.0, 10.0),
                exact_candidate(70.0, 1.0e9, 10.0, 12.0),
                exact_candidate(60.0, 1.0e9, 10.0, 12.0),
            ]
            selected = runner._select_voltage_scan_result(
                candidates, OBJECTIVE)
            self.assertEqual(selected["Vwork_V"], 60.0)

    def test_censored_or_overtemperature_candidate_is_not_eligible(self):
        for runner in self.runners():
            censored = exact_candidate(80.0, 2.0e9)
            censored.update({
                "status": "CENSORED_LIFETIME_CAP",
                "failureReached": False,
                "lifetimeExact": False,
                "censored": True,
                "capLimited": True,
            })
            overtemperature = exact_candidate(90.0, 3.0e9)
            overtemperature["maxErosionTmax_K"] = runner.temp_limit_K
            valid = exact_candidate(70.0, 1.0e9)
            selected = runner._select_voltage_scan_result(
                [censored, overtemperature, valid], OBJECTIVE)
            self.assertEqual(selected["Vwork_V"], 70.0)
            self.assertEqual(selected["ratedVoltageExactCandidateCount"], 1)

    def test_no_exact_candidate_fails_instead_of_promoting_diagnostic(self):
        for runner in self.runners():
            candidate = exact_candidate(80.0, 2.0e9)
            candidate.update({
                "status": "CENSORED_LIFETIME_CAP",
                "failureReached": False,
                "lifetimeExact": False,
                "censored": True,
                "capLimited": True,
            })
            selected = runner._select_voltage_scan_result(
                [candidate], OBJECTIVE)
            self.assertEqual(
                selected["status"], "FAIL_RATED_VOLTAGE_INCONCLUSIVE")
            self.assertEqual(
                selected["ratedVoltageSourceStatus"],
                "CENSORED_LIFETIME_CAP")
            self.assertFalse(selected["ratedVoltageEligible"])
            self.assertFalse(selected["lifetimeExact"])

    def test_candidate_generation_respects_upper_bound_and_deduplicates(self):
        for runner in self.runners():
            values = runner._build_voltage_candidates(
                80.0, [100.0, 80.01, 72.0, 72.02, -1.0, float("nan")])
            self.assertEqual(values, [80.0, 72.0])
            self.assertTrue(all(0.0 < value <= 80.0 for value in values))


class TemperatureAndBoundaryTests(unittest.TestCase):
    def runners(self):
        return (CylinderRunner(), ZigzagRunner())

    def test_temperature_formula_and_invalid_inputs(self):
        for runner in self.runners():
            self.assertAlmostEqual(
                runner._temperature_uniformity(3000.0, 1000.0, 2000.0),
                100.0)
            self.assertTrue(math.isnan(
                runner._temperature_uniformity(1000.0, 1100.0, 1050.0)))
            self.assertTrue(math.isnan(
                runner._temperature_uniformity(1000.0, 900.0, 0.0)))

    def test_half_space_coefficient_matches_declared_resistance(self):
        for runner in self.runners():
            radius = 2.5e-3
            h = runner._copper_spreading_h(radius)
            conductance_from_h = h * math.pi * radius ** 2
            expected_conductance = (
                4.0 * runner.copper_thermal_conductivity_W_mK * radius)
            self.assertAlmostEqual(
                conductance_from_h / expected_conductance, 1.0, places=12)

    def test_boundary_aliases_are_explicit_and_invalid_mode_is_rejected(self):
        for runner in self.runners():
            self.assertEqual(
                runner._canonical_electrode_boundary_mode("fixed"),
                "fixed_temperature")
            self.assertEqual(
                runner._canonical_electrode_boundary_mode(
                    "semi_infinite_copper"),
                "semi_infinite_copper_spreading")
            with self.assertRaises(ValueError):
                runner._canonical_electrode_boundary_mode("adiabatic")

    def test_formal_python_runners_contain_no_temperature_fallback(self):
        paths = (
            ROOT / "cylinder_family" / "ML" / "comsol_runner.py",
            ROOT / "zigzag_family" / "ML" / "zigzag_runner.py",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("Tmax * 0.95", source)
            self.assertNotIn("Tmax*0.95", source)
            self.assertIn("temperatureFallbackUsed", source)
            self.assertIn("HeatFluxBoundary", source)

    def test_optimizers_pin_the_formal_policy(self):
        paths = (
            ROOT / "cylinder_family" / "ML" / "optuna_optimize.py",
            ROOT / "zigzag_family" / "ML" / "optuna_optimize.py",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn('voltage_policy="rated_lifecycle_scan"', source)
            self.assertIn(
                'voltage_objective="lifeTotalP03escape_J"', source)
            self.assertIn(
                'electrode_boundary_mode="fixed_temperature"', source)


if __name__ == "__main__":
    unittest.main()
