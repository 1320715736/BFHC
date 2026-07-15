"""Pure-Python regression contract for D4 thermal physics."""

import importlib.util
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
    "d4_test_cylinder_runner",
    ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d4_test_zigzag_runner",
    ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")


RUNNERS = (CylinderRunner, ZigzagRunner)


class MaterialModelTests(unittest.TestCase):
    def test_both_families_publish_the_same_d4_contract(self):
        cylinder = CylinderRunner()
        zigzag = ZigzagRunner()
        for attribute in (
                "physics_version", "material_model",
                "material_uncertainty_version", "sublimation_heat_version",
                "transient_version", "sublimation_enthalpy_J_kg"):
            self.assertEqual(getattr(cylinder, attribute),
                             getattr(zigzag, attribute))
        self.assertEqual(cylinder.physics_version, "thermal_s2s_d4_v1")
        self.assertEqual(cylinder.material_model, "nist_reference_d4_v1")
        self.assertTrue(cylinder.sublimation_heat_enabled)

    def test_optimizers_archive_d4_material_and_latent_versions(self):
        required = (
            "materialModel", "materialUncertaintyVersion", "rhoeScale",
            "kScale", "cpScale", "sublimationHeatEnabled",
            "sublimationHeatVersion", "transientVersion",
            "initialSublimationHeat_W",
            "initialSublimationHeatToElectric_pct",
        )
        for path in (
                ROOT / "cylinder_family" / "ML" / "optuna_optimize.py",
                ROOT / "zigzag_family" / "ML" / "optuna_optimize.py"):
            source = path.read_text(encoding="utf-8")
            for field in required:
                self.assertIn(f'"{field}"', source, f"{path}: {field}")

    def test_transient_material_includes_relative_permittivity(self):
        for path in (
                ROOT / "cylinder_family" / "ML" / "comsol_runner.py",
                ROOT / "zigzag_family" / "ML" / "zigzag_runner.py"):
            source = path.read_text(encoding="utf-8")
            self.assertIn('material.set("relpermittivity", ["1"])', source)

    def test_nist_reference_values_have_expected_magnitudes(self):
        runner = CylinderRunner()
        values_3000 = runner.material_property_values(3000.0)
        self.assertAlmostEqual(
            values_3000["rhoe_ohm_m"], 9.137e-7, delta=1.0e-12)
        self.assertAlmostEqual(
            values_3000["k_W_mK"], 89.5, delta=0.1)
        self.assertAlmostEqual(
            values_3000["cp_J_kgK"], 222.5, delta=0.2)

        values_room = runner.material_property_values(293.15)
        self.assertAlmostEqual(
            values_room["rhoe_ohm_m"], 5.5e-8, delta=1.0e-14)
        self.assertGreater(values_room["k_W_mK"], 170.0)
        self.assertAlmostEqual(
            values_room["cp_J_kgK"], 132.2, delta=0.2)

    def test_nist_conductivity_fit_tracks_reference_table(self):
        reference = {
            300.0: 174.0,
            1000.0: 118.0,
            2000.0: 98.0,
            3000.0: 89.5,
            3273.0: 88.5,
            3600.0: 87.8,
        }
        for temperature, expected in reference.items():
            actual = CylinderRunner._nist_k_W_mK(temperature)
            self.assertLess(abs(actual - expected) / expected, 0.005)

    def test_material_switch_and_scales_are_explicit(self):
        for runner_type in RUNNERS:
            runner = runner_type()
            nominal = runner.material_property_values(3000.0)
            runner.configure_material_model(
                "legacy", rhoe_scale=0.99, k_scale=0.95, cp_scale=0.97)
            legacy = runner.material_property_values(3000.0)
            self.assertEqual(runner.material_model, "legacy_v1")
            self.assertAlmostEqual(
                legacy["rhoe_ohm_m"],
                runner._legacy_material_values(3000.0)["rhoe_ohm_m"] * 0.99)
            self.assertNotAlmostEqual(
                nominal["cp_J_kgK"], legacy["cp_J_kgK"], places=3)
            with self.assertRaises(ValueError):
                runner.configure_material_model(k_scale=0.0)
            with self.assertRaises(ValueError):
                runner.configure_material_model("unknown")


class LatentHeatTests(unittest.TestCase):
    def test_latent_heat_flux_is_monotonic_and_bounded_at_limit(self):
        for runner_type in RUNNERS:
            runner = runner_type()
            values = [runner.sublimation_heat_flux_W_m2(temperature)
                      for temperature in (2500.0, 3000.0, 3273.15)]
            self.assertEqual(values, sorted(values))
            self.assertAlmostEqual(values[1], 27.94, delta=0.1)
            self.assertLess(values[-1], 500.0)

    def test_latent_heat_configuration_is_validated(self):
        runner = CylinderRunner()
        runner.configure_sublimation_heat(False, scale=2.0)
        self.assertFalse(runner.sublimation_heat_enabled)
        self.assertEqual(runner.sublimation_heat_scale, 2.0)
        with self.assertRaises(ValueError):
            runner.configure_sublimation_heat(True, scale=-1.0)
        with self.assertRaises(ValueError):
            runner.sublimation_heat_flux_W_m2(0.0)

    def test_code_applies_latent_heat_only_to_free_surfaces(self):
        cylinder = (ROOT / "cylinder_family" / "ML" /
                    "comsol_runner.py").read_text(encoding="utf-8")
        zigzag = (ROOT / "zigzag_family" / "ML" /
                  "zigzag_runner.py").read_text(encoding="utf-8")
        self.assertIn(
            'feature("subHeatS2S").selection().named("selFreeS2S")',
            cylinder)
        self.assertIn(
            'feature("subHeatZZ").selection().named("selFreeZZ")',
            zigzag)
        self.assertIn("-latentHeatEnabled*latentHeatScale", cylinder)
        self.assertIn("-latentHeatEnabled*latentHeatScale", zigzag)


class StartupContractTests(unittest.TestCase):
    def test_output_grid_resolves_inrush_and_settling(self):
        for runner_type in RUNNERS:
            times = runner_type._transient_output_times(60.0)
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 60.0)
            self.assertTrue(all(right > left
                                for left, right in zip(times, times[1:])))
            self.assertLessEqual(times[1] - times[0], 0.01)
            self.assertGreater(len(times), 200)

    def test_monotonic_trace_has_zero_positive_overshoot(self):
        metrics = CylinderRunner._startup_metrics(
            [0.0, 1.0, 2.0, 3.0],
            [293.15, 1000.0, 1985.0, 2000.0],
            steady_temperature_K=2000.0,
            limit_K=3273.15)
        self.assertEqual(metrics["startupOvershoot_K"], 0.0)
        self.assertTrue(metrics["startupSettled"])
        self.assertEqual(metrics["startupSettlingTime_s"], 2.0)
        self.assertTrue(metrics["startupTemperatureOK"])

    def test_overshoot_and_overtemperature_are_not_hidden(self):
        metrics = ZigzagRunner._startup_metrics(
            [0.0, 1.0, 2.0, 3.0],
            [293.15, 3400.0, 3050.0, 3000.0],
            steady_temperature_K=3000.0,
            limit_K=3273.15)
        self.assertAlmostEqual(metrics["startupOvershoot_K"], 400.0)
        self.assertFalse(metrics["startupTemperatureOK"])
        self.assertLess(metrics["startupTemperatureMargin_K"], 0.0)

    def test_unsettled_trace_is_explicitly_censored(self):
        metrics = CylinderRunner._startup_metrics(
            [0.0, 1.0, 2.0], [293.15, 1000.0, 1500.0],
            steady_temperature_K=2000.0,
            limit_K=3273.15)
        self.assertFalse(metrics["startupSettled"])
        self.assertTrue(math.isnan(metrics["startupSettlingTime_s"]))

    def test_invalid_trace_is_rejected(self):
        with self.assertRaises(ValueError):
            CylinderRunner._startup_metrics(
                [0.0, 1.0], [293.15], 1000.0, 3273.15)
        with self.assertRaises(ValueError):
            CylinderRunner._startup_metrics(
                [0.0, 0.0], [293.15, 300.0], 1000.0, 3273.15)


if __name__ == "__main__":
    unittest.main()
