"""Pure-Python contract tests for D5 mechanics and geometry screening."""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verification.d5_mechanics import (  # noqa: E402
    HIGH_TEMPERATURE_MATERIAL,
    MECHANICS_VERSION,
    ROOM_TEMPERATURE_MATERIAL,
    close_geometry_contract,
    cylinder_manufacturability,
    mechanical_material_expressions,
    mechanical_property_values,
    zigzag_manufacturability,
)


def load_runner(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMSOLRunner


CylinderRunner = load_runner(
    "d5_test_cylinder_runner",
    ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d5_test_zigzag_runner",
    ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")


class MechanicalMaterialTests(unittest.TestCase):
    def test_high_temperature_properties_are_finite_and_soften(self):
        room = mechanical_property_values(293.15)
        hot = mechanical_property_values(3273.15)
        self.assertGreater(room["youngModulus_GPa"], hot["youngModulus_GPa"])
        self.assertGreater(hot["youngModulus_GPa"], 100.0)
        self.assertGreater(hot["thermalExpansion_1_K"],
                           room["thermalExpansion_1_K"])
        self.assertEqual(hot["yieldScreen_MPa"], 10.0)

    def test_room_temperature_sensitivity_is_explicit(self):
        values = mechanical_property_values(
            3000.0, ROOM_TEMPERATURE_MATERIAL)
        self.assertEqual(values["youngModulus_GPa"], 411.0)
        self.assertEqual(values["poissonRatio"], 0.28)
        self.assertEqual(values["thermalExpansion_1_K"], 4.5e-6)

    def test_comsol_expressions_are_temperature_dependent(self):
        expressions = mechanical_material_expressions(
            HIGH_TEMPERATURE_MATERIAL)
        self.assertIn("TcD5", expressions["young"])
        self.assertIn("TKD5", expressions["alpha"])
        self.assertIn("TcD5", expressions["yield"])
        self.assertEqual(MECHANICS_VERSION, "linear_thermoelastic_d5_v1")


class GeometryScreenTests(unittest.TestCase):
    def test_official_cylinder_passes_analytic_contract(self):
        runner = CylinderRunner()
        metrics = cylinder_manufacturability(
            [runner.reference_radius] * runner.seg_count,
            runner.Lseg, runner.reference_volume)
        self.assertAlmostEqual(metrics["endpointDistance_mm"], 15.0)
        self.assertLess(metrics["analyticTargetVolumeError_rel"], 1.0e-12)
        self.assertTrue(metrics["minimumFeaturePass"])
        self.assertTrue(metrics["parameterizationSymmetric"])

    def test_historical_trial68_preserves_volume_and_minimum_scale(self):
        runner = CylinderRunner()
        radii = [
            1.906383264e-3, 1.942386976e-3,
            3.647488445e-3, 2.070908913e-3,
            2.070908913e-3, 3.647488445e-3,
            1.942386976e-3, 1.906383264e-3,
        ]
        metrics = cylinder_manufacturability(
            radii, runner.Lseg, runner.reference_volume)
        self.assertLess(metrics["analyticTargetVolumeError_rel"], 1.0e-8)
        self.assertGreater(metrics["minFeature_mm"], 0.1)

    def test_long_zigzag_exposes_gap_and_slenderness(self):
        runner = ZigzagRunner()
        metrics = zigzag_manufacturability(
            runner, 12, 92.0111316015e-3, 2.2666310029e-3)
        self.assertAlmostEqual(metrics["side_mm"], 0.495829, delta=1.0e-5)
        self.assertGreater(metrics["minimumClearGap_mm"], 0.1)
        self.assertGreater(metrics["maximumSpanToSideRatio"], 150.0)
        self.assertTrue(metrics["minimumFeaturePass"])

    def test_comsol_domain_and_volume_close_the_contract(self):
        runner = CylinderRunner()
        metrics = cylinder_manufacturability(
            [runner.reference_radius] * runner.seg_count,
            runner.Lseg, runner.reference_volume)
        closed = close_geometry_contract(
            metrics,
            {"domainCount": 1, "finiteVoidCount": 0,
             "entitiesPerDimension": [18, 24, 8, 1]},
            runner.reference_volume, runner.reference_volume)
        self.assertTrue(closed["competitionGeometryPass"])
        self.assertTrue(closed["engineeringManufacturabilityScreenPass"])
        self.assertFalse(closed["manufacturingProcessValidated"])
        self.assertFalse(closed["voxelResolutionRequirementApplicable"])

    def test_disconnected_or_wrong_volume_geometry_is_rejected(self):
        runner = CylinderRunner()
        metrics = cylinder_manufacturability(
            [runner.reference_radius] * runner.seg_count,
            runner.Lseg, runner.reference_volume)
        closed = close_geometry_contract(
            metrics,
            {"domainCount": 2, "finiteVoidCount": 0,
             "entitiesPerDimension": [0, 0, 0, 2]},
            runner.reference_volume * 1.001, runner.reference_volume)
        self.assertFalse(closed["competitionGeometryPass"])

    def test_team_manufacturing_screen_is_not_an_official_void_ban(self):
        runner = CylinderRunner()
        metrics = cylinder_manufacturability(
            [runner.reference_radius] * runner.seg_count,
            runner.Lseg, runner.reference_volume)
        closed = close_geometry_contract(
            metrics,
            {"domainCount": 1, "finiteVoidCount": 1,
             "entitiesPerDimension": [0, 0, 0, 1]},
            runner.reference_volume, runner.reference_volume)
        self.assertTrue(closed["competitionGeometryPass"])
        self.assertFalse(closed["engineeringManufacturabilityScreenPass"])
        self.assertEqual(
            closed["minimumFeatureFloorBasis"],
            "team_engineering_screen_not_competition_hard_limit")


class CompetitionScopeTests(unittest.TestCase):
    def test_d5_is_postprocessing_not_thermal_geometry_coupling(self):
        source = (ROOT / "verification" / "d5_mechanics.py").read_text(
            encoding="utf-8")
        self.assertIn('"thermalExpansionGeometryCoupling": False', source)
        self.assertIn('"mechanicsInOptimizationLoop": False', source)
        self.assertIn('"Displacement2"', source)
        self.assertIn('["prescribed", "prescribed", "free"]', source)

    def test_d4_scoring_physics_version_remains_unchanged(self):
        self.assertEqual(CylinderRunner().physics_version, "thermal_s2s_d4_v1")
        self.assertEqual(ZigzagRunner().physics_version, "thermal_s2s_d4_v1")


if __name__ == "__main__":
    unittest.main()
