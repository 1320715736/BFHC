"""Regression tests for the D1 radiation escape metric."""

import csv
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
    "d1_cylinder_runner", ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d1_zigzag_runner", ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")


def successful_stationary_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["case_id"]: row for row in rows
        if row.get("runMode") == "stationary" and row.get("status") == "OK"
    }


class RadiationMetricMathTests(unittest.TestCase):
    def test_loss_is_physical_and_preserves_raw_residual(self):
        runner = CylinderRunner()
        loss = runner._radiation_loss_metrics(100.0, 80.0)
        self.assertAlmostEqual(loss["loss_pct"], 20.0)
        self.assertAlmostEqual(loss["self_absorbed"], 20.0)

        residual = runner._radiation_loss_metrics(100.0, 100.01)
        self.assertAlmostEqual(residual["loss_pct"], 0.0)
        self.assertLess(residual["loss_raw_pct"], 0.0)
        self.assertGreater(residual["numerical_excess_pct"], 0.0)

    def test_both_runners_use_metric_v2(self):
        for runner in (CylinderRunner(), ZigzagRunner()):
            self.assertEqual(runner.metric_version, "radiation_escape_v2")
            self.assertEqual(runner.radiation_escape_method,
                             "s2s_radiosity_famb")
            self.assertEqual(runner.spectral_split_um, 3.0)

    def test_black_enclosing_sphere_is_radius_invariant(self):
        # Independent two-surface radiosity enclosure. The black 0 K sphere
        # has zero radiosity, so its absorbed power must equal Famb escape.
        areas = (1.0, 1.0)
        emissivities = (0.35, 0.15)
        blackbody_powers = (1000.0, 400.0)
        mutual_view_factor = 0.35
        ambient_view_factors = (
            1.0 - mutual_view_factor,
            1.0 - mutual_view_factor,
        )

        a = 1.0
        b = -(1.0 - emissivities[0]) * mutual_view_factor
        c = -(1.0 - emissivities[1]) * mutual_view_factor
        d = 1.0
        rhs_1 = emissivities[0] * blackbody_powers[0]
        rhs_2 = emissivities[1] * blackbody_powers[1]
        determinant = a * d - b * c
        j_1 = (rhs_1 * d - b * rhs_2) / determinant
        j_2 = (a * rhs_2 - rhs_1 * c) / determinant

        escaped = sum(
            area * radiosity * f_ambient
            for area, radiosity, f_ambient in zip(
                areas, (j_1, j_2), ambient_view_factors)
        )

        for radius_ratio in (1.05, 1.5, 2.0):
            sphere_area = 4.0 * math.pi * radius_ratio**2
            sphere_to_device = tuple(
                area * f_ambient / sphere_area
                for area, f_ambient in zip(areas, ambient_view_factors)
            )
            sphere_self_view = 1.0 - sum(sphere_to_device)
            self.assertGreaterEqual(sphere_self_view, 0.0)

            sphere_irradiation = (
                sphere_to_device[0] * j_1
                + sphere_to_device[1] * j_2
                + sphere_self_view * 0.0
            )
            sphere_absorbed = sphere_area * sphere_irradiation
            self.assertAlmostEqual(sphere_absorbed, escaped, places=10)


class StationaryControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {}
        cls.rows.update(successful_stationary_rows(
            ROOT / "cylinder_family" / "ML" / "data"
            / "d1_metric_v2_controls.csv"))
        cls.rows.update(successful_stationary_rows(
            ROOT / "zigzag_family" / "ML" / "data"
            / "d1_metric_v2_controls.csv"))

    def value(self, case_id, key):
        self.assertIn(case_id, self.rows)
        value = float(self.rows[case_id][key])
        self.assertTrue(math.isfinite(value), f"{case_id}.{key}")
        return value

    def test_uniform_cylinder_has_negligible_self_view_loss(self):
        case = "D1_B2_official_cylinder"
        self.assertLess(self.value(case, "initialSelfViewLoss_pct"), 0.1)
        self.assertGreater(self.value(case, "initialFambAreaAvg"), 0.999)

    def test_nonconvex_controls_lose_escape_power(self):
        for case in (
            "D1_cylinder_trial68",
            "D1_B3_reference_zigzag",
            "D1_C10_zigzag_trial19",
        ):
            gross = self.value(case, "initialP03gross_W")
            escape = self.value(case, "initialP03escape_W")
            self.assertLess(escape, gross)
            self.assertGreater(self.value(case, "initialSelfViewLoss_pct"), 1.0)

    def test_more_folded_c10_has_more_loss_than_b3(self):
        b3 = self.value("D1_B3_reference_zigzag", "initialSelfViewLoss_pct")
        c10 = self.value("D1_C10_zigzag_trial19", "initialSelfViewLoss_pct")
        self.assertGreater(c10, b3)

    def test_second_spectral_band_is_nonzero(self):
        for case in self.rows:
            p03 = self.value(case, "initialP03gross_W")
            pall = self.value(case, "initialPradGross_W")
            self.assertGreater(pall, p03)

    def test_room_temperature_contamination_is_negligible_in_score_band(self):
        for case in self.rows:
            ratio_pct = self.value(case, "initialAmbient03ToEscape_pct")
            self.assertLess(ratio_pct, 1e-4)

    def test_control_csv_schema_and_lifecycle_rows_are_aligned(self):
        paths = (
            ROOT / "cylinder_family" / "ML" / "data"
            / "d1_metric_v2_controls.csv",
            ROOT / "zigzag_family" / "ML" / "data"
            / "d1_metric_v2_controls.csv",
        )
        completed_cases = set()
        for path in paths:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertIn("erosionSolveRetries", reader.fieldnames)
                rows = list(reader)
            for row in rows:
                self.assertNotIn(None, row, path.name)
                if (row.get("runMode") == "lifecycle"
                        and row.get("status") in {
                            "OK", "FAIL_OVERTEMP_DURING_EROSION"}):
                    self.assertEqual(row["runnerStatus"], row["status"])
                    self.assertTrue(row["erosionSolveRetries"].isdigit())
                    gross = float(row["lifeTotalP03gross_J"])
                    escape = float(row["lifeTotalP03escape_J"])
                    self.assertGreater(gross, 0.0)
                    self.assertGreater(escape, 0.0)
                    self.assertLessEqual(escape, gross * (1.0 + 1e-10))
                    completed_cases.add(row["case_id"])

        self.assertTrue({
            "D1_B2_official_cylinder",
            "D1_cylinder_trial68",
            "D1_B3_reference_zigzag",
            "D1_C10_zigzag_trial19",
        }.issubset(completed_cases))


class JavaParityTests(unittest.TestCase):
    def test_java_sources_use_escape_metric_and_numeric_spectral_split(self):
        sources = [
            ROOT / "cylinder_family" / "src" / "cylinder_baseline.java",
            ROOT / "cylinder_family" / "src" / "cylinder_process_views.java",
            ROOT / "zigzag_family" / "zigzag_baseline.java",
            ROOT / "zigzag_family" / "zigzag_process_views.java",
        ]
        for path in sources:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn('"lambda_r", "lam03"', text, path.name)
            self.assertNotIn('"lambda_r","lam03"', text, path.name)
            self.assertIn("rad.J_band1*rad.Famb1", text, path.name)
            self.assertIn("rad.epsilonu_band2*rad.ebu2", text, path.name)


if __name__ == "__main__":
    unittest.main()
