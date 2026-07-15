"""D5 thermoelastic and manufacturability verification.

The competition thermal model explicitly ignores geometry distortion caused by
thermal expansion.  This module therefore performs one-way, linear-elastic
post-processing on the undeformed D4 geometry.  It is intentionally separate
from the lifecycle/Optuna path and must not be interpreted as a scoring model.
"""

import math

from jpype.types import JInt


MECHANICS_VERSION = "linear_thermoelastic_d5_v1"
MANUFACTURABILITY_VERSION = "geometry_integrity_d5_v1"
HIGH_TEMPERATURE_MATERIAL = "uw_pppl_tungsten_d5_v1"
ROOM_TEMPERATURE_MATERIAL = "pppl_room_tungsten_d5_v1"
REFERENCE_TEMPERATURE_K = 293.15
MINIMUM_FEATURE_MM = 0.1
BUCKLING_SCREEN_FACTOR = 2.0

MATERIAL_SOURCES = {
    "elastic_modulus_poisson_yield": (
        "University of Wisconsin Fusion Technology Institute UWFDM-1237, "
        "Appendix: Material Properties Used in Analysis"),
    "thermal_expansion": (
        "Princeton Plasma Physics Laboratory ARIES tungsten property table, "
        "valid 293-2500 K"),
    "room_temperature": (
        "Princeton Plasma Physics Laboratory ARIES general tungsten properties"),
}

_ALPHA_TABLE = (
    (293.0, 5.250e-6),
    (400.0, 5.305e-6),
    (600.0, 5.419e-6),
    (800.0, 5.533e-6),
    (1000.0, 5.646e-6),
    (1200.0, 5.700e-6),
    (1500.0, 5.931e-6),
    (2000.0, 6.215e-6),
    (2500.0, 6.500e-6),
)

_YIELD_TABLE = (
    (0.0, 1385.0),
    (500.0, 853.0),
    (1000.0, 465.0),
    (1500.0, 204.0),
    (2000.0, 57.0),
    (2500.0, 10.0),
)


def _finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _interpolate_clamped(table, value):
    value = float(value)
    if value <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table[:-1], table[1:]):
        if value <= x1:
            fraction = (value - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return table[-1][1]


def canonical_material_model(model):
    aliases = {
        "high_temperature": HIGH_TEMPERATURE_MATERIAL,
        HIGH_TEMPERATURE_MATERIAL: HIGH_TEMPERATURE_MATERIAL,
        "room_temperature": ROOM_TEMPERATURE_MATERIAL,
        ROOM_TEMPERATURE_MATERIAL: ROOM_TEMPERATURE_MATERIAL,
    }
    try:
        return aliases[str(model)]
    except KeyError as exc:
        raise ValueError(
            "mechanical material must be high_temperature or room_temperature"
        ) from exc


def mechanical_property_values(temperature_K, model=HIGH_TEMPERATURE_MATERIAL):
    """Return the D5 screening properties used at a given temperature."""
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    model = canonical_material_model(model)
    if model == ROOM_TEMPERATURE_MATERIAL:
        return {
            "youngModulus_GPa": 411.0,
            "poissonRatio": 0.28,
            "thermalExpansion_1_K": 4.5e-6,
            "yieldScreen_MPa": 550.0,
        }

    temperature_C = min(3000.0, max(0.0, temperature - 273.15))
    young = max(
        50.0,
        398.0 - 0.00231 * temperature_C
        - 2.72e-5 * temperature_C ** 2,
    )
    poisson = min(0.34, max(0.27, 0.279 + 1.09e-5 * temperature_C))
    return {
        "youngModulus_GPa": young,
        "poissonRatio": poisson,
        "thermalExpansion_1_K": _interpolate_clamped(
            _ALPHA_TABLE, temperature),
        "yieldScreen_MPa": _interpolate_clamped(
            _YIELD_TABLE, temperature_C),
    }


def _piecewise_expression(table, variable, unit):
    result = f"{table[-1][1]:.12g}[{unit}]"
    for (x0, y0), (x1, y1) in reversed(list(zip(table[:-1], table[1:]))):
        segment = (
            f"({y0:.12g}+({y1 - y0:.12g})*"
            f"({variable}-{x0:.12g})/{x1 - x0:.12g})[{unit}]"
        )
        result = f"if({variable}<={x1:.12g},{segment},{result})"
    return f"if({variable}<={table[0][0]:.12g},{table[0][1]:.12g}[{unit}],{result})"


def mechanical_material_expressions(model=HIGH_TEMPERATURE_MATERIAL):
    model = canonical_material_model(model)
    if model == ROOM_TEMPERATURE_MATERIAL:
        return {
            "young": "411[GPa]",
            "poisson": "0.28",
            "alpha": "4.5e-6[1/K]",
            "yield": "550[MPa]",
        }
    temperature_C = "min(3000,max(0,(T-273.15[K])/1[K]))"
    temperature_K = "min(3273.15,max(293,T/1[K]))"
    return {
        "young": (
            "max(50[GPa],(398-0.00231*TcD5-2.72e-5*TcD5^2)*1[GPa])"
        ),
        "poisson": "min(0.34,max(0.27,0.279+1.09e-5*TcD5))",
        "alpha": _piecewise_expression(
            _ALPHA_TABLE, "TKD5", "1/K"),
        "yield": _piecewise_expression(
            _YIELD_TABLE, "TcD5", "MPa"),
        "temperature_C": temperature_C,
        "temperature_K": temperature_K,
    }


def cylinder_manufacturability(radii_m, segment_length_m, target_volume_m3):
    radii = [float(value) for value in radii_m]
    if not radii or any(not _finite(value) or value <= 0.0 for value in radii):
        raise ValueError("all cylinder radii must be finite and positive")
    volume = sum(math.pi * radius ** 2 * segment_length_m for radius in radii)
    jumps = [abs(right - left) for left, right in zip(radii[:-1], radii[1:])]
    min_feature_mm = min(2.0 * min(radii), segment_length_m) * 1.0e3
    return {
        "manufacturabilityVersion": MANUFACTURABILITY_VERSION,
        "family": "cylinder",
        "geometryRepresentation": "parametric_segmented_axisymmetric",
        "voxelResolutionRequirementApplicable": False,
        "endpointDistance_mm": len(radii) * segment_length_m * 1.0e3,
        "analyticVolume_m3": volume,
        "analyticTargetVolumeError_rel": abs(volume - target_volume_m3)
        / target_volume_m3,
        "minFeature_mm": min_feature_mm,
        "minimumFeatureFloor_mm": MINIMUM_FEATURE_MM,
        "minimumFeatureFloorBasis": (
            "team_engineering_screen_not_competition_hard_limit"),
        "minimumFeaturePass": min_feature_mm >= MINIMUM_FEATURE_MM,
        "analyticConnected": True,
        "maxAdjacentRadiusJump_mm": max(jumps, default=0.0) * 1.0e3,
        "maxRadiusRatio": max(radii) / min(radii),
        "parameterizationSymmetric": all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(radii, reversed(radii))),
        "screenScope": "geometric_manufacturability_only",
    }


def zigzag_manufacturability(runner, n_runs, run_length_m, z_first_m):
    n_runs = int(n_runs)
    run_length = float(run_length_m)
    z_first = float(z_first_m)
    if n_runs < 2 or run_length <= 0.0:
        raise ValueError("zigzag geometry parameters are invalid")
    side, blocks, path_length = runner.compute_side_and_blocks(
        n_runs, run_length, z_first)
    z_last = runner.L0 - z_first
    z_step = (z_last - z_first) / (n_runs - 1)
    clear_gap = z_step - side
    min_feature = min(side, runner.STUB_LEN, 2.0 * runner.R0)
    return {
        "manufacturabilityVersion": MANUFACTURABILITY_VERSION,
        "family": "zigzag",
        "geometryRepresentation": "parametric_union_blocks",
        "voxelResolutionRequirementApplicable": False,
        "endpointDistance_mm": runner.L0 * 1.0e3,
        "analyticVolume_m3": runner.V0,
        "analyticTargetVolumeError_rel": 0.0,
        "minFeature_mm": min_feature * 1.0e3,
        "minimumFeatureFloor_mm": MINIMUM_FEATURE_MM,
        "minimumFeatureFloorBasis": (
            "team_engineering_screen_not_competition_hard_limit"),
        "minimumFeaturePass": min_feature * 1.0e3 >= MINIMUM_FEATURE_MM,
        "minimumClearGap_mm": clear_gap * 1.0e3,
        "minimumClearGapPass": clear_gap * 1.0e3 >= MINIMUM_FEATURE_MM,
        "runPitch_mm": z_step * 1.0e3,
        "side_mm": side * 1.0e3,
        "pathLength_mm": path_length * 1.0e3,
        "maximumUnsupportedSpan_mm": run_length * 1.0e3,
        "maximumSpanToSideRatio": run_length / side,
        "pathLengthToSideRatio": path_length / side,
        "blockCount": len(blocks),
        "analyticConnected": clear_gap >= 0.0,
        "cornerTreatment": "sharp_union_connector_cube",
        "cornerStressMeshSensitive": True,
        "screenScope": "geometric_manufacturability_only",
    }


def close_geometry_contract(metrics, snapshot, comsol_volume_m3,
                            target_volume_m3, volume_tolerance=1.0e-4):
    result = dict(metrics)
    comsol_error = abs(float(comsol_volume_m3) - target_volume_m3) / target_volume_m3
    endpoint_ok = math.isclose(
        result["endpointDistance_mm"], 15.0, rel_tol=0.0, abs_tol=1.0e-9)
    result.update({
        "comsolDomainCount": snapshot["domainCount"],
        "comsolFiniteVoidCount": snapshot["finiteVoidCount"],
        "comsolEntitiesPerDimension": snapshot["entitiesPerDimension"],
        "comsolVolume_m3": float(comsol_volume_m3),
        "comsolTargetVolumeError_rel": comsol_error,
        "singleDomainPass": snapshot["domainCount"] == 1,
        "noFiniteVoidPass": snapshot["finiteVoidCount"] == 0,
        "endpointDistancePass": endpoint_ok,
        "strictVolumePass": comsol_error <= volume_tolerance,
    })
    result["competitionGeometryPass"] = all((
        result["analyticConnected"],
        result["singleDomainPass"],
        result["endpointDistancePass"],
        result["strictVolumePass"],
    ))
    result["engineeringManufacturabilityScreenPass"] = all((
        result["minimumFeaturePass"],
        result.get("minimumClearGapPass", True),
        result["noFiniteVoidPass"],
    ))
    result["competitionGeometryContract"] = (
        "equal_volume_fixed_electrode_positions_single_connected_body")
    result["engineeringScreenInterpretation"] = (
        "team_defined_screen_not_competition_acceptance_threshold")
    result["manufacturingProcessValidated"] = False
    return result


class COMSOLMechanicsVerifier:
    """Apply reproducible D5 mechanics post-processing to an initialized runner."""

    _FAMILY_CONFIG = {
        "cylinder": ("selInS2S", "selOutS2S", "volS2S"),
        "zigzag": ("selInZZ", "selOutZZ", "volZZ"),
    }

    def __init__(self, runner, family):
        if family not in self._FAMILY_CONFIG:
            raise ValueError("family must be cylinder or zigzag")
        if runner.j is None:
            raise RuntimeError("runner model must be initialized first")
        self.runner = runner
        self.family = family
        self.in_selection, self.out_selection, self.volume_tag = (
            self._FAMILY_CONFIG[family])
        self._buckling_index = 0

    @property
    def j(self):
        return self.runner.j

    @property
    def component(self):
        return self.j.component("comp1")

    @staticmethod
    def _remove_safe(container, tag):
        try:
            container.remove(tag)
        except Exception:
            pass

    def geometry_snapshot(self):
        geometry = self.component.geom("geom1")
        return {
            "domainCount": int(geometry.getNDomains()),
            "finiteVoidCount": int(geometry.getNFiniteVoids()),
            "entitiesPerDimension": [
                int(value) for value in geometry.getNEntities()],
        }

    def _clear_solutions(self):
        for tag in list(self.j.sol().tags()):
            try:
                self.j.sol().remove(str(tag))
            except Exception:
                try:
                    self.j.sol(str(tag)).clearSolution()
                except Exception:
                    pass

    def _set_voltage(self, voltage):
        voltage = float(voltage)
        if not 0.0 < voltage <= self.runner.voltage_upper:
            raise ValueError("D5 voltage must satisfy 0 < V <= 100 V")
        self.j.param().set("Vapp", f"{voltage:.16g}[V]")
        return voltage

    def _configure_material(self, material_model):
        model = canonical_material_model(material_model)
        expressions = mechanical_material_expressions(model)
        self._remove_safe(self.component.variable(), "varD5")
        self.component.variable().create("varD5")
        variables = self.component.variable("varD5")
        if model == HIGH_TEMPERATURE_MATERIAL:
            variables.set("TcD5", expressions["temperature_C"])
            variables.set("TKD5", expressions["temperature_K"])
        variables.set("EW_D5", expressions["young"])
        variables.set("nuW_D5", expressions["poisson"])
        variables.set("alphaW_D5", expressions["alpha"])
        variables.set("sigmaYieldW_D5", expressions["yield"])

        material = self.component.material("mat1").propertyGroup("def")
        material.set("youngsmodulus", ["EW_D5"])
        material.set("poissonsratio", ["nuW_D5"])
        material.set("thermalexpansioncoefficient", ["alphaW_D5"])
        return model

    def _configure_solid(self, boundary_mode, gravity, material_model):
        if boundary_mode not in ("fixed_fixed", "fixed_sliding"):
            raise ValueError("boundary_mode must be fixed_fixed or fixed_sliding")
        self._remove_safe(self.component.physics(), "solid")
        self.component.physics().create("solid", "SolidMechanics", "geom1")
        solid = self.component.physics("solid")
        material_model = self._configure_material(material_model)

        solid.create("fixInD5", "Fixed", 2)
        solid.feature("fixInD5").selection().named(self.in_selection)
        if boundary_mode == "fixed_fixed":
            solid.create("fixOutD5", "Fixed", 2)
            solid.feature("fixOutD5").selection().named(self.out_selection)
        else:
            # Electrode faces are normal to z. Fix x/y to remove rigid motion
            # while leaving z free for axial thermal expansion.
            solid.create("slideOutD5", "Displacement2", 2)
            sliding = solid.feature("slideOutD5")
            sliding.selection().named(self.out_selection)
            sliding.set("Direction", ["prescribed", "prescribed", "free"])
            sliding.set("U0", ["0", "0", "0"])

        solid.feature("lemm1").create("teD5", "ThermalExpansion", 3)
        thermal = solid.feature("lemm1").feature("teD5")
        thermal.set("InputType", "SecantCoefficient")
        thermal.set("minput_temperature_src", "root.comp1.T")
        thermal.set("minput_temperature", "T")
        thermal.set("minput_strainreferencetemperature_src", "userdef")
        thermal.set(
            "minput_strainreferencetemperature",
            f"{REFERENCE_TEMPERATURE_K}[K]")

        if gravity:
            solid.create("gravityD5", "Gravity", 3)
            solid.feature("gravityD5").set("g", ["0", "-g_const", "0"])
        return material_model

    def _numerical(self, tag, feature_type, expression, unit):
        self._remove_safe(self.j.result().numerical(), tag)
        numerical = self.j.result().numerical()
        numerical.create(tag, feature_type)
        feature = self.j.result().numerical(tag)
        feature.selection().all()
        feature.set("expr", [expression])
        try:
            feature.set("unit", [unit])
        except Exception:
            feature.set("unit", unit)
        return float(feature.getReal()[0][0])

    def _run_buckling(self):
        self._buckling_index += 1
        study_tag = f"stdD5Buck{self._buckling_index}"
        before = {str(tag) for tag in self.j.sol().tags()}
        self.j.study().create(study_tag)
        study = self.j.study(study_tag)
        study.create("stat", "Stationary")
        study.feature("stat").set("activate", [
            "ec", "on", "ht", "on", "rad", "on", "solid", "on",
            "frame:spatial1", "on", "frame:material1", "on",
        ])
        study.create("buck", "LinearBuckling")
        study.feature("buck").set("activate", [
            "ec", "off", "ht", "off", "rad", "off", "solid", "on",
            "frame:spatial1", "on", "frame:material1", "on",
        ])
        study.feature("buck").set("neigsactive", "on")
        study.feature("buck").set("neigs", JInt(3))
        study.run()

        factors = []
        solver_tag = ""
        for tag_value in self.j.sol().tags():
            tag = str(tag_value)
            if tag in before:
                continue
            try:
                values = [float(value) for value in self.j.sol(tag).getPVals()]
            except Exception:
                continue
            nonzero = [value for value in values
                       if math.isfinite(value) and abs(value) > 1.0e-10]
            if len(nonzero) > len(factors):
                factors = nonzero
                solver_tag = tag
        positive = sorted(value for value in factors if value > 0.0)
        critical = positive[0] if positive else None
        buckling_solved = bool(factors)
        return {
            "bucklingRequested": True,
            "bucklingSolveOK": buckling_solved,
            "positiveBucklingModeFound": bool(positive),
            "criticalLoadFactor": critical,
            "bucklingFactors": factors,
            "bucklingSolverTag": solver_tag,
            "linearBucklingPass": (
                critical > 1.0 if critical is not None else None),
            "bucklingEngineeringMarginPass": (
                critical >= BUCKLING_SCREEN_FACTOR
                if critical is not None else None),
            "bucklingEngineeringMargin": BUCKLING_SCREEN_FACTOR,
            "bucklingInterpretation": (
                "optional_linearized_screen_not_competition_score"),
        }

    def run_case(self, voltage, boundary_mode="fixed_fixed", gravity=False,
                 material_model=HIGH_TEMPERATURE_MATERIAL,
                 include_buckling=False):
        """Solve one undeformed-geometry linear thermoelastic D5 case."""
        voltage = self._set_voltage(voltage)
        material_model = self._configure_solid(
            boundary_mode, bool(gravity), material_model)
        self._clear_solutions()
        stationary = self.j.study("std1").feature("stat")
        stationary.set("activate", [
            "ec", "on", "ht", "on", "rad", "on", "solid", "on",
            "frame:spatial1", "on", "frame:material1", "on",
        ])
        stationary.set("geometricNonlinearity", "off")
        self.j.study("std1").run()

        max_temperature = self._numerical(
            "maxTD5", "MaxVolume", "T", "K")
        volume = self._numerical(
            "volD5", "IntVolume", "1", "m^3")
        max_mises = self._numerical(
            "maxMisesD5", "MaxVolume", "solid.mises", "MPa")
        mises_sq_integral = self._numerical(
            "intMisesSqD5", "IntVolume",
            "(solid.mises/1[MPa])^2", "m^3")
        rms_mises = math.sqrt(max(0.0, mises_sq_integral / volume))
        max_displacement = self._numerical(
            "maxDispD5", "MaxVolume", "solid.disp", "mm")
        max_ux = self._numerical(
            "maxUxD5", "MaxVolume", "abs(u)", "mm")
        max_uy = self._numerical(
            "maxUyD5", "MaxVolume", "abs(v)", "mm")
        max_uz = self._numerical(
            "maxUzD5", "MaxVolume", "abs(w)", "mm")
        min_principal = self._numerical(
            "minPrincipalD5", "MinVolume", "solid.sp3", "MPa")
        yield_utilization = self._numerical(
            "maxYieldUtilD5", "MaxVolume",
            "solid.mises/sigmaYieldW_D5", "1")
        thermal_strain = self._numerical(
            "maxThermalStrainD5", "MaxVolume",
            "abs(alphaW_D5*(T-293.15[K]))", "1")
        snapshot = self.geometry_snapshot()
        properties_at_tmax = mechanical_property_values(
            max_temperature, material_model)

        values = (
            max_temperature, volume, max_mises, rms_mises,
            max_displacement, max_ux, max_uy, max_uz, min_principal,
            yield_utilization, thermal_strain,
        )
        result = {
            "solveOK": all(_finite(value) for value in values),
            "mechanicsVersion": MECHANICS_VERSION,
            "mechanicalMaterialModel": material_model,
            "mechanicalMaterialSources": MATERIAL_SOURCES,
            "family": self.family,
            "voltage_V": voltage,
            "boundaryMode": boundary_mode,
            "gravityEnabled": bool(gravity),
            "gravityDirection": "-y" if gravity else "none",
            "Tmax_K": max_temperature,
            "temperatureLimit_K": self.runner.temp_limit_K,
            "temperatureLimitPass": max_temperature < self.runner.temp_limit_K,
            "volume_m3": volume,
            "domainCount": snapshot["domainCount"],
            "finiteVoidCount": snapshot["finiteVoidCount"],
            "maxMises_MPa": max_mises,
            "rmsMises_MPa": rms_mises,
            "stressConcentrationRatio": (
                max_mises / rms_mises if rms_mises > 0.0 else float("nan")),
            "mostCompressivePrincipalStress_MPa": min_principal,
            "maxDisplacement_mm": max_displacement,
            "maxAbsUx_mm": max_ux,
            "maxAbsUy_mm": max_uy,
            "maxAbsUz_mm": max_uz,
            "maxThermalStrain_pct": thermal_strain * 100.0,
            "maxYieldScreenUtilization": yield_utilization,
            "elasticYieldScreenPass": yield_utilization <= 1.0,
            "propertiesAtTmax": properties_at_tmax,
            "thermalExpansionGeometryCoupling": False,
            "undeformedThermalGeometry": True,
            "mechanicsInOptimizationLoop": False,
            "stressInterpretation": (
                "linear_elastic_comparison_not_strength_certification"),
            "bucklingRequested": False,
        }
        if include_buckling:
            try:
                result.update(self._run_buckling())
            except Exception as exc:
                result.update({
                    "bucklingRequested": True,
                    "bucklingSolveOK": False,
                    "positiveBucklingModeFound": False,
                    "criticalLoadFactor": None,
                    "linearBucklingPass": None,
                    "bucklingEngineeringMarginPass": None,
                    "bucklingFailure": str(exc),
                })
        return result
