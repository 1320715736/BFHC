"""Run D5 stress, stability, and geometry controls in real COMSOL."""

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verification.d5_mechanics import (  # noqa: E402
    HIGH_TEMPERATURE_MATERIAL,
    ROOM_TEMPERATURE_MATERIAL,
    COMSOLMechanicsVerifier,
    close_geometry_contract,
    cylinder_manufacturability,
    zigzag_manufacturability,
)


def load_runner(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMSOLRunner


CylinderRunner = load_runner(
    "d5_smoke_cylinder_runner",
    ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d5_smoke_zigzag_runner",
    ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")

TRIAL68_RADII_M = [
    1.906383264e-3, 1.942386976e-3,
    3.647488445e-3, 2.070908913e-3,
    2.070908913e-3, 3.647488445e-3,
    1.942386976e-3, 1.906383264e-3,
]

ZIGZAG_CONTROLS = {
    "d4_reference": {
        "N_RUNS": 8,
        "L_RUN_m": 104.0e-3,
        "z_first_m": 0.8e-3,
        "voltage_V": 90.0,
        "role": "current_reference_geometry",
    },
    "historical_c10_pressure_test": {
        "N_RUNS": 12,
        "L_RUN_m": 92.0111316015e-3,
        "z_first_m": 2.2666310029e-3,
        "voltage_V": 90.0,
        "role": "historical_long_path_pressure_test_not_formal_candidate",
    },
}


def finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def load_checkpoint(path, family, scope, resume):
    if not resume or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("family") != family or payload.get("scope") != scope:
        raise RuntimeError("D5 checkpoint family/scope does not match this run")
    return dict(payload.get("cases", {}))


def write_checkpoint(path, family, scope, cases):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "family": family,
        "scope": scope,
        "status": "RUNNING_CHECKPOINT",
        "cases": cases,
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def required_analysis_keys(family, case_id, scope):
    keys = {
        "fixed_fixed_high_temperature_no_gravity",
        "fixed_sliding_high_temperature_no_gravity",
    }
    if (family == "zigzag" and scope == "full"
            and case_id == "historical_c10_pressure_test"):
        keys.update({
            "fixed_sliding_high_temperature_no_gravity_100V",
            "fixed_sliding_high_temperature_gravity_100V",
            "fixed_sliding_room_temperature_no_gravity",
        })
    return keys


def checkpoint_case_complete(output, family, case_id, scope):
    if not output:
        return False
    geometry = output.get("geometry", {})
    analyses = output.get("analyses", {})
    return (
        "competitionGeometryPass" in geometry
        and required_analysis_keys(family, case_id, scope).issubset(analyses)
    )


def validate_analysis(case_id, analysis):
    required = (
        "Tmax_K", "volume_m3", "maxMises_MPa", "rmsMises_MPa",
        "maxDisplacement_mm", "maxYieldScreenUtilization",
    )
    invalid = [field for field in required
               if not finite(analysis.get(field))]
    if invalid:
        raise RuntimeError(
            f"{case_id} has non-finite D5 fields: {', '.join(invalid)}")
    if not analysis.get("solveOK"):
        raise RuntimeError(f"{case_id} mechanics solve failed")
    if not analysis.get("temperatureLimitPass"):
        raise RuntimeError(f"{case_id} exceeds the competition temperature limit")
    if analysis.get("bucklingRequested") and not analysis.get("bucklingSolveOK"):
        print(f"  WARN: {case_id} optional buckling screen did not solve",
              flush=True)


def run_mechanical_matrix(verifier, voltage, include_buckling=False,
                          analyses=None, checkpoint=None):
    analyses = analyses if analyses is not None else {}
    for boundary in ("fixed_fixed", "fixed_sliding"):
        key = f"{boundary}_high_temperature_no_gravity"
        if key in analyses:
            print(f"  D5 mechanics {key} resumed from checkpoint", flush=True)
            continue
        print(f"  D5 mechanics {key} started", flush=True)
        started = time.time()
        result = verifier.run_case(
            voltage, boundary_mode=boundary, gravity=False,
            material_model=HIGH_TEMPERATURE_MATERIAL,
            include_buckling=(include_buckling and boundary == "fixed_fixed"))
        result["elapsed_sec"] = round(time.time() - started, 1)
        validate_analysis(key, result)
        analyses[key] = result
        if checkpoint is not None:
            checkpoint()
        print(f"  D5 mechanics {key} completed in "
              f"{result['elapsed_sec']:.1f}s", flush=True)
    return analyses


def comparison(initial, candidate):
    output = {}
    for boundary in ("fixed_fixed", "fixed_sliding"):
        key = f"{boundary}_high_temperature_no_gravity"
        left = initial[key]
        right = candidate[key]
        output[boundary] = {
            "sameMechanicalBoundary": True,
            "sameMaterialModel": (
                left["mechanicalMaterialModel"]
                == right["mechanicalMaterialModel"]),
            "sameVoltage": left["voltage_V"] == right["voltage_V"],
            "initialMaxMises_MPa": left["maxMises_MPa"],
            "candidateMaxMises_MPa": right["maxMises_MPa"],
            "maxMisesChange_pct": 100.0 * (
                right["maxMises_MPa"] / left["maxMises_MPa"] - 1.0),
            "initialRmsMises_MPa": left["rmsMises_MPa"],
            "candidateRmsMises_MPa": right["rmsMises_MPa"],
            "rmsMisesChange_pct": 100.0 * (
                right["rmsMises_MPa"] / left["rmsMises_MPa"] - 1.0),
            "initialMaxDisplacement_mm": left["maxDisplacement_mm"],
            "candidateMaxDisplacement_mm": right["maxDisplacement_mm"],
            "maxDisplacementChange_pct": 100.0 * (
                right["maxDisplacement_mm"]
                / left["maxDisplacement_mm"] - 1.0),
        }
    return output


def percentage_change(initial, candidate):
    return 100.0 * (candidate / initial - 1.0)


def screen_summary(outputs):
    geometries = [case["geometry"] for case in outputs.values()]
    analyses = [
        analysis
        for case in outputs.values()
        for analysis in case["analyses"].values()
    ]
    requested_buckling = [
        analysis for analysis in analyses if analysis.get("bucklingRequested")]
    return {
        "allCompetitionGeometryPass": all(
            geometry["competitionGeometryPass"] for geometry in geometries),
        "allEngineeringManufacturabilityScreensPass": all(
            geometry["engineeringManufacturabilityScreenPass"]
            for geometry in geometries),
        "allMechanicsSolvesPass": all(
            analysis["solveOK"] for analysis in analyses),
        "allTemperatureLimitsPass": all(
            analysis["temperatureLimitPass"] for analysis in analyses),
        "allElasticYieldScreensPass": all(
            analysis["elasticYieldScreenPass"] for analysis in analyses),
        "requestedBucklingScreenCount": len(requested_buckling),
        "allRequestedBucklingScreensPass": (
            all(analysis.get("bucklingEngineeringMarginPass") is True
                for analysis in requested_buckling)
            if requested_buckling else None),
        "strengthCertificationClaimed": False,
    }


def zigzag_sensitivity(analyses):
    no_gravity = analyses[
        "fixed_sliding_high_temperature_no_gravity_100V"]
    gravity = analyses[
        "fixed_sliding_high_temperature_gravity_100V"]
    high_temperature = analyses[
        "fixed_sliding_high_temperature_no_gravity"]
    room_temperature = analyses[
        "fixed_sliding_room_temperature_no_gravity"]
    return {
        "gravityAt100VFixedSliding": {
            "pairedVoltage_V": 100.0,
            "maxMisesChange_pct": percentage_change(
                no_gravity["maxMises_MPa"], gravity["maxMises_MPa"]),
            "rmsMisesChange_pct": percentage_change(
                no_gravity["rmsMises_MPa"], gravity["rmsMises_MPa"]),
            "maxDisplacementChange_pct": percentage_change(
                no_gravity["maxDisplacement_mm"],
                gravity["maxDisplacement_mm"]),
        },
        "materialAt90VFixedSliding": {
            "pairedVoltage_V": 90.0,
            "maxMisesChangeRoomVsHighTemperature_pct": percentage_change(
                high_temperature["maxMises_MPa"],
                room_temperature["maxMises_MPa"]),
            "rmsMisesChangeRoomVsHighTemperature_pct": percentage_change(
                high_temperature["rmsMises_MPa"],
                room_temperature["rmsMises_MPa"]),
            "maxDisplacementChangeRoomVsHighTemperature_pct": (
                percentage_change(
                    high_temperature["maxDisplacement_mm"],
                    room_temperature["maxDisplacement_mm"])),
        },
    }


def initialize_cylinder(runner, radii):
    runner._r0 = max(radii)
    runner._initial_radii = list(radii)
    runner._fail_radii = [
        radius * (1.0 - runner.failure_fraction) for radius in radii]
    runner._init_model(radii)


def run_cylinder(scope, checkpoint_path, resume=False):
    runner = CylinderRunner()
    runner.start()
    started = time.time()
    try:
        cases = {
            "official_initial_cylinder": {
                "radii_m": [runner.reference_radius] * runner.seg_count,
                "voltage_V": 1.0,
                "role": "official_initial_geometry",
            },
            "historical_trial68_pressure_test": {
                "radii_m": TRIAL68_RADII_M,
                "voltage_V": 1.0,
                "role": "historical_shape_pressure_test_not_formal_candidate",
            },
        }
        outputs = load_checkpoint(
            checkpoint_path, "cylinder", scope, resume)
        for case_id, case in cases.items():
            if checkpoint_case_complete(
                    outputs.get(case_id), "cylinder", case_id, scope):
                print(f"D5 cylinder {case_id}: resumed from checkpoint",
                      flush=True)
                continue
            print(f"D5 cylinder {case_id}: model initialization", flush=True)
            initialize_cylinder(runner, case["radii_m"])
            verifier = COMSOLMechanicsVerifier(runner, "cylinder")
            metrics = cylinder_manufacturability(
                case["radii_m"], runner.Lseg, runner.reference_volume)
            existing = outputs.get(case_id, {})
            analyses = dict(existing.get("analyses", {}))
            outputs[case_id] = {
                "role": case["role"],
                "radii_mm": [value * 1.0e3 for value in case["radii_m"]],
                "voltage_V": case["voltage_V"],
                "analyses": analyses,
                "checkpointComplete": False,
            }
            persist = lambda: write_checkpoint(  # noqa: E731
                checkpoint_path, "cylinder", scope, outputs)
            analyses = run_mechanical_matrix(
                verifier, case["voltage_V"], analyses=analyses,
                checkpoint=persist)
            geometry = close_geometry_contract(
                metrics, verifier.geometry_snapshot(),
                analyses["fixed_fixed_high_temperature_no_gravity"]["volume_m3"],
                runner.reference_volume, runner.vol_tol)
            if not geometry["competitionGeometryPass"]:
                raise RuntimeError(f"{case_id} failed the D5 geometry contract")
            outputs[case_id]["geometry"] = geometry
            outputs[case_id]["checkpointComplete"] = True
            write_checkpoint(
                checkpoint_path, "cylinder", scope, outputs)
        return {
            "family": "cylinder",
            "scope": scope,
            "status": "D5_SCREEN_COMPLETE",
            "competitionScope": {
                "stressComparisonRequired": True,
                "thermalGeometryDistortionIgnored": True,
                "mechanicsOutsideOptimizationLoop": True,
                "strengthCertificationClaimed": False,
                "competitionStrengthThresholdSpecified": False,
            },
            "cases": outputs,
            "screenSummary": screen_summary(outputs),
            "sameBoundaryComparison": comparison(
                outputs["official_initial_cylinder"]["analyses"],
                outputs["historical_trial68_pressure_test"]["analyses"]),
            "lastInvocationElapsed_sec": round(time.time() - started, 1),
        }
    finally:
        runner.stop()


def run_zigzag(scope, checkpoint_path, resume=False):
    runner = ZigzagRunner()
    runner.start()
    started = time.time()
    try:
        outputs = load_checkpoint(
            checkpoint_path, "zigzag", scope, resume)
        for case_id, case in ZIGZAG_CONTROLS.items():
            if checkpoint_case_complete(
                    outputs.get(case_id), "zigzag", case_id, scope):
                print(f"D5 zigzag {case_id}: resumed from checkpoint",
                      flush=True)
                continue
            print(f"D5 zigzag {case_id}: model initialization", flush=True)
            runner._init_model(
                case["N_RUNS"], case["L_RUN_m"], case["z_first_m"])
            verifier = COMSOLMechanicsVerifier(runner, "zigzag")
            include_buckling = (
                scope == "full" and case_id == "historical_c10_pressure_test")
            existing = outputs.get(case_id, {})
            analyses = dict(existing.get("analyses", {}))
            outputs[case_id] = {
                "role": case["role"],
                "parameters": {
                    "N_RUNS": case["N_RUNS"],
                    "L_RUN_mm": case["L_RUN_m"] * 1.0e3,
                    "z_first_mm": case["z_first_m"] * 1.0e3,
                },
                "voltage_V": case["voltage_V"],
                "analyses": analyses,
                "checkpointComplete": False,
            }
            persist = lambda: write_checkpoint(  # noqa: E731
                checkpoint_path, "zigzag", scope, outputs)
            analyses = run_mechanical_matrix(
                verifier, case["voltage_V"], include_buckling,
                analyses=analyses, checkpoint=persist)

            if scope == "full" and case_id == "historical_c10_pressure_test":
                for key, kwargs in (
                    ("fixed_sliding_high_temperature_no_gravity_100V", {
                        "voltage": 100.0,
                        "boundary_mode": "fixed_sliding",
                        "gravity": False,
                        "material_model": HIGH_TEMPERATURE_MATERIAL,
                    }),
                    ("fixed_sliding_high_temperature_gravity_100V", {
                        "voltage": 100.0,
                        "boundary_mode": "fixed_sliding",
                        "gravity": True,
                        "material_model": HIGH_TEMPERATURE_MATERIAL,
                    }),
                    ("fixed_sliding_room_temperature_no_gravity", {
                        "voltage": case["voltage_V"],
                        "boundary_mode": "fixed_sliding",
                        "gravity": False,
                        "material_model": ROOM_TEMPERATURE_MATERIAL,
                    }),
                ):
                    if key in analyses:
                        print(f"  D5 mechanics sensitivity {key} resumed from "
                              "checkpoint", flush=True)
                        continue
                    print(f"  D5 mechanics sensitivity {key} started", flush=True)
                    sensitivity_started = time.time()
                    result = verifier.run_case(**kwargs)
                    result["elapsed_sec"] = round(
                        time.time() - sensitivity_started, 1)
                    validate_analysis(key, result)
                    analyses[key] = result
                    persist()
                    print(f"  D5 mechanics sensitivity {key} completed in "
                          f"{result['elapsed_sec']:.1f}s", flush=True)

            metrics = zigzag_manufacturability(
                runner, case["N_RUNS"], case["L_RUN_m"], case["z_first_m"])
            geometry = close_geometry_contract(
                metrics, verifier.geometry_snapshot(),
                analyses["fixed_fixed_high_temperature_no_gravity"]["volume_m3"],
                runner.V0, runner.vol_tol)
            if not geometry["competitionGeometryPass"]:
                raise RuntimeError(f"{case_id} failed the D5 geometry contract")
            outputs[case_id]["geometry"] = geometry
            outputs[case_id]["checkpointComplete"] = True
            write_checkpoint(
                checkpoint_path, "zigzag", scope, outputs)
        return {
            "family": "zigzag",
            "scope": scope,
            "status": "D5_SCREEN_COMPLETE",
            "competitionScope": {
                "stressComparisonRequired": True,
                "thermalGeometryDistortionIgnored": True,
                "mechanicsOutsideOptimizationLoop": True,
                "strengthCertificationClaimed": False,
                "gravityAndBucklingAreOptionalEngineeringScreens": True,
                "competitionStrengthThresholdSpecified": False,
            },
            "cases": outputs,
            "screenSummary": screen_summary(outputs),
            "sameBoundaryComparison": comparison(
                outputs["d4_reference"]["analyses"],
                outputs["historical_c10_pressure_test"]["analyses"]),
            "sensitivity": zigzag_sensitivity(
                outputs["historical_c10_pressure_test"]["analyses"]),
            "lastInvocationElapsed_sec": round(time.time() - started, 1),
        }
    finally:
        runner.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("cylinder", "zigzag"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--scope", choices=("required", "full"), default="full")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    payload = run_cylinder(
        args.scope, args.output, args.resume) if args.family == "cylinder" else (
            run_zigzag(args.scope, args.output, args.resume))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
