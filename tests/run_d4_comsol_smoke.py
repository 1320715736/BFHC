"""Run D4 transient, material, and latent-heat checks in real COMSOL."""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMSOLRunner


CylinderRunner = load_runner(
    "d4_smoke_cylinder_runner",
    ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d4_smoke_zigzag_runner",
    ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")


MATERIAL_CASES = {
    "nominal": {
        "material_model": "nist_reference_d4_v1",
        "rhoe_scale": 1.0,
        "k_scale": 1.0,
        "cp_scale": 1.0,
    },
    "hot_corner": {
        "material_model": "nist_reference_d4_v1",
        "rhoe_scale": 0.99,
        "k_scale": 0.90,
        "cp_scale": 0.97,
    },
    "cool_corner": {
        "material_model": "nist_reference_d4_v1",
        "rhoe_scale": 1.01,
        "k_scale": 1.10,
        "cp_scale": 1.03,
    },
}


def finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def configure_case(runner, case):
    runner.configure_material_model(**MATERIAL_CASES[case])
    runner.configure_sublimation_heat(True, scale=1.0)


def initialize_family(family, runner, duration_s, voltage_ramp_s):
    runner._active_electrode_boundary_mode = "fixed_temperature"
    if family == "cylinder":
        radii = [2.5e-3] * runner.seg_count
        runner._r0 = max(radii)
        runner._initial_radii = list(radii)
        runner._fail_radii = [
            radius * (1.0 - runner.failure_fraction) for radius in radii]
        runner._init_model(radii)
        return {
            "geometry": {
                "description": "uniform_official_cylinder",
                "radii_mm": [2.5] * runner.seg_count,
            },
            "voltage_V": 1.0,
            "startup": lambda: runner._run_startup_transient_prepared(
                radii, 1.0, duration_s=duration_s,
                voltage_ramp_s=voltage_ramp_s),
            "steady": lambda: runner._solve_prepared(radii, 1.0),
        }

    runner._init_model(8, 104.0e-3, 0.8e-3)
    return {
        "geometry": {
            "description": "d3_zigzag_control",
            "N_RUNS": 8,
            "L_RUN_mm": 104.0,
            "z_first_mm": 0.8,
        },
        "voltage_V": 90.0,
        "startup": lambda: runner._run_startup_transient_prepared(
            90.0, duration_s=duration_s,
            voltage_ramp_s=voltage_ramp_s),
        "steady": lambda: runner._solve_prepared(90.0),
    }


def validate_startup(case, result):
    required = (
        "transientPeakTmax_K", "steadyReferenceTmax_K",
        "startupOvershoot_K", "startupSettlingTime_s",
        "startupTemperatureMargin_K", "startupDuration_s",
        "steadyReferenceSublimationHeat_W",
        "steadyReferenceSublimationHeatToElectric_pct",
        "steadyReferenceSublimationHeatToGrossRadiation_pct",
    )
    missing = [field for field in required if not finite(result.get(field))]
    if missing:
        raise RuntimeError(
            f"{case} has non-finite D4 fields: " + ", ".join(missing))
    if not result.get("startupSettled", False):
        raise RuntimeError(f"{case} did not settle in the simulated window")
    if not result.get("startupTemperatureOK", False):
        raise RuntimeError(
            f"{case} exceeded the startup temperature limit: "
            f"peak={result.get('transientPeakTmax_K')} K")
    lengths = {
        len(result.get("startupTimes_s", [])),
        len(result.get("startupTmaxSeries_K", [])),
        len(result.get("startupTmeanSeries_K", [])),
        len(result.get("startupSublimationHeatSeries_W", [])),
    }
    if len(lengths) != 1 or 0 in lengths:
        raise RuntimeError(f"{case} transient arrays are inconsistent")


def summarize_steady(result):
    if not result.get("solve_ok", False):
        raise RuntimeError(result.get("failure", "steady solve failed"))
    fields = (
        "applied_V", "TmaxAll_K", "TminAll_K", "TmeanAll_K", "UAll_pct",
        "Pelec", "P03escape", "PradGross", "sublimationMassRate_kg_s",
        "sublimationHeat_W", "maxSublimationHeatFlux_W_m2",
        "sublimationHeatToElectric_pct",
        "sublimationHeatToGrossRadiation_pct",
    )
    summary = {field: result.get(field) for field in fields}
    invalid = [field for field, value in summary.items() if not finite(value)]
    if invalid:
        raise RuntimeError(
            "steady result has non-finite fields: " + ", ".join(invalid))
    return summary


def run(family, scope, duration_s=10.0, voltage_ramp_s=0.5):
    runner = CylinderRunner() if family == "cylinder" else ZigzagRunner()
    runner.start()
    started = time.time()
    try:
        configure_case(runner, "nominal")
        setup = initialize_family(
            family, runner, duration_s, voltage_ramp_s)

        startup_cases = {}
        cases = ("nominal",) if scope == "nominal" else (
            "nominal", "hot_corner", "cool_corner")
        for case in cases:
            print(f"D4 {family}: startup case {case} started", flush=True)
            configure_case(runner, case)
            case_started = time.time()
            result = setup["startup"]()
            validate_startup(case, result)
            result["elapsed_sec"] = round(time.time() - case_started, 1)
            startup_cases[case] = result
            print(
                f"D4 {family}: startup case {case} completed in "
                f"{result['elapsed_sec']:.1f}s",
                flush=True)

        steady_cases = {}
        if scope == "full":
            print(f"D4 {family}: steady sensitivity cases started", flush=True)
            configure_case(runner, "nominal")
            steady_nominal = summarize_steady(setup["steady"]())
            steady_cases["nominal_latent_on"] = steady_nominal

            runner.configure_material_model("legacy_v1")
            runner.configure_sublimation_heat(True)
            steady_cases["legacy_material_latent_on"] = summarize_steady(
                setup["steady"]())

            configure_case(runner, "nominal")
            runner.configure_sublimation_heat(False)
            steady_no_latent = summarize_steady(setup["steady"]())
            steady_cases["nominal_latent_off"] = steady_no_latent
            runner.configure_sublimation_heat(True)
            print(f"D4 {family}: steady sensitivity cases completed",
                  flush=True)

        heat_tags = [str(tag) for tag in
                     runner.j.component("comp1").physics("ht").feature().tags()]
        nominal = startup_cases["nominal"]
        payload = {
            "family": family,
            "status": "PASS",
            "scope": scope,
            "geometry": setup["geometry"],
            "voltage_V": setup["voltage_V"],
            "startupCases": startup_cases,
            "steadyCases": steady_cases,
            "nominalAcceptance": {
                "temperatureOK": nominal["startupTemperatureOK"],
                "settled": nominal["startupSettled"],
                "peakTmax_K": nominal["transientPeakTmax_K"],
                "temperatureMargin_K": nominal[
                    "startupTemperatureMargin_K"],
            },
            "robustAcceptance": {
                "allCasesTemperatureOK": all(
                    result["startupTemperatureOK"]
                    for result in startup_cases.values()),
                "allCasesSettled": all(
                    result["startupSettled"]
                    for result in startup_cases.values()),
                "worstCasePeakTmax_K": max(
                    result["transientPeakTmax_K"]
                    for result in startup_cases.values()),
                "minimumTemperatureMargin_K": min(
                    result["startupTemperatureMargin_K"]
                    for result in startup_cases.values()),
            },
            "heatBoundaryFeatureTags": heat_tags,
            "elapsed_sec": round(time.time() - started, 1),
        }

        if scope == "full":
            nominal_steady = steady_cases["nominal_latent_on"]
            no_latent = steady_cases["nominal_latent_off"]
            legacy = steady_cases["legacy_material_latent_on"]
            payload["sensitivity"] = {
                "latentHeatDeltaTmax_K": (
                    nominal_steady["TmaxAll_K"] - no_latent["TmaxAll_K"]),
                "latentHeatDeltaP03escape_W": (
                    nominal_steady["P03escape"] - no_latent["P03escape"]),
                "legacyMaterialDeltaTmax_K": (
                    legacy["TmaxAll_K"] - nominal_steady["TmaxAll_K"]),
                "legacyMaterialDeltaU_pctpt": (
                    legacy["UAll_pct"] - nominal_steady["UAll_pct"]),
                "hotCornerDeltaPeakTmax_K": (
                    startup_cases["hot_corner"]["transientPeakTmax_K"]
                    - nominal["transientPeakTmax_K"]),
                "coolCornerDeltaPeakTmax_K": (
                    startup_cases["cool_corner"]["transientPeakTmax_K"]
                    - nominal["transientPeakTmax_K"]),
                "hotCornerDeltaSettlingTime_s": (
                    startup_cases["hot_corner"]["startupSettlingTime_s"]
                    - nominal["startupSettlingTime_s"]),
                "coolCornerDeltaSettlingTime_s": (
                    startup_cases["cool_corner"]["startupSettlingTime_s"]
                    - nominal["startupSettlingTime_s"]),
            }
        return payload
    finally:
        runner.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("cylinder", "zigzag"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--scope", choices=("nominal", "full"),
                        default="full")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--ramp", type=float, default=0.5)
    args = parser.parse_args()

    payload = run(
        args.family, args.scope, duration_s=args.duration,
        voltage_ramp_s=args.ramp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
