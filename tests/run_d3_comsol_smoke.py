"""Run D3 steady temperature/boundary checks in a real COMSOL process."""

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
    "d3_smoke_cylinder_runner",
    ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d3_smoke_zigzag_runner",
    ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")


TEMPERATURE_FIELDS = (
    "TmaxAll_K", "TminAll_K", "TmeanAll_K", "UAll_pct",
    "TmaxActive_K", "TminActive_K", "TmeanActive_K", "UActive_pct",
    "activeVolumeFraction", "TmaxFreeSurface_K", "TminFreeSurface_K",
    "TmeanFreeSurface_K", "UFreeSurface_pct", "freeSurfaceArea_m2",
    "electrodeTemperatureUndershoot_K",
)


def finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def validate(result):
    if not result.get("solve_ok", False):
        raise RuntimeError(result.get("failure", "steady solve failed"))
    invalid = [field for field in TEMPERATURE_FIELDS
               if not finite(result.get(field))]
    if invalid:
        raise RuntimeError("non-finite D3 fields: " + ", ".join(invalid))
    if result.get("temperatureFallbackUsed") is not False:
        raise RuntimeError("formal temperature fallback was used")
    if not 0.0 < result["activeVolumeFraction"] <= 1.0 + 1.0e-6:
        raise RuntimeError("invalid active-volume fraction")
    expected_u = ((result["TmaxAll_K"] - result["TminAll_K"])
                  / result["TmeanAll_K"] * 100.0)
    if not math.isclose(
            result["UAll_pct"], expected_u, rel_tol=1.0e-10,
            abs_tol=1.0e-10):
        raise RuntimeError("formal U does not match the official formula")
    for alias, explicit in (
            ("Tmax", "TmaxAll_K"),
            ("Tmin", "TminAll_K"),
            ("Tmean", "TmeanAll_K"),
            ("U_pct", "UAll_pct")):
        if not math.isclose(
                result[alias], result[explicit],
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise RuntimeError(f"temperature alias mismatch: {alias}")


def extract(runner, result, elapsed_s):
    annotated = runner._annotate_voltage_result(
        dict(result), "fixed", runner.voltage_objective)
    fields = (
        "applied_V", "Tmax", "Tmin", "Tmean", "U_pct",
        *TEMPERATURE_FIELDS,
        "temperatureFallbackUsed", "temperatureStatisticVersion",
        "temperaturePrimaryDomain", "activeTemperatureTrim_mm",
        "electrodeBoundaryMode", "electrodeBoundaryVersion",
        "electrodeBoundaryApproximation", "electrodeTemperature_K",
        "copperThermalConductivity_W_mK", "electrodeContactRadiusIn_mm",
        "electrodeContactRadiusOut_mm", "electrodeSpreadingHIn_W_m2K",
        "electrodeSpreadingHOut_W_m2K", "physicsVersion",
    )
    summary = {field: annotated.get(field) for field in fields}
    summary["solve_ok"] = result["solve_ok"]
    summary["elapsed_sec"] = round(elapsed_s, 1)
    summary["heatBoundaryFeatureTags"] = [
        str(tag) for tag in
        runner.j.component("comp1").physics("ht").feature().tags()
    ]
    return summary


def solve_cylinder(runner, mode):
    radii = [2.5e-3] * runner.seg_count
    runner._active_electrode_boundary_mode = mode
    runner._r0 = max(radii)
    runner._initial_radii = list(radii)
    runner._fail_radii = [
        radius * (1.0 - runner.failure_fraction) for radius in radii]
    runner._init_model(radii)
    return runner._solve_prepared(radii, 1.0)


def solve_zigzag(runner, mode):
    runner._active_electrode_boundary_mode = mode
    runner._init_model(8, 104.0e-3, 0.8e-3)
    return runner._solve_prepared(90.0)


def run_family(family):
    runner = CylinderRunner() if family == "cylinder" else ZigzagRunner()
    solve = solve_cylinder if family == "cylinder" else solve_zigzag
    cases = {}
    runner.start()
    try:
        for mode in (
                "fixed_temperature", "semi_infinite_copper_spreading"):
            started = time.time()
            result = solve(runner, mode)
            validate(result)
            cases[mode] = extract(runner, result, time.time() - started)
    finally:
        runner.stop()

    fixed = cases["fixed_temperature"]
    spreading = cases["semi_infinite_copper_spreading"]
    return {
        "family": family,
        "status": "PASS",
        "cases": cases,
        "sensitivity": {
            "deltaTmax_K": (
                spreading["TmaxAll_K"] - fixed["TmaxAll_K"]),
            "deltaTmin_K": (
                spreading["TminAll_K"] - fixed["TminAll_K"]),
            "deltaTmean_K": (
                spreading["TmeanAll_K"] - fixed["TmeanAll_K"]),
            "deltaU_pctpt": (
                spreading["UAll_pct"] - fixed["UAll_pct"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("cylinder", "zigzag"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = run_family(args.family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
