"""Verify D3 fields survive one short, fixed-voltage lifecycle evaluation."""

import argparse
import importlib.util
import json
import math
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAP_H = 0.002


def load_runner(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMSOLRunner


CylinderRunner = load_runner(
    "d3_lifecycle_cylinder",
    ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d3_lifecycle_zigzag",
    ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")


D3_FIELDS = (
    "TmaxAll_K", "TminAll_K", "TmeanAll_K", "UAll_pct",
    "TmaxActive_K", "TminActive_K", "TmeanActive_K", "UActive_pct",
    "activeVolumeFraction", "TmaxFreeSurface_K", "TminFreeSurface_K",
    "TmeanFreeSurface_K", "UFreeSurface_pct",
    "electrodeTemperatureUndershoot_K",
)


def validate(result, family):
    expected = {
        "status": "CENSORED_LIFETIME_CAP",
        "capLimited": True,
        "censored": True,
        "failureReached": False,
        "lifetimeExact": False,
        "terminationReason": "lifetime_cap",
        "voltagePolicy": "fixed",
        "temperatureFallbackUsed": False,
        "temperatureStatisticVersion": "temperature_domains_v1",
        "temperaturePrimaryDomain": "all_tungsten_volume",
        "electrodeBoundaryMode": "fixed_temperature",
        "operatingPointVersion": "rated_lifecycle_energy_v1",
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(
                f"{family}.{key}: expected {value!r}, got {result.get(key)!r}")
    if not math.isclose(
            float(result["lifetimeH"]), CAP_H,
            rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"{family}: lifecycle cap was not exact")
    invalid = [key for key in D3_FIELDS
               if not math.isfinite(float(result.get(key, float("nan"))))]
    if invalid:
        raise RuntimeError(
            f"{family}: invalid D3 lifecycle fields: {', '.join(invalid)}")
    if not math.isclose(
            result["U_pct"], result["UAll_pct"],
            rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise RuntimeError(f"{family}: primary U alias changed")


def run(family):
    runner = CylinderRunner() if family == "cylinder" else ZigzagRunner()
    runner.max_lifetime_h = CAP_H
    runner.max_erosion_steps = 2
    runner.start()
    try:
        if family == "cylinder":
            result = runner.evaluate(
                [2.5e-3] * runner.seg_count,
                voltage_override=1.0,
                electrode_boundary_mode="fixed_temperature")
        else:
            result = runner.evaluate(
                8, 104.0e-3, 0.8e-3,
                voltage_override=90.0,
                electrode_boundary_mode="fixed_temperature")
        validate(result, family)
        return result
    finally:
        runner.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("cylinder", "zigzag"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = {"family": args.family, "cap_h": CAP_H, "status": "RUNNING"}
    started = time.time()
    try:
        payload["result"] = run(args.family)
        payload["status"] = "PASS"
    except Exception as exc:
        traceback.print_exc()
        payload["status"] = "FAIL"
        payload["failure"] = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        payload["elapsed_sec"] = round(time.time() - started, 1)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
