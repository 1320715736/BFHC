"""Run D2 stationary geometry-update smoke controls against COMSOL."""

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMSOLRunner


CylinderRunner = load_runner(
    "d2_smoke_cylinder",
    ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d2_smoke_zigzag",
    ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")


def finite_summary(result):
    fields = (
        "Tmax", "Tmin", "Tmean", "U_pct", "P03gross", "P03escape",
        "PradGross", "PradEscape", "volume_m3", "expectedVolume_m3",
        "volumeLossFromInitial_pct", "geometryVolumeError_rel",
        "targetVolumeDeviation_rel", "vol_err")
    summary = {field: result.get(field) for field in fields}
    summary["solve_ok"] = result.get("solve_ok", False)
    summary["failure"] = result.get("failure", "")
    for field in fields:
        value = summary[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"non-finite smoke field: {field}={value}")
    if not summary["solve_ok"]:
        raise RuntimeError(summary["failure"] or "COMSOL solve failed")
    if summary["vol_err"] > 1.0e-4:
        raise RuntimeError(
            f"geometry volume mismatch exceeds 1e-4: {summary['vol_err']}")
    return summary


def run_cylinder():
    radii_mm = [
        1.9063832640319742, 1.9423869762989763,
        3.6474884452463625, 2.0709089131869587,
        2.0709089131869587, 3.6474884452463625,
        1.9423869762989763, 1.9063832640319742,
    ]
    radii = [value * 1.0e-3 for value in radii_mm]
    voltage = 1.171875
    runner = CylinderRunner()
    runner.start()
    try:
        runner._r0 = max(radii)
        runner._initial_radii = list(radii)
        runner._fail_radii = [
            radius * (1.0 - runner.failure_fraction) for radius in radii]
        runner._init_model(radii)
        initial_result = runner._solve_prepared(radii, voltage)
        initial = finite_summary(initial_result)
        initial_mask_ratios = initial_result.get("segMaskAreaRatio", [])
        if len(initial_mask_ratios) != runner.seg_count:
            raise RuntimeError("cylinder segment mask diagnostics are missing")
        updated_radii = [
            radius * (1.0 - 0.002 * (1 + index % 3))
            for index, radius in enumerate(radii)]
        eroded_result = runner._solve_at_voltage(updated_radii, voltage)
        eroded = finite_summary(eroded_result)
        return {
            "family": "cylinder",
            "geometryVersion": runner.geometry_version,
            "lifecycleVersion": runner.lifecycle_version,
            "erosionModel": runner.erosion_model,
            "initialShoulderArea_m2": sum(runner._shoulder_areas(radii)),
            "updatedShoulderArea_m2": sum(
                runner._shoulder_areas(updated_radii)),
            "initialSegmentMaskAreaRatios": initial_mask_ratios,
            "updatedSegmentMaskAreaRatios": eroded_result.get(
                "segMaskAreaRatio", []),
            "initial": initial,
            "locallyEroded": eroded,
        }
    finally:
        runner.stop()


def run_zigzag():
    n_runs = 8
    run_length = 104.0e-3
    z_first = 0.8e-3
    voltage = 90.0
    runner = ZigzagRunner()
    runner.start()
    try:
        runner._init_model(n_runs, run_length, z_first)
        initial = finite_summary(runner._solve_prepared(voltage))

        initial_side = runner._init_side
        exact_block_sides = [
            initial_side * (1.0 - 0.01 * (1 + index % 3))
            for index in range(runner._n_blocks)]
        block_sides = runner._project_block_sides_to_geometry(
            exact_block_sides)
        stub_radii = [runner.R0 * 0.999, runner.R0 * 0.998]
        blocks = runner.eroded_blocks(block_sides)
        envelope = runner.compute_envelope(
            blocks, stub_radii, block_sides)
        runner._rebuild(
            blocks, block_sides, stub_radii, envelope, geom_only=True)
        eroded_result = runner._solve_prepared(voltage)
        topology_refresh = "clear_solution"
        if not eroded_result.get("solve_ok", False):
            # Unequal local sides can change the union boundary topology.
            # Force COMSOL to regenerate its S2S-aware solver before treating
            # the geometry as invalid.
            runner._clear_solutions(remove=True)
            eroded_result = runner._solve_prepared(voltage)
            topology_refresh = "regenerate_solver"
        if not eroded_result.get("solve_ok", False):
            eroded_result = runner._restart_at_geometry(
                n_runs, run_length, z_first,
                blocks, block_sides, stub_radii, envelope, voltage)
            topology_refresh = "rebuild_model_from_local_geometry"
        eroded = finite_summary(eroded_result)
        stub_temperatures = eroded_result.get("stub_Tavg", [])
        block_temperatures = eroded_result.get("block_Tavg", [])
        block_areas = eroded_result.get("block_A_lat", [])
        block_mask_ratios = eroded_result.get("blockMaskAreaRatio", [])
        if len(stub_temperatures) != 2:
            raise RuntimeError("stub temperature operators did not return two values")
        if len(block_temperatures) != runner._n_blocks:
            raise RuntimeError("block temperature operators are misaligned")
        if len(block_mask_ratios) != runner._n_blocks:
            raise RuntimeError("block mask diagnostics are misaligned")
        initial_rates, _, _ = runner._block_erosion_rates(
            block_sides, block_temperatures, runner._block_lengths,
            block_areas, block_sides)
        return {
            "family": "zigzag",
            "geometryVersion": runner.geometry_version,
            "lifecycleVersion": runner.lifecycle_version,
            "erosionModel": runner.erosion_model,
            "blockCount": runner._n_blocks,
            "initialSide_mm": initial_side * 1.0e3,
            "exactStateMinSide_mm": min(exact_block_sides) * 1.0e3,
            "exactStateMaxSide_mm": max(exact_block_sides) * 1.0e3,
            "updatedMinSide_mm": min(block_sides) * 1.0e3,
            "updatedMaxSide_mm": max(block_sides) * 1.0e3,
            "geometrySideQuantum_pct": (
                100.0 * runner.geometry_side_quantum_fraction),
            "maxProjectionError_pct": 100.0 * max(
                abs(exact - represented)
                for exact, represented in zip(
                    exact_block_sides, block_sides)) / initial_side,
            "updatedStubRadii_mm": [value * 1.0e3 for value in stub_radii],
            "stubTemperatures_K": stub_temperatures,
            "blockMaskAreaRatios": block_mask_ratios,
            "maxLocalBlockLossRate_pct_per_h": (
                100.0 * max(initial_rates) * 3600.0 / initial_side),
            "topologyRefresh": topology_refresh,
            "initial": initial,
            "locallyEroded": eroded,
        }
    finally:
        runner.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=("cylinder", "zigzag", "all"), default="all")
    args = parser.parse_args()

    if args.family == "all":
        failures = []
        for family in ("cylinder", "zigzag"):
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--family", family],
                cwd=ROOT, text=True, capture_output=True)
            print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            if completed.returncode != 0:
                failures.append(family)
        if failures:
            raise SystemExit(
                "D2 COMSOL smoke failed: " + ", ".join(failures))
        return

    requested = (args.family,)
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "RUNNING",
        "controls": [],
    }
    started = time.time()
    try:
        for family in requested:
            payload["controls"].append(
                run_cylinder() if family == "cylinder" else run_zigzag())
        payload["status"] = "OK"
    except Exception as exc:
        traceback.print_exc()
        payload["status"] = "FAIL"
        payload["failure"] = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        payload["elapsed_sec"] = round(time.time() - started, 1)
        for family in requested:
            output = (ROOT / f"{family}_family" / "ML" / "data"
                      / "d2_geometry_smoke.json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
