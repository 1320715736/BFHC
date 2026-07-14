"""Exercise the D2 lifecycle state machine with a short exact time cap."""

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
CAP_H = 0.01


def load_runner(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMSOLRunner


CylinderRunner = load_runner(
    "d2_lifecycle_cylinder",
    ROOT / "cylinder_family" / "ML" / "comsol_runner.py")
ZigzagRunner = load_runner(
    "d2_lifecycle_zigzag",
    ROOT / "zigzag_family" / "ML" / "zigzag_runner.py")


def validate(result, family):
    expected = {
        "status": "CENSORED_LIFETIME_CAP",
        "capLimited": True,
        "stepLimited": False,
        "censored": True,
        "failureReached": False,
        "lifetimeExact": False,
        "terminationReason": "lifetime_cap",
        "lifecycleVersion": "lifecycle_v2",
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(
                f"{family}.{key}: expected {value!r}, got {result.get(key)!r}")
    if not math.isclose(
            float(result["lifetimeH"]), CAP_H,
            rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(
            f"{family}.lifetimeH did not stop exactly at {CAP_H} h")
    for key in (
            "lifeTotalP03gross_J", "lifeTotalP03escape_J",
            "lifeAvgP03gross_W", "lifeAvgP03escape_W"):
        value = float(result[key])
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"{family}.{key} is invalid: {value}")
    if result["lifeTotalP03escape_J"] > result["lifeTotalP03gross_J"]:
        raise RuntimeError(f"{family}: escape energy exceeds gross energy")


def run_cylinder():
    radii = [value * 1.0e-3 for value in (
        1.9063832640319742, 1.9423869762989763,
        3.6474884452463625, 2.0709089131869587,
        2.0709089131869587, 3.6474884452463625,
        1.9423869762989763, 1.9063832640319742)]
    runner = CylinderRunner()
    runner.max_lifetime_h = CAP_H
    runner.max_erosion_steps = 3
    runner.start()
    try:
        result = runner.evaluate(radii, voltage_override=1.171875)
        validate(result, "cylinder")
        return result
    finally:
        runner.stop()


def run_zigzag():
    runner = ZigzagRunner()
    runner.max_lifetime_h = CAP_H
    runner.max_erosion_steps = 3
    runner.start()
    try:
        result = runner.evaluate(
            8, 104.0e-3, 0.8e-3, voltage_override=90.0)
        validate(result, "zigzag")
        return result
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
                "D2 lifecycle smoke failed: " + ", ".join(failures))
        return

    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "family": args.family,
        "cap_h": CAP_H,
        "status": "RUNNING",
    }
    started = time.time()
    try:
        result = run_cylinder() if args.family == "cylinder" else run_zigzag()
        payload["result"] = result
        payload["status"] = "OK"
    except Exception as exc:
        traceback.print_exc()
        payload["status"] = "FAIL"
        payload["failure"] = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        payload["elapsed_sec"] = round(time.time() - started, 1)
        output = (ROOT / f"{args.family}_family" / "ML" / "data"
                  / "d2_lifecycle_smoke.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
