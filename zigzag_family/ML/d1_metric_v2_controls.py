"""D1 radiation metric-v2 validation and control reruns for zigzags."""

import argparse
import csv
import json
import platform
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import mph

from zigzag_runner import COMSOLRunner


ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data"
OUTPUT_PATH = DATA_DIR / "d1_metric_v2_controls.csv"

CASES = {
    "B3": {
        "case_id": "D1_B3_reference_zigzag",
        "role": "reference_baseline_control",
        "N_RUNS": 8,
        "L_RUN_mm": 104.0,
        "z_first_mm": 0.8,
        "max_lifetime_h": 200.0,
        "max_erosion_steps": 50,
    },
    "C10": {
        "case_id": "D1_C10_zigzag_trial19",
        "role": "legacy_lead_control",
        "N_RUNS": 12,
        "L_RUN_mm": 92.01113160151677,
        "z_first_mm": 2.266631002891225,
        "max_lifetime_h": 1000.0,
        "max_erosion_steps": 150,
    },
}

FIELDS = [
    "case_id", "family", "role", "runMode", "timestamp", "N_RUNS",
    "L_RUN_mm", "z_first_mm", "side_mm", "pathLength_mm",
    "maxLifetimeCap_h", "maxErosionSteps", "physicsVersion",
    "metricVersion", "geometryVersion", "radiationEscapeMethod",
    "spectralSplit_um", "thermalAmbient_K", "scoreAmbientTarget_K",
    "gitCommit", "gitDirty", "pythonVersion", "mphVersion", "Vwork_V",
    "initialTmax_K", "Tmin_K", "Tmean_K", "U_pct",
    "maxErosionTmax_K", "lifetimeH", "initialP03gross_W",
    "initialP03escape_W", "initialP03sphere_W", "initialPradGross_W",
    "initialPradEscape_W", "initialPradSphere_W",
    "initialP03selfAbsorbed_W", "initialSelfViewLossRaw_pct",
    "initialSelfViewLoss_pct", "initialRadiationNumericalExcess_pct",
    "initialP03ambient_W", "initialAmbient03ToEscape_pct",
    "initialFambAreaAvg", "lifeAvgP03gross_W", "lifeAvgP03escape_W",
    "lifeAvgP03sphere_W", "lifeAvgPradGross_W", "lifeAvgPradEscape_W",
    "lifeAvgPradSphere_W", "lifeTotalP03gross_J",
    "lifeTotalP03escape_J", "lifeTotalP03sphere_J",
    "lifeTotalP03selfAbsorbed_J", "selfViewLossRaw_pct",
    "selfViewLoss_pct", "radiationNumericalExcess_pct", "failureReached",
    "capLimited", "stepLimited", "erosionSteps", "erosionSolveRetries",
    "voltagePolicy",
    "voltageObjective", "runnerStatus", "status", "failure", "elapsed_sec",
]


def ensure_output_schema():
    """Migrate the pre-retry D1 CSV without losing completed rows."""
    if not OUTPUT_PATH.exists():
        return
    with OUTPUT_PATH.open(newline="", encoding="utf-8") as handle:
        records = list(csv.reader(handle))
    if not records or records[0] == FIELDS:
        return

    legacy_fields = [
        field for field in FIELDS if field != "erosionSolveRetries"
    ]
    if records[0] != legacy_fields:
        raise RuntimeError(
            f"Unsupported D1 CSV schema in {OUTPUT_PATH}: {records[0]}")

    normalized = []
    for line_number, values in enumerate(records[1:], start=2):
        if len(values) == len(legacy_fields):
            source_fields = legacy_fields
        elif len(values) == len(FIELDS):
            # These rows were appended by the retry-aware writer before the
            # existing legacy header was migrated.
            source_fields = FIELDS
        else:
            raise RuntimeError(
                f"Unexpected column count at {OUTPUT_PATH}:{line_number}: "
                f"{len(values)}")
        normalized.append(dict(zip(source_fields, values)))

    temporary = OUTPUT_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
    temporary.replace(OUTPUT_PATH)
    print(f"Migrated D1 CSV schema: {OUTPUT_PATH}")


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ML_DIR, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def completed_keys():
    ensure_output_schema()
    if not OUTPUT_PATH.exists():
        return set()
    with OUTPUT_PATH.open(newline="", encoding="utf-8") as handle:
        return {
            (row.get("case_id"), row.get("runMode"))
            for row in csv.DictReader(handle)
            if row.get("status") in {"OK", "FAIL_OVERTEMP_DURING_EROSION"}
        }


def append_row(row):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_output_schema()
    exists = OUTPUT_PATH.exists()
    with OUTPUT_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def stationary_result(runner, case):
    runner._init_model(
        case["N_RUNS"], case["L_RUN_mm"] * 1e-3,
        case["z_first_mm"] * 1e-3)
    result = runner._solve_prepared(100.0)
    return {
        "Vwork_V": result.get("applied_V"),
        "initialTmax_K": result.get("Tmax"),
        "Tmin_K": result.get("Tmin"),
        "Tmean_K": result.get("Tmean"),
        "U_pct": result.get("U_pct"),
        "initialP03gross_W": result.get("P03gross"),
        "initialP03escape_W": result.get("P03escape"),
        "initialP03sphere_W": result.get("P03sphere"),
        "initialPradGross_W": result.get("PradGross"),
        "initialPradEscape_W": result.get("PradEscape"),
        "initialPradSphere_W": result.get("PradSphere"),
        "initialP03selfAbsorbed_W": result.get("P03selfAbsorbed"),
        "initialSelfViewLossRaw_pct": result.get("selfViewLossRaw_pct"),
        "initialSelfViewLoss_pct": result.get("selfViewLoss_pct"),
        "initialRadiationNumericalExcess_pct":
            result.get("radiationNumericalExcess_pct"),
        "initialP03ambient_W": result.get("P03ambient"),
        "initialAmbient03ToEscape_pct": result.get("ambient03ToEscape_pct"),
        "initialFambAreaAvg": result.get("FambAreaAvg"),
        "runnerStatus": "OK" if result.get("solve_ok") else "FAIL",
        "status": "OK" if result.get("solve_ok") else "FAIL_STATIONARY",
        "failure": result.get("failure", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["B3", "C10", "all"],
                        default="all")
    parser.add_argument("--mode", choices=["stationary", "lifecycle"],
                        default="stationary")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    requested = list(CASES) if args.case == "all" else [args.case]
    done = completed_keys()
    runner = COMSOLRunner()
    runner.start()
    try:
        for key in requested:
            case = CASES[key]
            resume_key = (case["case_id"], args.mode)
            if resume_key in done and not args.force:
                print(f"SKIP completed {case['case_id']} {args.mode}")
                continue

            side, _, path_length = runner.compute_side_and_blocks(
                case["N_RUNS"], case["L_RUN_mm"] * 1e-3,
                case["z_first_mm"] * 1e-3)
            started = time.time()
            base = {
                "case_id": case["case_id"],
                "family": "zigzag",
                "role": case["role"],
                "runMode": args.mode,
                "timestamp": datetime.now().astimezone().isoformat(),
                "N_RUNS": case["N_RUNS"],
                "L_RUN_mm": case["L_RUN_mm"],
                "z_first_mm": case["z_first_mm"],
                "side_mm": side * 1e3,
                "pathLength_mm": path_length * 1e3,
                "maxLifetimeCap_h": case["max_lifetime_h"],
                "maxErosionSteps": case["max_erosion_steps"],
                "physicsVersion": "thermal_s2s_v2",
                "metricVersion": runner.metric_version,
                "geometryVersion": "zigzag_blocks_v1",
                "radiationEscapeMethod": runner.radiation_escape_method,
                "spectralSplit_um": runner.spectral_split_um,
                "thermalAmbient_K": runner.thermal_ambient_K,
                "scoreAmbientTarget_K": runner.score_ambient_target_K,
                "gitCommit": git_value("rev-parse", "--short", "HEAD"),
                "gitDirty": bool(git_value("status", "--porcelain")),
                "pythonVersion": platform.python_version(),
                "mphVersion": mph.__version__,
            }
            try:
                runner.max_lifetime_h = case["max_lifetime_h"]
                runner.max_erosion_steps = case["max_erosion_steps"]
                if args.mode == "stationary":
                    result = stationary_result(runner, case)
                else:
                    result = runner.evaluate(
                        N_RUNS=case["N_RUNS"],
                        L_RUN_m=case["L_RUN_mm"] * 1e-3,
                        z_first_m=case["z_first_mm"] * 1e-3,
                    )
                    result["runnerStatus"] = result.get("status", "UNKNOWN")
                base.update(result)
            except Exception as exc:
                traceback.print_exc()
                base.update({
                    "runnerStatus": "EXCEPTION",
                    "status": "EXCEPTION",
                    "failure": str(exc),
                })
            base["elapsed_sec"] = round(time.time() - started, 1)
            append_row(base)
            print(json.dumps(base, ensure_ascii=False, default=str))
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
