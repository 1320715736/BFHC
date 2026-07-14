"""C8 high-lifetime-cap verification for zigzag trial 19.

This script re-evaluates the C7 lead zigzag geometry with a higher
``runner.max_lifetime_h``. It is a fixed-geometry verification run, not a new
Optuna search.
"""

import csv
import math
import os
import time
from pathlib import Path

from zigzag_runner import COMSOLRunner


ML_DIR = Path(r"D:\VScode\project\BFHC\zigzag_family\ML")
DATA_DIR = ML_DIR / "data"
CSV_PATH = DATA_DIR / "rematch_c8_trial19_verify.csv"

BASELINE_LIFETIME_H = 242.07911958397654
BASELINE_E_J = 105597676.11285222
BASELINE_U_PCT = 148.69276459515976
MAX_TEMP_K = 3273.15

CYLINDER_LEAD_E_J = 243679111.41385093
CYLINDER_LEAD_LIFETIME_H = 378.33572614862254
CYLINDER_LEAD_U_PCT = 127.03670684533598

TRIAL_ID = 19
N_RUNS = int(os.getenv("BFHC_C8_N_RUNS", "12"))
L_RUN_MM = float(os.getenv("BFHC_C8_L_RUN_MM", "89.09934228871522"))
Z_FIRST_MM = float(os.getenv("BFHC_C8_Z_FIRST_MM", "2.10495899560028"))
MAX_LIFETIME_H = float(os.getenv("BFHC_MAX_LIFETIME_H", "500.0"))


CSV_HEADER = [
    "case_id", "source_trial", "N_RUNS", "L_RUN_mm", "z_first_mm",
    "side_mm", "pathLength_mm", "maxLifetimeCap_h", "Vwork_V",
    "initialTmax_K", "Tmin_K", "Tmean_K", "U_pct",
    "maxErosionTmax_K", "overtemp_margin_K", "lifetimeH",
    "R_L_pct", "eta_L_pct", "initialP03sphere_W",
    "initialPradSphere_W", "lifeAvgP03sphere_W",
    "lifeAvgPradSphere_W", "lifeTotalP03sphere_J", "eta_E_pct",
    "objectiveScore", "uPenalty_pctpt", "U_limit_pct",
    "selfViewLoss_pct", "failureReached", "capLimited",
    "erosionSteps", "overtempStep", "overtempTimeH", "overtempTmax_K",
    "runnerStatus", "status", "voltagePolicy", "voltageObjective",
    "voltageCandidateCount", "voltageMaxSafe_V", "voltageScanSummary",
    "E_gain_pct_vs_cylinder_trial68",
    "life_ratio_pct_vs_cylinder_trial68",
    "U_reduction_pct_vs_cylinder_trial68",
    "elapsed_sec",
]


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def pct_change(value, baseline):
    return (value - baseline) / baseline * 100.0


def relative_reduction(value, baseline):
    return (baseline - value) / baseline * 100.0


def compute_scores(row):
    lifetime_h = float(row.get("lifetimeH", float("nan")))
    energy_j = float(row.get("lifeTotalP03sphere_J", float("nan")))
    u_pct = float(row.get("U_pct", float("nan")))

    r_l_pct = lifetime_h / BASELINE_LIFETIME_H * 100.0
    eta_l_pct = (lifetime_h - BASELINE_LIFETIME_H) / BASELINE_LIFETIME_H * 100.0
    eta_e_pct = (energy_j - BASELINE_E_J) / BASELINE_E_J * 100.0
    u_penalty = max(0.0, u_pct - BASELINE_U_PCT)
    score = eta_e_pct - 0.25 * u_penalty

    row.update({
        "R_L_pct": r_l_pct,
        "eta_L_pct": eta_l_pct,
        "eta_E_pct": eta_e_pct,
        "objectiveScore": score,
        "uPenalty_pctpt": u_penalty,
        "U_limit_pct": BASELINE_U_PCT * 1.20,
        "overtemp_margin_K": MAX_TEMP_K - float(row.get("maxErosionTmax_K")),
        "E_gain_pct_vs_cylinder_trial68": pct_change(
            energy_j, CYLINDER_LEAD_E_J),
        "life_ratio_pct_vs_cylinder_trial68": (
            lifetime_h / CYLINDER_LEAD_LIFETIME_H * 100.0),
        "U_reduction_pct_vs_cylinder_trial68": relative_reduction(
            u_pct, CYLINDER_LEAD_U_PCT),
    })
    return row


def write_row(row):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_HEADER})


def compact_status(status, limit=240):
    return " ".join(str(status).split())[:limit]


def main():
    runner = COMSOLRunner()
    t_start = time.time()

    side, _, path_len = runner.compute_side_and_blocks(
        N_RUNS, L_RUN_MM * 1e-3, Z_FIRST_MM * 1e-3)

    base_row = {
        "case_id": "C8_trial19_verify",
        "source_trial": TRIAL_ID,
        "N_RUNS": N_RUNS,
        "L_RUN_mm": L_RUN_MM,
        "z_first_mm": Z_FIRST_MM,
        "side_mm": side * 1e3,
        "pathLength_mm": path_len * 1e3,
        "maxLifetimeCap_h": MAX_LIFETIME_H,
        "status": "RUNNING",
        "runnerStatus": "RUNNING",
    }
    write_row(base_row)

    print("BFHC zigzag C8 trial 19 verification")
    print(f"CSV output:       {CSV_PATH}")
    print(f"Geometry:         N={N_RUNS}, L={L_RUN_MM:.4f} mm, "
          f"z_first={Z_FIRST_MM:.4f} mm")
    print(f"Computed side:    {side * 1e3:.4f} mm")
    print(f"Path length:      {path_len * 1e3:.4f} mm")
    print(f"Lifetime cap:     {MAX_LIFETIME_H:.1f} h")
    print()

    try:
        runner.start()
        runner.max_lifetime_h = MAX_LIFETIME_H
        result = runner.evaluate(
            N_RUNS=N_RUNS,
            L_RUN_m=L_RUN_MM * 1e-3,
            z_first_m=Z_FIRST_MM * 1e-3,
            voltage_policy="max_safe",
            electrode_boundary_mode="fixed_temperature",
        )
        elapsed = time.time() - t_start
        row = dict(base_row)
        row.update(result)
        row["runnerStatus"] = compact_status(result.get("status", "UNKNOWN"))
        row["status"] = row["runnerStatus"]
        row["elapsed_sec"] = round(elapsed, 1)

        if result.get("status") == "OK":
            required = [
                "lifetimeH", "lifeTotalP03sphere_J", "U_pct",
                "Vwork_V", "maxErosionTmax_K",
            ]
            invalid = [
                key for key in required
                if not finite_number(result.get(key))
            ]
            if invalid:
                row["status"] = "INVALID_METRIC"
                row["runnerStatus"] = "INVALID_METRIC"
            else:
                row = compute_scores(row)
                row["capLimited"] = (
                    result.get("failureReached") is False
                    and float(row["lifetimeH"]) >= MAX_LIFETIME_H - 1e-6
                )

        write_row(row)

        print("\n" + "=" * 60)
        print("  C8 VERIFY COMPLETE")
        print("=" * 60)
        print(f"Status:          {row.get('status')}")
        print(f"Lifetime:        {row.get('lifetimeH', '')} h")
        print(f"failureReached:  {row.get('failureReached', '')}")
        print(f"capLimited:      {row.get('capLimited', '')}")
        print(f"eta_E:           {row.get('eta_E_pct', '')}%")
        print(f"R_L:             {row.get('R_L_pct', '')}%")
        print(f"U:               {row.get('U_pct', '')}%")
    except Exception as exc:
        elapsed = time.time() - t_start
        row = dict(base_row)
        row.update({
            "runnerStatus": "EXCEPTION",
            "status": compact_status(f"ERROR: {exc}"),
            "elapsed_sec": round(elapsed, 1),
        })
        write_row(row)
        raise
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
